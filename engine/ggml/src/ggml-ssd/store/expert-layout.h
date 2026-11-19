// SPDX-License-Identifier: Apache-2.0
#pragma once
// The routed experts of a model, described once at load: each routed layer's up, gate and down (type, shape and the
// bytes of one expert) and where each expert's roles lie in the model's files. Whatever reads, holds or computes
// routed experts takes these facts from here; nothing derives them from a tensor per call or from the requests it has
// seen. The C part crosses from the model to the CPU backend and from there to a GPU tier, which borrow it for the
// lifetime of the context that built it.
#include <stddef.h>
#include <stdint.h>

#define SSD_EXPERT_LAYOUT_VERSION 1
enum { SSD_EXPERT_UP = 0, SSD_EXPERT_GATE = 1, SSD_EXPERT_DOWN = 2, SSD_EXPERT_ROLES = 3 };
// A direct read starts and ends on this unit (platform::open_direct).
#define SSD_EXPERT_READ_UNIT 4096u

struct ssd_expert_role {
    int32_t type;       // ggml_type
    int32_t columns;    // ne[0]: the latent for up and gate, the hidden for down
    int32_t rows;       // ne[1]
    uint64_t row_bytes; // ggml_row_size(type, columns)
    uint64_t bytes;     // one expert: row_bytes * rows
};
// The roles of one expert. Layers may differ in it (a quantization's mix), and so may gate from up within one.
struct ssd_expert_format { struct ssd_expert_role role[SSD_EXPERT_ROLES]; };
struct ssd_expert_extent {
    int32_t fd;      // the direct-read descriptor of the role's file, borrowed from the source index
    uint64_t offset; // the role's first byte in that file
};
struct ssd_expert_layer {
    int32_t id;     // the model's layer number
    int32_t format; // into ssd_expert_layout::formats
    const struct ssd_expert_extent * extents; // experts * SSD_EXPERT_ROLES, expert after expert
};
struct ssd_expert_layout {
    uint32_t version; // SSD_EXPERT_LAYOUT_VERSION
    uint32_t size;    // sizeof(struct ssd_expert_layout)
    // every routed layer has these: the latent (up and gate columns, down rows), the hidden (up and gate rows, down
    // columns) and its expert count
    int32_t latent, hidden, experts;
    int32_t n_formats;
    const struct ssd_expert_format * formats; // in the order their first layers come
    int32_t n_layers;
    const struct ssd_expert_layer * layers;   // the routed layers in model order
    uint64_t read_bytes[SSD_EXPERT_ROLES];    // each role's largest read (ssd_expert_read) over all layers and experts
};

// The read of a role whose bytes start at `offset`: whole units from the unit that holds its first byte; the role's
// bytes start data_offset bytes into it.
struct ssd_expert_read { uint64_t offset, bytes, data_offset; };
static inline struct ssd_expert_read ssd_expert_read_of(uint64_t offset, uint64_t bytes) {
    struct ssd_expert_read r;
    r.data_offset = offset % SSD_EXPERT_READ_UNIT;
    r.offset = offset - r.data_offset;
    r.bytes = (r.data_offset + bytes + SSD_EXPERT_READ_UNIT - 1) / SSD_EXPERT_READ_UNIT * SSD_EXPERT_READ_UNIT;
    return r;
}

#ifdef __cplusplus
#include "ggml.h"
#include "source.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ssd { namespace store {
// The index of layer `id` in layout.layers, or -1 when it is not a routed layer.
inline int expert_layer_index(const ssd_expert_layout & layout, int32_t id) noexcept {
    for (int i = 0; i < layout.n_layers; ++i) { if (layout.layers[i].id == id) { return i; } }
    return -1;
}
inline bool same_format(const ssd_expert_format & a, const ssd_expert_format & b) noexcept {
    for (int r = 0; r < SSD_EXPERT_ROLES; ++r) {
        const ssd_expert_role & x = a.role[r], & y = b.role[r];
        if (x.type != y.type || x.columns != y.columns || x.rows != y.rows) { return false; }
    }
    return true;
}
// "up/down" types, "up/gate/down" where gate's type is not up's
inline std::string format_name(const ssd_expert_format & f) {
    std::string s = ggml_type_name(ggml_type(f.role[SSD_EXPERT_UP].type));
    if (f.role[SSD_EXPERT_GATE].type != f.role[SSD_EXPERT_UP].type) {
        s += std::string("/") + ggml_type_name(ggml_type(f.role[SSD_EXPERT_GATE].type));
    }
    return s + "/" + ggml_type_name(ggml_type(f.role[SSD_EXPERT_DOWN].type));
}
// Checks a borrowed layout's version and size before a reader takes it.
inline const ssd_expert_layout & checked_layout(const ssd_expert_layout * layout) {
    if (!layout || layout->version != SSD_EXPERT_LAYOUT_VERSION || layout->size != sizeof(ssd_expert_layout) ||
            layout->n_layers <= 0 || !layout->layers || layout->n_formats <= 0 || !layout->formats) {
        throw std::runtime_error("expert layout ABI mismatch");
    }
    return *layout;
}

// Owns a layout: built from the routed layers' expert tensors (metadata only) and the source index, which every
// expert's every role is resolved in once.
class expert_layout {
public:
    struct routed_layer { int32_t id; const ggml_tensor * role[SSD_EXPERT_ROLES]; };

    // layers in model order. Throws, naming the layer, when one cannot be streamed: a role that is not contiguous or
    // whose rows are not whole blocks of its type, another geometry than the first layer's, or an expert the index
    // does not hold at a direct-read descriptor.
    static expert_layout build(const std::vector<routed_layer> & layers, const source_view & source) {
        if (layers.empty()) { throw std::runtime_error("the model has no routed expert layers"); }
        if (!source.resolve) { throw std::runtime_error("routed experts need a source index"); }
        expert_layout out;
        out.layers_.reserve(layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            const routed_layer & in = layers[i];
            const std::string where = "routed layer " + std::to_string(in.id);
            if (in.id < 0 || (i > 0 && in.id <= layers[i - 1].id)) {
                throw std::runtime_error(where + ": routed layers must be listed once, in model order");
            }
            ssd_expert_format f = {};
            for (int r = 0; r < SSD_EXPERT_ROLES; ++r) {
                const ggml_tensor * t = in.role[r];
                if (!t || !ggml_is_contiguous(t) || t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] <= 0 || t->ne[3] != 1 ||
                        t->ne[0] > 16384 || t->ne[1] > 16384 || t->ne[2] > 65536 ||
                        t->type < 0 || t->type >= GGML_TYPE_COUNT || ggml_blck_size(t->type) <= 0 ||
                        t->ne[0] % ggml_blck_size(t->type)) {
                    throw std::runtime_error(where + ": its experts are not contiguous rows of whole blocks");
                }
                ssd_expert_role & role = f.role[r];
                role.type = int32_t(t->type);
                role.columns = int32_t(t->ne[0]);
                role.rows = int32_t(t->ne[1]);
                role.row_bytes = ggml_row_size(t->type, t->ne[0]);
                role.bytes = role.row_bytes * uint64_t(t->ne[1]);
                if (role.bytes != t->nb[2]) { throw std::runtime_error(where + ": an expert's bytes are not its rows"); }
            }
            const ggml_tensor * up = in.role[SSD_EXPERT_UP], * gate = in.role[SSD_EXPERT_GATE], * down = in.role[SSD_EXPERT_DOWN];
            if (gate->ne[0] != up->ne[0] || gate->ne[1] != up->ne[1] || gate->ne[2] != up->ne[2] ||
                    down->ne[0] != up->ne[1] || down->ne[1] != up->ne[0] || down->ne[2] != up->ne[2]) {
                throw std::runtime_error(where + ": up, gate and down do not form one expert's shapes");
            }
            if (i == 0) {
                out.c_.latent = int32_t(up->ne[0]); out.c_.hidden = int32_t(up->ne[1]); out.c_.experts = int32_t(up->ne[2]);
            } else if (up->ne[0] != out.c_.latent || up->ne[1] != out.c_.hidden || up->ne[2] != out.c_.experts) {
                throw std::runtime_error(where + " has another latent, hidden or expert count than routed layer " +
                    std::to_string(layers[0].id));
            }
            size_t k = 0;
            while (k < out.formats_.size() && !same_format(out.formats_[k], f)) { ++k; }
            if (k == out.formats_.size()) { out.formats_.push_back(f); }

            std::vector<ssd_expert_extent> extents(size_t(out.c_.experts) * SSD_EXPERT_ROLES);
            for (int32_t e = 0; e < out.c_.experts; ++e) {
                for (int r = 0; r < SSD_EXPERT_ROLES; ++r) {
                    const uint64_t bytes = f.role[r].bytes;
                    int fd = -1; uint64_t offset = 0; size_t logical = 0;
                    if (!source.resolve(source.userdata, in.role[r]->name, e, size_t(bytes), out.c_.experts,
                            &fd, &offset, &logical) || logical != bytes || fd < 0) {
                        throw std::runtime_error(where + ": the source index does not hold expert " + std::to_string(e) +
                            " of " + in.role[r]->name);
                    }
                    const ssd_expert_read read = ssd_expert_read_of(offset, bytes);
                    if (read.bytes > uint64_t(std::numeric_limits<int>::max()) ||
                            read.offset > uint64_t(std::numeric_limits<int64_t>::max()) - read.bytes) {
                        throw std::runtime_error(where + ": an expert's read is out of range");
                    }
                    out.c_.read_bytes[r] = std::max(out.c_.read_bytes[r], read.bytes);
                    extents[size_t(e) * SSD_EXPERT_ROLES + size_t(r)] = { int32_t(fd), offset };
                }
            }
            out.extents_.push_back(std::move(extents));
            out.layers_.push_back({ in.id, int32_t(k), nullptr });
        }
        out.publish();
        return out;
    }

    expert_layout() = default;
    expert_layout(expert_layout && other) noexcept { *this = std::move(other); }
    expert_layout & operator=(expert_layout && other) noexcept {
        formats_ = std::move(other.formats_); layers_ = std::move(other.layers_); extents_ = std::move(other.extents_);
        c_ = other.c_; publish(); other.c_ = {};
        return *this;
    }
    expert_layout(const expert_layout &) = delete;
    expert_layout & operator=(const expert_layout &) = delete;

    const ssd_expert_layout * abi() const noexcept { return layers_.empty() ? nullptr : &c_; }

private:
    void publish() noexcept {
        for (size_t i = 0; i < layers_.size(); ++i) { layers_[i].extents = extents_[i].data(); }
        c_.version = SSD_EXPERT_LAYOUT_VERSION; c_.size = uint32_t(sizeof(ssd_expert_layout));
        c_.n_formats = int32_t(formats_.size()); c_.formats = formats_.data();
        c_.n_layers = int32_t(layers_.size()); c_.layers = layers_.data();
    }
    std::vector<ssd_expert_format> formats_;
    std::vector<ssd_expert_layer> layers_;
    std::vector<std::vector<ssd_expert_extent>> extents_;
    ssd_expert_layout c_ = {};
};
} } // namespace ssd::store
#endif
