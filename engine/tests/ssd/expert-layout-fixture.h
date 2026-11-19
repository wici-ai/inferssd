// SPDX-License-Identifier: Apache-2.0
#pragma once
// A test's expert layout: built as the engine builds it (ssd::store::expert_layout::build) from the fixture's routed
// layers and a role resolver that gives each expert role as its read (an aligned start and whole pages, the page
// prefix its bytes start at and its bytes, or a pack's aligned entry without them), as a source index places it.

#include "../../ggml/src/ggml-ssd/gpu/source.h"
#include "../../ggml/src/ggml-ssd/store/expert-layout.h"
#include "test-resources.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_layout {
using role_resolver = bool (*)(void * user, const char * tensor, int32_t expert, ssd_cpu_sink_source * read);
using routed_layer = ssd::store::expert_layout::routed_layer;

// The layout of `layers` (in model order), every expert's roles resolved once by `resolve`. Throws as the engine's
// build does, naming what it cannot describe.
inline std::shared_ptr<const ssd::store::expert_layout> build(const std::vector<routed_layer> & layers,
        role_resolver resolve, void * user) {
    struct adapter { role_resolver resolve; void * user; } a{ resolve, user };
    ssd::store::source_view view;
    view.userdata = &a;
    view.resolve = [](void * p, const char * name, int expert, size_t bytes, int64_t, int * fd, uint64_t * offset,
            size_t * logical) noexcept {
        const auto & in = *static_cast<const adapter *>(p);
        ssd_cpu_sink_source read = {};
        if (!in.resolve(in.user, name, expert, &read)) { return false; }
        *fd = read.fd_direct;
        *offset = read.offset + read.data_offset;
        // a pack's entry names no payload: its bytes are the role's when its read is their whole pages
        const size_t whole = (bytes + SSD_EXPERT_READ_UNIT - 1) / SSD_EXPERT_READ_UNIT * SSD_EXPERT_READ_UNIT;
        *logical = read.payload_nbytes ? read.payload_nbytes : read.nbytes == whole ? bytes : read.nbytes;
        return true;
    };
    return std::make_shared<const ssd::store::expert_layout>(ssd::store::expert_layout::build(layers, view));
}

// A role a test makes fall short while a sink runs on it, as a source that changes under a running sink: the layout
// places the role at the start of a file of its own, whose descriptor the test points at a one-page copy (shrink) and
// back (restore). A read of the role then ends after its first page.
class swappable_role {
public:
    swappable_role(const void * data, size_t bytes, size_t read_bytes) {
        if (read_bytes < bytes || read_bytes <= SSD_EXPERT_READ_UNIT) { throw std::runtime_error("a swappable role spans pages"); }
        std::vector<char> padded(read_bytes, 0);
        std::copy(static_cast<const char *>(data), static_cast<const char *>(data) + bytes, padded.begin());
        path_ = write(padded.data(), padded.size());
        short_path_ = write(padded.data(), SSD_EXPERT_READ_UNIT);
        fd_ = test_resources::open_direct(path_);
        good_ = test_resources::fd_dup(fd_);
        shrunk_ = test_resources::open_direct(short_path_);
        if (good_ < 0) { throw std::runtime_error("swappable role descriptor"); }
    }
    ~swappable_role() {
        for (int fd : { fd_, good_, shrunk_ }) { if (fd >= 0) { test_resources::fd_close(fd); } }
        test_resources::remove_file(path_); test_resources::remove_file(short_path_);
    }
    swappable_role(const swappable_role &) = delete;
    swappable_role & operator=(const swappable_role &) = delete;
    int fd() const { return fd_; } // the role starts at offset 0
    void shrink() const { if (test_resources::fd_dup2(shrunk_, fd_) != fd_) { throw std::runtime_error("swappable role shrink"); } }
    void restore() const { if (test_resources::fd_dup2(good_, fd_) != fd_) { throw std::runtime_error("swappable role restore"); } }
private:
    static std::string write(const char * data, size_t bytes) {
        const std::string path = test_resources::temp_file("ssd-swappable-role");
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(data, std::streamsize(bytes));
        file.close();
        if (!file) { throw std::runtime_error("swappable role write"); }
        return path;
    }
    std::string path_, short_path_;
    int fd_ = -1, good_ = -1, shrunk_ = -1;
};
// Shrinks a swappable role for a scope.
struct shrunk_role {
    const swappable_role & role;
    explicit shrunk_role(const swappable_role & r) : role(r) { role.shrink(); }
    ~shrunk_role() { role.restore(); }
};
} // namespace test_layout
