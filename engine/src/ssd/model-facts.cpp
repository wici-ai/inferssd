// SPDX-License-Identifier: Apache-2.0
// The facts of a model as the engine would load it (llama_ssd_model_facts), from its files' headers alone: a planner
// takes them before any weight is read or any backend starts. They come from the steps the load takes (the loader,
// the architecture's hyperparameters, the NextN blocks dropped, the routed experts' layout over their source index),
// so nothing outside the engine reads a GGUF again or keeps a copy of ggml's types.
#include "llama-ssd.h"

#include "llama-arch.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "ggml-ssd/store/expert-layout.h"
#if defined(LLAMA_SSD_STORE)
#include "ggml-ssd/store/index.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void quote(std::string & out, const char * text) {
    out += '"';
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
        switch (*p) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (*p < 0x20) {
                    char escaped[8];
                    std::snprintf(escaped, sizeof escaped, "\\u%04x", unsigned(*p));
                    out += escaped;
                } else {
                    out += char(*p);
                }
        }
    }
    out += '"';
}

void number(std::string & out, uint64_t value) { out += std::to_string(value); }
void number(std::string & out, int64_t value) { out += std::to_string(value); }
void real(std::string & out, double value) {
    if (!std::isfinite(value)) { out += "null"; return; }
    char text[32];
    std::snprintf(text, sizeof text, "%.17g", value);
    out += text;
}

// Element I of DATA, a scalar of TYPE (not a string).
void element(std::string & out, gguf_type type, const void * data, size_t i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   number(out, uint64_t(static_cast<const uint8_t *>(data)[i])); break;
        case GGUF_TYPE_INT8:    number(out, int64_t(static_cast<const int8_t *>(data)[i])); break;
        case GGUF_TYPE_UINT16:  number(out, uint64_t(static_cast<const uint16_t *>(data)[i])); break;
        case GGUF_TYPE_INT16:   number(out, int64_t(static_cast<const int16_t *>(data)[i])); break;
        case GGUF_TYPE_UINT32:  number(out, uint64_t(static_cast<const uint32_t *>(data)[i])); break;
        case GGUF_TYPE_INT32:   number(out, int64_t(static_cast<const int32_t *>(data)[i])); break;
        case GGUF_TYPE_UINT64:  number(out, static_cast<const uint64_t *>(data)[i]); break;
        case GGUF_TYPE_INT64:   number(out, static_cast<const int64_t *>(data)[i]); break;
        case GGUF_TYPE_FLOAT32: real(out, static_cast<const float *>(data)[i]); break;
        case GGUF_TYPE_FLOAT64: real(out, static_cast<const double *>(data)[i]); break;
        case GGUF_TYPE_BOOL:    out += static_cast<const int8_t *>(data)[i] ? "true" : "false"; break;
        default:                out += "null";
    }
}

// The model's own metadata: general.* and its architecture's keys, as the first file holds them. The tokenizer's
// tables and the split's bookkeeping stay out.
void metadata(std::string & out, const gguf_context * gguf, const std::string & arch) {
    out += '{';
    bool first = true;
    for (int64_t key = 0; key < gguf_get_n_kv(gguf); ++key) {
        const std::string name = gguf_get_key(gguf, key);
        if (name.rfind("general.", 0) != 0 && name.rfind(arch + ".", 0) != 0) { continue; }
        const gguf_type type = gguf_get_kv_type(gguf, key);
        const gguf_type items = type == GGUF_TYPE_ARRAY ? gguf_get_arr_type(gguf, key) : type;
        if (items == GGUF_TYPE_ARRAY) { continue; }
        out += first ? "" : ",";
        first = false;
        quote(out, name.c_str());
        out += ':';
        if (type == GGUF_TYPE_STRING) {
            quote(out, gguf_get_val_str(gguf, key));
        } else if (type != GGUF_TYPE_ARRAY) {
            element(out, type, gguf_get_val_data(gguf, key), 0);
        } else {
            out += '[';
            for (size_t i = 0; i < gguf_get_arr_n(gguf, key); ++i) {
                out += i ? "," : "";
                if (items == GGUF_TYPE_STRING) {
                    quote(out, gguf_get_arr_str(gguf, key, i));
                } else {
                    element(out, items, gguf_get_arr_data(gguf, key), i);
                }
            }
            out += ']';
        }
    }
    out += '}';
}

void role(std::string & out, const char * name, int32_t type, uint64_t bytes) {
    quote(out, name);
    out += ":{\"type\":";
    quote(out, ggml_type_name(ggml_type(type)));
    out += ",\"bytes\":";
    number(out, bytes);
    out += '}';
}

const char * const ROLE_NAMES[SSD_EXPERT_ROLES] = { "up", "gate", "down" };

// The routed experts: every layer whose up, gate and down are routed expert tensors, laid out as the engine lays
// them out at load (ssd::store::expert_layout), over the source index of the model's own files.
void experts(std::string & out, llama_model_loader & ml, const llama_hparams & hparams) {
    std::vector<ssd::store::expert_layout::routed_layer> routed;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        ssd::store::expert_layout::routed_layer layer = { int32_t(il), {} };
        for (int r = 0; r < SSD_EXPERT_ROLES; ++r) {
            const std::string name = "blk." + std::to_string(il) + ".ffn_" + ROLE_NAMES[r] + "_exps.weight";
            const auto * weight = ml.get_weight(name.c_str());
            layer.role[r] = weight ? weight->tensor : nullptr;
        }
        if (layer.role[SSD_EXPERT_UP] && layer.role[SSD_EXPERT_GATE] && layer.role[SSD_EXPERT_DOWN]) {
            routed.push_back(layer);
        }
    }
    if (routed.empty()) { out += "null"; return; }
#if defined(LLAMA_SSD_STORE)
    auto opened = ssd::store::source_index::open(ml.paths);
    if (!opened) { throw std::runtime_error(opened.error); }
    const auto built = ssd::store::expert_layout::build(routed, opened.value->view());
    const ssd_expert_layout & layout = *built.abi();
    // The largest expert's bytes (its three roles) and the most one expert's three role reads come to.
    uint64_t expert_bytes = 0, expert_read_bytes = 0;
    for (int i = 0; i < layout.n_layers; ++i) {
        const ssd_expert_layer & layer = layout.layers[i];
        const ssd_expert_format & format = layout.formats[layer.format];
        uint64_t bytes = 0;
        for (int r = 0; r < SSD_EXPERT_ROLES; ++r) { bytes += format.role[r].bytes; }
        expert_bytes = std::max(expert_bytes, bytes);
        for (int32_t e = 0; e < layout.experts; ++e) {
            uint64_t read = 0;
            for (int r = 0; r < SSD_EXPERT_ROLES; ++r) {
                read += ssd_expert_read_of(layer.extents[size_t(e) * SSD_EXPERT_ROLES + size_t(r)].offset,
                    format.role[r].bytes).bytes;
            }
            expert_read_bytes = std::max(expert_read_bytes, read);
        }
    }
    out += "{\"count\":";
    number(out, int64_t(layout.experts));
    out += ",\"used\":";
    number(out, int64_t(hparams.n_expert_used));
    out += ",\"latent\":";
    number(out, int64_t(layout.latent));
    out += ",\"hidden\":";
    number(out, int64_t(layout.hidden));
    out += ",\"formats\":[";
    for (int k = 0; k < layout.n_formats; ++k) {
        out += k ? ",{" : "{";
        for (int r = 0; r < SSD_EXPERT_ROLES; ++r) {
            out += r ? "," : "";
            role(out, ROLE_NAMES[r], layout.formats[k].role[r].type, layout.formats[k].role[r].bytes);
        }
        out += '}';
    }
    out += "],\"layers\":[";
    for (int i = 0; i < layout.n_layers; ++i) {
        out += i ? ",[" : "[";
        number(out, int64_t(layout.layers[i].id));
        out += ',';
        number(out, int64_t(layout.layers[i].format));
        out += ']';
    }
    out += "],\"read_bytes\":{";
    for (int r = 0; r < SSD_EXPERT_ROLES; ++r) {
        out += r ? "," : "";
        quote(out, ROLE_NAMES[r]);
        out += ':';
        number(out, layout.read_bytes[r]);
    }
    out += "},\"expert_bytes\":";
    number(out, expert_bytes);
    out += ",\"expert_read_bytes\":";
    number(out, expert_read_bytes);
    out += '}';
#else
    throw std::runtime_error("the routed experts' layout needs the SSD expert store (LLAMA_SSD_STORE=ON)");
#endif
}

std::string model_facts(const char * const * paths, size_t n_paths) {
    // as llama_model_load_from_file takes one path and llama_model_load_from_splits several
    std::vector<std::string> splits;
    if (n_paths > 1) { splits.assign(paths, paths + n_paths); }
    llama_model_loader ml(nullptr, nullptr, nullptr, paths[0], splits, nullptr, /* use_mmap */ false,
        /* use_direct_io */ false, /* check_tensors */ false, /* defer_experts */ false, /* no_alloc */ true,
        nullptr, nullptr);
    llama_model_params params = llama_model_default_params();
    params.use_mmap = false;
    params.no_alloc = true;
    llama_model model(params);
    model.load_arch(ml);
    model.load_hparams(ml);
    model.drop_nextn_blocks(ml);

    const std::string arch = llm_arch_name(model.arch);
    std::string out = "{\"version\":1,\"architecture\":";
    quote(out, arch.c_str());
    out += ",\"layers\":";
    number(out, int64_t(model.hparams.n_layer));
    out += ",\"files\":[";
    for (size_t i = 0; i < ml.paths.size(); ++i) {
        out += i ? "," : "";
        quote(out, ml.paths[i].c_str());
    }
    out += "],\"metadata\":";
    metadata(out, ml.metadata, ml.get_arch_name());
    // The tensors the engine loads, in the loader's order: each one's file (into files), first byte and bytes.
    out += ",\"tensors\":[";
    bool first = true;
    for (const auto & entry : ml.weights_map) {
        const ggml_tensor * t = entry.second.tensor;
        out += first ? "{\"name\":" : ",{\"name\":";
        first = false;
        quote(out, entry.first.c_str());
        out += ",\"type\":";
        quote(out, ggml_type_name(t->type));
        out += ",\"shape\":[";
        for (int d = 0; d < ggml_n_dims(t); ++d) {
            out += d ? "," : "";
            number(out, int64_t(t->ne[d]));
        }
        out += "],\"file\":";
        number(out, uint64_t(entry.second.idx));
        out += ",\"offset\":";
        number(out, uint64_t(entry.second.offs));
        out += ",\"bytes\":";
        number(out, uint64_t(ggml_nbytes(t)));
        out += '}';
    }
    out += "],\"experts\":";
    experts(out, ml, model.hparams);
    out += '}';
    return out;
}
} // namespace

bool llama_ssd_model_facts(const char * const * paths, size_t n_paths,
        void (*write)(const char * text, size_t size, void * user), void * user, char * error, size_t error_size) {
    if (error && error_size) { error[0] = '\0'; }
    const char * path = paths && n_paths ? paths[0] : nullptr;
    try {
        if (!path || !write || std::find(paths, paths + n_paths, nullptr) != paths + n_paths) {
            throw std::invalid_argument("model facts need a model path and a writer");
        }
        const std::string facts = model_facts(paths, n_paths);
        write(facts.data(), facts.size(), user);
        return true;
    } catch (const std::exception & exception) {
        if (error && error_size) { std::snprintf(error, error_size, "%s: %s", path ? path : "(null)", exception.what()); }
    } catch (...) {
        if (error && error_size) { std::snprintf(error, error_size, "%s: model facts failed", path ? path : "(null)"); }
    }
    return false;
}
