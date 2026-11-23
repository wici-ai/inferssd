// SPDX-License-Identifier: Apache-2.0
// A cache that grows past its resident cells moves to a private file on the SSD (llama_kv_cache::set_kv_spill): the
// cells it held keep their values, its storage reads as SSD-backed KV, the cells past them grow in the file, and a cache
// trimmed back to its resident cells returns to its own memory with the same values.

#include "llama-kv-cache.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr uint32_t n_layer  = 2;
constexpr uint32_t n_head   = 2;
constexpr uint32_t head_dim = 4;
constexpr uint32_t row      = n_head * head_dim; // a cell's K (and V) values a layer
constexpr uint32_t grow     = 16;                // the cells the cache allocates at a time, and the resident ones
constexpr uint32_t size     = 64;

bool require(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); }
    return ok;
}

float value(int kind, uint32_t il, uint32_t cell, uint32_t i) { return float(kind * 100000 + il * 10000 + cell * 10 + i); }

// The K (kind 0) or V (kind 1) view of layer il over the first n cells: the cache's storage is host memory (CPU, or the
// SSD file it maps), which the test writes and reads in place.
ggml_tensor * view(ggml_context * ctx, const llama_kv_cache & kv, int kind, uint32_t il, uint32_t n) {
    const llama_kv_cache::slot_info sinfo = {0, 0, {0}, {{}}};
    return kind == 0 ? kv.get_k(ctx, int32_t(il), n, sinfo) : kv.get_v(ctx, int32_t(il), n, sinfo);
}

void write(ggml_context * ctx, const llama_kv_cache & kv, uint32_t from, uint32_t to) {
    for (int kind = 0; kind < 2; kind++) {
        for (uint32_t il = 0; il < n_layer; il++) {
            auto * data = static_cast<float *>(view(ctx, kv, kind, il, to)->data);
            for (uint32_t cell = from; cell < to; cell++) {
                for (uint32_t i = 0; i < row; i++) { data[cell * row + i] = value(kind, il, cell, i); }
            }
        }
    }
}

// whether the first n cells hold what write() put there, and whether their storage is the SSD file's (spilled)
bool check(ggml_context * ctx, const llama_kv_cache & kv, uint32_t n, bool spilled, const char * when) {
    bool ok = true;
    for (int kind = 0; kind < 2; kind++) {
        for (uint32_t il = 0; il < n_layer; il++) {
            ggml_tensor * t = view(ctx, kv, kind, il, n);
            if (ggml_backend_tensor_is_ssd_kv(t) != spilled) {
                std::fprintf(stderr, "FAIL: %s: layer %u %s storage is %sSSD-backed\n", when, il, kind ? "V" : "K",
                    spilled ? "not " : "");
                ok = false;
            }
            const auto * data = static_cast<const float *>(t->data);
            for (uint32_t cell = 0; cell < n && ok; cell++) {
                for (uint32_t i = 0; i < row; i++) {
                    if (data[cell * row + i] != value(kind, il, cell, i)) {
                        std::fprintf(stderr, "FAIL: %s: layer %u %s cell %u value %u is %g\n", when, il, kind ? "V" : "K",
                            cell, i, data[cell * row + i]);
                        ok = false;
                        break;
                    }
                }
            }
        }
    }
    return ok;
}

} // namespace

int main() {
    llama_model model(llama_model_default_params());
    model.hparams = {};
    model.arch = LLM_ARCH_LLAMA;
    model.hparams.n_layer = n_layer;
    model.hparams.n_embd_head_k_full = head_dim;
    model.hparams.n_embd_head_v_full = head_dim;
    for (uint32_t il = 0; il < n_layer; il++) { model.hparams.n_head_kv_arr[il] = n_head; }

    llama_kv_cache kv(model, GGML_TYPE_F32, GGML_TYPE_F32, /*v_trans*/ false, /*offload*/ false, /*unified*/ true, size,
        /*n_seq_max*/ 1, /*n_pad*/ 1, /*n_swa*/ 0, LLAMA_SWA_TYPE_NONE, {}, {}, "", grow);
    if (!kv.can_spill()) {
        // the CPU on Windows has no growable buffers: the cache holds every cell from the start and never spills
        std::printf("kv_spill: skipped, the CPU's cache does not grow here\n");
        return 0;
    }

    std::error_code error;
    const auto directory = std::filesystem::temp_directory_path() / ("inferssd-kv-spill-test-" + std::to_string(std::rand()));
    std::filesystem::create_directories(directory, error);
    int spills = 0, restores = 0;
    bool ok = require(!error, "the spill directory was created") &&
        require(kv.set_kv_spill(directory.u8string(), grow, [&](bool spilling) { (spilling ? spills : restores)++; }),
            "the cache took its resident cells and spill directory");

    ggml_init_params params{ ggml_tensor_overhead() * 64, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (ok) {
        ok = require(kv.get_size() == grow, "the cache starts with its resident cells") && ok;
        write(ctx, kv, 0, grow);
        ok = check(ctx, kv, grow, false, "resident") && ok;

        // past the resident cells: the cache moves to the SSD with its values, and grows there
        ok = require(kv.grow_cells(2 * grow) && kv.get_size() == 2 * grow, "the cache grew past its resident cells") && ok;
        ok = require(spills == 1 && restores == 0, "the context was told of one spill") && ok;
        ok = check(ctx, kv, grow, true, "spilled") && ok;
        write(ctx, kv, grow, 2 * grow);
        ok = check(ctx, kv, 2 * grow, true, "grown on the SSD") && ok;

        // trimmed back to the resident cells (the ones past them hold nothing): back in its own memory, same values
        kv.trim_cells(grow);
        ok = require(kv.get_size() == grow && restores == 1, "the trimmed cache returned to its memory") && ok;
        ok = check(ctx, kv, grow, false, "restored") && ok;
    }
    ggml_free(ctx);
    std::filesystem::remove_all(directory, error);
    if (ok) { std::printf("kv_spill: ok\n"); }
    return ok ? 0 : 1;
}
