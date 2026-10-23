// The state an iSWA cache saves for a position inside the batch it decoded last (llama_memory_i::state_write_at): the
// same positions, K and V as a run stopped at that position holds in the full layers, and the window the position
// before it attends to in the sliding-window layers.  Restored into the running cache and truncated as the server
// does with a checkpoint, then decoded on, the cache holds what a run that never stopped holds.  A small cache: one
// sliding-window layer (n_swa 4) and one full layer, K and V rows that name their position, layer and kind, batches
// of 8 positions, enough of them that the sliding-window half (256 cells) gives early cells to later positions; a
// standard window and a chunked one.  The first position each state holds is the one the cache reports for it.
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-kv-cache-iswa.h"
#include "llama-model.h"
#include "llama-vocab.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & what) {
    if (!ok) {
        throw std::runtime_error(what);
    }
}

namespace {

constexpr uint32_t N_EMBD   = 4;
constexpr uint32_t N_SWA    = 4;
constexpr uint32_t N_BATCH  = 8;
constexpr uint32_t KV_SIZE  = 1024;
constexpr int32_t  L_SWA    = 0;   // layer 0 slides, layer 1 is full
constexpr int32_t  L_FULL   = 1;
constexpr llama_state_seq_flags PARTIAL = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

struct io_vector : llama_io_write_i {
    std::vector<uint8_t> data;
    void write(const void * src, size_t size) override {
        const auto * p = static_cast<const uint8_t *>(src);
        data.insert(data.end(), p, p + size);
    }
    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        std::vector<uint8_t> tmp(size);
        ggml_backend_tensor_get(tensor, tmp.data(), offset, size);
        write(tmp.data(), size);
    }
    size_t n_bytes() override { return data.size(); }
};

struct io_reader : llama_io_read_i {
    const std::vector<uint8_t> & data;
    size_t pos = 0;
    explicit io_reader(const std::vector<uint8_t> & data) : data(data) {}
    const uint8_t * read(size_t size) override {
        require(pos + size <= data.size(), "state read past its end");
        const uint8_t * p = data.data() + pos;
        pos += size;
        return p;
    }
    void read_to(void * dst, size_t size) override { std::memcpy(dst, read(size), size); }
    size_t n_bytes() override { return pos; }
};

// one half of a state (llama_kv_cache::state_write_range's format, one stream, V not transposed) as position -> the
// K rows then the V rows of its layers: what attention can read, whatever cells held it
using half_state = std::map<llama_pos, std::vector<float>>;

half_state read_half(io_reader & in) {
    half_state out;
    uint32_t n_stream = 0, n_cells = 0;
    in.read_to(&n_stream, sizeof(n_stream));
    require(n_stream == 1, "one stream expected");
    in.read_to(&n_cells, sizeof(n_cells));
    if (n_cells == 0) {
        return out;
    }
    std::vector<llama_pos> pos(n_cells);
    for (uint32_t i = 0; i < n_cells; ++i) {
        uint32_t n_seq_id = 0;
        in.read_to(&pos[i], sizeof(pos[i]));
        in.read_to(&n_seq_id, sizeof(n_seq_id));
        require(n_seq_id == 1, "one sequence per cell expected");
        llama_seq_id seq_id = -1;
        in.read_to(&seq_id, sizeof(seq_id));
        require(seq_id == 0, "sequence 0 expected");
        require(out.emplace(pos[i], std::vector<float>()).second, "a position saved twice");
    }
    uint32_t v_trans = 1, n_layer = 0;
    in.read_to(&v_trans, sizeof(v_trans));
    in.read_to(&n_layer, sizeof(n_layer));
    require(v_trans == 0, "V not transposed expected");
    for (uint32_t kind = 0; kind < 2; ++kind) {
        for (uint32_t l = 0; l < n_layer; ++l) {
            int32_t type = -1;
            uint64_t row = 0;
            in.read_to(&type, sizeof(type));
            in.read_to(&row, sizeof(row));
            require(type == GGML_TYPE_F32 && row == N_EMBD * sizeof(float), "F32 rows of N_EMBD expected");
            for (uint32_t i = 0; i < n_cells; ++i) {
                const auto * v = reinterpret_cast<const float *>(in.read(row));
                out[pos[i]].insert(out[pos[i]].end(), v, v + N_EMBD);
            }
        }
    }
    return out;
}

// a state of the whole iSWA cache: the full half unless partial, then the sliding-window half
struct iswa_state {
    half_state full, swa;
    bool operator==(const iswa_state & other) const { return full == other.full && swa == other.swa; }
};

iswa_state parse(const std::vector<uint8_t> & data, bool partial) {
    io_reader in(data);
    iswa_state out;
    if (!partial) {
        out.full = read_half(in);
    }
    out.swa = read_half(in);
    require(in.pos == data.size(), "state has trailing bytes");
    return out;
}

// what a cache holding positions [0, n) and restored at n holds afterwards for each layer kind: every position in the
// full layer, the window of the position before n in the sliding-window layer
iswa_state expected(llama_pos n, llama_pos swa_from, bool partial) {
    auto rows = [](llama_pos p, int32_t il) {
        std::vector<float> r;
        for (float kind : {0.0f, 1.0f}) {
            r.insert(r.end(), {float(p), float(il), kind, 1.0f});
        }
        return r;
    };
    iswa_state out;
    for (llama_pos p = 0; p < n && !partial; ++p) {
        out.full[p] = rows(p, L_FULL);
    }
    for (llama_pos p = swa_from; p < n; ++p) {
        out.swa[p] = rows(p, L_SWA);
    }
    return out;
}

struct fixture {
    llama_model model{llama_model_default_params()};
    llama_vocab vocab;
    std::unique_ptr<llama_kv_cache_iswa> kv;

    explicit fixture(llama_swa_type swa_type = LLAMA_SWA_TYPE_STANDARD) {
        auto & hp = model.hparams;
        hp.n_layer = 2;
        hp.n_embd = N_EMBD;
        hp.n_ctx_train = KV_SIZE;
        hp.n_head_arr.fill(1);
        hp.n_head_kv_arr.fill(1);
        hp.n_embd_head_k_full = hp.n_embd_head_v_full = N_EMBD;
        hp.n_embd_head_k_swa  = hp.n_embd_head_v_swa  = N_EMBD;
        hp.swa_type = swa_type;
        hp.n_swa = N_SWA;
        hp.swa_layers.fill(0);
        hp.swa_layers[L_SWA] = 1;
        kv = std::make_unique<llama_kv_cache_iswa>(model, GGML_TYPE_F32, GGML_TYPE_F32, /*v_trans*/ false,
            /*offload*/ false, /*swa_full*/ false, /*unified*/ true, KV_SIZE, /*n_seq_max*/ 1, N_BATCH, /*n_pad*/ 1,
            nullptr, nullptr);
    }

    // K and V of the ubatch's positions into the cells the cache gave them: (pos, layer, 0 = K / 1 = V, 1)
    static void write_rows(const llama_kv_cache_context * half, int32_t il, const llama_ubatch & ub) {
        ggml_init_params params = { 8 * ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr ctx(ggml_init(params));
        ggml_tensor * idxs = half->build_input_k_idxs(ctx.get(), ub);
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        half->set_input_k_idxs(idxs, &ub);
        ggml_tensor * k = half->get_k(ctx.get(), il)->view_src;
        ggml_tensor * v = half->get_v(ctx.get(), il)->view_src;
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const int64_t cell = static_cast<const int64_t *>(idxs->data)[i];
            for (ggml_tensor * t : { k, v }) {
                const float row[N_EMBD] = { float(ub.pos[i]), float(il), t == k ? 0.0f : 1.0f, 1.0f };
                ggml_backend_tensor_set(t, row, cell * t->nb[1], sizeof(row));
            }
        }
    }

    // positions [p0, p1) of sequence 0, in batches of N_BATCH counted from p0
    void decode(llama_pos p0, llama_pos p1) {
        for (llama_pos b0 = p0; b0 < p1; b0 += N_BATCH) {
            const int32_t n = std::min<llama_pos>(N_BATCH, p1 - b0);
            std::vector<float> embd(size_t(n) * N_EMBD, 0.0f);
            std::vector<llama_pos> pos(n);
            std::vector<int32_t> n_seq_id(n, 1);
            llama_seq_id seq0 = 0;
            std::vector<llama_seq_id *> seq_id(n + 1, &seq0);
            seq_id[n] = nullptr;
            std::vector<int8_t> output(n, 0);
            for (int32_t i = 0; i < n; ++i) {
                pos[i] = b0 + i;
            }
            llama_batch batch = { n, nullptr, embd.data(), pos.data(), n_seq_id.data(), seq_id.data(), output.data() };
            llama_batch_allocr balloc(1);
            require(balloc.init(batch, vocab, kv.get(), N_EMBD, 1, false), "batch rejected");
            auto mctx = kv->init_batch(balloc, N_BATCH, false);
            require(mctx && mctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS, "no room for a batch");
            do {
                require(mctx->apply(), "batch not applied");
                const auto * ctx = static_cast<const llama_kv_cache_iswa_context *>(mctx.get());
                write_rows(ctx->get_base(), L_FULL, mctx->get_ubatch());
                write_rows(ctx->get_swa(), L_SWA, mctx->get_ubatch());
            } while (mctx->next());
        }
    }

    std::vector<uint8_t> save_at(llama_pos n, llama_state_seq_flags flags) const {
        io_vector out;
        kv->state_write_at(out, 0, flags, n);
        return out.data;
    }

    // the window of the position before n as the sliding-window half holds it
    std::vector<uint8_t> save_window(llama_pos from, llama_pos n, llama_state_seq_flags flags) const {
        io_vector out;
        if ((flags & PARTIAL) == 0) {
            kv->get_base()->state_write_range(out, 0, flags, std::numeric_limits<llama_pos>::min(), n - 1);
        }
        kv->get_swa()->state_write_range(out, 0, flags, from, n - 1);
        return out.data;
    }
};

// the first position the window of the position before n reaches back to: n_swa positions for a standard window, the
// start of its chunk for a chunked one
llama_pos window_from(llama_swa_type swa_type, llama_pos n) {
    return swa_type == LLAMA_SWA_TYPE_CHUNKED ? (n - 1) / N_SWA * N_SWA : n - N_SWA;
}

// A decodes [0, 304) in batches of 8; the sliding-window half (256 cells) has given early cells away
int check_states(llama_swa_type swa_type) {
    const std::string kind = swa_type == LLAMA_SWA_TYPE_CHUNKED ? "chunked: " : "standard: ";
    const llama_pos m = 304, last = m - N_BATCH;
    fixture a(swa_type);
    a.decode(0, m);
    require(a.kv->seq_pos_max(0) == m - 1, kind + "A holds the wrong last position");
    for (llama_pos n = last + 1; n <= m; ++n) {
        require(a.kv->state_can_write_at(0, n), kind + "no earlier state inside the last batch at " + std::to_string(n));
    }
    for (llama_pos n : { 0, 20, m + 1 }) {
        require(!a.kv->state_can_write_at(0, n), kind + "an earlier state that cannot be at " + std::to_string(n));
    }
    require(!a.kv->state_can_write_at(-1, m), kind + "an earlier state of no sequence");

    int checked = 0;
    for (llama_pos n = last + 1; n < m; ++n) {
        const llama_pos from = window_from(swa_type, n);   // the window of position n - 1
        require(a.kv->state_pos_min_at(0, n) == from, kind + "the earlier state's first position at " + std::to_string(n));
        // B stops at n: its window and full layer, whatever else its cells still hold
        fixture b(swa_type);
        b.decode(0, n);
        for (llama_state_seq_flags flags : { PARTIAL, llama_state_seq_flags(0) }) {
            const bool partial = flags == PARTIAL;
            const auto at = a.save_at(n, flags);
            const auto stopped = b.save_window(from, n, flags);
            require(parse(at, partial) == parse(stopped, partial), kind + "earlier state differs from a run stopped at " +
                std::to_string(n));
            require(parse(at, partial) == expected(n, from, partial), kind + "earlier state holds other rows at " +
                std::to_string(n));
            ++checked;
        }

        // the server's checkpoint: restore the partial state into the running cache, drop the positions from the
        // one before n (the server evaluates it again), decode to m; the cache then holds what A held at m
        fixture c(swa_type);
        c.decode(0, m);
        const auto checkpoint = c.save_at(n, PARTIAL);
        io_reader in(checkpoint);
        c.kv->state_read(in, 0, PARTIAL);
        require(c.kv->seq_pos_min(0) == from && c.kv->seq_pos_max(0) == n - 1,
            kind + "restored window has other positions");
        require(c.kv->seq_rm(0, n - 1, -1), kind + "restored state cannot be truncated");
        c.decode(n - 1, m);
        const llama_pos from_m = window_from(swa_type, m);
        require(parse(c.save_window(from_m, m, 0), false) == parse(a.save_window(from_m, m, 0), false),
            kind + "decoding on from the checkpoint at " + std::to_string(n) + " differs from never stopping");
    }

    // a state saved at the current position is the whole state, as before
    io_vector whole;
    a.kv->state_write(whole, 0, 0);
    require(a.save_at(m, 0) == whole.data, kind + "the current state differs from state_write()");
    require(a.kv->state_pos_min_at(0, m) == a.kv->seq_pos_min(0), kind + "the current state's first position");
    return checked;
}

void run() {
    require(fixture(LLAMA_SWA_TYPE_STANDARD).kv->state_write_at_align() == 1, "standard window: no earlier states");
    require(fixture(LLAMA_SWA_TYPE_CHUNKED).kv->state_write_at_align() == 1, "chunked window: no earlier states");
    require(fixture(LLAMA_SWA_TYPE_SYMMETRIC).kv->state_write_at_align() == 0,
        "a symmetric window reaches forward: it can have no earlier states");
    const int standard = check_states(LLAMA_SWA_TYPE_STANDARD);
    const int chunked = check_states(LLAMA_SWA_TYPE_CHUNKED);

    std::printf("KV_CACHE_ISWA_STATE_AT_OK align=PASS can_write_at=PASS earlier_states=%d chunked=%d pos_min_at=PASS "
        "restore_decode=PASS current=PASS\n", standard, chunked);
}

} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "KV_CACHE_ISWA_STATE_AT_FAILED: %s\n", e.what());
        return 1;
    }
}
