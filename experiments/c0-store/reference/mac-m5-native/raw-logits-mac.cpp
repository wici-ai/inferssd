// Private tiny golden preparation. No model execution is admitted by this file.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

// Existing pinned-engine C lifecycle extension, also called by llama-server.
extern "C" bool llama_moe_prefill_phase_workspace_sized_ex(
    llama_context *, int active, uint32_t prompt_tokens, uint32_t prefill_tokens);

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "golden rows require IEEE binary32");
static volatile sig_atomic_t stopping = 0;
static void signal_intent(int value) { stopping = value; }
static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Poll lets TERM/HUP intent leave an idle command wait without doing C++ work
// inside the signal handler. The worker still owns deadlines and forced cleanup.
static std::string command() {
    std::string result;
    while (!stopping) {
        pollfd input = {STDIN_FILENO, POLLIN, 0};
        const int ready = poll(&input, 1, 100);
        if (ready < 0 && errno == EINTR) { continue; }
        require(ready >= 0, "command poll failed");
        if (!ready) { continue; }
        char byte;
        const ssize_t count = read(STDIN_FILENO, &byte, 1);
        if (count < 0 && errno == EINTR) { continue; }
        require(count == 1, "controller command pipe ended");
        if (byte == '\n') { return result; }
        require(result.size() < 32, "oversized command");
        result += byte;
    }
    return "TERMINATE";
}

static void exclusive_write(const std::string & path, const void * data, size_t bytes) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    require(fd >= 0, "cannot create raw output exclusively");
    const char * source = static_cast<const char *>(data);
    size_t offset = 0;
    try {
        while (offset < bytes) {
            const ssize_t count = write(fd, source + offset, bytes - offset);
            if (count < 0 && errno == EINTR) { continue; }
            require(count > 0, "raw output write failed");
            offset += static_cast<size_t>(count);
        }
        require(fsync(fd) == 0, "raw output fsync failed");
    } catch (...) { close(fd); throw; }
    require(close(fd) == 0, "raw output close failed");
}

struct backend_guard {
    backend_guard() { llama_backend_init(); }
    ~backend_guard() { llama_backend_free(); }
};
struct batch_guard {
    llama_batch batch = llama_batch_init(16, 0, 1);
    ~batch_guard() { llama_batch_free(batch); }
};

static uint64_t steady_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
static void boundary(const char * event, int request) {
    const auto stamp = static_cast<unsigned long long>(steady_ns());
    std::printf("{\"event\":\"%s\",\"request\":%d,\"steady_ns\":%llu}\n", event, request, stamp);
    std::fflush(stdout);
    std::fprintf(stderr, "[mac_tiny_boundary] event=%s request=%d steady_ns=%llu\n", event, request, stamp);
    std::fflush(stderr);
}

static std::string run_request(llama_context * ctx, int n_vocab, int request,
                               const std::string & out, llama_batch & batch) {
    auto memory = llama_get_memory(ctx);
    require(memory != nullptr, "tiny model has no KV memory");
    llama_memory_clear(memory, true);
    require(llama_memory_seq_pos_max(memory, 0) < 0, "sequence 0 not empty after KV clear");
    boundary("begin", request);
    require(llama_moe_prefill_phase_workspace_sized_ex(ctx, 1, 4, 4), "request begin phase failed");
    std::vector<llama_token> input = {3, 4, 5, 6};
    std::string rows = "[";
    int position = 0;
    for (int step = 0; step < 4; ++step) {
        require(!stopping, "termination requested during request");
        batch.n_tokens = static_cast<int32_t>(input.size());
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = input[i];
            batch.pos[i] = position + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == batch.n_tokens - 1;
        }
        require(llama_decode(ctx, batch) == 0, "llama_decode did not return zero");
        const float * logits = llama_get_logits_ith(ctx, -1);
        require(logits != nullptr, "missing raw logits row");
        // Copy before any later decode; no sampler or decimal conversion touches it.
        const std::vector<float> raw(logits, logits + n_vocab);
        llama_token selected = 0;
        for (int token = 0; token < n_vocab; ++token) {
            require(std::isfinite(raw[token]), "nonfinite raw logit");
            if (raw[token] > raw[selected]) { selected = token; }
        }
        const std::string name = "request-" + std::to_string(request) + "-step-" + std::to_string(step) + ".f32";
        exclusive_write(out + "/" + name, raw.data(), raw.size() * sizeof(float));
        rows += (step ? "," : "") + std::string("{\"step\":") + std::to_string(step) +
            ",\"position\":" + std::to_string(position + batch.n_tokens - 1) +
            ",\"batch_tokens\":" + std::to_string(batch.n_tokens) +
            ",\"selected_id\":" + std::to_string(selected) + ",\"file\":\"" + name + "\"}";
        if (step == 0) {
            require(llama_moe_prefill_phase_workspace_sized_ex(ctx, 0, 0, 0), "prefill done phase failed");
        }
        position += batch.n_tokens;
        input.assign(1, selected); // No EOS early exit in this fixed four-step protocol.
    }
    require(llama_moe_prefill_phase_workspace_sized_ex(ctx, 0, 0, 0), "request end phase failed");
    boundary("end", request);
    return rows + "]";
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: raw-logits-runner MODEL EXISTING_EMPTY_OUT PINNED_BACKEND_DIR\n");
        return 2;
    }
    struct sigaction action = {};
    action.sa_handler = signal_intent;
    sigemptyset(&action.sa_mask);
    for (int sig : {SIGINT, SIGTERM, SIGHUP}) {
        if (sigaction(sig, &action, nullptr)) { return 2; }
    }
    try {
        backend_guard backend;
        ggml_backend_load_all_from_path(argv[3]);
        auto mp = llama_model_default_params();
        const llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()},
            {"blk\\.[0-9]+\\.engram_embd\\.weight", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr},
        };
        mp.tensor_buft_overrides = overrides;
        mp.n_gpu_layers = 99;
        mp.use_mmap = true;
        mp.use_extra_bufts = false;
        mp.use_direct_io = false;
        mp.use_mlock = false;
        mp.check_tensors = false;
        mp.no_host = true;
        mp.no_alloc = false;
        mp.defer_experts = false;
        using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
        model_ptr model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "model load failed");
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(n_vocab > 6, "tiny vocabulary does not contain prompt IDs");
        auto cp = llama_context_default_params();
        cp.n_ctx = 128; cp.n_batch = 16; cp.n_ubatch = 16; cp.n_seq_max = 1;
        cp.n_threads = 2; cp.n_threads_batch = 2;
        cp.op_offload = false; cp.offload_kqv = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.swa_full = true; cp.kv_unified = false;
        cp.samplers = nullptr; cp.n_samplers = 0;
        using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
        context_ptr ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(ctx != nullptr, "context initialization failed");
        batch_guard storage;
        require(storage.batch.token != nullptr, "batch allocation failed");
        const uint16_t endian_probe = 1;
        const bool little = *reinterpret_cast<const unsigned char *>(&endian_probe) == 1;
        std::printf("{\"event\":\"ready\",\"n_vocab\":%d,\"float_bytes\":4,\"byte_order\":\"%s\","
                    "\"effective_n_ctx\":%u,\"effective_n_batch\":%u,\"effective_n_ubatch\":%u}\n",
                    n_vocab, little ? "little" : "big", llama_n_ctx(ctx.get()),
                    llama_n_batch(ctx.get()), llama_n_ubatch(ctx.get()));
        std::fflush(stdout);
        for (int request = 1; request <= 2; ++request) {
            const auto cmd = command();
            if (cmd == "TERMINATE") { return 0; }
            require(cmd == "RUN " + std::to_string(request), "unexpected request command");
            const auto rows = run_request(ctx.get(), n_vocab, request, argv[2], storage.batch);
            const auto receipt = "{\"request\":" + std::to_string(request) + ",\"rows\":" + rows + "}\n";
            exclusive_write(std::string(argv[2]) + "/request-" + std::to_string(request) + ".json",
                            receipt.data(), receipt.size());
            std::printf("{\"event\":\"done\",\"request\":%d}\n", request);
            std::fflush(stdout);
        }
        const auto exit_command = command();
        require(exit_command == "EXIT" || exit_command == "TERMINATE", "expected final EXIT");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "raw-logits-runner: %s\n", error.what());
        return 1;
    }
}
