#include "vram-compute-api.h"
#include "ggml-kimi-q3k-parity.h"
#include "ggml-kimi-q3k-chain.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const kimi_q3k_gpu_api_v1 * real_base;
const kimi_q3k_gpu_chain_api_v1 * real_chain;
kimi_q3k_gpu_api_v1 proxy_base;
kimi_q3k_gpu_chain_api_v1 proxy_chain;
enum class fault { none, submit, wait } inject = fault::none;
bool injected = false;
int submissions = 0, assertions = 0;
std::string destroyed;

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
    ++assertions;
}
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
int submitted(int status) {
    if (status) ++submissions;
    // The real implementation has already enqueued work using cache weights.
    if (status && inject == fault::submit) { injected = true; return 0; }
    return status;
}
int submit(kimi_q3k_gpu * p, const kimi_q3k_gpu_job * jobs, int n, const void * q8, size_t bytes) {
    return submitted(real_base->submit(p, jobs, n, q8, bytes));
}
int chain_submit(kimi_q3k_gpu_chain * p, const kimi_q3k_gpu_job * up,
        const kimi_q3k_gpu_job * down, int n, const void * q8, size_t bytes, float beta, float linear) {
    return submitted(real_chain->submit(p, up, down, n, q8, bytes, beta, linear));
}
int wait(kimi_q3k_gpu * p) {
    if (inject == fault::wait) { injected = true; return 0; }
    return real_base->wait(p);
}
const char * error(kimi_q3k_gpu * p) {
    return injected ? "injected error after real GPU submission" : real_base->error(p);
}
int destroy(kimi_q3k_gpu * p) {
    int status = real_base->destroy(p); // unconditional real stream synchronize
    if (status) destroyed += 'B';
    return status;
}
int chain_destroy(kimi_q3k_gpu_chain * p) {
    int status = real_chain->destroy(p);
    if (status) destroyed += 'C';
    return status;
}
struct device_payload {
    void * data = nullptr;
    ~device_payload() { if (data) cudaFree(data); }
};
struct context {
    const vram_compute_api * api;
    void * value;
    ~context() { if (value) api->destroy(value); }
};
}

// Both getters belong to this executable, satisfying the adapter's same-DSO
// guard. Every function except the injected return delegates to one real DSO.
extern "C" const kimi_q3k_gpu_api_v1 * kimi_q3k_gpu_get_api_v1() { return &proxy_base; }
extern "C" const kimi_q3k_gpu_chain_api_v1 * kimi_q3k_gpu_get_chain_api_v1() { return &proxy_chain; }

int main(int argc, char ** argv) { try {
    require(argc == 2, "usage: test-kimi-vram-component-error /path/to/libkimi-q3k-parity.so");
    void * library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    require(library != nullptr, "load real component DSO");
    auto get_base = reinterpret_cast<const kimi_q3k_gpu_api_v1 * (*)()>(dlsym(library, "kimi_q3k_gpu_get_api_v1"));
    auto get_chain = reinterpret_cast<const kimi_q3k_gpu_chain_api_v1 * (*)()>(dlsym(library, "kimi_q3k_gpu_get_chain_api_v1"));
    require(get_base && get_chain && get_base != kimi_q3k_gpu_get_api_v1 && get_chain != kimi_q3k_gpu_get_chain_api_v1,
            "resolve actual DSO APIs, not test proxies");
    real_base = get_base(); real_chain = get_chain();
    proxy_base = *real_base; proxy_chain = *real_chain;
    proxy_base.submit = submit; proxy_base.wait = wait; proxy_base.error = error; proxy_base.destroy = destroy;
    proxy_chain.submit = chain_submit; proxy_chain.destroy = chain_destroy;
    auto get_compute = reinterpret_cast<vram_compute_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_compute_get_api"));
    require(get_compute != nullptr, "preload production tier DSO");
    const auto * api = get_compute();
    require(api && api->version == 1 && api->size == sizeof(*api), "compute ABI");
    constexpr int latent = 3584, hidden = 3072;
    constexpr size_t role = size_t(latent / 256) * 110 * hidden;
    device_payload weights;
    cuda_check(cudaMalloc(&weights.data, 3 * role));
    cuda_check(cudaMemset(weights.data, 0, 3 * role));
    cuda_check(cudaDeviceSynchronize());
    auto * bytes = static_cast<unsigned char *>(weights.data);
    const vram_compute_expert expert{bytes, bytes + role, bytes + 2 * role};
    std::vector<unsigned char> q8(size_t(latent / 256) * 292, 0);
    std::vector<float> output(latent, 123.0f);
    for (int gpu_hidden : {0, 1}) {
        setenv("GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN", gpu_hidden ? "1" : "0", 1);
        for (fault point : {fault::submit, fault::wait}) {
            char message[512] = {};
            inject = point; injected = false; submissions = 0; destroyed.clear();
            std::fill(output.begin(), output.end(), 123.0f);
            context failed{api, api->create(0, latent, hidden, 1, message, sizeof(message))};
            require(failed.value != nullptr, message);
            const int status = api->compute(failed.value, &expert, 1, q8.data(), q8.size(), 4, 25,
                    output.data(), message, sizeof(message));
            require(status != 0 && injected && submissions == 1 && std::strstr(message, "injected error"),
                    "real queued work must propagate the injected failure");
            require(destroyed == (gpu_hidden ? "CB" : "B"), "component was not drained before compute returned error");
            require(api->device_scratch_bytes(failed.value) == 0, "failed component retained device scratch");
            require(std::all_of(output.begin(), output.end(), [](float x) { return x == 123.0f; }), "failure published output");
            require(api->compute(failed.value, &expert, 1, q8.data(), q8.size(), 4, 25,
                    output.data(), message, sizeof(message)) != 0 && submissions == 1 && std::strstr(message, "closed"),
                    "failed bridge silently reused or recreated component");
            api->destroy(failed.value); failed.value = nullptr;
            require(destroyed == (gpu_hidden ? "CB" : "B"), "component destroyed twice");
            std::printf("COMPONENT_ERROR_CASE_PASS gpu_hidden=%d point=%s\n", gpu_hidden, point == fault::submit ? "submit" : "wait");
        }
        // Only a separately created context may run after the failure.
        inject = fault::none; injected = false; destroyed.clear();
        char message[512] = {};
        context healthy{api, api->create(0, latent, hidden, 1, message, sizeof(message))};
        require(healthy.value != nullptr, message);
        require(api->compute(healthy.value, &expert, 1, q8.data(), q8.size(), 4, 25,
                output.data(), message, sizeof(message)) == 0, message);
        require(std::all_of(output.begin(), output.end(), [](float x) { return x == 0.0f; }), "new context numerical result");
    }
    std::printf("REAL_CUDA_COMPONENT_ERROR_PASS cases=4 assertions=%d\n", assertions);
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "COMPONENT_ERROR_FAIL: %s\n", e.what());
    return 1;
} }
