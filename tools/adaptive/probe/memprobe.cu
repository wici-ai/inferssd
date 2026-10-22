// Device memory/transfer probe for the DSV4.1 placement model.
// Prints: cudaMemGetInfo vs /proc/meminfo, whether cudaMalloc is charged to this process's cgroup,
// H2D bandwidth (pageable / pinned) for an 18.8 MB expert and a 6.27 MB role slice, D2D copy,
// GPU streaming-read bandwidth, and integrated/UMA attributes.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while (0)

static long long meminfo(const char * key) {
    std::ifstream f("/proc/meminfo");
    std::string k; long long v; std::string unit;
    while (f >> k >> v) { std::getline(f, unit); if (k == std::string(key) + ":") return v * 1024; }
    return -1;
}

static long long cgroup_current() {
    std::ifstream c("/proc/self/cgroup");
    std::string line, path;
    while (std::getline(c, line)) { if (line.rfind("0::", 0) == 0) path = line.substr(3); }
    std::ifstream f("/sys/fs/cgroup" + path + "/memory.current");
    long long v = -1; f >> v; return v;
}

__global__ void read_kernel(const uint4 * p, size_t n, unsigned long long * sink) {
    uint4 acc = {0, 0, 0, 0};
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) atomicAdd(sink, 1ull);
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    size_t big_gib = argc > 1 ? std::atoi(argv[1]) : 8;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int integrated = 0, pageable = 0, concurrent = 0, hostreg = 0;
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0);
    cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0);
    cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, 0);
    cudaDeviceGetAttribute(&hostreg, cudaDevAttrHostRegisterSupported, 0);
    std::printf("device=%s sm=%d.%d SMs=%d integrated=%d pageableMemoryAccess=%d concurrentManaged=%d hostRegister=%d\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount, integrated, pageable, concurrent, hostreg);
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    std::printf("before: cudaMemGetInfo free=%.2f GiB total=%.2f GiB | MemAvailable=%.2f GiB MemFree=%.2f GiB Cached=%.2f GiB | cgroup.current=%.2f GiB\n",
        fr / 1073741824.0, tot / 1073741824.0, meminfo("MemAvailable") / 1073741824.0, meminfo("MemFree") / 1073741824.0,
        meminfo("Cached") / 1073741824.0, cgroup_current() / 1073741824.0);

    // cudaMalloc accounting
    void * big = nullptr;
    size_t big_bytes = big_gib << 30;
    if (cudaMalloc(&big, big_bytes) == cudaSuccess) {
        CK(cudaMemset(big, 1, big_bytes)); CK(cudaDeviceSynchronize());
        CK(cudaMemGetInfo(&fr, &tot));
        std::printf("after cudaMalloc+memset %zu GiB: cudaMemGetInfo free=%.2f GiB | MemAvailable=%.2f GiB MemFree=%.2f GiB | cgroup.current=%.2f GiB\n",
            big_gib, fr / 1073741824.0, meminfo("MemAvailable") / 1073741824.0, meminfo("MemFree") / 1073741824.0,
            cgroup_current() / 1073741824.0);
        // GPU streaming read bandwidth over the big buffer
        unsigned long long * sink; CK(cudaMalloc(&sink, 8));
        int blocks = prop.multiProcessorCount * 8;
        read_kernel<<<blocks, 512>>>((const uint4 *) big, big_bytes / 16, sink); CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < 3; ++i) read_kernel<<<blocks, 512>>>((const uint4 *) big, big_bytes / 16, sink);
        cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b);
        std::printf("gpu_read_bw=%.1f GB/s\n", 3.0 * big_bytes / (ms / 1e3) / 1e9);
        // D2D copy
        cudaEventRecord(a);
        CK(cudaMemcpy((char *) big + big_bytes / 2, big, big_bytes / 2, cudaMemcpyDeviceToDevice));
        cudaEventRecord(b); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
        std::printf("d2d_copy_bw=%.1f GB/s (read+write counted once)\n", (big_bytes / 2) / (ms / 1e3) / 1e9);
        cudaFree(sink); cudaFree(big);
    } else {
        std::printf("cudaMalloc %zu GiB failed\n", big_gib);
    }

    // H2D for expert-sized copies
    const size_t sizes[2] = {18800640, 6266880};
    for (size_t s : sizes) {
        const int n = 64;
        std::vector<void *> host(n);
        for (int i = 0; i < n; ++i) { host[i] = std::aligned_alloc(4096, s); std::memset(host[i], i, s); }
        void * pinned; CK(cudaMallocHost(&pinned, s * 8)); std::memset(pinned, 3, s * 8);
        void * dev; CK(cudaMalloc(&dev, s * 8));
        cudaStream_t st; cudaStreamCreate(&st);
        // pageable (rotating sources so they are not cache-resident)
        CK(cudaMemcpyAsync(dev, host[0], s, cudaMemcpyHostToDevice, st)); cudaStreamSynchronize(st);
        double t0 = now_s();
        for (int i = 0; i < n; ++i) CK(cudaMemcpyAsync((char *) dev + (i % 8) * s, host[i], s, cudaMemcpyHostToDevice, st));
        cudaStreamSynchronize(st);
        double pg = n * (double) s / (now_s() - t0) / 1e9;
        t0 = now_s();
        for (int i = 0; i < n; ++i) CK(cudaMemcpyAsync((char *) dev + (i % 8) * s, (char *) pinned + (i % 8) * s, s, cudaMemcpyHostToDevice, st));
        cudaStreamSynchronize(st);
        double pn = n * (double) s / (now_s() - t0) / 1e9;
        // single-copy latency (pinned)
        t0 = now_s();
        CK(cudaMemcpyAsync(dev, pinned, s, cudaMemcpyHostToDevice, st)); cudaStreamSynchronize(st);
        double one = (now_s() - t0) * 1e6;
        std::printf("h2d bytes=%zu pageable=%.2f GB/s pinned=%.2f GB/s single_pinned_us=%.1f\n", s, pg, pn, one);
        cudaFree(dev); cudaFreeHost(pinned); for (auto p : host) std::free(p);
    }
    // host memory bandwidth (single-thread memcpy, for reference)
    {
        size_t s = 1ull << 30;
        char * a = (char *) std::aligned_alloc(4096, s); char * b = (char *) std::aligned_alloc(4096, s);
        std::memset(a, 1, s); std::memset(b, 2, s);
        double t0 = now_s(); std::memcpy(b, a, s); double t = now_s() - t0;
        std::printf("host_memcpy_1thread=%.1f GB/s\n", s / t / 1e9);
        std::free(a); std::free(b);
    }
    return 0;
}
