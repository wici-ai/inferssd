// SPDX-License-Identifier: Apache-2.0
// inferssd-memprobe: for every CUDA device, VRAM size, host-to-device and device-to-host copy bandwidth per copy size
// (pinned and pageable host memory), device-to-device copy bandwidth and streaming-read bandwidth. Every rate is
// cudaEvent time over back-to-back repeated operations on one stream. Prints one JSON object on stdout.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Streams the buffer once; the loaded data feeds a branch so the compiler keeps the loads.
__global__ void read_kernel(const uint4 * p, size_t n, unsigned long long * sink) {
    uint4 acc = { 0, 0, 0, 0 };
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) { atomicAdd(sink, 1ull); }
}

namespace {

const char * const USAGE =
    "usage: inferssd-memprobe\n"
    "\n"
    "Measures every visible CUDA device and prints one JSON object (schema inferssd-memprobe/1) on stdout:\n"
    "VRAM total and free, host-to-device copies from pinned and pageable memory, device-to-host copies to\n"
    "pinned memory (copy sizes 256 KiB to 64 MiB), device-to-device copy and streaming-read bandwidth.\n"
    "A device that cannot be measured (less than 512 MiB of VRAM free, or a CUDA error) is left out of\n"
    "\"devices\" and listed under \"skipped\" with the reason; the probe fails only when no device can be measured.\n";

const size_t COPY_SIZES[] = { 262144, 1048576, 4194304, 16777216, 67108864 };
// Copies cycle through the slots of 256 MiB regions so no source stays cache resident.
const size_t REGION_BYTES = 256ull << 20;
const int    DEVICE_REPS  = 10;

void check(cudaError_t e, const char * what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

#define CK(x) check((x), #x)

std::string num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    std::string s(buf);
    if (s.find_first_of(".eE") == std::string::npos) { s += ".0"; }
    return s;
}

std::string quote(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char) c; }
        else if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
        else { out += (char) c; }
    }
    return out + "\"";
}

// nvidia-smi's spelling: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.
std::string format_uuid(const cudaUUID_t & uuid) {
    const unsigned char * b = reinterpret_cast<const unsigned char *>(uuid.bytes);
    char buf[48];
    std::snprintf(buf, sizeof(buf), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return buf;
}

int attribute(cudaDeviceAttr attr, int device) {
    int value = 0;
    CK(cudaDeviceGetAttribute(&value, attr, device));
    return value;
}

struct timer {
    cudaStream_t stream = nullptr;
    cudaEvent_t  start  = nullptr;
    cudaEvent_t  stop   = nullptr;

    timer() {
        CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CK(cudaEventCreate(&start));
        CK(cudaEventCreate(&stop));
    }
    ~timer() {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaStreamDestroy(stream);
    }
    timer(const timer &) = delete;
    timer & operator=(const timer &) = delete;

    void begin() { CK(cudaEventRecord(start, stream)); }

    double end_seconds() {
        CK(cudaEventRecord(stop, stream));
        CK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, start, stop));
        if (!(ms > 0.0f)) { throw std::runtime_error("cudaEventElapsedTime returned no time"); }
        return ms * 1e-3;
    }
};

// Bytes per second of back-to-back copies of `size` bytes cycling through the slots of two REGION_BYTES regions.
double copy_Bps(timer & t, void * dst, const void * src, size_t size, cudaMemcpyKind kind) {
    const size_t slots = std::max<size_t>(1, REGION_BYTES / size);
    const size_t n = std::min<size_t>(1024, std::max<size_t>(16, REGION_BYTES / size));
    CK(cudaMemcpyAsync(dst, src, size, kind, t.stream));
    CK(cudaStreamSynchronize(t.stream));
    t.begin();
    for (size_t i = 0; i < n; ++i) {
        const size_t offset = (i % slots) * size;
        CK(cudaMemcpyAsync((char *) dst + offset, (const char *) src + offset, size, kind, t.stream));
    }
    return (double) (n * size) / t.end_seconds();
}

double read_Bps(timer & t, const void * buf, size_t bytes, int sms, unsigned long long * sink) {
    const size_t n = bytes / sizeof(uint4);
    const int blocks = sms * 8;
    read_kernel<<<blocks, 512, 0, t.stream>>>((const uint4 *) buf, n, sink);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(t.stream));
    t.begin();
    for (int i = 0; i < DEVICE_REPS; ++i) { read_kernel<<<blocks, 512, 0, t.stream>>>((const uint4 *) buf, n, sink); }
    CK(cudaGetLastError());
    return (double) DEVICE_REPS * (double) (n * sizeof(uint4)) / t.end_seconds();
}

// Copies one half of the buffer onto the other; bytes copied (read and write counted once) per second.
double d2d_Bps(timer & t, void * buf, size_t bytes) {
    const size_t half = bytes / 2;
    CK(cudaMemcpyAsync((char *) buf + half, buf, half, cudaMemcpyDeviceToDevice, t.stream));
    CK(cudaStreamSynchronize(t.stream));
    t.begin();
    for (int i = 0; i < DEVICE_REPS; ++i) {
        CK(cudaMemcpyAsync((char *) buf + half, buf, half, cudaMemcpyDeviceToDevice, t.stream));
    }
    return (double) DEVICE_REPS * (double) half / t.end_seconds();
}

std::string rate_list(const std::vector<std::pair<size_t, double>> & rates) {
    std::string out = "[";
    for (size_t i = 0; i < rates.size(); ++i) {
        out += std::string(i ? ", " : "") + "{\"bytes\": " + std::to_string(rates[i].first) +
               ", \"Bps\": " + num(rates[i].second) + "}";
    }
    return out + "]";
}

std::string measure_device(int device) {
    CK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, device));
    const int integrated = attribute(cudaDevAttrIntegrated, device);
    const int sms = attribute(cudaDevAttrMultiProcessorCount, device);
    char bus_id[32];
    std::snprintf(bus_id, sizeof(bus_id), "%04x:%02x:%02x.0", attribute(cudaDevAttrPciDomainId, device),
                  attribute(cudaDevAttrPciBusId, device), attribute(cudaDevAttrPciDeviceId, device));
    size_t free_bytes = 0, total_bytes = 0;
    CK(cudaMemGetInfo(&free_bytes, &total_bytes));

    // One device buffer serves every test: copy destinations use its first REGION_BYTES, the read and
    // device-to-device tests use all of it (up to 1 GiB, well beyond any L2).
    const size_t device_bytes = std::min<size_t>(1ull << 30, free_bytes / 2) & ~((64ull << 20) - 1);
    if (device_bytes < REGION_BYTES) {
        throw std::runtime_error(std::to_string(free_bytes) + " bytes of VRAM free, need at least " +
                                 std::to_string(2 * REGION_BYTES));
    }
    std::fprintf(stderr, "memprobe: device %d %s, %zu of %zu VRAM bytes free\n", device, prop.name, free_bytes,
                 total_bytes);

    void * dev = nullptr;
    void * pinned = nullptr;
    unsigned long long * sink = nullptr;
    std::vector<unsigned char> pageable(REGION_BYTES, 1);  // written, so every page is resident
    std::vector<std::pair<size_t, double>> h2d_pinned, h2d_pageable, d2h_pinned;
    double read = 0.0, d2d = 0.0;
    try {
        CK(cudaMalloc(&dev, device_bytes));
        CK(cudaMalloc(&sink, sizeof(*sink)));
        CK(cudaMallocHost(&pinned, REGION_BYTES));
        std::memset(pinned, 3, REGION_BYTES);
        CK(cudaMemset(dev, 1, device_bytes));
        CK(cudaMemset(sink, 0, sizeof(*sink)));
        CK(cudaDeviceSynchronize());
        timer t;
        for (size_t size : COPY_SIZES) {
            h2d_pinned.push_back({ size, copy_Bps(t, dev, pinned, size, cudaMemcpyHostToDevice) });
            h2d_pageable.push_back({ size, copy_Bps(t, dev, pageable.data(), size, cudaMemcpyHostToDevice) });
            d2h_pinned.push_back({ size, copy_Bps(t, pinned, dev, size, cudaMemcpyDeviceToHost) });
            std::fprintf(stderr, "memprobe: device %d bytes=%zu h2d pinned %.3g pageable %.3g d2h pinned %.3g B/s\n",
                         device, size, h2d_pinned.back().second, h2d_pageable.back().second, d2h_pinned.back().second);
        }
        read = read_Bps(t, dev, device_bytes, sms, sink);
        d2d = d2d_Bps(t, dev, device_bytes);
        std::fprintf(stderr, "memprobe: device %d read %.3g B/s, device-to-device copy %.3g B/s\n", device, read, d2d);
    } catch (...) {
        cudaFreeHost(pinned);
        cudaFree(sink);
        cudaFree(dev);
        throw;
    }
    CK(cudaFreeHost(pinned));
    CK(cudaFree(sink));
    CK(cudaFree(dev));

    return "{\"index\": " + std::to_string(device) + ", \"name\": " + quote(prop.name) +
           ", \"uuid\": " + quote(format_uuid(prop.uuid)) + ", \"pci_bus_id\": " + quote(bus_id) +
           ", \"integrated\": " + (integrated ? "true" : "false") +
           ",\n              \"vram_total_bytes\": " + std::to_string(total_bytes) +
           ", \"vram_free_bytes\": " + std::to_string(free_bytes) +
           ",\n              \"h2d\": {\"pinned\": " + rate_list(h2d_pinned) +
           ",\n                      \"pageable\": " + rate_list(h2d_pageable) + "}" +
           ",\n              \"d2h\": {\"pinned\": " + rate_list(d2h_pinned) + "}" +
           ",\n              \"d2d_Bps\": " + num(d2d) + ", \"read_Bps\": " + num(read) + "}";
}

int run(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::fputs(USAGE, stdout);
            return 0;
        }
        std::fprintf(stderr, "inferssd-memprobe: unknown option: %s (see --help)\n", argv[i]);
        return 2;
    }
    int count = 0;
    CK(cudaGetDeviceCount(&count));
    if (count <= 0) { throw std::runtime_error("no CUDA device"); }
    // One busy or failing device must not cost the measurements of the others.
    std::string devices, skipped;
    int measured = 0;
    for (int device = 0; device < count; ++device) {
        try {
            const std::string row = measure_device(device);
            devices += std::string(measured++ ? ",\n             " : "") + row;
        } catch (const std::exception & e) {
            cudaGetLastError();  // clears the error so the next device's launch checks do not report it
            std::fprintf(stderr, "inferssd-memprobe: device %d skipped: %s\n", device, e.what());
            skipped += std::string(skipped.empty() ? "" : ", ") + "{\"index\": " + std::to_string(device) +
                       ", \"reason\": " + quote(e.what()) + "}";
        }
    }
    if (measured == 0) {
        throw std::runtime_error("no CUDA device could be measured (" + std::to_string(count) + " found)");
    }
    const std::string json = "{\"schema\": \"inferssd-memprobe/1\",\n \"devices\": [" + devices +
                             "],\n \"skipped\": [" + skipped + "]}\n";
    std::fputs(json.c_str(), stdout);
    std::fflush(stdout);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "inferssd-memprobe: %s\n", e.what());
        return 1;
    }
}
