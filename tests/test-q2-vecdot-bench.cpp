// Synthetic DeepSeek-V4.1 expert vec_dot benchmark.
//
// The benchmark intentionally uses the real expert matrix orientations rather
// than one long vector: gate/up is 2304 rows of 5120 values, while down is 5120
// rows of 2304 values.  A 512 MiB ring keeps the streaming arm well beyond LLC;
// the hot arm repeatedly uses one role-sized matrix.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr int k_embd = 5120;
constexpr int k_ff   = 2304;
constexpr size_t k_stream_target = size_t(512) << 20;
constexpr int k_samples = 7;

struct aligned_buffer {
    uint8_t * data = nullptr;
    size_t size = 0;

    aligned_buffer() = default;
    explicit aligned_buffer(size_t n) : size(n) {
        if (posix_memalign(reinterpret_cast<void **>(&data), 4096, n) != 0) {
            throw std::bad_alloc();
        }
    }
    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;
    aligned_buffer(aligned_buffer && other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    aligned_buffer & operator=(aligned_buffer && other) noexcept {
        if (this != &other) {
            std::free(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
    ~aligned_buffer() { std::free(data); }
};

struct quant_data {
    ggml_type type;
    std::string name;
    size_t role_bytes;
    size_t roles;
    aligned_buffer weights;
};

struct bench_case {
    const quant_data * data;
    int columns;
    int rows;
    int threads;
    bool streaming;

    std::string key() const {
        return data->name + "/" + (columns == k_embd ? "gate" : "down") +
            "/t" + std::to_string(threads) + "/" + (streaming ? "stream" : "hot");
    }
};

struct result {
    double seconds;
    double gbps;
    double gflops;
    double checksum;
    size_t bytes;
    uint64_t weights;
};

void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void pin_this_thread(int cpu, std::atomic<int> & errors) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
        errors.fetch_add(1, std::memory_order_relaxed);
    }
#else
    (void) cpu;
    (void) errors;
#endif
}

aligned_buffer quantize_base(ggml_type type, const std::vector<float> & source,
                             const std::vector<float> & imatrix) {
    const size_t bytes = ggml_row_size(type, k_embd) * size_t(k_ff);
    aligned_buffer result(bytes);
    const float * importance = ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr;
    const size_t written = ggml_quantize_chunk(type, source.data(), result.data, 0, k_ff, k_embd, importance);
    require(written == bytes, "unexpected quantized role size");
    return result;
}

quant_data make_quant_data(ggml_type type, const char * name,
                           const std::vector<float> & source,
                           const std::vector<float> & imatrix) {
    aligned_buffer base = quantize_base(type, source, imatrix);
    const size_t roles = (k_stream_target + base.size - 1) / base.size;
    aligned_buffer ring(roles * base.size);
    for (size_t role = 0; role < roles; ++role) {
        std::memcpy(ring.data + role * base.size, base.data, base.size);
    }
    std::printf("DATA type=%s role_bytes=%zu roles=%zu ring_bytes=%zu\n",
                name, base.size, roles, ring.size);
    return {type, name, base.size, roles, std::move(ring)};
}

aligned_buffer quantize_input(ggml_type type, int columns) {
    const ggml_type input_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
    const auto * input_traits = ggml_get_type_traits_cpu(input_type);
    require(input_traits && input_traits->from_float, "missing vec_dot input quantizer");
    std::vector<float> input(static_cast<size_t>(columns));
    for (int i = 0; i < columns; ++i) {
        input[size_t(i)] = 0.35f * std::sin(float(i) * 0.013f) + 0.1f * std::cos(float(i) * 0.031f);
    }
    aligned_buffer q(ggml_row_size(input_type, columns));
    input_traits->from_float(input.data(), q.data, columns);
    return q;
}

result run_vecdot(const bench_case & c) {
    const auto * traits = ggml_get_type_traits_cpu(c.data->type);
    require(traits && traits->vec_dot, "missing vec_dot kernel");
    aligned_buffer input = quantize_input(c.data->type, c.columns);
    const size_t row_bytes = ggml_row_size(c.data->type, c.columns);
    require(row_bytes * size_t(c.rows) == c.data->role_bytes, "matrix orientation changed role bytes");

    const size_t rounds = c.data->roles;
    std::vector<double> partial(size_t(c.threads), 0.0);
    std::atomic<int> ready{0};
    std::atomic<int> pin_errors{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    workers.reserve(size_t(c.threads));

    for (int tid = 0; tid < c.threads; ++tid) {
        workers.emplace_back([&, tid] {
            pin_this_thread(tid, pin_errors);
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            double sum = 0.0;
            const int begin = c.rows * tid / c.threads;
            const int end   = c.rows * (tid + 1) / c.threads;
            for (size_t round = 0; round < rounds; ++round) {
                const size_t role = c.streaming ? round : 0;
                const uint8_t * base = c.data->weights.data + role * c.data->role_bytes;
                for (int row = begin; row < end; ++row) {
                    float value = 0.0f;
                    traits->vec_dot(c.columns, &value, 0, base + size_t(row) * row_bytes,
                                    0, input.data, 0, 1);
                    sum += value;
                }
            }
            partial[size_t(tid)] = sum;
        });
    }

    while (ready.load(std::memory_order_acquire) != c.threads) {
        std::this_thread::yield();
    }
    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto & worker : workers) {
        worker.join();
    }
    const auto end = std::chrono::steady_clock::now();
    require(pin_errors.load(std::memory_order_relaxed) == 0, "thread affinity failed");

    const double seconds = std::chrono::duration<double>(end - start).count();
    const size_t bytes = rounds * c.data->role_bytes;
    const uint64_t weights = uint64_t(rounds) * uint64_t(c.columns) * uint64_t(c.rows);
    return {seconds, double(bytes) / seconds / 1e9, 2.0 * double(weights) / seconds / 1e9,
            std::accumulate(partial.begin(), partial.end(), 0.0), bytes, weights};
}

uint64_t scan_bytes(const uint8_t * data, size_t size) {
#if defined(__AVX2__)
    __m256i accum0 = _mm256_setzero_si256();
    __m256i accum1 = _mm256_setzero_si256();
    __m256i accum2 = _mm256_setzero_si256();
    __m256i accum3 = _mm256_setzero_si256();
    __m256i accum4 = _mm256_setzero_si256();
    __m256i accum5 = _mm256_setzero_si256();
    __m256i accum6 = _mm256_setzero_si256();
    __m256i accum7 = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 256 <= size; i += 256) {
        accum0 = _mm256_xor_si256(accum0, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i +   0)));
        accum1 = _mm256_xor_si256(accum1, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i +  32)));
        accum2 = _mm256_xor_si256(accum2, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i +  64)));
        accum3 = _mm256_xor_si256(accum3, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i +  96)));
        accum4 = _mm256_xor_si256(accum4, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i + 128)));
        accum5 = _mm256_xor_si256(accum5, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i + 160)));
        accum6 = _mm256_xor_si256(accum6, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i + 192)));
        accum7 = _mm256_xor_si256(accum7, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i + 224)));
    }
    accum0 = _mm256_xor_si256(accum0, accum1);
    accum2 = _mm256_xor_si256(accum2, accum3);
    accum4 = _mm256_xor_si256(accum4, accum5);
    accum6 = _mm256_xor_si256(accum6, accum7);
    accum0 = _mm256_xor_si256(accum0, accum2);
    accum4 = _mm256_xor_si256(accum4, accum6);
    accum0 = _mm256_xor_si256(accum0, accum4);
    for (; i + 32 <= size; i += 32) {
        accum0 = _mm256_xor_si256(accum0, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i)));
    }
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), accum0);
    uint64_t result = lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3];
    for (; i < size; ++i) {
        result = (result << 1) ^ data[i];
    }
    return result;
#else
    uint64_t result = 0;
    for (size_t i = 0; i < size; ++i) {
        result = (result << 1) ^ data[i];
    }
    return result;
#endif
}

result run_raw_scan(const quant_data & data, int threads, bool streaming) {
    const size_t rounds = data.roles;
    std::vector<uint64_t> partial(static_cast<size_t>(threads));
    std::atomic<int> ready{0};
    std::atomic<int> pin_errors{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    workers.reserve(size_t(threads));
    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&, tid] {
            pin_this_thread(tid, pin_errors);
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            uint64_t sum = 0;
            const size_t begin = data.role_bytes * size_t(tid) / size_t(threads);
            const size_t end   = data.role_bytes * size_t(tid + 1) / size_t(threads);
            for (size_t round = 0; round < rounds; ++round) {
                const size_t role = streaming ? round : 0;
                sum ^= scan_bytes(data.weights.data + role * data.role_bytes + begin, end - begin);
            }
            partial[size_t(tid)] = sum;
        });
    }
    while (ready.load(std::memory_order_acquire) != threads) {
        std::this_thread::yield();
    }
    const auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto & worker : workers) {
        worker.join();
    }
    const auto end = std::chrono::steady_clock::now();
    require(pin_errors.load(std::memory_order_relaxed) == 0, "raw-scan affinity failed");
    const double seconds = std::chrono::duration<double>(end - start).count();
    const size_t bytes = rounds * data.role_bytes;
    const uint64_t checksum = std::accumulate(partial.begin(), partial.end(), uint64_t(0), std::bit_xor<uint64_t>());
    return {seconds, double(bytes) / seconds / 1e9, 0.0, double(checksum), bytes, 0};
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double pos = fraction * double(values.size() - 1);
    const size_t lo = size_t(pos);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    return values[lo] + (values[hi] - values[lo]) * (pos - double(lo));
}

} // namespace

int main() {
    try {
        ggml_cpu_init();
        std::printf("HOST avx2=%d avx_vnni=%d fma=%d stream_target=%zu samples=%d\n",
                    ggml_cpu_has_avx2(), ggml_cpu_has_avx_vnni(), ggml_cpu_has_fma(),
                    k_stream_target, k_samples);

        const size_t elements = size_t(k_embd) * size_t(k_ff);
        std::vector<float> source(elements);
        uint32_t state = 0x41d5a17u;
        for (size_t i = 0; i < elements; ++i) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            source[i] = (float(int32_t(state & 0xffffu)) - 32768.0f) / 32768.0f;
        }
        std::vector<float> imatrix(size_t(k_embd), 1.0f);

        std::vector<quant_data> datasets;
        datasets.emplace_back(make_quant_data(GGML_TYPE_IQ2_XXS, "iq2_xxs", source, imatrix));
        datasets.emplace_back(make_quant_data(GGML_TYPE_Q2_K, "q2_K", source, imatrix));
        datasets.emplace_back(make_quant_data(GGML_TYPE_MXFP4, "mxfp4", source, imatrix));
        source.clear();
        source.shrink_to_fit();

        std::vector<bench_case> cases;
        for (const auto & data : datasets) {
            for (int threads : {1, 9}) {
                for (bool streaming : {false, true}) {
                    cases.push_back({&data, k_embd, k_ff, threads, streaming});
                    cases.push_back({&data, k_ff, k_embd, threads, streaming});
                }
            }
        }

        std::map<std::string, std::vector<result>> observations;
        std::mt19937 rng(20260920);
        for (int sample = 0; sample < k_samples; ++sample) {
            std::shuffle(cases.begin(), cases.end(), rng);
            for (const auto & c : cases) {
                const result r = run_vecdot(c);
                observations[c.key()].push_back(r);
                std::printf("SAMPLE key=%s rep=%d seconds=%.9f bytes=%zu weights=%llu GBps=%.6f GFLOPs=%.6f checksum=%.9g\n",
                            c.key().c_str(), sample, r.seconds, r.bytes,
                            static_cast<unsigned long long>(r.weights), r.gbps, r.gflops, r.checksum);
                std::fflush(stdout);
            }
        }

        for (const auto & data : datasets) {
            for (bool streaming : {false, true}) {
                const std::string key = data.name + "/raw/t9/" + (streaming ? "stream" : "hot");
                for (int sample = 0; sample < k_samples; ++sample) {
                    const result r = run_raw_scan(data, 9, streaming);
                    observations[key].push_back(r);
                    std::printf("SAMPLE key=%s rep=%d seconds=%.9f bytes=%zu GBps=%.6f checksum=%.9g\n",
                                key.c_str(), sample, r.seconds, r.bytes, r.gbps, r.checksum);
                }
            }
        }

        for (const auto & item : observations) {
            std::vector<double> gbps;
            std::vector<double> gflops;
            std::vector<double> seconds;
            for (const result & r : item.second) {
                gbps.push_back(r.gbps);
                gflops.push_back(r.gflops);
                seconds.push_back(r.seconds);
            }
            std::printf("SUMMARY key=%s n=%zu seconds_med=%.9f GBps_p10=%.6f GBps_med=%.6f GBps_p90=%.6f GFLOPs_med=%.6f\n",
                        item.first.c_str(), item.second.size(), percentile(seconds, 0.5),
                        percentile(gbps, 0.1), percentile(gbps, 0.5), percentile(gbps, 0.9),
                        percentile(gflops, 0.5));
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "q2 vec_dot benchmark failed: %s\n", e.what());
        return 1;
    }
}
