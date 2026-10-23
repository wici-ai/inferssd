#include "gpu-kernels.h"
#include "explicit-fixture-inputs.h"
// Real-pack DeepSeek-V4.1 prefill crossover benchmark.
//
// The timed region has no storage access.  Setup reads one expert triplet,
// admits it once to the decode RAM cache, and loads a separate aligned copy for
// the GPU tier.  Every CPU sample must report one read-only RAM source and zero
// SSD bytes.  Run under the project's 7 GB/s H2D governor and 16 GiB cgroup.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main
#include "vram-tier.cpp"
#include "kimi-mxfp4-batch.h"

#include <atomic>
#include <filesystem>
#include <mutex>

namespace crossover {

using steady = std::chrono::steady_clock;

struct options {
    std::string format;
    std::string pack;
    std::string jsonl;
    std::string real_input;
    int threads = 10;
    int warmup = 2;
    int repeats = 7;
    bool batch = false;
    bool integration_check = false;
};

struct aligned_bytes {
    uint8_t * data = nullptr;
    size_t size = 0;
    aligned_bytes() = default;
    explicit aligned_bytes(size_t n) : size(n) {
        if (posix_memalign(reinterpret_cast<void **>(&data), 4096, n) != 0) {
            throw std::bad_alloc();
        }
    }
    aligned_bytes(const aligned_bytes &) = delete;
    aligned_bytes & operator=(const aligned_bytes &) = delete;
    aligned_bytes(aligned_bytes && other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    aligned_bytes & operator=(aligned_bytes && other) noexcept {
        if (this != &other) {
            std::free(data);
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
    ~aligned_bytes() { std::free(data); }
};

struct real_expert {
    fixture pack;
    int layer = 1;
    int expert = 0;
    size_t role_bytes[3] = {};
    aligned_bytes role[3];
    struct stat status = {};
};

static void read_exact(int fd, void * destination, size_t bytes, uint64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        const ssize_t got = pread(fd, static_cast<uint8_t *>(destination) + done,
                bytes - done, off_t(offset + done));
        require(got > 0, "short expert-pack read during untimed setup");
        done += size_t(got);
    }
}

static void load_expert(real_expert & out, const options & o) {
    out.pack.latent = 5120;
    out.pack.hidden = 2304;
    out.pack.experts = 384;
    if (o.format == "mxfp4") {
        out.pack.type = out.pack.down_type = GGML_TYPE_MXFP4;
    } else if (o.format == "q2") {
        out.pack.type = GGML_TYPE_IQ2_XXS;
        out.pack.down_type = GGML_TYPE_Q2_K;
    } else {
        throw std::runtime_error("--format must be mxfp4 or q2");
    }

    std::ifstream file(o.pack, std::ios::binary);
    require(bool(file), "cannot open expert pack");
    char magic[16] = {};
    uint32_t version = 0, header = 0;
    uint64_t count = 0, data_start = 0;
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&header), sizeof(header));
    file.read(reinterpret_cast<char *>(&count), sizeof(count));
    file.read(reinterpret_cast<char *>(&data_start), sizeof(data_start));
    require(bool(file) && !std::memcmp(magic, "GGMLMOEPACKv1", 12) &&
            version == 1 && header == 40 && count < 1000000 &&
            data_start >= 40 + count * 152, "invalid expert-pack header");

    for (uint64_t i = 0; i < count; ++i) {
        char name[129] = {};
        int32_t expert = -1;
        uint32_t reserved = 0;
        uint64_t offset = 0, bytes = 0;
        file.read(name, 128);
        file.read(reinterpret_cast<char *>(&expert), sizeof(expert));
        file.read(reinterpret_cast<char *>(&reserved), sizeof(reserved));
        file.read(reinterpret_cast<char *>(&offset), sizeof(offset));
        file.read(reinterpret_cast<char *>(&bytes), sizeof(bytes));
        require(bool(file), "truncated expert-pack index");
        if (expert != out.expert) { continue; }
        for (int role = 0; role < 3; ++role) {
            if (name != tensor_name(out.layer, role)) { continue; }
            const ggml_type type = role == 2 ? out.pack.down_type : out.pack.type;
            const int columns = role == 2 ? out.pack.hidden : out.pack.latent;
            const int rows = role == 2 ? out.pack.latent : out.pack.hidden;
            const size_t logical = ggml_row_size(type, columns) * size_t(rows);
            const size_t physical = aligned(logical);
            require(bytes == logical && offset % 4096 == 0,
                    "expert-pack role geometry differs from the real tensor shape");
            entry item;
            item.offset = offset;
            item.raw_bytes = size_t(bytes);
            item.read_bytes = physical;
            require(out.pack.entries.emplace(key { name, expert }, std::move(item)).second,
                    "duplicate expert-pack role");
            out.role_bytes[role] = physical;
        }
    }
    require(out.pack.entries.size() == 3, "selected expert triplet is missing");
    file.close();

    out.pack.path = o.pack;
    out.pack.fd = open(o.pack.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    require(out.pack.fd >= 0, "cannot open expert pack with O_DIRECT");
    require(fstat(out.pack.fd, &out.status) == 0, "cannot stat expert pack");
    for (int role = 0; role < 3; ++role) {
        out.role[role] = aligned_bytes(out.role_bytes[role]);
        const auto & item = out.pack.entries.at({ tensor_name(out.layer, role), out.expert });
        read_exact(out.pack.fd, out.role[role].data, item.read_bytes, item.offset);
    }
}

struct sample {
    double wall_us = 0;
    double internal_us = 0;
    double checksum = 0;
    uint64_t ram_experts = 0;
    uint64_t ssd_bytes = 0;
};

struct json_writer {
    std::ofstream output;
    explicit json_writer(const std::string & path) {
        if (!path.empty()) {
            output.open(path, std::ios::out | std::ios::trunc);
            require(bool(output), "cannot create crossover JSONL");
        }
    }
    void write(const std::string & format, const std::string & kind, int multiplicity,
            int iteration, const sample & value, uint64_t bytes = 0, int lanes = 1) {
        if (!output.is_open()) { return; }
        output << std::setprecision(12)
            << "{\"schema\":1,\"format\":\"" << format << "\",\"kind\":\"" << kind
            << "\",\"M\":" << multiplicity << ",\"iteration\":" << iteration
            << ",\"wall_us\":" << value.wall_us << ",\"internal_us\":" << value.internal_us
            << ",\"checksum\":" << value.checksum << ",\"ram_experts\":" << value.ram_experts
            << ",\"ssd_bytes\":" << value.ssd_bytes << ",\"bytes\":" << bytes
            << ",\"lanes\":" << lanes << "}\n";
        output.flush();
        require(bool(output), "cannot write crossover JSONL");
    }
};

static void configure_environment(fixture_inputs & inputs) {
    inputs.set("ready_first", "1");
    inputs.set("split_cv", "1");
    inputs.set("early_up_gate", "1");
    inputs.set("role_major", "1");
    inputs.set("nonblocking_down_sched", "1");
    inputs.set("dynamic_rows", "1");
    inputs.set("dynamic_chunk_rows", "64");
    inputs.set("cache_policy", "lru");
    inputs.set("cache_allocation", "uniform");
    inputs.set("cache_keep", "1");
    inputs.set("cache_prefill_read", "1");
    inputs.set("cache_reserve_bytes", "536870912");
    inputs.set("vram_budget", "0");
    inputs.set("vram_prefill", "0");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    inputs.set("tier.keep_across_requests", "1");
    inputs.set("vram_async", "0");
    inputs.set("vram_deferred", "0");
    inputs.set("tier.global", "0");
    inputs.set("tier.elastic", "0");
    inputs.set("tier.safety_mib", "32");
    inputs.set("tier.mxfp4_gpu_hidden", "1");
    inputs.set("tier.timing", "1");
}

struct cpu_runner {
    real_expert & expert;
    tensors metadata;
    sink_ptr sink;
    std::unique_ptr<kimi_cpu_prefill, decltype(&kimi_cpu_prefill_destroy)> prefill;
    kimi_cpu_sink_activation activation = {
        KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f,
    };
    std::vector<float> input;
    std::vector<int32_t> ids;
    std::vector<float> weights;
    std::vector<float> output;

    cpu_runner(fixture_inputs & inputs, real_expert & value, int threads, int maximum) :
            expert(value), metadata(expert.pack, expert.layer),
            sink(make_sink(inputs, expert.pack, threads, 8)),
            prefill(nullptr, kimi_cpu_prefill_destroy),
            input(size_t(maximum) * size_t(expert.pack.latent)),
            ids(size_t(maximum), expert.expert), weights(size_t(maximum), 1.0f),
            output(size_t(maximum) * size_t(expert.pack.latent)) {
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = 0.4f * std::sin(float(i % 8191) * 0.011f) +
                0.17f * std::cos(float(i % 4093) * 0.019f);
        }
        const uint64_t stride = expert.role_bytes[0] + expert.role_bytes[1] + expert.role_bytes[2];
        configure_cache(sink.get(), stride, { expert.layer }, 1);
        begin_cache(sink.get(), 20260921, 1, true);
        std::vector<float> seed_output(size_t(expert.pack.latent));
        kimi_cpu_sink_stats seed = {};
        char error[512] = {};
        require(kimi_cpu_sink_run_activation(sink.get(), metadata.role[0], metadata.role[1], metadata.role[2],
                input.data(), ids.data(), weights.data(), 1, &activation, seed_output.data(),
                &seed, error, sizeof(error)), std::string("RAM seed: ") + error);
        require(seed.cache_admitted == 1 && seed.read_requests == 3,
                "RAM seed did not admit exactly one complete expert");
        begin_cache(sink.get(), 20260921, 1, false);
        prefill.reset(kimi_cpu_prefill_create(sink.get(), error, sizeof(error)));
        require(bool(prefill), std::string("prefill create: ") + error);
        std::fprintf(stderr, "CROSSOVER_SETUP source=ram_cache format=%s seed_read_bytes=%llu timed_ssd_bytes=0\n",
            expert.pack.type == GGML_TYPE_MXFP4 ? "mxfp4" : "q2",
            (unsigned long long) seed.read_bytes);
    }

    sample run(int multiplicity) {
        kimi_cpu_prefill_stats stats = {};
        char error[512] = {};
        const auto start = steady::now();
        require(kimi_cpu_prefill_run_activation(prefill.get(), metadata.role[0], metadata.role[1], metadata.role[2],
                input.data(), ids.data(), weights.data(), 1, multiplicity, &activation, output.data(),
                &stats, error, sizeof(error)), std::string("RAM prefill: ") + error);
        sample result;
        result.wall_us = std::chrono::duration<double, std::micro>(steady::now() - start).count();
        result.internal_us = stats.stream.compute_us;
        result.ram_experts = stats.ram_experts;
        result.ssd_bytes = stats.ssd_bytes;
        require(stats.ram_experts == 1 && stats.ssd_experts == 0 && stats.ssd_bytes == 0 &&
                stats.stream.read_requests == 0 && stats.stream.pending_reads == 0,
                "timed CPU sample touched SSD or missed the RAM lease");
        for (size_t i = 0; i < size_t(multiplicity) * size_t(expert.pack.latent); ++i) {
            require(std::isfinite(output[i]), "CPU sample produced a non-finite output");
            result.checksum += output[i];
        }
        return result;
    }
};

static kimi_vram_request make_request(const real_expert & expert, int id) {
    kimi_vram_request request = {};
    request.layer = expert.layer;
    request.expert = id;
    request.weight = 1.0f;
    for (int role = 0; role < 3; ++role) {
        const auto & item = expert.pack.entries.at({ tensor_name(expert.layer, role), expert.expert });
        request.role[role] = {
            uint64_t(expert.status.st_dev), uint64_t(expert.status.st_ino), uint64_t(expert.status.st_size),
            item.offset, item.read_bytes,
            expert.status.st_mtim.tv_sec, expert.status.st_mtim.tv_nsec,
            expert.status.st_ctim.tv_sec, expert.status.st_ctim.tv_nsec,
        };
    }
    return request;
}

struct gpu_runner {
    real_expert & expert;
    std::unique_ptr<tier> cache;
    std::array<int32_t, 1> layers = { 1 };
    kimi_vram_expert_format format = {};
    kimi_vram_prefill_launch activation = {
        { KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP, 0.0f, 0.0f, 10.0f }, nullptr, 0.0f, 0,
    };
    std::vector<float> & f32;
    aligned_bytes quantized;
    size_t input_stride = 0;
    std::vector<int32_t> routes;
    std::vector<float> weights;
    std::vector<float> output;
    kimi_vram_request prepared = {};
    uint64_t epoch = 20260921;
    uint64_t token = 0;
    const kimi_mxfp4_batch_api * batch_api = nullptr;
    kimi_mxfp4_batch * batch = nullptr;

    gpu_runner(real_expert & value, std::vector<float> & input, int maximum, bool use_batch = false) :
            expert(value), f32(input), routes(size_t(maximum), 0), weights(size_t(maximum), 1.0f),
            output(size_t(maximum) * size_t(value.pack.latent)) {
        const uint64_t stride = expert.role_bytes[0] + expert.role_bytes[1] + expert.role_bytes[2];
        cache = std::make_unique<tier>(stride);
        layers[0] = expert.layer;
        format = { expert.pack.latent, expert.pack.hidden, int32_t(expert.pack.type),
            int32_t(expert.pack.down_type), expert.pack.experts, int32_t(layers.size()), layers.data() };
        if (expert.pack.type == GGML_TYPE_IQ2_XXS) {
            input_stride = size_t(expert.pack.latent) * sizeof(float);
        } else {
            const auto * traits = ggml_get_type_traits_cpu(expert.pack.type);
            const auto * input_traits = traits ? ggml_get_type_traits_cpu(traits->vec_dot_type) : nullptr;
            require(traits && traits->vec_dot_type == GGML_TYPE_Q8_0 && input_traits && input_traits->from_float,
                    "MXFP4 Q8_0 input quantizer is missing");
            input_stride = ggml_row_size(traits->vec_dot_type, expert.pack.latent);
            quantized = aligned_bytes(size_t(maximum) * input_stride);
            for (int row = 0; row < maximum; ++row) {
                input_traits->from_float(f32.data() + size_t(row) * size_t(expert.pack.latent),
                        quantized.data + size_t(row) * input_stride, expert.pack.latent);
            }
        }
        auto get_batch = kimi_mxfp4_batch_get_api;
        if (use_batch && get_batch && expert.pack.type == GGML_TYPE_MXFP4) {
            batch_api = get_batch();
            require(batch_api && batch_api->version == 3 && batch_api->size == sizeof(*batch_api),
                    "MXFP4 batch ABI mismatch");
            char error[512] = {};
            batch = batch_api->create(0, expert.pack.latent, expert.pack.hidden,
                                      std::min(maximum, 256), error, sizeof(error));
            require(batch != nullptr, std::string("MXFP4 batch create: ") + error);
        }
    }
    ~gpu_runner() {
        if (batch && !batch_api->destroy(batch)) { std::abort(); }
    }

    const void * input_data() const {
        return expert.pack.type == GGML_TYPE_IQ2_XXS ?
            static_cast<const void *>(f32.data()) : static_cast<const void *>(quantized.data);
    }

    void prepare_upload(int id) {
        prepared = make_request(expert, id);
        cache->begin(epoch, ++token, true);
        int slot = -2;
        cache->prepare_format(&prepared, 1, &format, &slot);
        require(slot == -1, "upload sample unexpectedly hit a resident expert");
    }

    sample perform_upload() {
        const void * data[3] = { expert.role[0].data, expert.role[1].data, expert.role[2].data };
        const double before_internal = cache->upload_us;
        const auto start = steady::now();
        cache->admit(prepared, data, false);
        cache->finish(nullptr, size_t(expert.pack.latent));
        sample result;
        result.wall_us = std::chrono::duration<double, std::micro>(steady::now() - start).count();
        result.internal_us = cache->upload_us - before_internal;
        result.checksum = double(cache->upload_bytes);
        return result;
    }

    void enter_prefill() { cache->begin(epoch, token, false); }

    sample run_resident(int multiplicity) {
        int slot = -2;
        cache->prepare_prefill(&prepared, 1, &format, &slot);
        require(slot >= 0, "new upload was not resident for grouped prefill");
        std::fill(output.begin(), output.begin() + size_t(multiplicity) * size_t(expert.pack.latent), 0.0f);
        const double before_internal = cache->compute_us;
        const auto start = steady::now();
        cache->launch_prefill(input_data(), input_stride, routes.data(), weights.data(), 1,
                multiplicity, nullptr, 0, &activation);
        kimi_vram_prefill_stats prefill_stats = {};
        cache->finish_prefill(output.data(), size_t(expert.pack.latent), &prefill_stats);
        sample result;
        result.wall_us = std::chrono::duration<double, std::micro>(steady::now() - start).count();
        result.internal_us = cache->compute_us - before_internal;
        for (size_t i = 0; i < size_t(multiplicity) * size_t(expert.pack.latent); ++i) {
            require(std::isfinite(output[i]), "GPU sample produced a non-finite output");
            result.checksum += output[i];
        }
        return result;
    }
    sample run_batch(int multiplicity) {
        require(batch != nullptr, "MXFP4 batch engine missing");
        int slot = -2;
        cache->prepare_prefill(&prepared, 1, &format, &slot);
        require(slot >= 0, "batch weight was not resident");
        char * base = cache->slot_base(size_t(slot));
        char error[512] = {};
        const auto start = steady::now();
        require(batch_api->compute(batch, base, base + expert.role_bytes[0],
                base + expert.role_bytes[0] + expert.role_bytes[1],
                input_data(), multiplicity, activation.activation.swiglu_limit, output.data(), error, sizeof(error)),
                std::string("MXFP4 batch compute: ") + error);
        sample result;
        result.internal_us = std::chrono::duration<double, std::micro>(steady::now() - start).count();
        cache->cancel_prefill();
        result.wall_us = std::chrono::duration<double, std::micro>(steady::now() - start).count();
        for (size_t i = 0; i < size_t(multiplicity) * size_t(expert.pack.latent); ++i) {
            require(std::isfinite(output[i]), "GPU batch produced a non-finite output");
            result.checksum += output[i];
        }
        return result;
    }
};

static sample merge_sample(int multiplicity, int latent) {
    const size_t values = size_t(multiplicity) * size_t(latent);
    std::vector<float> source(values), destination(values);
    for (size_t i = 0; i < values; ++i) { source[i] = std::sin(float(i % 4093)); }
    const int repeats = std::max<int>(1, int(10000000 / std::max<size_t>(1, values)));
    const auto start = steady::now();
    double checksum = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        std::fill(destination.begin(), destination.end(), 0.0f);
        for (size_t i = 0; i < values; ++i) {
            volatile float scaled = source[i] * 0.9375f;
            destination[i] += scaled;
        }
        checksum += destination[size_t(repeat) % values];
    }
    sample result;
    result.wall_us = std::chrono::duration<double, std::micro>(steady::now() - start).count() / repeats;
    result.internal_us = result.wall_us;
    result.checksum = checksum;
    return result;
}

static double relative_l2(const float * a, const float * b, size_t size) {
    double error = 0.0, norm = 0.0;
    for (size_t i = 0; i < size; ++i) {
        const double delta = double(a[i]) - double(b[i]);
        error += delta * delta;
        norm += double(a[i]) * double(a[i]);
    }
    return norm > 0.0 ? std::sqrt(error / norm) : std::sqrt(error);
}

static sample concurrent_sample(cpu_runner & cpu, gpu_runner & gpu, int multiplicity,
        sample & cpu_part, sample & gpu_part) {
    std::atomic<int> ready { 0 };
    std::atomic<bool> go { false };
    auto wait = [&] {
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) { std::this_thread::yield(); }
    };
    std::thread cpu_thread([&] { wait(); cpu_part = cpu.run(multiplicity); });
    std::thread gpu_thread([&] { wait(); gpu_part = gpu.run_resident(multiplicity); });
    while (ready.load(std::memory_order_acquire) != 2) { std::this_thread::yield(); }
    const auto start = steady::now();
    go.store(true, std::memory_order_release);
    cpu_thread.join();
    gpu_thread.join();
    sample result;
    result.wall_us = std::chrono::duration<double, std::micro>(steady::now() - start).count();
    result.internal_us = std::max(cpu_part.wall_us, gpu_part.wall_us);
    result.checksum = cpu_part.checksum + gpu_part.checksum;
    return result;
}

static sample upload_lanes(std::vector<std::unique_ptr<gpu_runner>> & lanes, int id_base) {
    for (size_t i = 0; i < lanes.size(); ++i) { lanes[i]->prepare_upload(id_base + int(i)); }
    std::atomic<int> ready { 0 };
    std::atomic<bool> go { false };
    std::vector<sample> parts(lanes.size());
    std::vector<std::thread> workers;
    for (size_t i = 0; i < lanes.size(); ++i) {
        workers.emplace_back([&, i] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            parts[i] = lanes[i]->perform_upload();
        });
    }
    while (ready.load(std::memory_order_acquire) != int(lanes.size())) { std::this_thread::yield(); }
    const auto start = steady::now();
    go.store(true, std::memory_order_release);
    for (auto & worker : workers) { worker.join(); }
    sample result;
    result.wall_us = std::chrono::duration<double, std::micro>(steady::now() - start).count();
    for (const auto & part : parts) {
        result.internal_us += part.internal_us;
        result.checksum += part.checksum;
    }
    return result;
}

static options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char * name) -> std::string {
            require(i + 1 < argc, std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--format") result.format = value("--format");
        else if (arg == "--pack") result.pack = value("--pack");
        else if (arg == "--jsonl") result.jsonl = value("--jsonl");
        else if (arg == "--threads") result.threads = std::stoi(value("--threads"));
        else if (arg == "--warmup") result.warmup = std::stoi(value("--warmup"));
        else if (arg == "--repeats") result.repeats = std::stoi(value("--repeats"));
        else if (arg == "--batch") result.batch = true;
        else if (arg == "--integration-check") result.integration_check = true;
        else if (arg == "--real-input") result.real_input = value("--real-input");
        else throw std::runtime_error("unknown option: " + arg);
    }
    require((result.format == "mxfp4" || result.format == "q2") && !result.pack.empty() &&
            result.threads >= 2 && result.threads <= 64 && result.warmup >= 0 &&
            result.repeats >= 3 && result.repeats <= 31,
            "usage: test-kimi-prefill-crossover --format mxfp4|q2 --pack FILE [--jsonl FILE] "
            "[--threads 10] [--warmup 2] [--repeats 7]");
    return result;
}

static int run(fixture_inputs & inputs, int argc, char ** argv) {
    const options o = parse_options(argc, argv);
    configure_environment(inputs);
    ggml_cpu_init();
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu_backend(
        ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(cpu_backend), "failed to initialize CPU backend");
    real_expert expert;
    load_expert(expert, o);
    constexpr int maximum = 2048;
    const std::array<int, 11> multiplicities = { 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
    cpu_runner cpu(inputs, expert, o.threads, maximum);
    if (!o.real_input.empty()) {
        std::ifstream stream(o.real_input, std::ios::binary);
        require(bool(stream), "cannot open real activation dump");
        uint32_t header[3] = {};
        stream.read(reinterpret_cast<char *>(header), sizeof(header));
        require(bool(stream) && header[2] == 0x4d584631 && header[0] >= maximum &&
                header[1] == size_t(expert.pack.latent) * sizeof(float),
                "real activation dump has incompatible geometry");
        stream.read(reinterpret_cast<char *>(cpu.input.data()),
            std::streamsize(cpu.input.size() * sizeof(float)));
        require(bool(stream), "real activation dump is truncated");
    }
    gpu_runner gpu(expert, cpu.input, maximum, o.batch);
    json_writer json(o.jsonl);
    int expert_id = 1;
    double worst_relative = 0.0;

    std::printf("CROSSOVER_BEGIN format=%s threads=%d governor_expected_gbps=7 timed_ssd_bytes=0 role_bytes=%zu,%zu,%zu\n",
        o.format.c_str(), o.threads, expert.role_bytes[0], expert.role_bytes[1], expert.role_bytes[2]);
    for (int multiplicity : multiplicities) {
        for (int iteration = -o.warmup; iteration < o.repeats; ++iteration) {
            const sample cpu_value = cpu.run(multiplicity);
            gpu.prepare_upload(expert_id++);
            const sample upload = gpu.perform_upload();
            gpu.enter_prefill();
            const sample resident = o.batch ? gpu.run_batch(multiplicity) : gpu.run_resident(multiplicity);
            const sample merge = merge_sample(multiplicity, expert.pack.latent);
            if (iteration >= 0) {
                const double rel = relative_l2(cpu.output.data(), gpu.output.data(),
                        size_t(multiplicity) * size_t(expert.pack.latent));
                double max_abs = 0;
                for (size_t value = 0; value < size_t(multiplicity) * size_t(expert.pack.latent); ++value) {
                    max_abs = std::max(max_abs, std::abs(double(cpu.output[value]) - gpu.output[value]));
                }
                worst_relative = std::max(worst_relative, rel);
                require(std::isfinite(rel), "GPU crossover output has invalid relative L2");
                json.write(o.format, "cpu_ram", multiplicity, iteration, cpu_value);
                json.write(o.format, "gpu_upload", multiplicity, iteration, upload,
                    expert.role_bytes[0] + expert.role_bytes[1] + expert.role_bytes[2]);
                json.write(o.format, o.batch ? "gpu_batch" : "gpu_resident", multiplicity, iteration, resident);
                json.write(o.format, "merge", multiplicity, iteration, merge);
                sample combined = resident;
                combined.wall_us += upload.wall_us;
                combined.internal_us += upload.internal_us;
                json.write(o.format, "gpu_upload_plus_compute", multiplicity, iteration, combined,
                    expert.role_bytes[0] + expert.role_bytes[1] + expert.role_bytes[2]);
                std::printf("CROSSOVER_SAMPLE format=%s M=%d i=%d cpu_us=%.3f upload_us=%.3f gpu_resident_us=%.3f upload_gpu_us=%.3f merge_us=%.3f rel_l2=%.6g max_abs=%.6g\n",
                    o.format.c_str(), multiplicity, iteration, cpu_value.wall_us, upload.wall_us,
                    resident.wall_us, upload.wall_us + resident.wall_us, merge.wall_us, rel, max_abs);
            }
        }

        for (int iteration = 0; iteration < o.repeats; ++iteration) {
            sample cpu_part, gpu_part;
            if (o.batch) break;
            const sample concurrent = concurrent_sample(cpu, gpu, multiplicity, cpu_part, gpu_part);
            json.write(o.format, "concurrent_cpu_gpu", multiplicity, iteration, concurrent);
            json.write(o.format, "concurrent_cpu_part", multiplicity, iteration, cpu_part);
            json.write(o.format, "concurrent_gpu_part", multiplicity, iteration, gpu_part);
        }
    }

    if (o.batch) {
        if (o.integration_check) {
            constexpr int count = 32;
            cpu.run(count);
            gpu.run_resident(count);
            const double resident_error = relative_l2(cpu.output.data(), gpu.output.data(),
                size_t(count) * size_t(expert.pack.latent));
            require(resident_error < 0.001, "tier resident batch output differs from CPU");
            const uint64_t stride = expert.role_bytes[0] + expert.role_bytes[1] + expert.role_bytes[2];
            tier cold(stride, inputs.tier_policy(), nullptr, test_cpu_kernels());
            cold.begin(20260921, 0, false);
            const kimi_vram_request request = make_request(expert, 0);
            int slot = -2;
            cold.prepare_prefill(&request, 1, &gpu.format, &slot);
            require(slot == -1 && cold.transient_prefill_available(),
                "cold MXFP4 transient scratch is unavailable");
            kimi_vram_prefill_transient transient = {};
            transient.request_index = 0;
            transient.source = KIMI_VRAM_PREFILL_TRANSIENT_RAM;
            for (int role = 0; role < 3; ++role) transient.role[role] = expert.role[role].data;
            std::vector<float> cold_output(size_t(count) * size_t(expert.pack.latent), 0.0f);
            cold.launch_prefill(gpu.input_data(), gpu.input_stride, gpu.routes.data(), gpu.weights.data(),
                1, count, &transient, 1, &gpu.activation);
            kimi_vram_prefill_stats stats = {};
            cold.finish_prefill(cold_output.data(), size_t(expert.pack.latent), &stats);
            const double transient_error = relative_l2(cpu.output.data(), cold_output.data(),
                cold_output.size());
            require(transient_error < 0.001 && stats.transient_experts == 1 &&
                stats.ram_transient_routes == count && stats.transient_upload_bytes == stride,
                "tier cold transient batch result or accounting differs from CPU");
            std::printf("CROSSOVER_INTEGRATION resident_rel_l2=%.9g transient_rel_l2=%.9g "
                "transient_upload_bytes=%llu\n", resident_error, transient_error,
                (unsigned long long)stats.transient_upload_bytes);
        }
        std::printf("CROSSOVER_PASS format=%s worst_rel_l2=%.6g timed_ssd_bytes=0 samples_per_point=%d\n",
                o.format.c_str(), worst_relative, o.repeats);
        return 0;
    }
    std::vector<std::unique_ptr<gpu_runner>> lanes;
    for (int i = 0; i < 4; ++i) { lanes.emplace_back(std::make_unique<gpu_runner>(expert, cpu.input, maximum)); }
    for (int iteration = -o.warmup; iteration < o.repeats; ++iteration) {
        const sample queued = upload_lanes(lanes, 32 + (iteration + o.warmup) * 4);
        if (iteration >= 0) {
            json.write(o.format, "gpu_upload_4way", 0, iteration, queued,
                4 * (expert.role_bytes[0] + expert.role_bytes[1] + expert.role_bytes[2]), 4);
            std::printf("CROSSOVER_UPLOAD4 format=%s i=%d wall_us=%.3f aggregate_internal_us=%.3f\n",
                o.format.c_str(), iteration, queued.wall_us, queued.internal_us);
        }
    }
    std::printf("CROSSOVER_PASS format=%s worst_rel_l2=%.6g timed_ssd_bytes=0 samples_per_point=%d\n",
        o.format.c_str(), worst_relative, o.repeats);
    return 0;
}

} // namespace crossover

#ifndef KIMI_CROSSOVER_NO_MAIN
int main(int argc, char ** argv) {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services());
    try {
        return crossover::run(inputs, argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "CROSSOVER_FAIL: %s\n", error.what());
        return 1;
    }
}
#endif
