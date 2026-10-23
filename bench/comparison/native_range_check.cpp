// Standalone compatibility check; compile against the frozen v0.1.13 reference.
// Its existing scalar/graph oracle is reused without changing production sources.
#define main release_fixture_main
#include "tests/test-kimi-cpu-sink.cpp"
#undef main

static bool native_range(void * user, const char * name, int32_t expert, kimi_cpu_sink_source * out) {
    auto & f = *static_cast<fixture *>(user);
    auto found = f.entries.find({name, expert});
    if (found == f.entries.end()) { return false; }
    const auto & item = found->second;
    *out = {};
    out->fd_direct = f.fd;
    out->data_offset = item.offset % 4096;
    out->offset = item.offset - out->data_offset;
    out->payload_nbytes = item.raw_bytes;
    out->nbytes = aligned(out->data_offset + item.raw_bytes);
    return true;
}

int main() {
    try {
        ggml_cpu_init();
        setenv("GGML_KIMI_CPU_SINK_CACHE_POLICY", "lru", 1);
        setenv("GGML_KIMI_CPU_SINK_CACHE_ALLOCATION", "uniform", 1);
        setenv("GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES", "536870912", 1);
        fixture f; f.type = f.down_type = GGML_TYPE_MXFP4; make_synthetic(f);
        close(f.fd); f.fd = -1;
        std::ofstream stream(f.path, std::ios::binary | std::ios::trunc);
        const char prefix[32] = {};
        for (auto & [key, item] : f.entries) {
            put_bytes(stream, prefix, sizeof(prefix));
            item.offset = static_cast<uint64_t>(stream.tellp());
            put_bytes(stream, item.data.data(), item.raw_bytes);
        }
        stream.close();
        f.fd = open(f.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        require(f.fd >= 0, "opening native-range fixture");
        char error[512] = {};
        sink_ptr sink(kimi_cpu_sink_create(2, 2, native_range, &f, error, sizeof(error)), kimi_cpu_sink_destroy);
        require(bool(sink), error);
        const kimi_cpu_sink_activation activation = {KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP, 0, 0, 10};
        const std::vector<int32_t> ids = {19, 0, 8, 2, 4, 17};
        const std::vector<float> weights = {.3f, .25f, .15f, .12f, .1f, .08f};
        const std::vector<float> input(f.latent, .13f);
        configure_cache(sink.get(), 8 * 1024 * 1024, {2}, 6);
        tensors metadata(f, 2);
        const auto expected = reference_moe_graph(f, 2, input, ids, weights, activation, 1);
        for (int pass = 0; pass < 2; ++pass) {
            begin_cache(sink.get(), 1, 1 + pass);
            std::vector<float> actual(f.latent); kimi_cpu_sink_stats stats = {};
            require(kimi_cpu_sink_run_activation(sink.get(), metadata.role[0], metadata.role[1], metadata.role[2],
                input.data(), ids.data(), weights.data(), int(ids.size()), &activation,
                actual.data(), &stats, error, sizeof(error)), error);
            require(!std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)),
                "native payload or cache output differs from the release graph oracle");
            require(stats.cache_hits == (pass ? 6 : 0) && stats.cache_admitted == (pass ? 0 : 6)
                && stats.read_errors == 0, "native cold/cache-hit counters differ");
        }
        sink.reset();
        struct stat info = {};
        require(!fstat(f.fd, &info) && !truncate(f.path.c_str(), info.st_size - 1), "truncate last payload");
        sink.reset(kimi_cpu_sink_create(2, 2, native_range, &f, error, sizeof(error)));
        require(bool(sink), error);
        tensors meta(f, 2); std::vector<float> output(f.latent); kimi_cpu_sink_stats stats = {};
        require(!kimi_cpu_sink_run_activation(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(),
            ids.data(), weights.data(), int(ids.size()), &activation, output.data(), &stats, error, sizeof(error)),
            "truncated logical weight unexpectedly accepted");
        require(stats.read_errors > 0, "truncated weight must produce a read error");
        std::puts("RELEASE_GGUF_RANGE_OK unaligned=1 complete_eof=1 cache_hit=1 truncated_rejected=1");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "RELEASE_GGUF_RANGE_FAILED: %s\n", error.what());
        return 1;
    }
}
