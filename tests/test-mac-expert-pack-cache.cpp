// Exercise the cache implementation without a model, GPU or Apple-only I/O.
#define GGML_MAC_EXPERT_CACHE_TESTING
#include "../ggml/src/ggml-cpu/ssd/mac-expert-source.cpp"

#include <functional>
#include <csignal>
#include <sys/resource.h>

namespace {
void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
struct fixture {
    char path[64] = "/tmp/ssdash-pack-test-XXXXXX";
    int fd;
    static constexpr size_t slice_bytes = 1 << 20;
    fixture() {
        fd = mkstemp(path);
        require(fd >= 0, "create temporary pack");
        require(ftruncate(fd, alignment + 3 * slice_bytes) == 0, "size pack");
        pack_header header {};
        std::memcpy(header.magic, "GGMLMOEPACKv1\0\0\0", 16);
        header.version = 1;
        header.header_size = sizeof(header);
        header.entry_count = 3;
        header.data_start = alignment;
        write(&header, sizeof(header), 0);
        for (int i = 0; i < 3; ++i) {
            pack_entry entry {};
            std::strcpy(entry.name, "blk.0.ffn_up_exps.weight");
            entry.expert = i;
            entry.offset = alignment + i * slice_bytes;
            entry.bytes = slice_bytes;
            write(&entry, sizeof(entry), sizeof(header) + i * sizeof(entry));
            fill(i, 0x11 + i);
        }
    }
    ~fixture() { close(fd); unlink(path); }
    void write(const void * data, size_t size, uint64_t offset) {
        require(pwrite(fd, data, size, static_cast<off_t>(offset)) == static_cast<ssize_t>(size), "write pack");
    }
    void fill(int expert, unsigned char value) { write(&value, 1, alignment + expert * slice_bytes); }
};
const char * tensor = "blk.0.ffn_up_exps.weight";
cache_options small_cache_options() { cache_options options; options.budget = size_t(1) << 20; return options; }
void rejects(const std::function<void()> & run) {
    bool rejected = false;
    try { run(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "invalid input accepted");
}
void read_byte(pack_cache & cache, int id, unsigned char expected) {
    const auto * data = static_cast<const unsigned char *>(cache.acquire(tensor, id, fixture::slice_bytes));
    require(data[0] == expected, "incorrect expert byte");
    cache.release(data);
}

void rejects_foreign_batch() {
    fixture file;
    pack_cache first({file.path}, small_cache_options()), second({file.path}, small_cache_options());
    const int64_t counts[] {1};
    auto * batch = first.begin(tensor, counts, 1, fixture::slice_bytes, true);
    rejects([&] { second.acquire(tensor, 0, fixture::slice_bytes, batch); });
    rejects([&] { second.end(batch); });
    const auto * payload = first.acquire(tensor, 0, fixture::slice_bytes, batch);
    require(static_cast<const unsigned char *>(payload)[0] == 0x11, "foreign rejection mutated original batch");
    first.release(payload); first.end(batch);
}

// A real regular writer hits EFBIG in a reader thread. Limit/handler restoration
// precedes destructor diagnostics, so the rest of the fixture process is intact.
void writer_failure_drains(bool acquire_boundary) {
    fixture file;
    const std::string path = std::string(file.path) + ".full.jsonl";
    auto opened = ssd::telemetry::writer::open(path); require(bool(opened), "open writer-full fixture");
    auto writer = std::move(opened.value);
    require(bool(writer->event("ssd.fixture.ready", "{}")), "prime real writer");
    struct stat info {}; require(stat(path.c_str(), &info) == 0, "stat real writer");
    {
        std::promise<void> start;
        auto options = small_cache_options(); options.events = writer.get(); options.debug.profile = true;
        options.read_gate = start.get_future().share();
        pack_cache cache({file.path}, options);
        struct bounded_file_size {
            struct rlimit previous {};
            struct sigaction signal_before {};
            explicit bounded_file_size(rlim_t bytes) {
                require(getrlimit(RLIMIT_FSIZE, &previous) == 0, "read file-size limit");
                struct sigaction ignore {}; ignore.sa_handler = SIG_IGN; sigemptyset(&ignore.sa_mask);
                require(sigaction(SIGXFSZ, &ignore, &signal_before) == 0, "ignore expected SIGXFSZ");
                auto limit = previous; limit.rlim_cur = bytes;
                if (setrlimit(RLIMIT_FSIZE, &limit) != 0) {
                    sigaction(SIGXFSZ, &signal_before, nullptr);
                    throw std::runtime_error("set actual file-size limit");
                }
            }
            ~bounded_file_size() {
                if (setrlimit(RLIMIT_FSIZE, &previous) != 0 || sigaction(SIGXFSZ, &signal_before, nullptr) != 0) { std::abort(); }
            }
        };
        const int64_t counts[] {1,1,1};
        auto * batch = cache.begin(tensor, counts, 3, fixture::slice_bytes, true);
        {
            bounded_file_size limit(rlim_t(info.st_size));
            start.set_value();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!cache.fixture_publication_failed() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            require(cache.fixture_publication_failed(), "real writer did not fail under RLIMIT_FSIZE");
            bool rejected = false;
            try {
                if (acquire_boundary) { cache.acquire(tensor, 0, fixture::slice_bytes, batch); }
                else { auto * owned = batch; batch = nullptr; cache.end(owned); }
            } catch (const std::exception & e) {
                rejected = std::strstr(e.what(), "telemetry publication failed after reader drain") != nullptr;
            }
            require(rejected && cache.fixture_readers_joined(), "writer failure returned before reader threads joined");
            require(cache.fixture_submitted_reads() == 2 && cache.fixture_read_count() == 2,
                    "writer failure cancelled or left an issued real pread incomplete");
            const auto before = cache.fixture_read_count();
            rejects([&] { cache.acquire(tensor, 0, fixture::slice_bytes); });
            require(cache.fixture_read_count() == before && cache.fixture_readers_joined(), "failed cache accepted another read");
            if (batch) { rejects([&] { auto * owned = batch; batch = nullptr; cache.end(owned); }); }
        }
        require(!writer->flush(), "real writer I/O failure was not sticky after resource restoration");
    } // cache drains and drops its borrowed writer before writer destruction.
    writer.reset(); unlink(path.c_str());
}

void configured_context_isolation() {
    fixture file;
    auto snapshot = [&](int mib) {
        auto parsed = ssd::settings::parse("inferssd-settings 1\nexpert.source = [" + ssd::json::quote(file.path) +
                "]\nhost.expert_cache_mib = " + std::to_string(mib) + "\n", "fixture", {}, "darwin");
        require(bool(parsed), "parse configured Mac settings"); return std::move(parsed.value);
    };
    const auto hot_settings = snapshot(1), transient_settings = snapshot(0);
    const std::string event_path = std::string(file.path) + ".events.jsonl";
    auto opened = ssd::telemetry::writer::open(event_path);
    require(bool(opened), "open per-context telemetry writer");
    auto writer = std::move(opened.value);
    ssd::telemetry::debug_channels debug; debug.trace = true;
    const ssd::runtime_services services{writer.get(), &debug};
    char error[512] {};
    std::unique_ptr<ggml_ssd_mac_cache, decltype(&ssd::mac::destroy)> hot(
            ssd::mac::create_configured(&hot_settings, &services, error, sizeof(error)), ssd::mac::destroy);
    require(bool(hot), error);
    std::unique_ptr<ggml_ssd_mac_cache, decltype(&ssd::mac::destroy)> transient(
            ssd::mac::create_configured(&transient_settings, nullptr, error, sizeof(error)), ssd::mac::destroy);
    require(bool(transient), error);
    debug.trace = false; // The cache owns its debug selection, not this mutable caller object.
    require(ggml_mac_expert_pack_enabled(hot.get()) && !ggml_mac_expert_pack_enabled(nullptr) &&
            ggml_mac_expert_pack_prefill_enabled(hot.get()), "configured hook enable/prefill policy");
    auto read = [&](ggml_ssd_mac_cache * owner, unsigned char expected) {
        const auto * data = static_cast<const unsigned char *>(
                ggml_mac_expert_pack_acquire(owner, tensor, 0, fixture::slice_bytes));
        require(data[0] == expected, "two configured contexts mixed cached bytes");
        ggml_mac_expert_pack_release(owner, data);
    };
    read(hot.get(), 0x11); file.fill(0, 0x66);
    read(transient.get(), 0x66); read(hot.get(), 0x11);
    auto other = std::async(std::launch::async, [&] {
        for (int i = 0; i < 16; ++i) { read(transient.get(), 0x66); }
    });
    hot.reset(); // Joins only this context's readers and closes only this source index.
    other.get(); read(transient.get(), 0x66); transient.reset();
    require(bool(writer->flush()), "flush context telemetry after its workers joined");
    writer.reset();
    std::ifstream events(event_path); std::string line; bool copied_trace = false, own_budget = false;
    while (std::getline(events, line)) {
        auto row = ssd::json::parse(line); require(bool(row), "invalid Mac telemetry JSONL");
        copied_trace = copied_trace || row.value.object.at("ev").text == "ssd.mac.expert_request";
        own_budget = own_budget || (row.value.object.count("budget_bytes") && row.value.object.at("budget_bytes").text == "1048576");
    }
    require(copied_trace && own_budget, "Mac telemetry lost copied debug selection or context budget");
    unlink(event_path.c_str());
}

void concurrent_batches() {
    fixture file;
    for (int expert = 0; expert < 3; ++expert) {
        std::vector<unsigned char> payload(fixture::slice_bytes, 0x21 + expert);
        file.write(payload.data(), payload.size(), alignment + expert * fixture::slice_bytes);
    }
    pack_cache cache({file.path}, small_cache_options());
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::vector<std::future<void>> workers;
    try {
        for (int caller = 0; caller < 4; ++caller) {
            workers.emplace_back(std::async(std::launch::async, [&, caller] {
                ready.wait();
                for (int iteration = 0; iteration < 16; ++iteration) {
                    const int64_t counts[] {1, 1, 1};
                    std::unique_ptr<expert_batch> batch(cache.begin(tensor, counts, 3, fixture::slice_bytes, caller % 2 != 0));
                    const unsigned char * pinned[3];
                    for (int expert = 0; expert < 3; ++expert) {
                        pinned[expert] = static_cast<const unsigned char *>(cache.acquire(tensor, expert, fixture::slice_bytes, batch.get()));
                        std::this_thread::yield();
                    }
                    bool valid = true;
                    for (int expert = 2; expert >= 0; --expert) {
                        for (size_t byte = 0; byte < fixture::slice_bytes; byte += 4096) {
                            valid = valid && pinned[expert][byte] == 0x21 + expert;
                        }
                        valid = valid && pinned[expert][fixture::slice_bytes - 1] == 0x21 + expert;
                        cache.release(pinned[expert]);
                    }
                    cache.end(batch.release());
                    require(valid, "concurrent prefill/decode changed pinned bytes");
                }
            }));
        }
    } catch (...) {
        // Release callers even if thread creation fails, before future cleanup.
        start.set_value();
        throw;
    }
    start.set_value();
    for (auto & worker : workers) { worker.get(); }
}
}

int main() {
    try {
        {
            fixture file;
            pack_cache cache({file.path}, small_cache_options());
            read_byte(cache, 0, 0x11);
            read_byte(cache, 0, 0x11);
            file.fill(0, 0x44);
            read_byte(cache, 1, 0x12);
            read_byte(cache, 0, 0x11); // LFU retains the hotter expert.
            for (int i = 0; i < 5; ++i) { read_byte(cache, 1, 0x12); }
            read_byte(cache, 0, 0x44); // A hotter expert eventually evicts it.
            rejects([&] { cache.acquire(tensor, 9, fixture::slice_bytes); });
            rejects([&] { cache.acquire(tensor, 0, 1); });
        }
        {
            fixture file;
            pack_cache cache({file.path}, small_cache_options());
            const auto * pinned = static_cast<const unsigned char *>(cache.acquire(tensor, 0, fixture::slice_bytes));
            file.fill(0, 0x55);
            for (int i = 0; i < 5; ++i) { read_byte(cache, 1, 0x12); }
            require(pinned[0] == 0x11, "pinned expert was evicted");
            const void * again = cache.acquire(tensor, 0, fixture::slice_bytes);
            require(again == pinned, "pinned expert was replaced");
            cache.release(again);
            cache.release(pinned);
            rejects([&] { cache.release(pinned); });
        }
        concurrent_batches();
        configured_context_isolation();
        rejects_foreign_batch();
        writer_failure_drains(false);
        writer_failure_drains(true);
        {
            fixture file;
            const unsigned char invalid = 'X';
            file.write(&invalid, 1, 0);
            rejects([&] { pack_cache cache({file.path}, small_cache_options()); });
        }
        {
            fixture file;
            uint64_t start = 0;
            file.write(&start, sizeof(start), offsetof(pack_header, data_start));
            rejects([&] { pack_cache cache({file.path}, small_cache_options()); });
        }
        {
            fixture file;
            uint64_t count = max_entries + 1;
            file.write(&count, sizeof(count), offsetof(pack_header, entry_count));
            rejects([&] { pack_cache cache({file.path}, small_cache_options()); });
        }
        {
            fixture file;
            uint64_t offset = UINT64_MAX;
            file.write(&offset, sizeof(offset), sizeof(pack_header) + offsetof(pack_entry, offset));
            rejects([&] { pack_cache cache({file.path}, small_cache_options()); });
        }
        {
            fixture file;
            int32_t duplicate = 0;
            file.write(&duplicate, sizeof(duplicate), sizeof(pack_header) + sizeof(pack_entry) + offsetof(pack_entry, expert));
            rejects([&] { pack_cache cache({file.path}, small_cache_options()); });
        }
        std::puts("MAC_EXPERT_PACK_CACHE_OK header=PASS lfu=PASS pins=PASS concurrent_batches=PASS contexts=PASS copied_debug=PASS telemetry=PASS foreign_batch=PASS writer_full_drains=PASS invalid_entries=PASS");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "MAC_EXPERT_PACK_CACHE_FAILED: %s\n", error.what());
        return 1;
    }
}
