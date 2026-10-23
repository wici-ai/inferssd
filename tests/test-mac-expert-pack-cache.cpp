// Exercise the cache implementation without a model, GPU or Apple-only I/O.
#define GGML_MAC_EXPERT_CACHE_TESTING
#include "../ggml/src/ggml-cpu/ssd/mac-expert-source.cpp"

#include <functional>

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

void concurrent_batches() {
    fixture file;
    for (int expert = 0; expert < 3; ++expert) {
        std::vector<unsigned char> payload(fixture::slice_bytes, 0x21 + expert);
        file.write(payload.data(), payload.size(), alignment + expert * fixture::slice_bytes);
    }
    setenv("SSD_LLAMA_MAC_EXPERT_READ_AHEAD", "2", 1);
    pack_cache cache(file.path, false);
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
    unsetenv("SSD_LLAMA_MAC_EXPERT_READ_AHEAD");
}
}

int main() {
    try {
        setenv("SSD_LLAMA_MAC_EXPERT_CACHE_MIB", "1", 1);
        setenv("SSD_LLAMA_MAC_EXPERT_CACHE_POLICY", "lfu", 1);
        unsetenv("SSD_LLAMA_MAC_EXPERT_TRACE");
        {
            fixture file;
            pack_cache cache(file.path, false);
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
            pack_cache cache(file.path, false);
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
        {
            fixture file;
            const unsigned char invalid = 'X';
            file.write(&invalid, 1, 0);
            rejects([&] { pack_cache cache(file.path, false); });
        }
        {
            fixture file;
            uint64_t start = 0;
            file.write(&start, sizeof(start), offsetof(pack_header, data_start));
            rejects([&] { pack_cache cache(file.path, false); });
        }
        {
            fixture file;
            uint64_t count = max_entries + 1;
            file.write(&count, sizeof(count), offsetof(pack_header, entry_count));
            rejects([&] { pack_cache cache(file.path, false); });
        }
        {
            fixture file;
            uint64_t offset = UINT64_MAX;
            file.write(&offset, sizeof(offset), sizeof(pack_header) + offsetof(pack_entry, offset));
            rejects([&] { pack_cache cache(file.path, false); });
        }
        {
            fixture file;
            int32_t duplicate = 0;
            file.write(&duplicate, sizeof(duplicate), sizeof(pack_header) + sizeof(pack_entry) + offsetof(pack_entry, expert));
            rejects([&] { pack_cache cache(file.path, false); });
        }
        std::puts("MAC_EXPERT_PACK_CACHE_OK header=PASS lfu=PASS pins=PASS concurrent_batches=PASS invalid_entries=PASS");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "MAC_EXPERT_PACK_CACHE_FAILED: %s\n", error.what());
        return 1;
    }
}
