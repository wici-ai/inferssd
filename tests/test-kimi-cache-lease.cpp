// Inspect private metadata without creating workers, a GPU context or a payload
// arena. Real file identities exercise version rejection and RAM lease vetoes.
#include "kimi-cpu-sink.cpp"
int main() { try {
    int checks = 0;
    auto require = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("cache lease assertion"); };
    setenv("GGML_KIMI_CPU_SINK_CACHE_POLICY","lru",1);
    char path[] = "/tmp/kimi-cache-lease-XXXXXX";
    const int fd = mkstemp(path); require(fd >= 0); unlink(path);
    sink_cache cache; sink_cache::layer layer; layer.id = 1; layer.count = 1;
    cache.layers.push_back(layer); cache.entries.resize(1); cache.stride = 12288;
    sink_request r; r.id = 5;
    auto & e = cache.entries[0]; e.id = 5; e.valid = e.resident = e.enabled = true;
    for (int role = 0; role < 3; ++role) {
        r.role[role] = {fd,uint64_t(role)*4096,4096}; e.source[role] = sink_cache::key(r.role[role]);
    }
    cache.stats.valid_bytes = cache.stats.committed_bytes = cache.stride;
    cache.acquire_lease(0,r); require(cache.drop_duplicate(1,r) == -1);
    e.protected_now = true; cache.release_lease(0); require(!e.protected_now);
    e.reserved = true; require(cache.drop_duplicate(1,r) == -1); e.reserved = false;
    e.protected_now = true; require(cache.drop_duplicate(1,r) == -1); e.protected_now = false;
    auto stale = r; stale.role[0].offset += 4096;
    require(cache.drop_duplicate(1,stale) == 0 && e.valid);
    require(cache.drop_duplicate(2,r) == 0 && e.valid);
    require(cache.drop_duplicate(1,r) == 1 && !e.valid && e.resident);
    require(cache.stats.valid_bytes == 0 && cache.stats.committed_bytes == cache.stride);
    require(cache.drop_duplicate(1,r) == 0);
    close(fd); std::printf("CACHE_LEASE_PASS assertions=%d payload_allocated=0\n",checks); return 0;
} catch (const std::exception & e) { std::fprintf(stderr,"CACHE_LEASE_FAIL: %s\n",e.what()); return 1; } }
