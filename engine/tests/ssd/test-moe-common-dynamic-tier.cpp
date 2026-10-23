#include "fixtures/legacy-cuda/moe-common-dynamic-tier.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <thread>

#undef NDEBUG
#include <cassert>

using tier = ggml_moe_common::common_dynamic_tier_lru;

static tier::reservation begin_fill(tier & cache, size_t key) {
    tier::reservation fill;
    assert(cache.reserve(key, fill));
    assert(fill.valid());
    return fill;
}

static void insert(tier & cache, size_t key) {
    auto fill = begin_fill(cache, key);
    assert(cache.publish(fill));
    assert(!fill.valid());
}

static void test_lru() {
    tier cache(3);
    insert(cache, 10);
    insert(cache, 20);
    insert(cache, 30);
    tier::lease held;
    assert(cache.acquire(10, held));
    assert(cache.release(held));
    auto fill = begin_fill(cache, 40);
    assert(fill.evicted && fill.evicted_key == 20);
    assert(!cache.acquire(20, held));
    assert(cache.publish(fill));
    fill = begin_fill(cache, 50);
    assert(fill.evicted && fill.evicted_key == 30);
    assert(cache.publish(fill));
    const auto stats = cache.get_stats();
    assert(stats.admissions == 5 && stats.evictions == 2);
    assert(stats.ready == 3 && stats.filling == 0 && stats.leased == 0);
}

static void test_leases_and_busy() {
    tier cache(2);
    insert(cache, 1);
    insert(cache, 2);
    tier::lease first;
    tier::lease second;
    tier::lease extra_reader;
    assert(cache.acquire(1, first));
    assert(cache.acquire(2, second));
    assert(cache.acquire(1, extra_reader));
    assert(cache.get_stats().leased == 3);
    assert(!cache.reset(3));
    tier::reservation fill;
    assert(!cache.reserve(3, fill));
    assert(cache.release(first));
    assert(!cache.release(first));
    assert(!cache.reserve(3, fill));
    assert(cache.release(extra_reader));
    // 2 is the least recently used, but its H2D reader still owns it.
    assert(cache.reserve(3, fill));
    assert(fill.evicted && fill.evicted_key == 1);
    tier::reservation blocked;
    assert(!cache.reserve(4, blocked));
    assert(cache.publish(fill));
    assert(cache.release(second));
    fill = begin_fill(cache, 4);
    assert(fill.evicted_key == 2);
    assert(cache.cancel(fill));
    assert(cache.get_stats().leased == 0);
}

static void test_publication() {
    tier cache(1);
    std::array<unsigned char, 4096> payload{};
    auto fill = begin_fill(cache, 7);
    tier::lease held;
    tier::reservation duplicate;
    assert(!cache.acquire(7, held));
    assert(!cache.reserve(7, duplicate));
    assert(!cache.reserve(8, duplicate));
    assert(!cache.reset(1));

    std::mutex mutex;
    std::condition_variable condition;
    bool half_written = false;
    bool finish_write = false;
    std::thread writer([&] {
        std::fill(payload.begin(), payload.begin() + payload.size() / 2, 0xa5);
        {
            std::unique_lock<std::mutex> lock(mutex);
            half_written = true;
            condition.notify_one();
            condition.wait(lock, [&] { return finish_write; });
        }
        std::fill(payload.begin() + payload.size() / 2, payload.end(), 0xa5);
        assert(cache.publish(fill));
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return half_written; });
        // An incomplete asynchronous fill must stay invisible.
        assert(!cache.acquire(7, held));
        finish_write = true;
        condition.notify_one();
    }
    while (!cache.acquire(7, held)) {
        std::this_thread::yield();
    }
    // Acquire/publish establish visibility without joining the producer first.
    for (const auto byte : payload) {
        assert(byte == 0xa5);
    }
    assert(!cache.reserve(8, duplicate));
    assert(cache.release(held));
    writer.join();
}

static void test_cancel_and_stale_handles() {
    tier cache(1);
    insert(cache, 1);
    tier::lease held;
    assert(cache.acquire(1, held));
    auto stale_lease = held;
    assert(cache.release(held));
    auto fill = begin_fill(cache, 2);
    auto stale_fill = fill;
    assert(fill.evicted_key == 1);
    // A read/copy failure cannot resurrect the possibly overwritten victim.
    assert(cache.cancel(fill));
    assert(!cache.cancel(fill));
    assert(!cache.acquire(1, held));
    assert(!cache.acquire(2, held));
    fill = begin_fill(cache, 3);
    assert(!fill.evicted);
    assert(!cache.publish(stale_fill));
    assert(!cache.cancel(stale_fill));
    assert(!cache.release(stale_lease));
    assert(cache.publish(fill));
    assert(cache.acquire(3, held));
    assert(!cache.release(stale_lease));
    assert(cache.release(held));
    const auto stats = cache.get_stats();
    assert(stats.admissions == 2 && stats.evictions == 1 && stats.ready == 1);
    assert(cache.reset(1));
    fill = begin_fill(cache, 2);
    assert(!cache.publish(stale_fill));
    assert(cache.cancel(fill));
}

static void test_duplicate_concurrent_miss() {
    tier cache(8);
    std::array<tier::reservation, 8> fills;
    std::array<std::thread, 8> workers;
    std::atomic<bool> start{false};
    std::atomic<size_t> winners{0};
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i] {
            while (!start.load()) {
                std::this_thread::yield();
            }
            if (cache.reserve(42, fills[i])) {
                winners.fetch_add(1);
            }
        });
    }
    start.store(true);
    for (auto & worker : workers) {
        worker.join();
    }
    assert(winners.load() == 1);
    assert(cache.get_stats().filling == 1);
    for (auto & fill : fills) {
        if (fill.valid()) {
            assert(cache.publish(fill));
        }
    }
    assert(cache.get_stats().admissions == 1);
}

static void test_payload_stress() {
    tier cache(4);
    std::array<std::array<size_t, 64>, 4> payload{};
    std::array<std::thread, 4> workers;
    std::atomic<size_t> hits{0};
    for (size_t worker = 0; worker < workers.size(); ++worker) {
        workers[worker] = std::thread([&, worker] {
            uint32_t state = uint32_t(worker + 1);
            for (size_t i = 0; i < 10000; ++i) {
                state = state * 1664525u + 1013904223u;
                const size_t key = (state >> 16) % 12;
                tier::lease held;
                if (cache.acquire(key, held)) {
                    for (const size_t word : payload[held.slot]) {
                        assert(word == key);
                    }
                    if (i % 8 == 0) {
                        std::this_thread::yield();
                    }
                    // Another writer cannot change any byte while leased.
                    for (const size_t word : payload[held.slot]) {
                        assert(word == key);
                    }
                    assert(cache.release(held));
                    hits.fetch_add(1);
                } else {
                    tier::reservation fill;
                    if (cache.reserve(key, fill)) {
                        payload[fill.slot].fill(key);
                        if (i % 13 == 0) {
                            assert(cache.cancel(fill));
                        } else {
                            assert(cache.publish(fill));
                        }
                    }
                }
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    const auto stats = cache.get_stats();
    assert(hits.load() > 0 && stats.admissions > 0 && stats.evictions > 0);
    assert(stats.ready <= 4 && stats.filling == 0 && stats.leased == 0);
}

static void test_disabled() {
    tier cache;
    tier::lease held;
    tier::reservation fill;
    assert(cache.size() == 0);
    assert(!cache.acquire(0, held));
    assert(!cache.reserve(0, fill));
    assert(!cache.publish(fill));
    assert(!cache.cancel(fill));
    assert(!cache.release(held));
    assert(cache.reset(2));
    insert(cache, 0);
    insert(cache, SIZE_MAX);
    assert(cache.acquire(SIZE_MAX, held));
    assert(cache.release(held));
}

int main() {
    test_disabled();
    test_lru();
    test_leases_and_busy();
    test_publication();
    test_cancel_and_stale_handles();
    test_duplicate_concurrent_miss();
    test_payload_stress();
    std::puts("common dynamic tier: 7 CPU tests passed");
    return 0;
}
