// SPDX-License-Identifier: MIT
#include "../../ggml/src/ggml-ssd/host/arbiter.h"
#include "../../ggml/src/ggml-ssd/host/ledger.h"
#include <atomic>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ssd::host;
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
static void require(bool value, const char * why) { if (!value) { throw std::runtime_error(why); } }
template<class F> static void must_throw(F f, const char * why) {
    bool threw = false; try { f(); } catch (const std::exception &) { threw = true; }
    require(threw, why);
}
template<class T> static T checked(ssd::result<T> value) {
    if (!value) { throw std::runtime_error(value.error); }
    return std::move(value.value);
}
static void checked(ssd::result<void> value) { if (!value) { throw std::runtime_error(value.error); } }
// Tests adapt result/status explicitly; product code never receives these
// throwing helpers. This also replays the pre-migration equations verbatim.
struct checked_client {
    std::unique_ptr<context_client> raw;
    explicit checked_client(std::unique_ptr<context_client> value) : raw(std::move(value)) {}
    void configure(const client_config & c) { checked(raw->configure(c)); }
    snapshot state() const { return checked(raw->state()); }
    void set_prefill_pages(uint64_t p) { checked(raw->set_prefill_pages(p)); }
    void clear_prefill_pages() { checked(raw->clear_prefill_pages()); }
    room measure(uint64_t x, bool full = false) const { return checked(raw->measure(x, full)); }
    room project_decode_capacity(uint64_t x = 0, bool full = false) const { return checked(raw->project_decode_capacity(x, full)); }
    allocation_ticket take(uint64_t h, uint64_t d = 0, const actions & a = {}, bool finite = false,
            allocation_class kind = allocation_class::external) {
        auto out = raw->take(h, d, a, finite, kind);
        if (!out && out.code == ssd::status::invalid) { throw std::runtime_error(out.error); }
        return std::move(out.value);
    }
    bool update_file_reserve() { return checked(raw->update_file_reserve()); }
    void begin_interval(uint64_t h) { checked(raw->begin_interval(h)); }
    bool end_interval(bool consecutive, uint64_t h, uint64_t & unknown) { return checked(raw->end_interval(consecutive, h, unknown)); }
    void close() { raw->close(); }
};
struct test_owner {
    std::shared_ptr<process_arbiter> raw;
    explicit test_owner(process_arbiter::reader reader) : raw(std::make_shared<process_arbiter>(std::move(reader))) {}
    std::unique_ptr<checked_client> make_client() {
        return std::make_unique<checked_client>(checked(raw->make_client()));
    }
};
struct files {
    std::atomic<uint64_t> limit{8*GiB}, current{GiB}, available{8*GiB}, file{0}, refault{0};
    std::atomic<bool> readable{true};
    std::atomic<bool> fail_usage{false};
    bool read(const std::string & path, std::string & text) const {
        if (!readable) { return false; }
        if (path == "/proc/self/cgroup") { text = "0::/fixture\n"; }
        else if (path == "/sys/fs/cgroup/fixture/memory.max") { text = std::to_string(limit.load()); }
        else if (path == "/sys/fs/cgroup/fixture/memory.current") {
            if (fail_usage) { return false; }
            text = std::to_string(current.load());
        }
        else if (path == "/sys/fs/cgroup/fixture/memory.stat") {
            text = "active_file 0\ninactive_file 0\nfile " + std::to_string(file.load()) +
                "\nworkingset_refault_file " + std::to_string(refault.load()) + "\n";
        } else if (path == "/proc/meminfo") { text = "MemAvailable: " + std::to_string(available.load()/1024) + " kB\n"; }
        else { return false; }
        return true;
    }
    process_arbiter::reader reader() { return [this](const std::string & path, std::string & out) { return read(path, out); }; }
};
struct barrier {
    std::mutex mutex; std::condition_variable changed;
    unsigned count, generation = 0;
    explicit barrier(unsigned n) : count(n), original(n) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex); const auto old = generation;
        if (!--count) { ++generation; count = original; changed.notify_all(); }
        else { changed.wait(lock, [&] { return generation != old; }); }
    }
  private: const unsigned original;
};
static client_config config(bool unified = false) { client_config c; c.unified = unified; return c; }
static void equal_room(const room & a, const kimi_host_ledger::room & b) {
    require(int(a.cgroup) == int(b.cgroup) && a.cgroup_shortage == b.cgroup_shortage &&
        a.cgroup_surplus == b.cgroup_surplus && a.system == b.system && a.available == b.available &&
        a.floor == b.floor && a.system_shortage == b.system_shortage && a.system_surplus == b.system_surplus &&
        a.boost_headroom == b.boost_headroom && a.fits() == b.fits() && a.surplus() == b.surplus(), "one-client equation changed");
}
static void serial_equivalence() {
    files f; auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client();
    must_throw([&] { auto t = a->take(1); }, "unconfigured client admitted");
    a->configure(config(true)); a->configure(config(true));
    kimi_host_ledger old; old.read = f.reader(); old.cgroup = "/sys/fs/cgroup/fixture";
    old.unified = true; old.reserve = old.margin = 512*MiB; old.reclaimable_file = true;
    old.page_size = uint64_t(sysconf(_SC_PAGESIZE));
    for (const uint64_t pages : {uint64_t(0), GiB, 3*GiB}) {
        a->set_prefill_pages(pages); old.prefill_pages = pages;
        for (const uint64_t extra : {uint64_t(0), GiB, 7*GiB}) {
            equal_room(a->measure(extra), old.measure(extra));
            equal_room(a->measure(extra, true), old.measure(extra, true));
        }
    }
    a->clear_prefill_pages(); old.prefill_pages = 0;
    f.file = 256*MiB; f.refault = 7;
    require(a->update_file_reserve() == old.update_file_reserve(), "first file reserve changed");
    f.refault = 10;
    require(a->update_file_reserve() == old.update_file_reserve(), "refault growth changed");
    require(a->state().file_reserve == old.file_reserve, "refault reserve differs");
    old.begin_interval(GiB); a->begin_interval(GiB);
    f.available -= 2*GiB; uint64_t x = 0, y = 0;
    require(a->end_interval(true, GiB, x) == old.end_interval(true, GiB, y) && x == y &&
        a->state().margin == old.margin, "serial interval changed");
    f.readable = false; equal_room(a->measure(1), old.measure(1));
    require(!a->take(1), "unreadable budget admitted real allocation");
}
static void holds_and_pending() {
    files f; auto owner = std::make_shared<test_owner>(f.reader());
    auto a = owner->make_client(), b = owner->make_client(); a->configure(config()); b->configure(config());
    a->set_prefill_pages(GiB); b->set_prefill_pages(2*GiB);
    require(a->state().all_prefill_pages == 3*GiB, "holds were not summed");
    const auto before = b->measure(0);
    require(a->project_decode_capacity().cgroup_surplus == before.cgroup_surplus + GiB, "projection omitted wrong hold");
    require(b->measure(0).cgroup_surplus == before.cgroup_surplus, "projection mutated published hold");
    a->clear_prefill_pages(); require(b->state().all_prefill_pages == 2*GiB, "A cleared B hold");
    b->set_prefill_pages(GiB); b->set_prefill_pages(GiB);
    require(b->state().all_prefill_pages == GiB, "repeated hold added");
    b->set_prefill_pages(9*GiB); require(!b->take(1) && b->state().prefill_pages == 9*GiB, "refusal lost prefill hold");
    b->clear_prefill_pages(); f.limit = 2*GiB; f.current = 0; f.available = 2*GiB;
    auto t = a->take(GiB); require(bool(t), "first take failed");
    require(!b->take(GiB), "unmaterialized allocation granted twice");
    require(a->state().pending_bytes == GiB && a->state().granted == GiB, "pending/grant not published");
    allocation_ticket moved = std::move(t); require(!t && bool(moved), "ticket move failed");
    moved.cancel(); moved.commit(); require(a->state().pending_bytes == 0 && a->state().granted == 0, "cancel did not roll back");
    auto committed = b->take(GiB); f.current = GiB; f.available = GiB; committed.commit();
    require(!a->take(GiB) && b->state().pending_bytes == 0 && b->state().granted == GiB, "commit lost OS accounting");
    a->set_prefill_pages(UINT64_MAX); require(!b->take(1), "saturated holds admitted");
    a->clear_prefill_pages(); a->close(); require(b->state().all_prefill_pages == 0, "close left peer hold");
    must_throw([&] { a->measure(0); }, "closed client remained live");
}
static void races_and_reclaim() {
    files f; f.limit = f.available = 2*GiB; f.current = 0;
    auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    a->configure(config(true)); b->configure(config(true));
    barrier ready(2), held(2); std::atomic<unsigned> granted{0};
    auto run = [&](checked_client & client) { ready.wait(); auto t = client.take(GiB); granted += bool(t); held.wait(); };
    std::thread ta(run, std::ref(*a)), tb(run, std::ref(*b)); ta.join(); tb.join();
    require(granted == 1 && a->state().pending_bytes == 0, "barrier race over-admitted");
    require(!a->take(UINT64_MAX, 1), "DRAM overflow admitted");
    std::vector<int> order; f.current = 2*GiB; f.available = 0;
    actions reclaim;
    reclaim.collect_deferred = [&] { (void)b->state(); order.push_back(1); };
    reclaim.reclaim_tier = [&](uint64_t) { (void)a->measure(0); order.push_back(2); f.available = GiB; return true; };
    reclaim.reclaim_cache = [&](uint64_t) { (void)b->state(); order.push_back(3); f.current = 0; return true; };
    auto ticket = a->take(256*MiB, 0, reclaim);
    require(bool(ticket) && order == std::vector<int>({1,2,3}), "reclaim order or unlocked reentry changed");
    ticket.cancel();
    auto changed = config(true); changed.margin_bytes = GiB;
    must_throw([&] { a->configure(changed); }, "client changed immutable config");
    auto c = owner->make_client(); must_throw([&] { c->configure(config(false)); }, "incompatible physical domain admitted");
}
static void process_reserves_and_rollback() {
    files f; auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    auto ca = config(true), cb = config(true);
    ca.margin_bytes = GiB; ca.explicit_reserve_bytes = 256*MiB;
    cb.reserve_bytes = GiB; cb.margin_bytes = 2*GiB; cb.explicit_reserve_bytes = 512*MiB;
    a->configure(ca); b->configure(cb);
    require(a->state().reserve == GiB && a->state().margin == 2*GiB && a->state().explicit_reserve == 512*MiB,
        "common reserves were added instead of applied once");
    try { auto t = a->take(MiB); require(bool(t), "exception setup take failed"); throw std::bad_alloc(); }
    catch (const std::bad_alloc &) {}
    require(a->state().pending_bytes == 0 && a->state().granted == 0, "exception leaked a pending grant");
    a->close(); b->close(); auto next = owner->make_client();
    must_throw([&] { next->configure(config(false)); }, "physical budget domain changed after close");
    next->configure(config(true));
    require(next->state().reserve == 512*MiB && next->state().margin == 2*GiB && next->state().explicit_reserve == 0,
        "reserve release or monotonic margin changed");
}
static void interval_isolation_and_close() {
    files f; auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    a->configure(config(true)); b->configure(config(true));
    a->begin_interval(0); b->begin_interval(0); f.available -= 2*GiB; uint64_t untracked = 99;
    require(!a->end_interval(true, 0, untracked) && !untracked, "A learned B overlap");
    require(!b->end_interval(true, 0, untracked) && a->state().overlap_skips == 1 && b->state().overlap_skips == 1, "intervals were mixed");
    a->begin_interval(0); b->set_prefill_pages(1);
    require(!a->end_interval(true, 0, untracked) && a->state().overlap_skips == 2, "peer hold did not invalidate sample");
    b->clear_prefill_pages(); a->begin_interval(0);
    auto t = a->take(256*MiB); f.available -= 256*MiB; t.commit();
    require(!a->end_interval(true, 0, untracked) && untracked == 0, "own grant was learned as driver growth");
    auto pending = b->take(1); require(bool(pending), "close setup failed");
    std::promise<void> started; std::atomic<bool> closed{false};
    std::thread closer([&] { started.set_value(); b->close(); closed = true; }); started.get_future().wait();
    // Wait for the actual closing flag, rather than asserting elapsed sleeps.
    for (;;) {
        try { (void)b->state(); std::this_thread::yield(); }
        catch (const std::runtime_error &) { break; }
    }
    require(!closed, "close returned with pending allocation");
    pending.cancel(); closer.join(); require(closed, "close did not finish after ticket cancellation");
    require(a->state().pending_bytes == 0, "closed client left process pending");
}
static void close_waits_for_reclaimer() {
    files f; f.current = f.limit.load(); f.available = 0;
    auto owner = std::make_shared<test_owner>(f.reader()); auto a = owner->make_client(), b = owner->make_client();
    a->configure(config(true)); b->configure(config(true));
    std::promise<void> entered, release; auto proceed = release.get_future();
    actions reclaim;
    reclaim.collect_deferred = [&] { entered.set_value(); proceed.wait(); };
    reclaim.reclaim_tier = [](uint64_t) { return false; };
    std::atomic<bool> rejected{false}, closed{false};
    std::thread worker([&] {
        try { auto t = a->take(1, 0, reclaim); }
        catch (const std::runtime_error &) { rejected = true; }
    });
    entered.get_future().wait();
    (void)b->measure(0); // a blocked user callback cannot own the arbiter mutex
    std::thread closer([&] { a->close(); closed = true; });
    for (;;) {
        try { (void)a->state(); std::this_thread::yield(); }
        catch (const std::runtime_error &) { break; }
    }
    const bool waited = !closed.load(); release.set_value(); worker.join(); closer.join();
    require(waited && rejected && closed, "close did not wait for an active reclaimer");
}
static void result_errors_and_tracked_allocations() {
    files f;
    auto owner = std::make_shared<process_arbiter>(f.reader());
    auto a = checked(owner->make_client());
    require(a->take(1).code == ssd::status::invalid, "unconfigured core admission did not return invalid");
    checked(a->configure(config(true)));
    f.fail_usage = true;
    require(a->take(1).code == ssd::status::io, "missing bounded cgroup usage fell through to system-only admission");
    f.fail_usage = false;
    checked(a->begin_interval(0));
    auto tracked = a->take(GiB, 0, {}, false, allocation_class::tracked);
    require(bool(tracked), "tracked ticket setup failed");
    f.available = 6*GiB; f.current = 2*GiB; tracked.value.commit();
    uint64_t unknown = 0;
    require(checked(a->end_interval(true, GiB, unknown)) && unknown == GiB && checked(a->state()).granted == 0,
        "cache held plus ticket grants hid real driver growth");
    auto crossing = a->take(256*MiB); require(bool(crossing), "crossing ticket setup failed");
    checked(a->begin_interval(0)); f.available -= 256*MiB; crossing.value.commit();
    require(!checked(a->end_interval(true, 0, unknown)) && unknown == 0 && checked(a->state()).overlap_skips == 1,
        "unmaterialized grant at interval start became driver growth");
    checked(a->begin_interval(0));
    // An external free can hide unknown driver growth if only the cumulative
    // grant counter is consulted; notify must invalidate the actor, not just peers.
    f.available += 128*MiB; checked(a->note_known_memory_change());
    require(!checked(a->end_interval(true, 0, unknown)) && unknown == 0 && checked(a->state()).overlap_skips == 2,
        "external free did not invalidate actor learning");
    checked(a->begin_interval(0)); f.available -= 2*GiB;
    require(checked(a->end_interval(true, 0, unknown)) && unknown == 2*GiB,
        "historical external grants masked later driver growth");
    auto zero = a->take(0); require(bool(zero) && bool(zero.value), "zero-byte successful result lost ticket status");
    zero.value.cancel();
    f.current = f.limit.load(); f.available = 0;
    actions reclaim; reclaim.reclaim_cache = [](uint64_t) -> bool { throw std::runtime_error("fixture callback"); };
    auto refused = a->take(1, 0, reclaim);
    require(!refused && refused.code == ssd::status::io && checked(a->state()).pending_bytes == 0,
        "callback exception escaped core or leaked pending");
    auto broken = std::make_shared<process_arbiter>([](const std::string &, std::string &) -> bool { throw std::runtime_error("fixture reader"); });
    require(broken->make_client().code == ssd::status::io, "reader exception escaped owner construction");
    auto exhausted = std::make_shared<process_arbiter>([](const std::string &, std::string &) -> bool { throw std::bad_alloc(); });
    require(exhausted->make_client().code == ssd::status::resource, "allocation failure escaped owner construction");
}
int main() {
    try {
        serial_equivalence(); holds_and_pending(); races_and_reclaim(); process_reserves_and_rollback();
        interval_isolation_and_close(); close_waits_for_reclaimer();
        result_errors_and_tracked_allocations();
        std::cout << "host arbiter: serial equations, holds, atomic tickets, reclaim, interval and close gates passed\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
