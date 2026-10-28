// SPDX-License-Identifier: Apache-2.0
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

static void require(bool value, const char * why) { if (!value) { throw std::runtime_error(why); } }
struct host_type;
struct allocation { host_type * owner; void * data; size_t size; };
static bool resident(const allocation & a) {
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
#if defined(__APPLE__)
    std::vector<char> bits((a.size + page - 1)/page);
#else
    std::vector<unsigned char> bits((a.size + page - 1)/page);
#endif
    require(mincore(a.data, a.size, bits.data()) == 0, "mincore failed");
    return std::all_of(bits.begin(), bits.end(), [](auto byte) { return byte & 1; });
}
struct host_type {
    bool host = true, initialize = true, expose_base = true;
    unsigned fail_after = UINT32_MAX;
    unsigned allocations = 0, initially_unresident = 0;
    std::vector<ggml_backend_buffer_t> buffers;
    ggml_backend_buffer_type type = {};
    host_type() {
        type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "host_prefault_fixture"; };
        type.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(32); };
        type.iface.get_max_size = [](ggml_backend_buffer_type_t) { return size_t(8192); };
        type.iface.is_host = [](ggml_backend_buffer_type_t buft) { return static_cast<host_type *>(buft->context)->host; };
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, size_t size) -> ggml_backend_buffer_t {
            auto & owner = *static_cast<host_type *>(buft->context);
            if (owner.allocations == owner.fail_after) { return nullptr; }
            void * data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (data == MAP_FAILED) { return nullptr; }
            auto * a = new allocation{&owner, data, size};
            if (!resident(*a)) { ++owner.initially_unresident; }
            if (owner.initialize) { std::memset(data, 0xa5, size); }
            ggml_backend_buffer_i methods = {};
            methods.get_base = [](ggml_backend_buffer_t b) -> void * {
                auto & item = *static_cast<allocation *>(b->context);
                return item.data;
            };
            if (!owner.expose_base) { methods.get_base = nullptr; }
            methods.free_buffer = [](ggml_backend_buffer_t b) {
                auto * item = static_cast<allocation *>(b->context);
                auto & list = item->owner->buffers;
                list.erase(std::find(list.begin(), list.end(), b));
                require(munmap(item->data, item->size) == 0, "munmap failed"); delete item;
            };
            auto buffer = ggml_backend_buffer_init(buft, methods, a, size);
            owner.buffers.push_back(buffer); ++owner.allocations; return buffer;
        };
    }
    ~host_type() { if (!buffers.empty()) { std::terminate(); } }
};
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using allocator_ptr = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;
static ggml_cgraph * graph(ggml_context * ctx, int64_t elements = 1536) {
    auto * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
    auto * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
    auto * z = ggml_add(ctx, x, y);
    ggml_set_input(x); ggml_set_input(y);
    ggml_set_output(x); ggml_set_output(y); ggml_set_output(z);
    auto * out = ggml_new_graph_custom(ctx, 16, false); ggml_build_forward_expand(out, z); return out;
}
static context_ptr context() { return context_ptr(ggml_init({1u << 20, nullptr, true}), &ggml_free); }
static void assert_bytes(const host_type & type, unsigned char value, bool touched, size_t page) {
    for (auto * buffer : type.buffers) {
        const auto & item = *static_cast<allocation *>(buffer->context);
        const auto * bytes = static_cast<const unsigned char *>(item.data);
        for (size_t i = 0; i < item.size; ++i) {
            const bool mark = touched && (i % page == 0 || i == item.size - 1);
            require(bytes[i] == (mark ? 0 : value), "prefault changed the wrong byte or skipped a page");
        }
    }
}
static void multi_chunk_and_reuse() {
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    host_type type; auto ctx = context(); auto * g = graph(ctx.get());
    allocator_ptr alloc(ggml_gallocr_new(&type.type), &ggml_gallocr_free);
    ggml_gallocr_set_host_prefault(alloc.get(), page);
    require(ggml_gallocr_reserve(alloc.get(), g), "multi-chunk reserve failed");
    require(type.buffers.size() >= 3, "fixture did not create multiple chunks");
    assert_bytes(type, 0xa5, true, page);
    require(ggml_gallocr_alloc_graph(alloc.get(), g), "assign tensor chunks failed");
    for (auto * buffer : type.buffers) {
        const auto & item = *static_cast<allocation *>(buffer->context); std::memset(item.data, 0x5a, item.size);
    }
    auto * weights = ggml_backend_buft_alloc_buffer(&type.type, 512);
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const unsigned before = type.allocations;
    require(ggml_gallocr_reserve(alloc.get(), g) && type.allocations == before, "reserve reallocated identical geometry");
    for (auto * buffer : type.buffers) {
        const auto & item = *static_cast<allocation *>(buffer->context);
        const auto * bytes = static_cast<const unsigned char *>(item.data);
        const unsigned char expected = buffer == weights ? 0xa5 : 0x5a;
        require(std::all_of(bytes, bytes + item.size, [&](auto value) { return value == expected; }), "reused data or weights were overwritten");
    }
    ggml_backend_buffer_free(weights);
}
static void disabled_and_device() {
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    for (const bool host : {false, true}) {
        host_type type; type.host = host; auto ctx = context(); auto * g = graph(ctx.get());
        allocator_ptr alloc(ggml_gallocr_new(&type.type), &ggml_gallocr_free);
        ggml_gallocr_set_host_prefault(alloc.get(), host ? 0 : page);
        require(ggml_gallocr_reserve(alloc.get(), g), "disabled/device reserve failed");
        assert_bytes(type, 0xa5, false, page);
    }
}
static void real_page_population_and_failure() {
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    {
        host_type type; type.initialize = false; auto ctx = context(); auto * g = graph(ctx.get());
        allocator_ptr alloc(ggml_gallocr_new(&type.type), &ggml_gallocr_free);
        ggml_gallocr_set_host_prefault(alloc.get(), page);
        require(ggml_gallocr_reserve(alloc.get(), g), "real prefault reserve failed");
        require(type.initially_unresident == type.allocations && type.allocations >= 3, "fresh mmap was already resident");
        for (auto * buffer : type.buffers) { require(resident(*static_cast<allocation *>(buffer->context)), "new host page remained unfaulted"); }
    }
    {
        host_type type; type.fail_after = 1; auto ctx = context(); auto * g = graph(ctx.get());
        allocator_ptr alloc(ggml_gallocr_new(&type.type), &ggml_gallocr_free);
        ggml_gallocr_set_host_prefault(alloc.get(), page);
        require(!ggml_gallocr_reserve(alloc.get(), g) && type.buffers.empty(), "partial multi-chunk allocation failure leaked");
    }
}
static void missing_host_base() {
    host_type type; type.expose_base = false; auto ctx = context(); auto * g = graph(ctx.get());
    allocator_ptr alloc(ggml_gallocr_new(&type.type), &ggml_gallocr_free);
    ggml_gallocr_set_host_prefault(alloc.get(), size_t(sysconf(_SC_PAGESIZE)));
    require(!ggml_gallocr_reserve(alloc.get(), g) && type.buffers.empty(),
        "host buffer without a base was not rejected or leaked a chunk");
}
static void scheduler_explicit_reserve() {
    host_type type;
    ggml_backend_device device = {};
    device.context = &type;
    device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; };
    device.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
    device.iface.supports_buft = [](ggml_backend_dev_t, ggml_backend_buffer_type_t) { return true; };
    device.iface.get_buffer_type = [](ggml_backend_dev_t d) { return &static_cast<host_type *>(d->context)->type; };
    ggml_backend backend = {}; backend.device = &device;
    backend.iface.get_name = [](ggml_backend_t) { return "host_prefault_fixture"; };
    ggml_backend_t backends[] = {&backend};
    using sched_ptr = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
    {
        sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 32, false, false), &ggml_backend_sched_free);
        ggml_backend_sched_set_auto_reserve(sched.get(), false);
        ggml_backend_sched_set_host_prefault(sched.get(), size_t(sysconf(_SC_PAGESIZE)));
        auto small = context(); auto * sg = graph(small.get(), 128);
        require(!ggml_backend_sched_alloc_graph(sched.get(), sg) && type.allocations == 0,
            "disabled fallback allocated without an explicit reservation");
        require(ggml_backend_sched_reserve(sched.get(), sg) && ggml_backend_sched_alloc_graph(sched.get(), sg),
            "explicit reserve did not admit graph");
        ggml_backend_sched_reset(sched.get());
        const unsigned before = type.allocations;
        auto large = context(); auto * lg = graph(large.get());
        require(!ggml_backend_sched_alloc_graph(sched.get(), lg) && type.allocations == before,
            "larger graph bypassed explicit reservation policy");
        require(ggml_backend_sched_reserve(sched.get(), lg) && ggml_backend_sched_alloc_graph(sched.get(), lg),
            "larger explicit reservation failed");
    }
    {
        auto ctx = context(); auto * g = graph(ctx.get(), 128);
        sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 1, 32, false, false), &ggml_backend_sched_free);
        require(ggml_backend_sched_alloc_graph(sched.get(), g), "default generic scheduler lost automatic reserve");
    }
}
int main() {
    try {
        multi_chunk_and_reuse(); disabled_and_device(); real_page_population_and_failure(); missing_host_base(); scheduler_explicit_reserve();
        std::cout << "host prefault: real multichunk pages, reuse, weights, disabled/device and failure cleanup passed\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
