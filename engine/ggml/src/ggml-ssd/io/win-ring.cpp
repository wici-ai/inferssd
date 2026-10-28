// SPDX-License-Identifier: Apache-2.0
#include "win-ring.h"
#include "ring-common.h"
#include "../platform/file.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <ioringapi.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <string>

#if !defined(NTDDI_WIN10_CO) || NTDDI_VERSION < NTDDI_WIN10_CO
#error "win-ring needs the IoRing declarations of the Windows 10.0.22000 SDK or later"
#endif
static_assert(sizeof(UINT_PTR) == sizeof(uint64_t), "ring tags are 64-bit user data");

namespace ssd { namespace io {
namespace {

std::atomic<bool> g_force_fallback{false};
std::atomic<bool> g_backend_logged{false};

std::runtime_error win_error(const char * operation, DWORD code) {
    return std::runtime_error(std::string(operation) + " failed: Win32 error " + std::to_string(code));
}
std::runtime_error hr_error(const char * operation, HRESULT hr) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08lx", static_cast<unsigned long>(hr));
    return std::runtime_error(std::string(operation) + " failed: HRESULT " + text);
}

// Completion results follow io_uring: bytes, 0 at end of file, or -errno.
int32_t win32_result(DWORD error) {
    return error == ERROR_HANDLE_EOF ? 0 : -platform::errno_from_win32(error);
}
int32_t hresult_result(HRESULT hr) {
    if (hr & FACILITY_NT_BIT) {
        // An NTSTATUS passed through unchanged.
        switch (static_cast<uint32_t>(hr) & ~static_cast<uint32_t>(FACILITY_NT_BIT)) {
            case 0xC0000011u: return 0;          // STATUS_END_OF_FILE
            case 0xC0000008u: return -EBADF;     // STATUS_INVALID_HANDLE
            case 0xC000000Du: return -EINVAL;    // STATUS_INVALID_PARAMETER
            case 0xC0000017u:                    // STATUS_NO_MEMORY
            case 0xC000009Au: return -ENOMEM;    // STATUS_INSUFFICIENT_RESOURCES
            case 0xC0000120u: return -ECANCELED; // STATUS_CANCELLED
            default:          return -EIO;
        }
    }
    if (HRESULT_FACILITY(hr) == FACILITY_WIN32) { return win32_result(DWORD(HRESULT_CODE(hr))); }
    return -EIO;
}

// Resolved at run time, so the binary still starts where IoRing is absent and
// the overlapped backend takes over.
struct ioring_api {
    decltype(&::QueryIoRingCapabilities) query_capabilities = nullptr;
    decltype(&::CreateIoRing) create = nullptr;
    decltype(&::IsIoRingOpSupported) op_supported = nullptr;
    decltype(&::SetIoRingCompletionEvent) set_completion_event = nullptr;
    decltype(&::BuildIoRingReadFile) build_read = nullptr;
    decltype(&::BuildIoRingRegisterBuffers) build_register_buffers = nullptr;
    decltype(&::SubmitIoRing) submit = nullptr;
    decltype(&::PopIoRingCompletion) pop = nullptr;
    decltype(&::CloseIoRing) close = nullptr;
    bool complete = false;
};
template<class F> bool resolve(HMODULE module, const char * name, F & function) {
    // Through void (*)(), the generic function pointer type, as GetProcAddress results are meant to be cast.
    function = reinterpret_cast<F>(reinterpret_cast<void (*)()>(GetProcAddress(module, name)));
    return function != nullptr;
}
const ioring_api * ioring() {
    static const ioring_api api = [] {
        ioring_api value;
        // The SDK import library binds these to kernel32, which forwards them.
        const HMODULE module = GetModuleHandleW(L"kernel32.dll");
        value.complete = module &&
            resolve(module, "QueryIoRingCapabilities", value.query_capabilities) &&
            resolve(module, "CreateIoRing", value.create) &&
            resolve(module, "IsIoRingOpSupported", value.op_supported) &&
            resolve(module, "SetIoRingCompletionEvent", value.set_completion_event) &&
            resolve(module, "BuildIoRingReadFile", value.build_read) &&
            resolve(module, "BuildIoRingRegisterBuffers", value.build_register_buffers) &&
            resolve(module, "SubmitIoRing", value.submit) &&
            resolve(module, "PopIoRingCompletion", value.pop) &&
            resolve(module, "CloseIoRing", value.close);
        return value;
    }();
    return api.complete ? &api : nullptr;
}

// IoRing: one SubmitIoRing for a batch, completions in a shared queue, and
// registered buffers for fixed reads. Its I/O is not bound to the issuing thread.
class ioring_backend final : public detail::ring_backend {
public:
    // nullptr, with the reason, when IoRing cannot serve this ring.
    static std::unique_ptr<ioring_backend> create(unsigned depth, std::string & why) {
        const ioring_api * api = ioring();
        if (!api) { why = "IoRing is not available"; return nullptr; }
        IORING_CAPABILITIES caps {};
        const HRESULT hr = api->query_capabilities(&caps);
        if (FAILED(hr)) { why = hr_error("QueryIoRingCapabilities", hr).what(); return nullptr; }
        // Version 1 can miss the completion event when the queue turns non-empty.
        if (caps.MaxVersion < IORING_VERSION_2) { why = "IoRing version is too old"; return nullptr; }
        // User-mode emulation would only add a layer over overlapped reads.
        if (caps.FeatureFlags & IORING_FEATURE_UM_EMULATION) { why = "IoRing is emulated in user mode"; return nullptr; }
        if (!(caps.FeatureFlags & IORING_FEATURE_SET_COMPLETION_EVENT)) { why = "IoRing has no completion event"; return nullptr; }
        if (caps.MaxSubmissionQueueSize < depth || caps.MaxCompletionQueueSize < 2 * uint64_t(depth)) {
            why = "IoRing queues are smaller than the ring depth";
            return nullptr;
        }
        std::unique_ptr<ioring_backend> backend(new ioring_backend(*api, depth, caps.MaxVersion));
        try {
            backend->open();
        } catch (const std::exception & e) {
            why = e.what();
            return nullptr;
        }
        return backend;
    }
    ~ioring_backend() override { close(); }

    const char * name() const noexcept override { return "ioring"; }

    void start(const detail::staged_read & read) override {
        if (!ring_) { open(); }
        const HANDLE handle = static_cast<HANDLE>(platform::native_handle(read.source.fd_direct));
        if (handle == INVALID_HANDLE_VALUE) {
            rejected_.push_back({read.tag, -EBADF, 0});
            return;
        }
        const IORING_BUFFER_REF buffer = read.fixed ?
            IoRingBufferRefFromIndexAndOffset(static_cast<UINT32>(read.tag), 0) : IoRingBufferRefFromPointer(read.data);
        const HRESULT hr = api_.build_read(ring_, IoRingHandleRefFromHandle(handle), buffer,
                                           static_cast<UINT32>(read.source.nbytes), read.source.offset,
                                           static_cast<UINT_PTR>(read.tag), IOSQE_FLAGS_NONE);
        if (FAILED(hr)) { throw hr_error("BuildIoRingReadFile", hr); }
        ++unsubmitted_;
    }

    void flush() override {
        while (unsubmitted_) {
            UINT32 submitted = 0;
            const HRESULT hr = api_.submit(ring_, 0, 0, &submitted);
            if (FAILED(hr)) { throw hr_error("SubmitIoRing", hr); }
            if (!submitted) { throw std::runtime_error("IoRing submission made no progress"); }
            unsubmitted_ -= std::min(unsubmitted_, unsigned(submitted));
        }
    }

    bool collect(completion & out) override {
        if (!rejected_.empty()) {
            out = rejected_.front();
            rejected_.pop_front();
            return true;
        }
        IORING_CQE entry {};
        if (stashed_) {
            entry = stash_;
            stashed_ = false;
        } else if (!next(entry)) {
            return false;
        }
        const int32_t res = SUCCEEDED(entry.ResultCode) ? static_cast<int32_t>(entry.Information) : hresult_result(entry.ResultCode);
        out = {static_cast<uint64_t>(entry.UserData), res, 0};
        return true;
    }

    void block() override {
        // Popping before each wait makes the event's edge irrelevant: a
        // completion posted after the pop signals it.
        while (rejected_.empty() && !stashed_) {
            if (next(stash_)) { stashed_ = true; return; }
            if (WaitForSingleObject(event_, INFINITE) != WAIT_OBJECT_0) { throw win_error("WaitForSingleObject", GetLastError()); }
        }
    }

    void register_buffers(const std::vector<buffer_span> & buffers) override {
        if (!ring_) { open(); }
        std::vector<IORING_BUFFER_INFO> info(buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (buffers[i].len > UINT32_MAX) { throw std::runtime_error("IoRing buffer exceeds 4 GiB"); }
            info[i].Address = buffers[i].base;
            info[i].Length = static_cast<UINT32>(buffers[i].len);
        }
        HRESULT hr = api_.build_register_buffers(ring_, static_cast<UINT32>(info.size()), info.data(), registration_tag);
        if (FAILED(hr)) { throw hr_error("BuildIoRingRegisterBuffers", hr); }
        // No reads are in flight (the core checks), so the one completion is this one.
        UINT32 submitted = 0;
        hr = api_.submit(ring_, 1, INFINITE, &submitted);
        if (FAILED(hr)) { throw hr_error("SubmitIoRing (register buffers)", hr); }
        IORING_CQE entry {};
        if (!next(entry) || entry.UserData != registration_tag) {
            throw std::runtime_error("IoRing buffer registration did not complete");
        }
        if (FAILED(entry.ResultCode)) { throw hr_error("IoRing buffer registration (no fallback)", entry.ResultCode); }
    }

    // IoRing has no unregister operation. Closing the ring drops the
    // registration; the next read or registration opens a new ring.
    void unregister_buffers() override { close(); }

private:
    static constexpr UINT_PTR registration_tag = ~UINT_PTR(0);

    ioring_backend(const ioring_api & api, unsigned depth, IORING_VERSION version)
        : api_(api), depth_(depth), version_(version) {}

    void open() {
        const IORING_CREATE_FLAGS flags = {IORING_CREATE_REQUIRED_FLAGS_NONE, IORING_CREATE_ADVISORY_FLAGS_NONE};
        // A CQ twice the SQ, as io_uring sizes it; the core bounds pending reads to it.
        HRESULT hr = api_.create(version_, flags, depth_, 2 * depth_, &ring_);
        if (FAILED(hr)) { ring_ = nullptr; throw hr_error("CreateIoRing", hr); }
        if (!api_.op_supported(ring_, IORING_OP_READ) || !api_.op_supported(ring_, IORING_OP_REGISTER_BUFFERS)) {
            close();
            throw std::runtime_error("IoRing lacks read or buffer registration");
        }
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_) {
            const DWORD code = GetLastError();
            close();
            throw win_error("CreateEventW", code);
        }
        hr = api_.set_completion_event(ring_, event_);
        if (FAILED(hr)) {
            close();
            throw hr_error("SetIoRingCompletionEvent", hr);
        }
    }

    void close() noexcept {
        if (ring_) { api_.close(ring_); }
        if (event_) { CloseHandle(event_); }
        ring_ = nullptr;
        event_ = nullptr;
        unsubmitted_ = 0;
    }

    bool next(IORING_CQE & entry) {
        if (!ring_) { return false; }
        const HRESULT hr = api_.pop(ring_, &entry);
        if (hr == S_OK) { return true; }
        if (hr == S_FALSE) { return false; }
        throw hr_error("PopIoRingCompletion", hr);
    }

    const ioring_api & api_;
    const unsigned depth_;
    const IORING_VERSION version_;
    HIORING ring_ = nullptr;
    HANDLE event_ = nullptr;
    unsigned unsubmitted_ = 0;
    bool stashed_ = false;            // one completion block() popped for collect()
    IORING_CQE stash_ {};
    std::deque<completion> rejected_; // reads that never reached the ring
};

// Overlapped ReadFile signalling one auto-reset event per ring. No completion
// port: a handle binds to one port for life, and descriptors are shared by
// every ring and reader in the process. The reads belong to the issuing
// thread, which is the ring's owner and drains before it exits.
class overlapped_backend final : public detail::ring_backend {
public:
    explicit overlapped_backend(unsigned depth) : slots_(2 * size_t(depth)) {
        free_.reserve(slots_.size());
        active_.reserve(slots_.size());
        for (size_t i = slots_.size(); i-- > 0;) { free_.push_back(i); }
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_) { throw win_error("CreateEventW", GetLastError()); }
    }
    ~overlapped_backend() override { if (event_) { CloseHandle(event_); } }

    const char * name() const noexcept override { return "overlapped"; }

    void start(const detail::staged_read & read) override {
        const HANDLE handle = static_cast<HANDLE>(platform::native_handle(read.source.fd_direct));
        if (handle == INVALID_HANDLE_VALUE) {
            rejected_.push_back({read.tag, -EBADF, 0});
            return;
        }
        if (free_.empty()) { throw std::runtime_error("overlapped ring has no free request"); }
        request & slot = slots_[free_.back()];
        slot.overlapped = OVERLAPPED {};
        slot.overlapped.Offset = DWORD(read.source.offset);
        slot.overlapped.OffsetHigh = DWORD(read.source.offset >> 32);
        slot.overlapped.hEvent = event_;
        slot.handle = handle;
        slot.tag = read.tag;
        if (!ReadFile(handle, read.data, static_cast<DWORD>(read.source.nbytes), nullptr, &slot.overlapped)) {
            const DWORD error = GetLastError();
            if (error != ERROR_IO_PENDING) {
                rejected_.push_back({read.tag, win32_result(error), 0});
                return;
            }
        }
        active_.push_back(free_.back());
        free_.pop_back();
    }

    void flush() override {} // ReadFile already issued each read

    bool collect(completion & out) override {
        if (!rejected_.empty()) {
            out = rejected_.front();
            rejected_.pop_front();
            return true;
        }
        for (size_t i = 0; i < active_.size(); ++i) {
            request & slot = slots_[active_[i]];
            if (!HasOverlappedIoCompleted(&slot.overlapped)) { continue; }
            DWORD got = 0;
            const int32_t res = GetOverlappedResult(slot.handle, &slot.overlapped, &got, FALSE) ?
                static_cast<int32_t>(got) : win32_result(GetLastError());
            out = {slot.tag, res, 0};
            free_.push_back(active_[i]);
            active_[i] = active_.back();
            active_.pop_back();
            return true;
        }
        return false;
    }

    void block() override {
        // Every ReadFile resets the shared event; checking right before each
        // wait, on the only thread that issues reads, keeps that harmless.
        for (;;) {
            if (!rejected_.empty()) { return; }
            for (const size_t index : active_) {
                if (HasOverlappedIoCompleted(&slots_[index].overlapped)) { return; }
            }
            if (WaitForSingleObject(event_, INFINITE) != WAIT_OBJECT_0) { throw win_error("WaitForSingleObject", GetLastError()); }
        }
    }

    // ReadFile takes plain addresses; registration is only the core's bounds checks.
    void register_buffers(const std::vector<buffer_span> &) override {}
    void unregister_buffers() override {}

private:
    struct request {
        OVERLAPPED overlapped {};
        HANDLE handle = nullptr;
        uint64_t tag = 0;
    };
    std::vector<request> slots_; // fixed size: OVERLAPPED addresses stay valid while in flight
    std::vector<size_t> free_, active_;
    std::deque<completion> rejected_;
    HANDLE event_ = nullptr;
};

} // namespace

struct win_ring::impl : detail::ring_core {};

win_ring::win_ring() noexcept = default;
win_ring::~win_ring() = default;

void win_ring::force_fallback_for_testing(bool force) noexcept { g_force_fallback.store(force); }

result<void> win_ring::init(unsigned depth, bool fixed) noexcept {
    return detail::boundary<void>([&] {
        if (state_) { throw std::runtime_error("read ring initialized twice"); }
        if (!depth) { throw std::runtime_error("read ring depth must be positive"); }
        std::unique_ptr<detail::ring_backend> backend;
        std::string why = "IoRing disabled for testing";
        if (!g_force_fallback.load()) { backend = ioring_backend::create(depth, why); }
        const bool fallback = !backend;
        if (fallback) { backend = std::make_unique<overlapped_backend>(depth); }
        if (!g_backend_logged.exchange(true)) {
            std::fprintf(stderr, "[kimi_cpu_sink_io] read_ring backend=%s%s%s\n", backend->name(),
                         fallback ? " reason=" : "", fallback ? why.c_str() : "");
        }
        auto state = std::make_unique<impl>();
        state->init(std::move(backend), depth, fixed);
        state_ = std::move(state);
    });
}
void win_ring::close_ring() noexcept { if (state_) { state_->close(); } }
result<void> win_ring::register_buffers(const std::vector<buffer_span> & buffers) noexcept {
    return detail::boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->register_buffers(buffers); });
}
result<void> win_ring::unregister_buffers() noexcept {
    return detail::boundary<void>([&] { if (state_) { state_->unregister_buffers(); } });
}
result<void> win_ring::queue(const read_range & source, void * data, uint64_t tag, bool fixed) noexcept {
    return detail::boundary<void>([&] {
        if (!state_) { throw std::runtime_error("ring not initialized"); }
        // FILE_FLAG_NO_BUFFERING reads whole sectors; 4096 is the O_DIRECT
        // granularity callers already pad to and covers common sector sizes.
        if (source.offset % 4096 || source.nbytes % 4096) {
            throw std::runtime_error("unbuffered read offset or length is not 4096-byte aligned");
        }
        state_->queue(source, data, tag, fixed);
    });
}
result<void> win_ring::submit(unsigned count) noexcept {
    return detail::boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->submit(count); });
}
result<bool> win_ring::pop(completion & entry) noexcept {
    return detail::boundary<bool>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } return state_->pop(entry); });
}
result<void> win_ring::wait() noexcept {
    return detail::boundary<void>([&] { if (!state_) { throw std::runtime_error("ring not initialized"); } state_->wait(); });
}
result<void> win_ring::drain(void * userdata, void (*callback)(void *, const completion &)) noexcept {
    return detail::boundary<void>([&] { if (state_) { state_->drain(userdata, callback); } });
}
uint64_t win_ring::pending() const noexcept { return state_ ? state_->pending() : 0; }
uint64_t win_ring::completed() const noexcept { return state_ ? state_->completed() : 0; }
bool win_ring::read_fixed() const noexcept { return state_ && state_->read_fixed(); }
bool win_ring::buffers_registered() const noexcept { return state_ && state_->buffers_registered(); }
const char * win_ring::backend_name() const noexcept { return state_ ? state_->backend_name() : "none"; }
}} // namespace ssd::io
