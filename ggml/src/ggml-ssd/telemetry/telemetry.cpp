// SPDX-License-Identifier: MIT
#include "telemetry.h"

#include "../settings/json.h"
#include "../settings/settings.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace ssd {
namespace telemetry {

result<std::unique_ptr<writer>> writer::open(const std::string & path) noexcept {
    // The fd and FILE must have exactly one owner even while constructing an
    // error result or allocating the writer. In particular fdopen failure must
    // never leave a closed descriptor available for a catch block to close again.
    struct file_owner {
        int         fd     = -1;
        std::FILE * stream = nullptr;

        ~file_owner() {
            if (stream) {
                std::fclose(stream);
            } else if (fd >= 0) {
                ::close(fd);
            }
        }
    } owned;

    try {
        owned.fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (owned.fd < 0) {
            return { nullptr, status::io, path + ": " + std::strerror(errno) };
        }
        owned.stream = fdopen(owned.fd, "w");
        if (!owned.stream) {
            const int code = errno;
            return { nullptr, status::io, path + ": " + std::strerror(code) };
        }
        owned.fd = -1;
        std::unique_ptr<writer> out(new writer(owned.stream));
        owned.stream = nullptr;
        return { std::move(out), status::ok, {} };
    } catch (const std::bad_alloc &) {
        return { nullptr, status::resource, "oom" };
    }
}

writer::~writer() {
    if (stream_) {
        std::fclose(stream_);
    }
}

result<void> writer::event(const std::string & name, const std::string & payload) noexcept {
    try {
        if (name.compare(0, 4, "ssd.") != 0 || name.size() == 4 || name.back() == '.') {
            return { status::invalid, "telemetry event must start with ssd. and name an event" };
        }
        for (char c : name) {
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_')) {
                return { status::invalid, "invalid telemetry event name" };
            }
        }
        if (name.find("..") != std::string::npos) {
            return { status::invalid, "empty telemetry event name segment" };
        }
        const auto parsed = json::parse(payload);
        if (!parsed) {
            return { parsed.code, parsed.error };
        }
        if (parsed.value.type != 'o') {
            return { status::invalid, "telemetry payload must be an object" };
        }
        for (const char * key : { "v", "ev", "seq" }) {
            if (parsed.value.object.count(key)) {
                return { status::invalid, std::string("reserved telemetry field ") + key };
            }
        }
        const size_t                begin = payload.find('{'), end = payload.find_last_of('}');
        const std::string           body = payload.substr(begin + 1, end - begin - 1);
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            return { status::io, "telemetry stream previously failed" };
        }
        const std::string line = "{\"v\":1,\"ev\":" + json::quote(name) + ",\"seq\":" + std::to_string(sequence_) +
                                 (parsed.value.object.empty() ? "" : "," + body) + "}\n";
        // One mutex protects a complete line and its sequence. Flush errors are
        // reported while the runtime still owns the writer and can fail its arm.
        if (std::fwrite(line.data(), 1, line.size(), stream_) != line.size() || std::fflush(stream_) != 0) {
            failed_ = true;
            return { status::io, "telemetry write/flush failed" };
        }
        ++sequence_;
        return {};
    } catch (const std::bad_alloc &) {
        return { status::resource, "oom" };
    } catch (const std::exception & error) {
        return { status::invalid, error.what() };
    }
}

result<void> writer::settings(const ssd::settings::snapshot & values) noexcept {
    const auto payload = values.json();
    if (!payload) {
        return { payload.code, payload.error };
    }
    return event("ssd.settings", payload.value);
}

result<void> writer::flush() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            return { status::io, "telemetry stream previously failed" };
        }
        if (std::fflush(stream_) != 0) {
            failed_ = true;
            return { status::io, "telemetry flush failed" };
        }
        return {};
    } catch (const std::exception & error) {
        return { status::io, error.what() };
    }
}

}  // namespace telemetry
}  // namespace ssd
