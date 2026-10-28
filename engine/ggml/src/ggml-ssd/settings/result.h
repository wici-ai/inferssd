// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <utility>

namespace ssd {
enum class status { ok, invalid, io, resource };

template <class T> struct [[nodiscard]] result {
    T           value{};
    status      code = status::ok;
    std::string error;

    explicit operator bool() const noexcept { return code == status::ok; }
};

template <> struct [[nodiscard]] result<void> {
    status      code = status::ok;
    std::string error;

    explicit operator bool() const noexcept { return code == status::ok; }
};
}  // namespace ssd
