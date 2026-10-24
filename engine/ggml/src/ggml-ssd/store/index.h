// SPDX-License-Identifier: MIT
#pragma once
#include "source.h"
#include "../settings/result.h"
#include <memory>
#include <string>
#include <vector>

namespace ssd { namespace store {
class source_index {
public:
    [[nodiscard]] static result<std::unique_ptr<source_index>> open(const std::vector<std::string> & paths) noexcept;
    ~source_index();
    source_index(const source_index &) = delete;
    source_index & operator=(const source_index &) = delete;
    source_view view() noexcept;
private:
    struct implementation;
    explicit source_index(std::unique_ptr<implementation> state);
    std::unique_ptr<implementation> state_;
};
} }
