// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

namespace ssd {

// A private, zero-filled temporary file mapping. The file's disk space is reserved
// before mapping; creating the mapping does not fault in its pages.
class kv_backing {
public:
    kv_backing(const std::filesystem::path & directory, size_t bytes);
    ~kv_backing();

    kv_backing(const kv_backing &) = delete;
    kv_backing & operator=(const kv_backing &) = delete;

    void * data() const noexcept;
    size_t size() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

} // namespace ssd
