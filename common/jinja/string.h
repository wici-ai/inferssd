#pragma once

#include <optional>
#include <string>
#include <vector>

#include "utils.h"

namespace jinja {

// allow differentiate between user input strings and template strings
// Transformations retain the origin of copied bytes and transformed input.
// Concatenation and slicing must not flatten mixed template/input provenance.
struct string_part {
    bool is_input = false; // may skip parsing special tokens if true
    std::string val;

    bool is_uppercase() const;
    bool is_lowercase() const;
};

struct string {
    std::vector<string_part> parts;
    string() = default;
    string(const std::string & v, bool user_input = false) {
        parts.push_back({user_input, v});
    }
    string(int v) {
        parts.push_back({false, std::to_string(v)});
    }
    string(double v) {
        parts.push_back({false, std::to_string(v)});
    }

    // mark all parts as user input
    void mark_input();

    std::string str() const;
    size_t length() const;
    void hash_update(hasher & hash) const noexcept;
    bool all_parts_are_input() const;
    bool any_parts_are_input() const;
    bool is_uppercase() const;
    bool is_lowercase() const;

    // Byte slice of a possibly mixed string; retain every source boundary.
    string substr(size_t offset, size_t count = std::string::npos) const;

    string append(const string & other);

    // in-place transformations

    string uppercase();
    string lowercase();
    string capitalize();
    string titlecase();
    string strip(bool left, bool right, std::optional<const std::string_view> chars = std::nullopt);
};

} // namespace jinja
