// SPDX-License-Identifier: MIT
#pragma once

// Small strict JSON boundary shared by settings path lists and JSONL payloads.
// No host observations, global state or third-party dependency.
#include "result.h"

#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace ssd {
namespace json {

inline bool valid_utf8(const std::string & text) noexcept {
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char first = text[i++];
        if (first < 128) {
            continue;
        }
        unsigned count = 0, cp = 0, minimum = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            count   = 1;
            cp      = first & 31;
            minimum = 128;
        } else if (first >= 0xe0 && first <= 0xef) {
            count   = 2;
            cp      = first & 15;
            minimum = 2048;
        } else if (first >= 0xf0 && first <= 0xf4) {
            count   = 3;
            cp      = first & 7;
            minimum = 65536;
        } else {
            return false;
        }
        if (text.size() - i < count) {
            return false;
        }
        while (count--) {
            const unsigned char next = text[i++];
            if ((next & 0xc0) != 0x80) {
                return false;
            }
            cp = (cp << 6) | (next & 63);
        }
        if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
            return false;
        }
    }
    return true;
}

inline std::string quote(const std::string & text) {
    const char * hex = "0123456789abcdef";
    std::string  out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += char(c);
        } else if (c < 32) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else {
            out += char(c);
        }
    }
    return out + '"';
}

struct node {
    char                        type = 'n';  // s string, a array, o object, d number, b boolean, n null
    std::string                 text;
    std::vector<node>           array;
    std::map<std::string, node> object;
};

class parser {
    const std::string & text_;
    size_t              at_  = 0;
    bool                bad_ = false;

    void fail() noexcept { bad_ = true; }

    void space() {
        while (at_ < text_.size() &&
               (text_[at_] == ' ' || text_[at_] == '\t' || text_[at_] == '\r' || text_[at_] == '\n')) {
            ++at_;
        }
    }

    bool take(char c) {
        space();
        if (at_ < text_.size() && text_[at_] == c) {
            ++at_;
            return true;
        }
        return false;
    }

    unsigned hex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            if (at_ == text_.size()) {
                fail();
                return {};
            }
            const char c     = text_[at_++];
            const int  digit = c >= '0' && c <= '9' ? c - '0' :
                               c >= 'a' && c <= 'f' ? c - 'a' + 10 :
                               c >= 'A' && c <= 'F' ? c - 'A' + 10 :
                                                      -1;
            if (digit < 0) {
                fail();
                return {};
            }
            value = value * 16 + unsigned(digit);
        }
        return value;
    }

    static void utf8(std::string & out, unsigned cp) {
        if (cp < 128) {
            out += char(cp);
        } else if (cp < 2048) {
            out += char(0xc0 | (cp >> 6));
            out += char(0x80 | (cp & 63));
        } else if (cp < 65536) {
            out += char(0xe0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 63));
            out += char(0x80 | (cp & 63));
        } else {
            out += char(0xf0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 63));
            out += char(0x80 | ((cp >> 6) & 63));
            out += char(0x80 | (cp & 63));
        }
    }

    std::string string() {
        if (!take('"')) {
            fail();
            return {};
        }
        std::string out;
        while (at_ < text_.size()) {
            unsigned char c = text_[at_++];
            if (c == '"') {
                return out;
            }
            if (c < 32) {
                fail();
                return {};
            }
            if (c != '\\') {
                out += char(c);
                continue;
            }
            if (at_ == text_.size()) {
                fail();
                return {};
            }
            c = text_[at_++];
            if (c == '"' || c == '\\' || c == '/') {
                out += char(c);
            } else if (c == 'b') {
                out += '\b';
            } else if (c == 'f') {
                out += '\f';
            } else if (c == 'n') {
                out += '\n';
            } else if (c == 'r') {
                out += '\r';
            } else if (c == 't') {
                out += '\t';
            } else if (c == 'u') {
                unsigned cp = hex4();
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (text_.substr(at_, 2) != "\\u") {
                        fail();
                        return {};
                    }
                    at_ += 2;
                    const unsigned low = hex4();
                    if (low < 0xdc00 || low > 0xdfff) {
                        fail();
                        return {};
                    }
                    cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
                } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                    fail();
                    return {};
                }
                utf8(out, cp);
            } else {
                fail();
                return {};
            }
        }
        fail();
        return {};
    }

    node number() {
        node out;
        out.type           = 'd';
        const size_t begin = at_;
        if (text_[at_] == '-') {
            ++at_;
        }
        if (at_ == text_.size()) {
            fail();
            return {};
        }
        if (text_[at_] == '0') {
            ++at_;
        } else {
            if (text_[at_] < '1' || text_[at_] > '9') {
                fail();
                return {};
            }
            while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9') {
                ++at_;
            }
        }
        if (at_ < text_.size() && text_[at_] == '.') {
            const size_t from = ++at_;
            while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9') {
                ++at_;
            }
            if (at_ == from) {
                fail();
                return {};
            }
        }
        if (at_ < text_.size() && (text_[at_] == 'e' || text_[at_] == 'E')) {
            ++at_;
            if (at_ < text_.size() && (text_[at_] == '+' || text_[at_] == '-')) {
                ++at_;
            }
            const size_t from = at_;
            while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9') {
                ++at_;
            }
            if (at_ == from) {
                fail();
                return {};
            }
        }
        out.text = text_.substr(begin, at_ - begin);
        if (!std::isfinite(std::strtod(out.text.c_str(), nullptr))) {
            fail();
            return {};
        }
        return out;
    }

    node parse_node(unsigned depth) {
        if (depth > 64) {
            fail();
            return {};
        }
        space();
        if (at_ == text_.size()) {
            fail();
            return {};
        }
        node out;
        if (text_[at_] == '"') {
            out.type = 's';
            out.text = string();
            return out;
        }
        if (take('[')) {
            out.type = 'a';
            if (take(']')) {
                return out;
            }
            do {
                out.array.push_back(parse_node(depth + 1));
                if (bad_) {
                    return {};
                }
            } while (take(','));
            if (!take(']')) {
                fail();
                return {};
            }
            return out;
        }
        if (take('{')) {
            out.type = 'o';
            if (take('}')) {
                return out;
            }
            do {
                const std::string key = string();
                if (!take(':') || out.object.count(key)) {
                    fail();
                    return {};
                }
                out.object.emplace(key, parse_node(depth + 1));
                if (bad_) {
                    return {};
                }
            } while (take(','));
            if (!take('}')) {
                fail();
                return {};
            }
            return out;
        }
        for (const char * literal : { "true", "false", "null" }) {
            const std::string word(literal);
            if (text_.compare(at_, word.size(), word) == 0) {
                at_ += word.size();
                out.type = word == "null" ? 'n' : 'b';
                out.text = word;
                return out;
            }
        }
        return number();
    }
  public:
    explicit parser(const std::string & text) : text_(text) {}

    result<node> read() {
        if (!valid_utf8(text_)) {
            return { {}, status::invalid, "invalid UTF-8 JSON" };
        }
        node out = parse_node(0);
        space();
        if (bad_ || at_ != text_.size()) {
            return { {}, status::invalid, "invalid JSON at byte " + std::to_string(at_) };
        }
        return { std::move(out), status::ok, {} };
    }
};

inline result<node> parse(const std::string & text) {
    return parser(text).read();
}

}  // namespace json
}  // namespace ssd
