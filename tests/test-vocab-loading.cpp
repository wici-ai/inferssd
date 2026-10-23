// SPDX-License-Identifier: MIT
#include "llama-impl.h"
#include "unicode.h"
#include "gguf.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void test_byte_mapping() {
    for (unsigned byte = 0; byte < 256; ++byte) {
        const auto encoded = unicode_byte_to_utf8(byte);
        size_t offset = 0;
        const auto cpt = unicode_cpt_from_utf8(encoded, offset);
        GGML_ASSERT(offset == encoded.size());
        GGML_ASSERT(unicode_cpt_to_byte(cpt) == byte);
    }
    // Include holes inside the lookup table and code points outside it.
    std::vector<uint32_t> codepoints;
    for (uint32_t cpt = 0; cpt < 512; ++cpt) {
        codepoints.push_back(cpt);
    }
    codepoints.insert(codepoints.end(), {0x400, 0x2581, 0x1f600, 0x10ffff});
    for (const auto cpt : codepoints) {
        bool old_throws = false, new_throws = false;
        uint8_t old_value = 0, new_value = 0;
        try { old_value = unicode_utf8_to_byte(unicode_cpt_to_utf8(cpt)); }
        catch (const std::out_of_range &) { old_throws = true; }
        try { new_value = unicode_cpt_to_byte(cpt); }
        catch (const std::out_of_range &) { new_throws = true; }
        GGML_ASSERT(old_throws == new_throws);
        GGML_ASSERT(old_throws || old_value == new_value);
    }
}

static void test_metadata_rendering() {
    auto * ctx = gguf_init_empty();
    GGML_ASSERT(ctx);
    const char * words[] = {"", "a\\b", "a\"b", "\n", "token"};
    gguf_set_arr_str(ctx, "strings", words, 5);
    const std::string expected = "[\"\", \"a\\\\b\", \"a\\\"b\", \"\n\", \"token\"]";
    GGML_ASSERT(gguf_kv_to_str(ctx, 0) == expected);
    const int32_t numbers[] = {-1, 0, 100};
    gguf_set_arr_data(ctx, "numbers", GGUF_TYPE_INT32, numbers, 3);
    GGML_ASSERT(gguf_kv_to_str(ctx, 1) == "[-1, 0, 100]");
    gguf_set_arr_str(ctx, "empty", words, 0);
    GGML_ASSERT(gguf_kv_to_str(ctx, 2, 0) == "[]");
    const std::string scalar(1000, 'x');
    gguf_set_val_str(ctx, "scalar", scalar.c_str());
    GGML_ASSERT(gguf_kv_to_str(ctx, 3, 40) == scalar);
    gguf_set_val_bool(ctx, "boolean", true);
    GGML_ASSERT(gguf_kv_to_str(ctx, 4, 0) == "true");

    for (int key : {0, 1}) {
        const auto full = gguf_kv_to_str(ctx, key);
        for (size_t limit = 0; limit <= full.size(); ++limit) {
            const auto shortened = gguf_kv_to_str(ctx, key, limit);
            GGML_ASSERT(full.compare(0, shortened.size(), shortened) == 0);
            GGML_ASSERT(shortened == full || shortened.size() > limit);
        }
    }
    // Logging should not format the remainder of a large tokenizer array.
    std::vector<const char *> many(100000, "token");
    gguf_set_arr_str(ctx, "large", many.data(), many.size());
    const auto preview = gguf_kv_to_str(ctx, 5, 40);
    GGML_ASSERT(preview.size() > 40 && preview.size() < 60);
    GGML_ASSERT(preview == gguf_kv_to_str(ctx, 5).substr(0, preview.size()));
    gguf_free(ctx);
}

int main() {
    test_byte_mapping();
    test_metadata_rendering();
    std::puts("vocabulary loading helpers: OK");
}
