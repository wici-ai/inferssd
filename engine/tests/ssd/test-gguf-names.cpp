// SPDX-License-Identifier: Apache-2.0
#include "ggml.h"
#include "gguf.h"
#include "test-resources.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

template<typename T>
static void write_value(FILE * file, T value) {
    GGML_ASSERT(fwrite(&value, sizeof(value), 1, file) == 1);
}

static FILE * catalog(const std::string & path, const size_t count, const bool duplicate,
        const bool embedded_nul, const bool empty_name) {
    FILE * file = fopen(path.c_str(), "w+b");
    GGML_ASSERT(file);
    GGML_ASSERT(fwrite("GGUF", 1, 4, file) == 4);
    write_value<uint32_t>(file, 3);
    write_value<uint64_t>(file, count);
    write_value<uint64_t>(file, 0);
    for (size_t i = 0; i < count; ++i) {
        std::string name = "layer.tensor." + std::to_string(duplicate && i + 1 == count ? 0 : i);
        if (empty_name && (i == 0 || (duplicate && i + 1 == count))) {
            name.clear();
        }
        if (embedded_nul && i + 1 == count) {
            name += std::string("\0suffix", 7);
        }
        write_value<uint64_t>(file, name.size());
        GGML_ASSERT(fwrite(name.data(), 1, name.size(), file) == name.size());
        write_value<uint32_t>(file, 1);
        write_value<uint64_t>(file, 1);
        write_value<uint32_t>(file, GGML_TYPE_F32);
        write_value<uint64_t>(file, i * GGUF_DEFAULT_ALIGNMENT);
    }
    while (ftell(file) % GGUF_DEFAULT_ALIGNMENT) {
        write_value<uint8_t>(file, 0);
    }
    rewind(file);
    return file;
}

static void capture_log(enum ggml_log_level, const char * text, void * data) {
    *static_cast<std::string *>(data) += text;
}

static double check(const size_t count, const bool duplicate, const bool embedded_nul, const bool empty_name = false) {
    const std::string path = test_resources::temp_file("gguf-names");
    FILE * file = catalog(path, count, duplicate, embedded_nul, empty_name);
    std::string logs;
    ggml_log_set(capture_log, &logs);
    const auto start = std::chrono::steady_clock::now();
    gguf_context * ctx = gguf_init_from_file_ptr(file, {true, nullptr});
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    ggml_log_set(nullptr, nullptr);
    if (duplicate) {
        GGML_ASSERT(ctx == nullptr);
        GGML_ASSERT(logs.find("for tensors 0 and " + std::to_string(count - 1)) != std::string::npos);
    } else {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(gguf_get_n_tensors(ctx) == int64_t(count));
        for (size_t i = 0; i < count; ++i) {
            const std::string expected = empty_name && i == 0 ? "" : "layer.tensor." + std::to_string(i);
            GGML_ASSERT(strcmp(gguf_get_tensor_name(ctx, i), expected.c_str()) == 0);
            GGML_ASSERT(gguf_get_tensor_offset(ctx, i) == i * GGUF_DEFAULT_ALIGNMENT);
            GGML_ASSERT(gguf_get_tensor_type(ctx, i) == GGML_TYPE_F32);
        }
    }
    gguf_free(ctx);
    fclose(file);
    test_resources::remove_file(path);
    return elapsed_ms;
}

int main(int argc, char ** argv) {
    if (argc == 2) {
        // Optional reproducible parser timing; fixture creation and verification
        // are outside the measured interval. No tensor payload or GPU is used.
        const size_t count = std::stoul(argv[1]);
        GGML_ASSERT(count > 0 && count <= 100000);
        printf("tensors=%zu parse_ms=%.6f\n", count, check(count, false, false));
        return 0;
    }
    GGML_ASSERT(argc == 1);
    check(0, false, false);
    check(1, false, false);
    check(4096, false, false);
    check(4096, true, false);
    // ggml uses C-string names: a suffix after NUL must not bypass duplicate checks.
    check(4096, true, true);
    check(2, false, true);
    check(1, false, false, true);
    check(2, true, false, true);
    puts("GGUF tensor-name identity checks passed");
    return 0;
}
