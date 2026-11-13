// SPDX-License-Identifier: Apache-2.0
#include "inferssd/prompt-cache.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
static void append(std::vector<uint8_t> & bytes, T value) {
    const auto * data = reinterpret_cast<const uint8_t *>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

struct prompt_file {
    std::vector<uint8_t> bytes;
    size_t tokens_offset;
    size_t checkpoint_size_offset;
    size_t state_size_offset;
    size_t data_offset;

    prompt_file() {
        append(bytes, uint32_t(0x4643504c));
        append(bytes, uint32_t(1));
        const std::string fingerprint = "fixture";
        append(bytes, uint32_t(fingerprint.size()));
        bytes.insert(bytes.end(), fingerprint.begin(), fingerprint.end());
        append(bytes, uint32_t(3));
        tokens_offset = bytes.size();
        for (llama_token token : { 1, 2, 3 }) {
            append(bytes, token);
        }
        append(bytes, uint32_t(1));
        append(bytes, llama_pos(0));
        append(bytes, llama_pos(1));
        append(bytes, int64_t(2));
        checkpoint_size_offset = bytes.size();
        append(bytes, uint64_t(4));
        state_size_offset = bytes.size();
        append(bytes, uint64_t(5));
        data_offset = bytes.size();
        bytes.insert(bytes.end(), { 10, 11, 12, 13, 20, 21, 22, 23, 24 });
    }

    void replace_size(size_t offset, uint64_t value) {
        std::vector<uint8_t> encoded;
        append(encoded, value);
        std::copy(encoded.begin(), encoded.end(), bytes.begin() + offset);
    }
};

struct temporary_directory {
    std::filesystem::path path;

    temporary_directory() {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("prompt-cache-file-" + std::to_string(unique));
        require(std::filesystem::create_directory(path), "cannot create test directory");
    }

    ~temporary_directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

static void check_file(const prompt_file & fixture, bool accepted, const std::string & fingerprint = "fixture") {
    temporary_directory directory;
    const auto path = directory.path / "fixture.prompt";
    {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char *>(fixture.bytes.data()), fixture.bytes.size());
        output.close();
        require(bool(output), "cannot write prompt fixture");
    }

    server_prompt_cache cache(0, 0);
    cache.use_dir(directory.path.string(), fingerprint);
    require(cache.states.size() == size_t(accepted), "unexpected cached prompt count");
    require(std::filesystem::exists(path) == accepted, "invalid prompt should be removed on discovery");
    if (accepted) {
        const auto & prompt = cache.states.front();
        require(prompt.tokens.get_tokens() == llama_tokens({ 1, 2, 3 }), "token metadata changed");
        require(prompt.checkpoints.size() == 1, "checkpoint metadata missing");
        require(prompt.checkpoints.front().file_offset == fixture.data_offset, "wrong checkpoint offset");
        require(prompt.checkpoints.front().file_size == 4, "wrong checkpoint size");
        require(prompt.file_data_offset == fixture.data_offset + 4, "wrong main state offset");
        require(prompt.file_data_size == 5, "wrong main state size");
    }
}

int main() {
    int failed = 0;
    for (int scenario = 0; scenario < 9; ++scenario) {
        try {
            prompt_file fixture;
            bool accepted = false;
            std::string fingerprint = "fixture";
            switch (scenario) {
                case 0: accepted = true; break; // Exact end-of-file boundary.
                case 1: fixture.bytes.pop_back(); break; // Main state truncated.
                case 2: fixture.bytes.resize(fixture.data_offset + 2); break; // Checkpoint truncated.
                case 3: fixture.replace_size(fixture.checkpoint_size_offset, std::numeric_limits<uint64_t>::max()); break;
                case 4: fixture.replace_size(fixture.state_size_offset, std::numeric_limits<uint64_t>::max()); break;
                case 5: fixture.bytes.resize(fixture.tokens_offset + 1); break;
                case 6: fixture.bytes.resize(fixture.state_size_offset - 1); break;
                case 7: fingerprint = "different"; break;
                case 8: fixture.bytes.push_back(99); accepted = true; break; // Extra bytes remain allowed.
            }
            check_file(fixture, accepted, fingerprint);
            std::printf("prompt-file case %d passed\n", scenario);
        } catch (const std::exception & error) {
            ++failed;
            std::fprintf(stderr, "prompt-file case %d failed: %s\n", scenario, error.what());
        }
    }
    return failed == 0 ? 0 : 1;
}
