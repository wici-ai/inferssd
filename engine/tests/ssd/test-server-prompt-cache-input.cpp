// SPDX-License-Identifier: Apache-2.0
#include "inferssd/prompt-cache.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct temporary_directory {
    std::filesystem::path path;

    temporary_directory() {
        std::string pattern = (std::filesystem::temp_directory_path() / "prompt-input-XXXXXX").string();
        require(mkdtemp(pattern.data()) != nullptr, "cannot create test directory");
        path = pattern;
    }

    ~temporary_directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

template <typename T>
static void write_value(std::ofstream & output, T value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

// A real on-disk header and payload; only discovery is exercised, not state deserialization.
static void write_prompt(const std::filesystem::path & path) {
    std::ofstream output(path, std::ios::binary);
    write_value(output, uint32_t(0x4643504c));
    write_value(output, uint32_t(1));
    const std::string fingerprint = "fixture";
    write_value(output, uint32_t(fingerprint.size()));
    output.write(fingerprint.data(), fingerprint.size());
    write_value(output, uint32_t(3));
    for (llama_token token : { 1, 2, 3 }) {
        write_value(output, token);
    }
    write_value(output, uint32_t(0)); // No checkpoints.
    write_value(output, uint64_t(4));
    write_value(output, uint32_t(123));
    output.close();
    require(bool(output), "cannot write prompt fixture");
}

static void check_input(int scenario) {
    temporary_directory root;
    const auto cache_dir = root.path / "cache";
    require(std::filesystem::create_directory(cache_dir), "cannot create cache directory");
    const auto entry = cache_dir / (scenario == 1 ? u8"caf\u00e9.prompt" : "fixture.prompt");
    const auto target = root.path / "target";
    const bool accepted = scenario < 3;
    switch (scenario) {
        case 0:
        case 1: write_prompt(entry); break;
        case 2:
            write_prompt(target);
            std::filesystem::create_symlink(target, entry);
            break;
        case 3: require(mkfifo(entry.c_str(), 0600) == 0, "cannot create FIFO"); break;
        case 4:
            require(mkfifo(target.c_str(), 0600) == 0, "cannot create FIFO target");
            std::filesystem::create_symlink(target, entry);
            break;
        case 5: std::filesystem::create_directory(entry); break;
        case 6: std::ofstream(entry) << "not a prompt"; break;
        case 7: std::filesystem::create_symlink(target, entry); break;
    }

    // Keep the parent available to report a blocked open, and leave no FIFO writer that could hide it.
    const pid_t child = fork();
    require(child >= 0, "cannot fork cache discovery");
    if (child == 0) {
        alarm(5);
        try {
            server_prompt_cache cache(0, 0);
            cache.use_dir(cache_dir.string(), "fixture");
            require(cache.states.size() == size_t(accepted), "unexpected cached prompt count");
            require(std::filesystem::exists(std::filesystem::symlink_status(entry)) == accepted,
                    "invalid cache entry should be removed");
            if (accepted) {
                const auto & prompt = cache.states.front();
                require(prompt.tokens.get_tokens() == llama_tokens({ 1, 2, 3 }), "token metadata changed");
                require(prompt.file_data_size == 4, "state size changed");
                require(prompt.file == entry.string(), "cached path changed");
            }
            if (scenario == 2 || scenario == 4) {
                require(std::filesystem::exists(target), "symlink target was removed");
            }
            _exit(0);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "cache discovery: %s\n", error.what());
            _exit(1);
        }
    }
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    require(waited == child, "cannot wait for cache discovery");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "cache discovery blocked or failed");
}

int main() {
    int failed = 0;
    for (int scenario = 0; scenario < 8; ++scenario) {
        try {
            check_input(scenario);
            std::printf("prompt-input case %d passed\n", scenario);
        } catch (const std::exception & error) {
            ++failed;
            std::fprintf(stderr, "prompt-input case %d failed: %s\n", scenario, error.what());
        }
    }
    return failed == 0 ? 0 : 1;
}
