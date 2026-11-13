// SPDX-License-Identifier: Apache-2.0
// Exercise the real cache writer; only the context state payload is interposed.
#include "inferssd/prompt-cache.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

static bool short_state_write = false;
static mode_t state_file_mode = 0;
static int state_fd_flags = -1;
static constexpr unsigned char payload[] = {1, 2, 3, 4, 5, 6, 7, 8};

extern "C" size_t __wrap_llama_state_seq_get_size_at(llama_context *, llama_seq_id, uint32_t, int32_t) {
    return sizeof(payload);
}

extern "C" size_t __wrap_llama_state_seq_fwrite_at(llama_context *, FILE * file, llama_seq_id, uint32_t, int32_t) {
    struct stat st {};
    if (fstat(fileno(file), &st) != 0) return 0;
    state_file_mode = st.st_mode & 0777;
    state_fd_flags = fcntl(fileno(file), F_GETFD);
    const size_t size = short_state_write ? 3 : sizeof(payload);
    return std::fwrite(payload, 1, size, file);
}

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct mask_scope {
    mode_t previous;
    explicit mask_scope(mode_t value) : previous(umask(value)) {}
    ~mask_scope() { umask(previous); }
};

static std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

static void write_file(const fs::path & path, const char * value) {
    std::ofstream out(path, std::ios::binary);
    out << value;
    require(bool(out), "fixture file write failed");
}

struct fixture {
    fs::path root;
    fs::path final;
    fs::path temporary;
    server_prompt_cache cache{0, 0};
    server_prompt prompt;

    fixture() {
        const char * base = std::getenv("TMPDIR");
        const std::string name = (fs::path(base ? base : "/tmp") / "prompt-write-XXXXXX").string();
        std::vector<char> buffer(name.begin(), name.end());
        buffer.push_back('\0');
        require(mkdtemp(buffer.data()) != nullptr, "fixture directory creation failed");
        root = buffer.data();
        cache.use_dir(root.string(), "write-fixture");
        prompt.tokens = server_tokens({3, 19, 27, 51}, false);
        // Obtain the production token-hash path without duplicating its algorithm.
        save();
        require(cache.states.size() == 1, "fixture cache priming failed");
        final = cache.states.front().file;
        temporary = final.string() + ".tmp";
        cache.states.clear();
        require(fs::remove(final), "fixture priming cleanup failed");
    }

    ~fixture() { std::error_code ec; fs::remove_all(root, ec); }

    void save() { cache.save(prompt, true, nullptr, 0, COMMON_CONTEXT_SEQ_RM_TYPE_PART); }
    void saved() {
        require(cache.states.size() == 1 && fs::is_regular_file(final), "prompt was not published");
        require(!fs::exists(temporary), "temporary entry remains after publish");
    }
    void rejected() {
        require(cache.states.empty(), "failed write published a cache entry");
    }
};

static std::vector<int> descriptors() {
    std::vector<int> result;
    for (int fd = 0; fd < 512; ++fd) {
        if (fcntl(fd, F_GETFD) >= 0 || errno != EBADF) result.push_back(fd);
    }
    return result;
}

int main() {
    const std::vector<std::pair<const char *, std::function<void()>>> cases = {
        {"private under umask 000", [] {
            fixture f; mask_scope mask(0000); f.save(); f.saved();
            require(state_file_mode == 0600, "new prompt permits access by other users");
        }},
        {"private under umask 022", [] {
            fixture f; mask_scope mask(0022); f.save(); f.saved();
            require(state_file_mode == 0600, "new prompt permits access by other users");
        }},
        {"private under umask 077", [] {
            fixture f; mask_scope mask(0077); f.save(); f.saved();
            require(state_file_mode == 0600, "restrictive umask changed private permissions");
        }},
        {"honor stricter umask", [] {
            fixture f; mask_scope mask(0777); f.save(); f.saved();
            require(state_file_mode == 0000, "writer widened the requested permissions");
        }},
        {"replace old public final atomically", [] {
            fixture f; write_file(f.final, "old-final");
            require(chmod(f.final.c_str(), 0644) == 0, "fixture chmod failed");
            mask_scope mask(0022); f.save(); f.saved();
            struct stat st {};
            require(stat(f.final.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600,
                    "replacement prompt retained public permissions");
            require(read_file(f.final) != "old-final", "replacement did not publish new state");
        }},
        {"preserve existing temporary file", [] {
            fixture f; write_file(f.temporary, "other-writer"); f.save(); f.rejected();
            require(read_file(f.temporary) == "other-writer" && !fs::exists(f.final),
                    "writer modified an existing temporary file");
        }},
        {"preserve existing temporary symlink", [] {
            fixture f; const auto target = f.root / "target";
            write_file(target, "unrelated"); fs::create_symlink(target, f.temporary);
            f.save(); f.rejected();
            require(fs::is_symlink(f.temporary) && read_file(target) == "unrelated" && !fs::exists(f.final),
                    "writer followed or removed an existing temporary symlink");
        }},
        {"preserve dangling temporary symlink", [] {
            fixture f; const auto target = f.root / "absent";
            fs::create_symlink(target, f.temporary); f.save(); f.rejected();
            require(fs::is_symlink(f.temporary) && !fs::exists(target) && !fs::exists(f.final),
                    "writer created the dangling symlink target");
        }},
        {"preserve unowned temporary directory", [] {
            fixture f; fs::create_directory(f.temporary); f.save(); f.rejected();
            require(fs::is_directory(f.temporary) && !fs::exists(f.final),
                    "failed open removed an unowned temporary directory");
        }},
        {"clean owned file after short state write", [] {
            fixture f; write_file(f.final, "old-final"); short_state_write = true;
            f.save(); f.rejected();
            require(!fs::exists(f.temporary) && read_file(f.final) == "old-final",
                    "short write left a temporary file or replaced the old final");
        }},
        {"clean owned file after failed rename", [] {
            fixture f; fs::create_directory(f.final); f.save(); f.rejected();
            require(!fs::exists(f.temporary) && fs::is_directory(f.final),
                    "failed rename left a temporary file or removed the final directory");
        }},
        {"close descriptor on exec", [] {
            fixture f; f.save(); f.saved();
            require((state_fd_flags & FD_CLOEXEC) != 0, "temporary descriptor can survive exec");
        }},
    };
    size_t failed = 0;
    for (const auto & test : cases) {
        short_state_write = false;
        const auto before = descriptors();
        try {
            test.second();
            require(descriptors() == before, "cache writer leaked a descriptor");
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception & error) {
            ++failed;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << "PROMPT_WRITE_RESULT passed=" << cases.size() - failed << " failed=" << failed << '\n';
    return failed ? 1 : 0;
}
