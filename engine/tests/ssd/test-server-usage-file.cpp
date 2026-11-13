// SPDX-License-Identifier: Apache-2.0
#include "inferssd/usage.h"
#include <nlohmann/json.hpp>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

using descriptor_set = std::array<bool, 256>;
static descriptor_set descriptors() {
    descriptor_set out{};
    for (size_t i = 0; i < out.size(); ++i) { out[i] = fcntl(int(i), F_GETFD) != -1; }
    return out;
}

struct temporary_directory {
    std::filesystem::path path;
    temporary_directory() {
        std::string name = (std::filesystem::temp_directory_path() / "usage-file-XXXXXX").string();
        require(mkdtemp(name.data()) != nullptr, "cannot create test directory");
        path = name;
    }
    ~temporary_directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

static void exercise(const std::filesystem::path & root, int scenario) {
    const auto directory = root / (scenario == 4 ? u8"caf\u00e9" : "usage");
    const auto path = directory / "usage-events.jsonl";
    const auto target = root / "target";
    const std::string prefix = "{\"existing\":true}\n";
    const bool accepted = scenario >= 2 && scenario <= 5;
    int reader = -1;
    if (scenario > 2) { std::filesystem::create_directory(directory); }
    switch (scenario) {
        case 0: require(unsetenv("INFERSSD_USAGE_DIR") == 0, "cannot clear usage directory"); break;
        case 1: require(setenv("INFERSSD_USAGE_DIR", "", 1) == 0, "cannot clear usage directory"); break;
        case 3: std::ofstream(path) << prefix; break;
        case 5:
            std::ofstream(target) << prefix;
            std::filesystem::create_symlink(target, path);
            break;
        case 6: require(mkfifo(path.c_str(), 0600) == 0, "cannot create FIFO"); break;
        case 7:
            require(mkfifo(target.c_str(), 0600) == 0, "cannot create FIFO target");
            std::filesystem::create_symlink(target, path);
            break;
        case 8:
            require(mkfifo(path.c_str(), 0600) == 0, "cannot create FIFO with reader");
            reader = open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
            require(reader >= 0, "cannot open FIFO reader");
            break;
        case 9: std::filesystem::create_directory(path); break;
        case 10: std::filesystem::create_symlink("/dev/null", path); break;
    }
    if (scenario >= 2) {
        require(setenv("INFERSSD_USAGE_DIR", directory.c_str(), 1) == 0, "cannot set usage directory");
    }
    const auto before = descriptors();
    {
        auto log = server_usage_log::from_environment();
        require(bool(log) == accepted, "unexpected usage log admission");
        if (log) {
            auto request = log->begin();
            request->add(3, 5);
            request->set_outcome(SERVER_USAGE_COMPLETED);
            request->finish();
            request->finish();
        }
    }
    require(before == descriptors(), "usage log leaked a descriptor");
    if (reader >= 0) { require(close(reader) == 0, "cannot close FIFO reader"); }
    if (accepted) {
        std::ifstream input(path);
        std::string line;
        if (scenario == 3 || scenario == 5) {
            require(bool(std::getline(input, line)) && line + '\n' == prefix, "existing log changed");
        }
        require(bool(std::getline(input, line)), "request event missing");
        const auto received = nlohmann::json::parse(line);
        require(received.at("type") == "request_received", "wrong request event");
        require(bool(std::getline(input, line)), "usage event missing");
        const auto final = nlohmann::json::parse(line);
        require(final.at("type") == "usage_finalized" && final.at("outcome") == "completed" &&
                    final.at("input_tokens") == 3 && final.at("output_tokens") == 5 && final.at("token_count") == 8 &&
                    final.at("request_id") == received.at("request_id"), "usage values changed");
        require(!std::getline(input, line), "usage finalized more than once");
        if (scenario == 2 || scenario == 4) {
            struct stat info{};
            require(stat(path.c_str(), &info) == 0 && (info.st_mode & 0777) == 0600, "new log is not private");
        }
    } else if (scenario >= 6) {
        const auto kind = std::filesystem::symlink_status(path).type();
        require(kind == (scenario == 7 || scenario == 10 ? std::filesystem::file_type::symlink :
                    scenario == 9 ? std::filesystem::file_type::directory : std::filesystem::file_type::fifo),
                "rejected log entry was changed");
    }
}

static void check_case(int scenario) {
    temporary_directory root;
    const pid_t child = fork();
    require(child >= 0, "cannot fork usage log fixture");
    if (child == 0) {
        alarm(3);
        try { exercise(root.path, scenario); _exit(0); }
        catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); _exit(1); }
    }
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    require(waited == child, "cannot wait for usage log fixture");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "usage log blocked or failed");
}

int main() {
    int failed = 0;
    for (int scenario = 0; scenario < 11; ++scenario) {
        try { check_case(scenario); std::printf("usage-file case %d passed\n", scenario); }
        catch (const std::exception & error) {
            ++failed;
            std::fprintf(stderr, "usage-file case %d failed: %s\n", scenario, error.what());
        }
    }
    return failed == 0 ? 0 : 1;
}
