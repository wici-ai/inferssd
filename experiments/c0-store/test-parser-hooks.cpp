#include "parser.h"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
unsigned io_reports = 0;
unsigned page_reports = 0;
void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}
void report_io(const char * label, const char * path) {
    check(std::strcmp(label, "fixture") == 0 && path && path[0], "wrong callback arguments");
    ++io_reports;
}
void report_page() { ++page_reports; }
struct fixture {
    std::filesystem::path directory;
    std::string data;
    fixture() {
        char name[] = "/tmp/ssdash-c0-parser-hooks-XXXXXX";
        char * result = ::mkdtemp(name);
        if (!result) { throw std::runtime_error("mkdtemp failed"); }
        directory = result;
        data = (directory / "data.bin").string();
        std::ofstream out(data, std::ios::binary);
        const std::string bytes(8192, 'x');
        out.write(bytes.data(), bytes.size());
        check(bool(out), "fixture write failed");
    }
    ~fixture() { std::filesystem::remove_all(directory); }
};
}

int main() {
    try {
        fixture files;
        c0::source_open_hooks hooks{report_io, report_page};
        c0::parser_state pages;
        pages.io_backend = 2;
        check(c0::open_data_source(pages, files.data.c_str(), "fixture", true, false, hooks), "page source failed");
        check(pages.io_backend == 2 && pages.page_cache_sources == 1 && page_reports == 1 && io_reports == 0,
                "page source changed transport mode or accounting");
        const int page_fd = ::fileno(pages.sources.at(0).file);
        const auto missing = (files.directory / "missing.bin").string();
        check(!c0::open_data_source(pages, missing.c_str(), "fixture", true, false, hooks), "missing source succeeded");
        check(pages.sources.size() == 1 && pages.page_cache_sources == 1 && page_reports == 1, "failed open reported a source");
        pages.reset();
        errno = 0;
        check(::fcntl(page_fd, F_GETFD) == -1 && errno == EBADF, "reset did not close partial catalog");

#if defined(O_DIRECT)
        // Probe the actual fixture file and filesystem, not an unrelated mount.
        const int probe = ::open(files.data.c_str(), O_RDONLY | O_DIRECT);
        if (probe < 0) {
            if (errno == EINVAL || errno == EOPNOTSUPP || errno == ENOSYS) {
                std::cerr << "SKIP: fixture filesystem does not support O_DIRECT\n";
                return 77;
            }
            throw std::runtime_error("unexpected O_DIRECT preflight error");
        }
        ::close(probe);
        c0::parser_state capable;
        capable.io_backend = 2;
        for (int i = 0; i < 2; ++i) {
            check(c0::open_data_source(capable, files.data.c_str(), "fixture", false, true, hooks), "capable source failed");
        }
        check(capable.io_backend == 2 && io_reports == 2 && page_reports == 1, "provider capability was lost");
        c0::parser_state fallback;
        fallback.io_backend = 2;
        check(c0::open_data_source(fallback, files.data.c_str(), "fixture", false, true), "direct fallback failed");
        check(fallback.io_backend == 1, "missing capability did not preserve direct fallback");
        check(c0::open_data_source(fallback, files.data.c_str(), "fixture", false, true, hooks), "later source failed");
        check(fallback.io_backend == 1 && io_reports == 2, "later source ignored prior transport fallback");
        std::cout << "page accounting, failure/reset, io_uring capability and ordered fallback: PASS\n";
#else
        std::cout << "page accounting and failure/reset: PASS (O_DIRECT cases require Linux)\n";
#endif
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
