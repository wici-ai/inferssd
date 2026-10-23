#include "parser.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <unistd.h>

// The parser is linked directly into this fixture so allocation failure is
// injected at its actual container operations, without DSO interposition rules.
static thread_local int fail_after = -1;
void * operator new(std::size_t size) {
    if (fail_after == 0) { throw std::bad_alloc(); }
    if (fail_after > 0) { --fail_after; }
    if (void * value = std::malloc(size ? size : 1)) { return value; }
    throw std::bad_alloc();
}
void * operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void * value) noexcept { std::free(value); }
void operator delete[](void * value) noexcept { std::free(value); }
void operator delete(void * value, std::size_t) noexcept { std::free(value); }
void operator delete[](void * value, std::size_t) noexcept { std::free(value); }

namespace {
using fd_set = std::array<bool, 1024>;
fd_set open_fds() {
    fd_set result{};
    for (int fd = 0; fd < int(result.size()); ++fd) { result[fd] = ::fcntl(fd, F_GETFD) != -1; }
    return result;
}
void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}
struct fixture {
    std::filesystem::path directory;
    std::string data;
    std::string alias;
    fixture() {
        char name[] = "/tmp/ssdash-c0-parser-failures-XXXXXX";
        char * result = ::mkdtemp(name);
        if (!result) { throw std::runtime_error("mkdtemp failed"); }
        directory = result;
        data = (directory / "data.bin").string();
        alias = (directory / "alias.tsv").string();
        std::ofstream(data, std::ios::binary) << std::string(8192, 'x');
        std::ofstream index(alias);
        index << "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
              << "blk.0.ffn_up_exps.weight\t0\t4096\t" << data << "\t0\n";
        check(bool(index), "alias fixture write failed");
    }
    ~fixture() { std::filesystem::remove_all(directory); }
};

void throwing_hook() { throw std::runtime_error("injected hook failure"); }
void check_hook_cleanup(const fixture & files) {
    const auto before = open_fds();
    c0::parser_state state;
    c0::source_open_hooks hooks;
    hooks.page_cache_source_opened = throwing_hook;
    bool threw = false;
    try { c0::open_data_source(state, files.data.c_str(), "fixture", true, false, hooks); }
    catch (const std::runtime_error &) { threw = true; }
    state.reset();
    check(threw && before == open_fds(), "throwing source hook leaked a file descriptor");
}

unsigned check_allocations(const fixture & files, bool alias) {
    c0::description desc;
    desc.aliases.push_back(files.alias);
    for (int position = 0; position < 64; ++position) {
        const auto before = open_fds();
        c0::parser_state state;
        std::vector<c0::expert_pack_entry> entries;
        bool threw = false;
        bool ok = false;
        fail_after = position;
        try {
            ok = alias ? c0::append_sources(state, desc, entries) :
                c0::open_data_source(state, files.data.c_str(), "fixture");
        } catch (const std::bad_alloc &) { threw = true; }
        catch (...) { fail_after = -1; throw; }
        fail_after = -1;
        state.reset();
        check(before == open_fds(), "parser allocation failure leaked a file descriptor");
        if (!threw) {
            check(ok, "fixture parsing failed without allocation injection");
            check(position >= (alias ? 4 : 1), "fixture did not exercise expected allocations");
            return unsigned(position);
        }
    }
    throw std::runtime_error("fixture allocation bound exceeded");
}
}

int main(int argc, char ** argv) {
    try {
        fixture files;
        const bool alias_only = argc == 2 && std::strcmp(argv[1], "--alias-only") == 0;
        check(argc == 1 || alias_only, "unexpected argument");
        if (!alias_only) { check_hook_cleanup(files); }
        const unsigned source = alias_only ? 0 : check_allocations(files, false);
        const unsigned alias = check_allocations(files, true);
        std::cout << "source allocation failures=" << source << " alias allocation failures=" << alias
                  << "; hook and all allocation points released handles: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        fail_after = -1;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
