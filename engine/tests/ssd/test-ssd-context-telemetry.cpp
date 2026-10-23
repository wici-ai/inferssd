// SPDX-License-Identifier: MIT
#include "ssd/telemetry.h"

#include <cassert>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <new>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static bool fail_next_allocation = false;
void * operator new(std::size_t size) {
    if (fail_next_allocation) { fail_next_allocation = false; throw std::bad_alloc(); }
    if (void * value = std::malloc(size ? size : 1)) { return value; }
    throw std::bad_alloc();
}
void operator delete(void * value) noexcept { std::free(value); }
void operator delete(void * value, std::size_t) noexcept { std::free(value); }

static std::string read(const std::string & path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), {}};
}

int main() {
    char directory[] = "/tmp/ssdash-context-telemetry-XXXXXX";
    assert(mkdtemp(directory));
    const std::string first = std::string(directory) + "/first.jsonl";
    const std::string second = std::string(directory) + "/second.jsonl";
    auto a = ssd::telemetry::writer::open(first);
    auto b = ssd::telemetry::writer::open(second);
    assert(a && b);
    ssd::telemetry::debug_channels debug;
    debug.trace = debug.profile = true;
    llama_ssd_telemetry missing;
    assert(!missing.bind(debug, nullptr));
    llama_ssd_telemetry left, right;
    assert(left.bind(debug, a.value.get()));
    assert(right.bind(debug, b.value.get()));
    uint32_t bits[] = {0x3f800000, 0x80000000, 0x7fc12345}; // exact 1, negative zero, NaN payload
    float values[3];
    std::memcpy(values, bits, sizeof(values));
    assert(left.logits(values, 3, 7));
    assert(left.logits(values, 3, 7)); // dedupe belongs only to this context
    assert(right.logits(values, 3, 9));
    assert(left.graph(true, 17, 0, 0));
    assert(right.graph(false, 23, 11, 0));
    const auto l = read(first), r = read(second);
    assert(l.find("\"seq\":0") != std::string::npos && r.find("\"seq\":0") != std::string::npos);
    assert(l.find("\"seq\":2") == std::string::npos); // two events, no duplicate logits
    assert(l.find("\"output_index\":7") != std::string::npos && r.find("\"output_index\":9") != std::string::npos);
    static const char digits[] = "0123456789abcdef";
    std::string expected;
    for (auto byte : reinterpret_cast<const unsigned char (&)[sizeof(values)]>(values)) {
        expected += digits[byte >> 4]; expected += digits[byte & 15];
    }
    assert(l.find("\"data\":\"" + expected + "\"") != std::string::npos);
    assert(l.find("ssd.context.graph_submit") != std::string::npos);
    assert(r.find("\"tokens\":11") != std::string::npos);
    // Production observations must still be emitted with both debug channels off.
    llama_ssd_telemetry controller;
    assert(controller.bind({}, a.value.get()));
    assert(controller.prefill_graph(128, 8192, 256));
    assert(controller.prefill_copy(4096, 7, 1, 128));
    const auto observed = read(first);
    assert(observed.find("ssd.prefill.graph") != std::string::npos);
    assert(observed.find("\"requested_tokens\":128") != std::string::npos);
    assert(observed.find("ssd.prefill.copy") != std::string::npos);
    assert(observed.find("\"bytes\":4096") != std::string::npos);
    llama_ssd_telemetry allocation;
    assert(allocation.bind(debug, a.value.get()));
    fail_next_allocation = true;
    assert(!allocation.logits(values, 3, 1));
    assert(!allocation.good());
    assert(!allocation.graph(true, 1, 0, 0));
    assert(right.good());

    // The failure is caused by a real FILE flush, in a separate process so the
    // resource limit and SIGXFSZ disposition cannot affect another test/context.
    const std::string failed_path = std::string(directory) + "/failed.jsonl";
    const pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        auto output = ssd::telemetry::writer::open(failed_path);
        if (!output) { _exit(10); }
        llama_ssd_telemetry failed;
        if (!failed.bind(debug, output.value.get())) { _exit(11); }
        signal(SIGXFSZ, SIG_IGN);
        const rlimit limit{0, 0};
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) { _exit(12); }
        if (failed.logits(values, 3, 0) || failed.good() || failed.graph(true, 1, 0, 0)) { _exit(13); }
        _exit(0);
    }
    int status;
    assert(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    a.value.reset(); b.value.reset();
    unlink(first.c_str()); unlink(second.c_str()); unlink(failed_path.c_str()); rmdir(directory);
}
