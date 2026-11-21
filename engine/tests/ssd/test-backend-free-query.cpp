// SPDX-License-Identifier: Apache-2.0
// The queries the parser answers before it loads backends (--ssd-schema, --ssd-model-facts, --version): the real
// parser, compiled with its backend loader redirected to a tripwire.
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool expect_backend = false;
static bool loaded_backend = false;

static void verify_exit() {
    if (loaded_backend != expect_backend) {
        std::fputs("unexpected backend loading behavior\n", stderr);
        std::_Exit(1);
    }
}

// Declared by the test-only forced-include header, not a backend DLL export.
void inferssd_test_backend_load_all() {
    loaded_backend = true;
    if (!expect_backend) {
        std::fputs("the query attempted to load compute backends\n", stderr);
        std::exit(1);
    }
    // Control cases stop here: no real backend or model is needed for this test.
    std::exit(0);
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--expect-backend") == 0) {
        expect_backend = true;
        --argc;
        ++argv;
    }
    if (std::atexit(verify_exit) != 0) {
        return 1;
    }
    common_init();
    common_params params;
    common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER);
    return 1; // Either the query or the tripwire must exit first.
}
