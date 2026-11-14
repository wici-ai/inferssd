// SPDX-License-Identifier: Apache-2.0
#include "ggml-backend-impl.h"

#include <cstdio>

static ggml_backend_reg_t schema_test_backend() {
    std::fputs("schema test backend initialized\n", stderr);
    return nullptr;
}

GGML_BACKEND_DL_IMPL(schema_test_backend)
