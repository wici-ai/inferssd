// SPDX-License-Identifier: Apache-2.0
#pragma once

// Keep the real imported declarations intact, including on Windows DLL builds.
// Only calls compiled into this test's parser are redirected.
#include "ggml-backend.h"
[[noreturn]] void inferssd_test_backend_load_all();
#define ggml_backend_load_all inferssd_test_backend_load_all
