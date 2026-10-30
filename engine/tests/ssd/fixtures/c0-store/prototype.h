#pragma once
// Production index view; test.cpp separately compiles the fixed legacy reference.
#include "../../../../ggml/src/ggml-ssd/store/legacy-index.h"
// The POSIX calls of the reference and these fixtures; the store itself is portable.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
