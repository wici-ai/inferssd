// SPDX-License-Identifier: Apache-2.0
#pragma once
// The read ring for this OS: io_uring on Linux, IoRing (or overlapped reads)
// on Windows, and a worker pool elsewhere. All share linux_ring's interface.
#if defined(__linux__)
#include "linux-ring.h"
namespace ssd { namespace io { using read_ring = linux_ring; } }
#elif defined(_WIN32)
#include "win-ring.h"
namespace ssd { namespace io { using read_ring = win_ring; } }
#else
#include "pool-ring.h"
namespace ssd { namespace io { using read_ring = pool_ring; } }
#endif
