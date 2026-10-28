// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>

namespace ssd { namespace platform {

// The commit and discard granularity: 4 KiB on x86, 16 KiB on Apple silicon.
size_t page_size();

// Heap blocks with a power-of-two alignment that is a multiple of
// sizeof(void *); nullptr on failure. Release only with aligned_free.
void * aligned_alloc_bytes(size_t alignment, size_t bytes);
void aligned_free(void * p);

// An elastic arena such as the RAM expert cache: address space is reserved
// once, and a slot holds memory and accounting only while it holds data.
//
//   reserve -> [no_hugepage] -> per slot: commit -> [populate] -> use
//                                         -> discard -> commit -> use ...
//   -> release
//
// commit is where Windows charges commit, as the host ledger charges; Linux
// charges at first touch, macOS takes the slot back into phys_footprint.
// discard drops the slot's memory and accounting without keeping its contents;
// a discarded slot must be committed again before it is touched. Ranges are
// page aligned.

// Address space only: the ledger admits each resident slot, so the OS must not
// refuse an arena larger than RAM plus swap. nullptr, with the reason in *why.
void * arena_reserve(size_t bytes, std::string * why = nullptr);
bool arena_commit(void * p, size_t bytes);
bool arena_discard(void * p, size_t bytes);
// Ordinary pages, so a discard returns exactly one slot (Linux THP).
bool arena_no_hugepage(void * p, size_t bytes);
// Makes every page of a committed range resident before its first use. True
// when the kernel populated it in one call (Linux MADV_POPULATE_WRITE), false
// when it was touched page by page; either way the range is resident.
bool arena_populate(void * begin, size_t bytes);
// Aborts on failure: the arena handle would be corrupt.
void arena_release(void * p, size_t bytes);

} } // namespace ssd::platform
