#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
bool mock_register(const char * file, int mode);
const void * mock_catalog();
const void * mock_entries();
const void * mock_sources();
unsigned mock_calls();
int mock_fd();
int mock_direct();
const void * mock_mapping();
bool mock_half_ready();
void mock_finish_load();
}
