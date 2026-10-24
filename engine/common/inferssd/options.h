// SPDX-License-Identifier: MIT
#pragma once

#include <string>
struct common_params;

std::string common_ssd_option_name(const char * key);
void common_ssd_prepare(common_params & params);
void common_ssd_print_schema();
