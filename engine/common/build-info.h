#pragma once

int llama_build_number(void);

const char * llama_commit(void);
const char * llama_compiler(void);

const char * llama_build_target(void);
const char * llama_build_info(void);
const char * inferssd_version(void);    // empty before the first release
const char * inferssd_build_name(void); // "InferSSD [version ](commit)"

void llama_print_build_info(void);
