// SPDX-License-Identifier: Apache-2.0
#include "arg.h"
#include "common.h"
#include "inferssd/options.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "schema-reference") {
        common_ssd_print_schema();
        return 0;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    std::printf("model=%s\n", params.model.path.c_str());
    return 0;
}
