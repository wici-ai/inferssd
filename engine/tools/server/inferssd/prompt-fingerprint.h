// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common.h"

#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

// These options change position-dependent K/V values without changing the
// model file or build. Include the resolved context sizes as well as overrides:
// automatic fitting can resolve the same requested size differently.
inline std::string server_prompt_cache_position_key(
        const common_params & params, uint32_t n_ctx, uint32_t n_ctx_seq) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "position-v1|" << n_ctx << '|' << n_ctx_seq << '|'
        << int(params.rope_scaling_type) << '|' << params.rope_freq_base << '|'
        << params.rope_freq_scale << '|' << params.yarn_ext_factor << '|'
        << params.yarn_attn_factor << '|' << params.yarn_beta_fast << '|'
        << params.yarn_beta_slow << '|' << params.yarn_orig_ctx;
    return out.str();
}
