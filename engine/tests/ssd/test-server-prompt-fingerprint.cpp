// SPDX-License-Identifier: Apache-2.0
#include "inferssd/prompt-fingerprint.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

struct comma_punctuation : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '_'; }
    std::string do_grouping() const override { return "\3"; }
};

int main() {
    try {
        common_params params;
        const auto key = server_prompt_cache_position_key(params, 4096, 2048);
        const std::vector<std::function<void(common_params &)>> changes = {
            [](common_params & p) { p.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; },
            [](common_params & p) { p.rope_freq_base = 100; },
            [](common_params & p) { p.rope_freq_scale = 0.5f; },
            [](common_params & p) { p.yarn_ext_factor = 0.5f; },
            [](common_params & p) { p.yarn_attn_factor = 0.5f; },
            [](common_params & p) { p.yarn_beta_fast = 16; },
            [](common_params & p) { p.yarn_beta_slow = 2; },
            [](common_params & p) { p.yarn_orig_ctx = 1024; },
        };
        for (const auto & change : changes) {
            auto different = params;
            change(different);
            require(server_prompt_cache_position_key(different, 4096, 2048) != key,
                    "position override did not change the cache key");
        }
        require(server_prompt_cache_position_key(params, 8192, 2048) != key, "resolved context size omitted");
        require(server_prompt_cache_position_key(params, 4096, 4096) != key, "resolved sequence size omitted");
        auto unrelated = params;
        unrelated.cpuparams.n_threads = 3;
        unrelated.sampling.seed = 42;
        unrelated.sampling.temp = 0.25f;
        require(server_prompt_cache_position_key(unrelated, 4096, 2048) == key,
                "thread or sampling options invalidated reusable K/V");
        params.rope_freq_scale = 1;
        const auto exact = server_prompt_cache_position_key(params, 4096, 2048);
        params.rope_freq_scale = std::nextafter(1.0f, 2.0f);
        require(server_prompt_cache_position_key(params, 4096, 2048) != exact,
                "distinct adjacent float settings share a key");
        const auto stable = server_prompt_cache_position_key(params, 4096, 2048);
        const auto previous = std::locale::global(std::locale(std::locale::classic(), new comma_punctuation));
        const auto localized = server_prompt_cache_position_key(params, 4096, 2048);
        std::locale::global(previous);
        require(localized == stable, "locale changed the cache key");
        std::puts("PROMPT_FINGERPRINT_PASS checks=13");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "PROMPT_FINGERPRINT_FAIL: %s\n", error.what());
        return 1;
    }
}
