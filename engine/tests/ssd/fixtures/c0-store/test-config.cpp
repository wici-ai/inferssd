#include "legacy-runtime/legacy-config.h"
#include <stdexcept>

int main() {
    try {
        const char * names[] = {"GGML_MOE_EXPERT_PACK", "GGML_MOE_EXPERT_PACK_OVERLAY", "GGML_MOE_EXPERT_PACK_OVERLAY_EXTRA",
            "GGML_MOE_EXPERT_PACK_LIST", "GGML_MOE_PAGECACHE_PACK", "GGML_MOE_PAGECACHE_PACK_LIST",
            "GGML_MOE_EXPERT_GGUF_ALIAS_TSV", "GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES", "GGML_MOE_IO_BACKEND",
            "GGML_MOE_LOSSLESS_PACK", "GGML_MOE_GLM52_LOSSLESS_PACK", "GGML_MOE_LOSSLESS_MODE", "GGML_MOE_GLM52_LOSSLESS_MODE"};
        // This executable is its own process; do not mutate a parent test environment.
        for (const char * name : names) { ::unsetenv(name); }
        unsigned checks = 0;
        auto check = [&checks](bool ok, const char * text) { ++checks; if (!ok) { throw std::runtime_error(text); } };
        auto config = c0::description_from_env();
        check(config.packs.empty() && config.aliases.empty() && config.page_cache_packs.empty() && config.io_backend == 0,
                "empty configuration must stay disabled/buffered");
        ::setenv("GGML_MOE_EXPERT_PACK_OVERLAY", "orphan", 1);
        check(c0::description_from_env().packs.empty(), "overlay requires a base pack");
        ::setenv("GGML_MOE_EXPERT_PACK", "single:path.pack", 1);
        ::setenv("GGML_MOE_EXPERT_PACK_OVERLAY_EXTRA", "extra", 1);
        ::setenv("GGML_MOE_EXPERT_PACK_LIST", ":list1::list2:", 1);
        ::setenv("GGML_MOE_PAGECACHE_PACK", "page:literal", 1);
        ::setenv("GGML_MOE_PAGECACHE_PACK_LIST", "page2::page3", 1);
        ::setenv("GGML_MOE_EXPERT_GGUF_ALIAS_TSV", ":a.tsv::b.tsv:", 1);
        config = c0::description_from_env();
        check(config.packs == std::vector<std::string>({"single:path.pack", "orphan", "extra", "list1", "list2"}),
                "single path and overlay/list order changed");
        check(config.page_cache_packs == std::vector<std::string>({"page:literal", "page2", "page3"}), "page-cache order changed");
        check(config.aliases == std::vector<std::string>({"a.tsv", "b.tsv"}), "alias list empty-item handling changed");
        for (const auto & pair : std::vector<std::pair<const char *, int>>{{"direct",1},{"iouring",2},{"io_uring",2},{"mmap",0},{"unknown",0},{"",0}}) {
            ::setenv("GGML_MOE_IO_BACKEND", pair.first, 1);
            check(c0::description_from_env().io_backend == pair.second, "legacy IO backend mapping changed");
        }
        for (const auto & pair : std::vector<std::pair<const char *, bool>>{{"",false},{"0",false},{"0tail",false},{"1",true},{"false",true}}) {
            ::setenv("GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES", pair.first, 1);
            check(c0::description_from_env().replace_duplicates == pair.second, "legacy first-character bool changed");
        }
        check(!c0::lossless_requested(), "empty lossless config must be inactive");
        ::setenv("GGML_MOE_GLM52_LOSSLESS_PACK", "sidecar", 1);
        check(c0::lossless_requested(), "sidecar defaults to verify mode");
        for (const char * disabled : {"0", "off", "disabled"}) {
            ::setenv("GGML_MOE_LOSSLESS_MODE", disabled, 1);
            check(!c0::lossless_requested(), "explicit disabled mode must override sidecar");
        }
        ::unsetenv("GGML_MOE_LOSSLESS_MODE");
        ::setenv("GGML_MOE_GLM52_LOSSLESS_MODE", "off", 1);
        check(!c0::lossless_requested(), "fallback disabled mode must be honored");
        ::setenv("GGML_MOE_LOSSLESS_MODE", "cuda", 1);
        check(c0::lossless_requested(), "primary mode must override fallback disabled mode");
        ::unsetenv("GGML_MOE_GLM52_LOSSLESS_PACK");
        for (const char * active : {"verify", "cuda", "invalid"}) {
            ::setenv("GGML_MOE_LOSSLESS_MODE", active, 1);
            check(c0::lossless_requested(), "active or invalid mode must not become plain alias");
        }
        std::printf("%u legacy source configuration checks passed\n", checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
