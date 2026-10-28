// Historical environment precedence adapter; test reference only.
#include "legacy-config.h"
namespace c0 { namespace {
static void expert_pack_append_env_list(std::vector<std::string> &paths, const char *env_name) {
    const char *env = std::getenv(env_name);
    if (!env || !env[0]) return;

    const std::string value(env);
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find(':', start);
        const size_t stop = end == std::string::npos ? value.size() : end;
        if (stop > start) {
            paths.emplace_back(value.substr(start, stop - start));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}
} // namespace
description source_description_from_env() {
    description desc;
    const char * path = std::getenv("GGML_MOE_EXPERT_PACK");
    if (path && path[0]) {
        desc.packs.emplace_back(path);
        for (const char * name : {"GGML_MOE_EXPERT_PACK_OVERLAY", "GGML_MOE_EXPERT_PACK_OVERLAY_EXTRA"}) {
            const char * overlay = std::getenv(name);
            if (overlay && overlay[0]) { desc.packs.emplace_back(overlay); }
        }
    }
    expert_pack_append_env_list(desc.packs, "GGML_MOE_EXPERT_PACK_LIST");
    const char * page = std::getenv("GGML_MOE_PAGECACHE_PACK");
    if (page && page[0]) { desc.page_cache_packs.emplace_back(page); }
    expert_pack_append_env_list(desc.page_cache_packs, "GGML_MOE_PAGECACHE_PACK_LIST");
    expert_pack_append_env_list(desc.aliases, "GGML_MOE_EXPERT_GGUF_ALIAS_TSV");
    desc.require_direct = [] { const char * value = std::getenv("GGML_MOE_IO_REQUIRE_DIRECT"); return value && value[0] && value[0] != '0'; }();
    desc.replace_duplicates = [] { const char * value = std::getenv("GGML_MOE_EXPERT_PACK_REPLACE_DUPLICATES"); return value && value[0] && value[0] != '0'; }();
    return desc;
}
} // namespace c0
