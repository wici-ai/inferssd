struct expert_pack_entry {
    char tensor[128] = {};
    int32_t expert_idx = -1;
    int32_t source_idx = 0;
    int32_t layer_idx = -1;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    uint64_t object_offset = 0;
    uint64_t object_nbytes = 0;
    uint64_t object_relative_offset = 0;
    bool alias_source = false;
};

struct expert_pack_source {
    FILE *file = nullptr;
#if !defined(_WIN32)
    int fd_direct = -1;
#endif
    void *mmap_base = nullptr;
    size_t mmap_size = 0;
    bool mmap_attempted = false;
    bool mmap_enabled = false;
    bool page_cache = false;
    bool direct_required = false;
    uint64_t file_size = 0;
    char path[512] = {};
};
