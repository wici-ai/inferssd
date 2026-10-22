// Included after sink_cache. Private, default-off GPU tier adapter.
static_assert(int(KIMI_VRAM_ACTIVATION_SITU) == int(KIMI_CPU_SINK_ACTIVATION_SITU) &&
    int(KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP) == int(KIMI_CPU_SINK_ACTIVATION_SWIGLU_CLAMP),
    "VRAM tier and CPU sink activation kinds must match");
struct sink_vram {
    const kimi_vram_tier_api * api = nullptr;
    const kimi_vram_residency_api * residency = nullptr;
    const kimi_vram_location_api * location_api = nullptr;
    const kimi_vram_format_api * format = nullptr; // optional explicit-format extension
    const kimi_vram_prefill_api * prefill_api = nullptr;
    const kimi_vram_elastic_api * elastic = nullptr;
    bool format_path = false;
    void * context = nullptr;
    std::vector<kimi_vram_request> requests;
    std::vector<int> slots;
    int hits = 0;
    bool format_f32_input = false;
    bool format_mxfp4_input = false;
    bool mxfp4_batch_enabled = true;
    bool exclusive = false, decode = false, prefill_open = false, async_upload = false, prefill_enabled = false;
    bool prefill_job_active = false;
    int q2_prefill_ram_gpu_min_routes = 0;
    int q2_prefill_ssd_gpu_min_routes = 0;
    int q2_prefill_ssd_gpu_max_experts_per_layer = 0;
    int q2_prefill_cold_admit_per_layer = 0;
    // MXFP4 transient candidates and the per-layer SSD transient cap: the makespan decides by default
    // (1 route, no cap: a streamed transient holds only a ring slot and a device slot, whatever the count);
    // the environment keeps them as A/B switches, and the legacy selector keeps 16/16/64.
    int mxfp4_prefill_ram_gpu_min_routes = 1;
    int mxfp4_prefill_ssd_gpu_min_routes = 1;
    int mxfp4_prefill_ssd_gpu_max_experts_per_layer = -1; // -1: no cap
    bool mxfp4_prefill_legacy_schedule = false;
    // The MXFP4 cost model's terms (kimi-prefill-cost.h; select_prefill_transient). GGML_KIMI_CPU_PREFILL_MXFP4_COST
    // sets any of them by name (cpu_fixed_us=..,link_gbps=..), and each calibrated line prints them by the same names.
    struct mxfp4_prefill_costs {
        // Start values only: 166, 10 CPU threads, a2f1fbb5e, 2026-09-26; see
        // 20260925-deepseek-prefill-sched-assets/{cpu-calibration.txt,gpu-micro.jsonl}.
        double cpu_fixed_us = 204.0, cpu_route_us = 87.0; // the CPU's time per expert, and per route of it
        double gpu_fixed_us = 25.0, gpu_route_us = 8.2;   // the GPU worker's time per expert, and per route of it
        double link_gbps = 6.74;  // a transient's upload from the ring
        double drive_gbps = 0.0;  // the drive while reads are in flight; 0: unknown until measured
        double drive_fixed_us = 0.0; // the drive's time per layer beyond its bytes: the read queue filling and draining
        double gpu_token_us = -1.0; // the GPU work's time per token outside its experts; -1: the output row over the link
    } mxfp4_cost, mxfp4_prior;
    // Calibration: every grouped-prefill layer adds one sample, and the terms are refit over the samples of the last
    // forward pass (one per layer of the sink): the CPU and GPU lines by least squares over each expert's time against
    // its routes, the drive's line over each layer's read time against its bytes, the link and the GPU overhead as
    // ratios of their sums.
    struct prefill_sample {
        kimi_prefill_fit cpu, gpu;
        double link_bytes, link_us, read_bytes, read_us; // link: pageable copies only
        double gpu_overhead_us, gpu_tokens; // gpu_tokens: the layer's tokens when it had GPU work, else 0
    };
    kimi_prefill_fit prefill_cpu_fit; // this layer's CPU experts, from observe_prefill_cpu
    size_t prefill_window = 0;
    std::vector<prefill_sample> prefill_samples;
    size_t prefill_sample_next = 0;
    uint64_t cost_calibrations = 0;
    // This layer's plan, set by select_prefill_transient for the makespan rule.
    struct prefill_plan_state {
        bool valid;
        int ntokens;
        double cpu_us, gpu_us, link_us, drive_us, fill_us, gpu_overhead_us, layer_us;
        bool ram_locked; // a RAM transient of the layer comes from a page-locked slot
    } prefill_plan = {};
    // Whether a RAM cache slot is page-locked (the sink's decode registrar; unset: none is).
    std::function<bool(int)> ram_slot_locked;
    int prefill_latent = 0;
    std::vector<kimi_vram_prefill_transient> prefill_transient;
    kimi_vram_prefill_stats prefill_stats = {};
    uint64_t epoch = 0, ram_drops = 0, ram_drop_blocked = 0, ready_confirmations = 0;
    std::vector<int> ram_leases;
    std::vector<sink_request> location_sources;
    // Opt-in deferred uploads: the RAM source of an upload that outlives its layer stays leased until collected.
    const kimi_vram_deferred_api * deferred_api = nullptr;
    bool deferred_upload = false, have_token = false;
    uint64_t token = 0, deferred_queued = 0, deferred_collected = 0, deferred_stale = 0, deferred_aborted = 0;
    struct deferred_source { kimi_vram_request key; sink_request source; int lease; uint64_t cache_generation; };
    std::vector<deferred_source> deferred;
    std::vector<uint8_t> upload_queued; // per request of the current layer
    // Optional: the GPU computes experts straight from host memory (kimi_vram_host_api).
    const kimi_vram_host_api * host_api = nullptr;
    uint64_t host_experts = 0;
    const kimi_vram_policy_api * policy_api = nullptr; // optional: miss cost, host demand and floor, host reclaim

    // "name=value" items separated by commas or spaces, names as in mxfp4_prefill_costs; the others keep their values.
    static void parse_mxfp4_costs(const char * spec, mxfp4_prefill_costs & cost) {
        static const struct { const char * name; double mxfp4_prefill_costs::* field; } terms[] = {
            { "cpu_fixed_us", &mxfp4_prefill_costs::cpu_fixed_us }, { "cpu_route_us", &mxfp4_prefill_costs::cpu_route_us },
            { "gpu_fixed_us", &mxfp4_prefill_costs::gpu_fixed_us }, { "gpu_route_us", &mxfp4_prefill_costs::gpu_route_us },
            { "link_gbps", &mxfp4_prefill_costs::link_gbps }, { "drive_gbps", &mxfp4_prefill_costs::drive_gbps },
            { "drive_fixed_us", &mxfp4_prefill_costs::drive_fixed_us }, { "gpu_token_us", &mxfp4_prefill_costs::gpu_token_us },
        };
        auto separator = [](char c) { return c == ',' || c == ' ' || c == '\t' || c == '\n'; };
        for (const char * cursor = spec; *cursor;) {
            if (separator(*cursor)) { ++cursor; continue; }
            const char * equals = cursor;
            while (*equals && *equals != '=' && !separator(*equals)) { ++equals; }
            const std::string name(cursor, equals);
            auto term = std::find_if(std::begin(terms), std::end(terms),
                [&](const auto & t) { return name == t.name; });
            if (*equals != '=' || term == std::end(terms)) {
                throw std::runtime_error("GGML_KIMI_CPU_PREFILL_MXFP4_COST: unknown item " + name);
            }
            char * end = nullptr;
            errno = 0;
            const double value = std::strtod(equals + 1, &end);
            if (errno || end == equals + 1 || (*end && !separator(*end)) || !std::isfinite(value) || value < 0.0 ||
                    value > 1e6 || (term->field == &mxfp4_prefill_costs::link_gbps && value == 0.0)) {
                throw std::runtime_error("GGML_KIMI_CPU_PREFILL_MXFP4_COST: invalid value of " + name);
            }
            cost.*(term->field) = value;
            cursor = end;
        }
    }

    void initialize(bool ready_first) {
        bool use_locations = false;
        if (const char * flag = std::getenv("GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS")) {
            if (std::strcmp(flag,"0") && std::strcmp(flag,"1")) { throw std::runtime_error("VRAM_CACHE_LOCATIONS must be 0 or 1"); }
            use_locations = !std::strcmp(flag,"1");
        }
        if (const char * flag = std::getenv("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD")) {
            if (std::strcmp(flag, "0") && std::strcmp(flag, "1")) { throw std::runtime_error("VRAM_ASYNC_UPLOAD must be 0 or 1"); }
            async_upload = !std::strcmp(flag, "1");
        }
        if (const char * flag = std::getenv("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE")) {
            if (std::strcmp(flag, "0") && std::strcmp(flag, "1")) { throw std::runtime_error("VRAM_EXCLUSIVE must be 0 or 1"); }
            exclusive = !std::strcmp(flag, "1");
        }
        if (const char * flag = std::getenv("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD")) {
            if (std::strcmp(flag, "0") && std::strcmp(flag, "1")) { throw std::runtime_error("VRAM_DEFERRED_UPLOAD must be 0 or 1"); }
            deferred_upload = !std::strcmp(flag, "1");
        }
        if (const char * flag = std::getenv("GGML_KIMI_CPU_SINK_VRAM_PREFILL")) {
            if (std::strcmp(flag, "0") && std::strcmp(flag, "1")) { throw std::runtime_error("VRAM_PREFILL must be 0 or 1"); }
            prefill_enabled = !std::strcmp(flag, "1");
        }
        if (const char * flag = std::getenv("GGML_KIMI_MXFP4_PREFILL_BATCH")) {
            if (std::strcmp(flag, "0") && std::strcmp(flag, "1")) {
                throw std::runtime_error("GGML_KIMI_MXFP4_PREFILL_BATCH must be 0 or 1");
            }
            mxfp4_batch_enabled = !std::strcmp(flag, "1");
        }
        auto mxfp4_setting = [](const char * name, int fallback, long minimum, long maximum) {
            const char * value = std::getenv(name);
            if (!value) { return fallback; }
            char * end = nullptr;
            errno = 0;
            const long parsed = std::strtol(value, &end, 10);
            if (errno || end == value || *end || parsed < minimum || parsed > maximum) {
                throw std::runtime_error(std::string(name) + " is out of range");
            }
            return int(parsed);
        };
        mxfp4_prefill_legacy_schedule = mxfp4_setting(
            "GGML_KIMI_CPU_PREFILL_MXFP4_SCHED_LEGACY", 0, 0, 1) != 0;
        mxfp4_prefill_ram_gpu_min_routes = mxfp4_setting(
            "GGML_KIMI_CPU_PREFILL_MXFP4_RAM_GPU_MIN_ROUTES", mxfp4_prefill_legacy_schedule ? 16 : 1, 0, 2048);
        mxfp4_prefill_ssd_gpu_min_routes = mxfp4_setting(
            "GGML_KIMI_CPU_PREFILL_MXFP4_SSD_GPU_MIN_ROUTES", mxfp4_prefill_legacy_schedule ? 16 : 1, 0, 2048);
        mxfp4_prefill_ssd_gpu_max_experts_per_layer = mxfp4_setting(
            "GGML_KIMI_CPU_PREFILL_MXFP4_SSD_GPU_MAX_EXPERTS_PER_LAYER",
            mxfp4_prefill_legacy_schedule ? 64 : -1, -1, 65536);
        if (const char * spec = std::getenv("GGML_KIMI_CPU_PREFILL_MXFP4_COST")) {
            parse_mxfp4_costs(spec, mxfp4_cost);
        }
        mxfp4_prior = mxfp4_cost;
        std::fprintf(stderr,
            "[kimi_cpu_prefill_mxfp4_cost] rule=%s cpu_fixed_us=%.3f cpu_route_us=%.3f gpu_fixed_us=%.3f "
            "gpu_route_us=%.3f link_gbps=%.3f drive_gbps=%.3f drive_fixed_us=%.3f gpu_token_us=%.3f ram_min_routes=%d "
            "ssd_min_routes=%d ssd_limit=%d\n",
            mxfp4_prefill_legacy_schedule ? "legacy" : "makespan",
            mxfp4_cost.cpu_fixed_us, mxfp4_cost.cpu_route_us, mxfp4_cost.gpu_fixed_us, mxfp4_cost.gpu_route_us,
            mxfp4_cost.link_gbps, mxfp4_cost.drive_gbps, mxfp4_cost.drive_fixed_us, mxfp4_cost.gpu_token_us,
            mxfp4_prefill_ram_gpu_min_routes,
            mxfp4_prefill_ssd_gpu_min_routes, mxfp4_prefill_ssd_gpu_max_experts_per_layer);
        if (const char * threshold = std::getenv("GGML_KIMI_CPU_PREFILL_Q2_RAM_GPU_MIN_ROUTES")) {
            char * threshold_end = nullptr;
            errno = 0;
            const long parsed = std::strtol(threshold, &threshold_end, 10);
            if (errno || threshold_end == threshold || *threshold_end || parsed < 0 || parsed > 2048) {
                throw std::runtime_error("GGML_KIMI_CPU_PREFILL_Q2_RAM_GPU_MIN_ROUTES must be in 0..2048");
            }
            q2_prefill_ram_gpu_min_routes = int(parsed);
        }
        if (const char * threshold = std::getenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES")) {
            char * threshold_end = nullptr;
            errno = 0;
            const long parsed = std::strtol(threshold, &threshold_end, 10);
            if (errno || threshold_end == threshold || *threshold_end || parsed < 0 || parsed > 2048) {
                throw std::runtime_error("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES must be in 0..2048");
            }
            q2_prefill_ssd_gpu_min_routes = int(parsed);
        }
        if (const char * count = std::getenv("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MAX_EXPERTS_PER_LAYER")) {
            char * count_end = nullptr;
            errno = 0;
            const long parsed = std::strtol(count, &count_end, 10);
            if (errno || count_end == count || *count_end || parsed < 0 || parsed > 64) {
                throw std::runtime_error("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MAX_EXPERTS_PER_LAYER must be in 0..64");
            }
            q2_prefill_ssd_gpu_max_experts_per_layer = int(parsed);
        }
        if (!!q2_prefill_ssd_gpu_min_routes != !!q2_prefill_ssd_gpu_max_experts_per_layer) {
            throw std::runtime_error("Q2 SSD GPU scheduling requires both MIN_ROUTES and MAX_EXPERTS_PER_LAYER");
        }
        if (const char * count = std::getenv("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_PER_LAYER")) {
            char * count_end = nullptr;
            errno = 0;
            const long parsed = std::strtol(count, &count_end, 10);
            if (errno || count_end == count || *count_end || parsed < 0 || parsed > 384) {
                throw std::runtime_error("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_PER_LAYER must be in 0..384");
            }
            q2_prefill_cold_admit_per_layer = int(parsed);
        }
        if ((q2_prefill_ram_gpu_min_routes || q2_prefill_ssd_gpu_min_routes ||
                q2_prefill_cold_admit_per_layer) && !prefill_enabled) {
            throw std::runtime_error("Q2 prefill GPU scheduling/admission requires VRAM_PREFILL=1");
        }
        const char * value = std::getenv("GGML_KIMI_CPU_SINK_VRAM_BYTES");
        if (!value || !value[0] || !std::strcmp(value, "0")) {
            if (exclusive || use_locations) { throw std::runtime_error("VRAM_EXCLUSIVE/CACHE_LOCATIONS require a nonzero VRAM budget"); }
            if (deferred_upload) { throw std::runtime_error("VRAM_DEFERRED_UPLOAD requires a nonzero VRAM budget"); }
            if (prefill_enabled) { throw std::runtime_error("VRAM_PREFILL requires a nonzero VRAM budget"); }
            return;
        }
        // auto: no byte budget. An elastic tier starts from one block and grows into free memory; a number keeps
        // its legacy meaning, the first allocation.
        uint64_t bytes = UINT64_MAX;
        if (std::strcmp(value, "auto")) {
            char * end = nullptr;
            errno = 0;
            bytes = std::strtoull(value, &end, 10);
            // The tier clamps the arena to free VRAM minus its safety margin; this is only a sanity bound.
            if (errno || end == value || *end || value[0] == '-' || !bytes || bytes > (64ull << 30)) {
                throw std::runtime_error("invalid GGML_KIMI_CPU_SINK_VRAM_BYTES (auto or 1..64 GiB)");
            }
        }
        if (!ready_first) { throw std::runtime_error("GPU tier requires READY_FIRST=1 for ordered reduction"); }
        auto get_api = reinterpret_cast<kimi_vram_tier_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_tier_get_api"));
        if (!get_api) { throw std::runtime_error("GPU tier DSO is missing; no fallback"); }
        api = get_api();
        if (!api || api->version != 2 || api->size != sizeof(*api) || !api->create || !api->destroy ||
                !api->begin || !api->prepare || !api->launch || !api->admit || !api->finish || !api->cancel) {
            throw std::runtime_error("GPU tier ABI mismatch");
        }
        if (auto get_format = reinterpret_cast<kimi_vram_format_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_format_get_api"))) {
            format = get_format();
            if (!format || format->version != 1 || format->size != sizeof(*format) || !format->prepare || !format->launch) {
                throw std::runtime_error("GPU tier explicit-format ABI mismatch");
            }
        }
        if (prefill_enabled) {
            auto get_prefill = reinterpret_cast<kimi_vram_prefill_get_api_fn>(
                dlsym(RTLD_DEFAULT, "kimi_vram_prefill_get_api"));
            prefill_api = get_prefill ? get_prefill() : nullptr;
            if (!prefill_api || prefill_api->version != 5 || prefill_api->size != sizeof(*prefill_api) ||
                    !prefill_api->prepare || !prefill_api->transient_available || !prefill_api->launch ||
                    !prefill_api->finish || !prefill_api->cancel || !prefill_api->cold_admission_active ||
                    !prefill_api->admit_cold) {
                throw std::runtime_error("GPU tier grouped-prefill ABI missing or incompatible; no fallback");
            }
            if (!format) { throw std::runtime_error("GPU tier grouped prefill requires the explicit-format ABI"); }
        }
        if (auto get_elastic = reinterpret_cast<kimi_vram_elastic_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_elastic_get_api"))) {
            elastic = get_elastic();
            if (!elastic || elastic->version != 3 || elastic->size != sizeof(*elastic) ||
                    !elastic->phase_complete || !elastic->reserve_accumulate || !elastic->reserve) {
                throw std::runtime_error("GPU tier elastic lifecycle ABI mismatch");
            }
        }
        {
            // what the tier holds (the next-layer prefetch asks), and with VRAM_EXCLUSIVE its ready leases
            auto get = reinterpret_cast<kimi_vram_residency_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_residency_get_api"));
            residency = get ? get() : nullptr;
            if (!residency || residency->version != 2 || residency->size != sizeof(*residency) || !residency->held ||
                    (exclusive && (!residency->acquire || !residency->validate || !residency->release))) {
                throw std::runtime_error("GPU residency ABI missing or incompatible; no fallback");
            }
        }
        if (deferred_upload) {
            if (!async_upload || use_locations) { throw std::runtime_error("VRAM_DEFERRED_UPLOAD requires ASYNC_UPLOAD=1 and no CACHE_LOCATIONS"); }
            auto get = reinterpret_cast<kimi_vram_deferred_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_deferred_get_api"));
            deferred_api = get ? get() : nullptr;
            if (!deferred_api || deferred_api->version != 2 || deferred_api->size != sizeof(*deferred_api) ||
                    !deferred_api->admit || !deferred_api->collect || !deferred_api->wait) {
                throw std::runtime_error("GPU tier deferred-upload ABI missing or incompatible; no fallback");
            }
            deferred.reserve(64);
        }
        if (exclusive || async_upload) { ram_leases.reserve(16); }
        char error[512] = {};
        context = api->create(bytes, error, sizeof(error));
        if (!context) { throw std::runtime_error(std::string("GPU tier create: ") + error); }
        if (auto get_policy = reinterpret_cast<kimi_vram_policy_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_policy_get_api"))) {
            policy_api = get_policy();
            if (!policy_api || policy_api->version != 3 || policy_api->size != sizeof(*policy_api) ||
                    !policy_api->set_miss_cost || !policy_api->set_host_demand || !policy_api->set_host_floor ||
                    !policy_api->reclaim_host || !policy_api->host_bytes) {
                throw std::runtime_error("GPU tier admission-policy ABI mismatch");
            }
        }
        if (auto get_host = reinterpret_cast<kimi_vram_host_get_api_fn>(dlsym(RTLD_DEFAULT, "kimi_vram_host_get_api"))) {
            host_api = get_host();
            if (!host_api || host_api->version != 2 || host_api->size != sizeof(*host_api) ||
                    !host_api->available || !host_api->launch || !host_api->costs || !host_api->register_host ||
                    !host_api->unregister_host || !host_api->costs2) {
                throw std::runtime_error("GPU tier host-memory ABI mismatch");
            }
        }
        if (use_locations) {
            auto get=reinterpret_cast<kimi_vram_location_get_api_fn>(dlsym(RTLD_DEFAULT,"kimi_vram_location_get_api"));
            location_api=get?get():nullptr;
            if(!location_api || location_api->version!=1 || location_api->size!=sizeof(*location_api) ||
                    !location_api->configure || !location_api->view || location_api->view->version!=1 ||
                    location_api->view->size!=sizeof(kimi_cache_location_api)) {
                throw std::runtime_error("VRAM location provider ABI missing or incompatible");
            }
        }
    }
    ~sink_vram() {
        if (host_experts) { std::fprintf(stderr, "[kimi_vram_host_sink] gpu_host_experts=%llu\n", (unsigned long long) host_experts); }
        if (context) { api->destroy(context); }
        if (exclusive) { std::fprintf(stderr, "[kimi_vram_exclusive] ram_drops=%llu ram_drop_blocked=%llu ready_confirmations=%llu\n",
            (unsigned long long) ram_drops, (unsigned long long) ram_drop_blocked, (unsigned long long) ready_confirmations); }
        if (deferred_upload) { std::fprintf(stderr, "[kimi_vram_deferred_sink] queued=%llu collected=%llu aborted=%llu stale_leases=%llu\n",
            (unsigned long long) deferred_queued, (unsigned long long) deferred_collected, (unsigned long long) deferred_aborted,
            (unsigned long long) deferred_stale); }
    }
    void check(int result, const char * error) {
        if (result) { throw std::runtime_error(std::string("GPU tier: ") + error); }
    }
    void begin(uint64_t epoch, uint64_t token, bool decode, sink_cache & cache) {
        if (!context) { return; }
        // Graph-layer prefill calls repeat the context boundary transition.
        // The first transition owns shrink/reclaimer arming; repeating it after
        // scheduler reserve would incorrectly re-arm OOM reclamation mid-graph.
        // A prefill phase stays open until phase_complete(true) or a decode
        // begin, not until the next decode graph: a request that samples no
        // decode step must still let the next request reopen prefill.
        if (prefill_open && !decode) { return; }
        if (deferred_upload) {
            // Anything but the next decode token of this epoch may release the arena or the RAM cache.
            const bool next = decode && this->decode && have_token && epoch == this->epoch && token == this->token + 1;
            collect_deferred(cache, next ? -2 : -1);
        }
        char error[512] = {};
        const int result = api->begin(context, epoch, token, decode, error, sizeof(error));
        check(result, error);
        this->epoch = epoch;
        this->decode = decode;
        prefill_open = !decode;
        this->token = token;
        have_token = decode;
    }
    void reserve_accumulate(int tokens) {
        if (!context || !elastic) { return; }
        char error[512] = {};
        check(elastic->reserve_accumulate(context, tokens, error, sizeof(error)), error);
    }
    // At load (unified memory): the tier's storage before any request, for experts of this geometry. The role
    // bytes are the tensors' per-expert strides, as the decode requests carry them; the key is not needed.
    void reserve(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down,
                 const std::vector<int32_t> & layer_ids) {
        if (!context || !elastic || !format || layer_ids.empty()) { return; }
        kimi_vram_request r = {};
        r.layer = layer_ids.front(); r.weight = 1.0f;
        const ggml_tensor * tensors[] = { up, gate, down };
        for (int role = 0; role < 3; ++role) { r.role[role].bytes = tensors[role]->nb[2]; }
        const kimi_vram_expert_format f = { int32_t(up->ne[0]), int32_t(up->ne[1]), int32_t(up->type),
            int32_t(down->type), int32_t(up->ne[2]), int32_t(layer_ids.size()), layer_ids.data() };
        char error[512] = {};
        check(elastic->reserve(context, &r, &f, error, sizeof(error)), error);
    }
    void phase_complete(bool decode) {
        if (decode) { prefill_open = false; } // the tier leaves prefill: scratch released, arena grown
        if (!context || !elastic) { return; }
        char error[512] = {};
        check(elastic->phase_complete(context, decode ? 1 : 0, error, sizeof(error)), error);
    }
    // Q3_K SiTU experts (legacy) keep transport ABI 2 and the K3 layer range;
    // every other format uses the tier's explicit-format extension.
    void prepare(std::vector<sink_request> & source, const ggml_tensor * up, const ggml_tensor * down,
                 int latent, int hidden, bool legacy, const std::vector<int32_t> & layer_ids, sink_cache & cache, bool unweighted = false) {
        hits = 0;
        if (!context) { return; }
        int layer = -1, consumed = 0;
        const char * tensor_name = up->name;
        if (std::sscanf(tensor_name, "blk.%d.ffn_up_exps.weight%n", &layer, &consumed) != 1 ||
                !consumed || tensor_name[consumed] || source.size() > 16 ||
                (legacy ? (layer < 1 || layer > 92) : (layer < 0 || layer > 1023))) {
            throw std::runtime_error("GPU tier requires a canonical routed layer and <=16 slots");
        }
        if (!legacy && !format) {
            throw std::runtime_error("GPU tier lacks the explicit expert format ABI for these experts; no fallback");
        }
        if (deferred_upload && decode) {
            // The RAM cache refuses a layer's prepare while that layer's entries are leased.
            collect_deferred(cache, cache.allocation == sink_cache::global ? -1 : layer);
        }
        upload_queued.assign(source.size(), 0);
        requests.resize(source.size()); slots.assign(source.size(), -1);
        for (size_t i = 0; i < source.size(); ++i) {
            auto & r = requests[i]; r.layer = layer; r.expert = source[i].id; r.weight = unweighted ? 1.0f : source[i].weight;
            for (int role = 0; role < 3; ++role) {
                const auto k = sink_cache::key(source[i].role[role]);
                r.role[role] = { k.device, k.inode, k.size, k.offset, k.bytes,
                                k.mtime_s, k.mtime_ns, k.ctime_s, k.ctime_ns };
            }
        }
        char error[512] = {};
        int result;
        if (legacy) {
            result = api->prepare(context, requests.data(), int(requests.size()), latent, hidden,
                                  slots.data(), error, sizeof(error));
        } else {
            // MXFP4's engine takes the input quantized, the generic one (every other format the CUDA
            // backend has a matrix-vector kernel for; it refuses the rest) takes it as f32
            const bool mxfp4 = up->type == GGML_TYPE_MXFP4 && down->type == GGML_TYPE_MXFP4;
            const kimi_vram_expert_format f = { latent, hidden, int32_t(up->type), int32_t(down->type),
                int32_t(up->ne[2]), int32_t(layer_ids.size()), layer_ids.data() };
            result = format->prepare(context, requests.data(), int(requests.size()), &f, slots.data(), error, sizeof(error));
            format_f32_input = !mxfp4;
        }
        format_path = !legacy;
        check(result, error);
        for (size_t i = 0; i < source.size(); ++i) { source[i].vram_slot = slots[i]; hits += slots[i] >= 0; }
    }
    void launch(const void * quantized, size_t quantized_bytes, const float * f32, size_t f32_bytes,
                const kimi_cpu_sink_activation & activation) {
        if (!context) { return; }
        char error[512] = {};
        int result;
        if (format_path) {
            const kimi_vram_activation a = { activation.kind, activation.situ_beta, activation.situ_linear_beta, activation.swiglu_limit };
            const void * input = format_f32_input ? static_cast<const void *>(f32) : quantized;
            const size_t bytes = format_f32_input ? f32_bytes : quantized_bytes;
            result = format->launch(context, input, bytes, &a, error, sizeof(error));
        } else {
            result = api->launch(context, quantized, quantized_bytes,
                                 activation.situ_beta, activation.situ_linear_beta, error, sizeof(error));
        }
        check(result, error);
    }
    void pin_ram_batch(sink_cache & cache, const std::vector<sink_request> & source) {
        if ((!exclusive && !async_upload && !location_api) || !decode) { return; }
        if (!ram_leases.empty()) { throw std::runtime_error("GPU tier still holds a previous RAM batch"); }
        // One lease per physical slot covers CPU, its read CQEs and a possible
        // borrowed upload. Repeated router IDs cannot release it twice.
        for (const auto & r : source) {
            const int index = r.cache_hit >= 0 ? r.cache_hit : r.cache_admit;
            if (index < 0 || std::find(ram_leases.begin(), ram_leases.end(), index) != ram_leases.end()) { continue; }
            cache.acquire_lease(index, r); ram_leases.push_back(index);
        }
    }
    void admit(size_t index, const void * const * data, bool borrow_until_finish) {
        if (!context) { return; }
        char error[512] = {};
        if (deferred_upload && borrow_until_finish) {
            int queued = 0;
            check(deferred_api->admit(context, &requests.at(index), data[0], data[1], data[2], &queued, error, sizeof(error)), error);
            if (queued) { upload_queued.at(index) = 1; ++deferred_queued; }
            return;
        }
        const int result = api->admit(context, &requests.at(index), data[0], data[1], data[2],
                                      borrow_until_finish, error, sizeof(error));
        check(result, error);
    }
    // Which of `count` experts of `layer` the tier holds now (out: 1 or 0); none without a tier.
    void held(int32_t layer, const int32_t * experts, int count, uint8_t * out) const {
        std::fill(out, out + count, uint8_t(0));
        if (context && residency) { residency->held(context, layer, experts, count, out); }
    }
    void set_host_demand(uint64_t bytes) {
        if (context && policy_api) { policy_api->set_host_demand(context, bytes); }
    }
    // The host-memory ledger's floor, which the tier keeps free on unified memory.
    void set_host_floor(uint64_t bytes) {
        if (context && policy_api) { policy_api->set_host_floor(context, bytes); }
    }
    // Unified memory: tier blocks go so that `bytes` of host memory can be allocated (false: a layer is live, or not
    // enough could go). Deferred uploads are collected first by the caller.
    bool reclaim_host(uint64_t bytes) {
        return context && policy_api && !policy_api->reclaim_host(context, bytes);
    }
    // DRAM the tier holds on unified memory (0 on a discrete GPU).
    uint64_t host_bytes() {
        return context && policy_api ? policy_api->host_bytes(context) : 0;
    }
    void set_miss_cost(double us) {
        if (context && policy_api && decode && us > 0) { policy_api->set_miss_cost(context, us); }
    }
    // The GPU can compute a decode expert from host memory in this layer (in place, or after a copy: costs2 says).
    bool host_ready() {
        return context && host_api && decode && format_path && host_api->available(context);
    }
    void host_costs(double & resident_us, double & host_us) {
        resident_us = host_us = 0;
        if (context && host_api) { host_api->costs(context, &resident_us, &host_us); }
    }
    // Page-locks host memory for the device's copies (no-op where it reads host memory in place). false: refused.
    bool register_host(const void * pointer, size_t bytes, std::string * why = nullptr) {
        if (!context || !host_api) { return false; }
        char error[256] = {};
        const bool ok = host_api->register_host(context, pointer, bytes, error, sizeof(error)) == 0;
        if (!ok && why) { *why = error; }
        return ok;
    }
    bool unregister_host(const void * pointer) {
        if (!context || !host_api) { return false; }
        char error[256] = {};
        return host_api->unregister_host(context, pointer, error, sizeof(error)) == 0;
    }
    // Queues experts behind the layer's resident batch, as one engine batch where possible; false if the tier
    // has no launched layer to follow.
    bool launch_host(const std::vector<kimi_vram_host_job> & jobs) {
        char error[512] = {};
        int accepted = 0;
        check(host_api->launch(context, jobs.data(), int(jobs.size()), &accepted, error, sizeof(error)), error);
        if (accepted) { host_experts += jobs.size(); }
        return accepted != 0;
    }
    void finish(float * contributions, size_t stride, sink_cache & cache, const std::vector<sink_request> & source) {
        if (!context) { return; }
        char error[512] = {};
        const int result = api->finish(context, contributions, stride, error, sizeof(error));
        check(result, error);
        if (!decode) { return; }
        if (deferred_upload) { retain_deferred(cache, source); }
        release_ram_batch(cache);
        drop_ready(cache,source);
        if (deferred_upload) { collect_deferred(cache, -2); }
    }
    int prepare_prefill(std::vector<sink_request> & source, const ggml_tensor * up, const ggml_tensor * down,
            int latent, int hidden, const std::vector<int32_t> & layer_ids) {
        hits = 0;
        prefill_plan.valid = false;
        prefill_cpu_fit = {};
        prefill_stats = {};
        if (!context || !prefill_enabled || source.empty()) { return 0; }
        int layer = -1, consumed = 0;
        const char * tensor_name = up->name;
        if (std::sscanf(tensor_name, "blk.%d.ffn_up_exps.weight%n", &layer, &consumed) != 1 ||
                !consumed || tensor_name[consumed] || layer < 0 || layer > 1023 ||
                source.size() > static_cast<size_t>(up->ne[2])) {
            throw std::runtime_error("GPU grouped prefill requires a canonical routed layer and a distinct expert union");
        }
        const bool q2 = up->type == GGML_TYPE_IQ2_XXS && down->type == GGML_TYPE_Q2_K;
        const bool mxfp4 = up->type == GGML_TYPE_MXFP4 && down->type == GGML_TYPE_MXFP4;
        if (!q2 && !mxfp4) { return 0; }
        prefill_latent = latent;
        if (!prefill_window) { prefill_window = std::max<size_t>(1, layer_ids.size()); }
        requests.resize(source.size()); slots.assign(source.size(), -1);
        for (size_t i = 0; i < source.size(); ++i) {
            auto & r = requests[i]; r.layer = layer; r.expert = source[i].id; r.weight = 1.0f;
            for (int role = 0; role < 3; ++role) {
                const auto k = sink_cache::key(source[i].role[role]);
                r.role[role] = { k.device, k.inode, k.size, k.offset, k.bytes,
                                k.mtime_s, k.mtime_ns, k.ctime_s, k.ctime_ns };
            }
        }
        const kimi_vram_expert_format f = { latent, hidden, int32_t(up->type), int32_t(down->type),
            int32_t(up->ne[2]), int32_t(layer_ids.size()), layer_ids.data() };
        char error[512] = {};
        check(prefill_api->prepare(context, requests.data(), int(requests.size()), &f,
                slots.data(), error, sizeof(error)), error);
        format_f32_input = q2;
        format_mxfp4_input = mxfp4;
        for (size_t i = 0; i < source.size(); ++i) {
            source[i].vram_slot = slots[i];
            hits += slots[i] >= 0;
        }
        return hits;
    }
    void select_prefill_transient(std::vector<sink_request> & source,
            kimi_cpu_sink_prefill_batch & batch, sink_cache & cache, size_t ssd_staging_limit, size_t ring_slots) {
        prefill_transient.clear();
        for (auto & request : source) {
            request.vram_transient = false;
            request.vram_ssd_transient = false;
            request.vram_ssd_streaming = false;
            request.vram_ssd_staging = -1;
            request.vram_cold_admit = false;
            request.vram_cold_admit_routes = 0;
            request.lane_start_us = 0.0;
            request.prefill_routes = 0;
        }
        const bool mxfp4 = format_mxfp4_input && mxfp4_batch_enabled;
        if (!context || !prefill_enabled || (!format_f32_input && !mxfp4) || source.empty()) { return; }
        const int ram_threshold = mxfp4 ? mxfp4_prefill_ram_gpu_min_routes : q2_prefill_ram_gpu_min_routes;
        const int ssd_threshold = mxfp4 ? mxfp4_prefill_ssd_gpu_min_routes : q2_prefill_ssd_gpu_min_routes;
        const int ssd_cap = mxfp4 ? mxfp4_prefill_ssd_gpu_max_experts_per_layer : q2_prefill_ssd_gpu_max_experts_per_layer;
        const bool transient_ready = (ram_threshold || ssd_threshold) &&
            prefill_api->transient_available(context);
        const bool cold_admission_ready = format_f32_input && q2_prefill_cold_admit_per_layer &&
            prefill_api->cold_admission_active(context);
        if (!transient_ready && !cold_admission_ready) { return; }

        std::vector<int> token_routes(source.size(), 0);
        std::vector<int> last_token(source.size(), -1);
        for (int token_index = 0; token_index < batch.ntokens; ++token_index) {
            for (int slot = 0; slot < batch.nslots; ++slot) {
                const size_t route = size_t(token_index) * size_t(batch.nslots) + size_t(slot);
                const int32_t request = batch.route_request_index[route];
                if (request >= 0 && size_t(request) < source.size() &&
                        last_token[size_t(request)] != token_index) {
                    ++token_routes[size_t(request)];
                    last_token[size_t(request)] = token_index;
                }
            }
        }
        for (size_t i = 0; i < source.size(); ++i) { source[i].prefill_routes = token_routes[i]; }

        if (transient_ready) {
            // Q2 and the legacy MXFP4 selector keep their fixed 2026-09 fits and serial greedy rule. The makespan rule
            // uses the calibrated MXFP4 cost model (calibrate_prefill, kimi-prefill-cost.h), per layer:
            //   CPU lane  C = sum over CPU experts of cpu_fixed + cpu_route * routes
            //   GPU lane  G = sum over GPU experts of gpu_fixed + gpu_route * routes (the worker's time per expert),
            //                 plus each transient's upload, bytes / link: the worker copies it from the pageable ring
            //   link      L = (transient bytes + routed input rows) / link
            //   drive     D = the layer's SSD reads, for both lanes, / drive rate
            //   layer     T = fill + coupled(D, max(C, G, L), ring) + the GPU overhead once the layer has GPU work.
            //             The drive and the lanes hand the reads over through the ring, and the closer their times
            //             the more each waits for the other (kimi_prefill_coupled), so balanced lanes below D still
            //             gain, and lanes far below D (a short batch) gain nothing worth the GPU overhead.
            // The candidates move to the GPU in order of routes, and the selector takes the number of them that gives
            // the lowest T. The reads are then issued in the order the lanes need them (lane_start_us), so that the
            // lanes overlap.
            const bool makespan = mxfp4 && !mxfp4_prefill_legacy_schedule;
            auto cpu_cost = [this, mxfp4, makespan](int routes) {
                return routes ? (makespan ? mxfp4_cost.cpu_fixed_us + mxfp4_cost.cpu_route_us * routes :
                    mxfp4 ? 70.0 + 95.0 * routes : 70.0 + 184.0 * routes) : 0.0;
            };
            auto gpu_cost = [this, mxfp4, makespan](int routes) {
                return routes ? (makespan ? mxfp4_cost.gpu_fixed_us + mxfp4_cost.gpu_route_us * routes :
                    mxfp4 ? 55.0 + 7.0 * routes : 15.0 + 39.0 * routes) : 0.0;
            };
            const double link_bytes_per_us = mxfp4_cost.link_gbps * 1000.0;
            const double drive_bytes_per_us = mxfp4_cost.drive_gbps * 1000.0;
            // One routed input row as the tier copies it: the row, its token index and its weight.
            const double activation_bytes_per_route = double(format_f32_input ? batch.f32_stride :
                batch.quantized_stride) + sizeof(int32_t) + sizeof(float);
            auto expert_bytes = [&source](size_t i) {
                return source[i].role[0].nbytes + source[i].role[1].nbytes + source[i].role[2].nbytes;
            };
            auto upload_cost = [&](uint64_t bytes) {
                return makespan ? double(bytes) / link_bytes_per_us : double(bytes) / (mxfp4 ? 6740.0 : 7000.0) + 10.0;
            };
            auto link_cost = [&](uint64_t bytes, int routes) {
                return makespan ? (double(bytes) + activation_bytes_per_route * routes) / link_bytes_per_us : 0.0;
            };
            kimi_prefill_layer layer = {};
            double drive_bytes = 0.0, first_read_bytes = 0.0;
            for (size_t i = 0; i < source.size(); ++i) {
                if (source[i].vram_slot >= 0) {
                    layer.gpu += gpu_cost(token_routes[i]);
                    layer.link += link_cost(0, token_routes[i]);
                    layer.gpu_work = true;
                    continue;
                }
                layer.cpu += cpu_cost(token_routes[i]);
                if (source[i].cache_hit < 0) {
                    drive_bytes += double(expert_bytes(i));
                    if (first_read_bytes == 0.0) { first_read_bytes = double(expert_bytes(i)); }
                }
            }
            if (makespan && drive_bytes_per_us > 0.0 && drive_bytes > 0.0) {
                layer.drive = mxfp4_cost.drive_fixed_us + drive_bytes / drive_bytes_per_us;
                layer.fill = first_read_bytes / drive_bytes_per_us;
            }
            layer.ring = double(ring_slots);
            layer.gpu_overhead = !makespan ? 0.0 : double(batch.ntokens) * (mxfp4_cost.gpu_token_us >= 0.0 ?
                mxfp4_cost.gpu_token_us : double(prefill_latent) * sizeof(float) / link_bytes_per_us);
            struct candidate { size_t index; int source; };
            std::vector<candidate> candidates;
            size_t ram_candidates = 0, ssd_candidates = 0;
            for (size_t i = 0; i < source.size(); ++i) {
                if (source[i].vram_slot >= 0) { continue; }
                if (ram_threshold && source[i].cache_hit >= 0 && source[i].cache_lease &&
                        token_routes[i] >= ram_threshold) {
                    candidates.push_back({i, KIMI_VRAM_PREFILL_TRANSIENT_RAM});
                    ++ram_candidates;
                } else if (ssd_threshold && source[i].cache_hit < 0 &&
                        token_routes[i] >= ssd_threshold) {
                    candidates.push_back({i, KIMI_VRAM_PREFILL_TRANSIENT_SSD});
                    ++ssd_candidates;
                }
            }
            batch.gpu_ssd_transient_candidates = ssd_candidates;
            std::stable_sort(candidates.begin(), candidates.end(), [&](const candidate & a, const candidate & b) {
                if (token_routes[a.index] != token_routes[b.index]) {
                    return token_routes[a.index] > token_routes[b.index];
                }
                if (a.source != b.source) { return a.source < b.source; }
                return source[a.index].id < source[b.index].id;
            });
            const size_t ssd_limit = ssd_cap < 0 ? ssd_staging_limit :
                std::min(ssd_staging_limit, size_t(ssd_cap));
            std::vector<candidate> chosen;
            double time_before = 0.0, time_after = 0.0;
            // The layer's reads: every expert neither resident nor in the RAM cache, on either lane.
            size_t reads = 0;
            for (size_t i = 0; i < source.size(); ++i) { reads += source[i].vram_slot < 0 && source[i].cache_hit < 0; }
            const bool in_ring = reads <= ring_slots;
            if (makespan) {
                std::vector<kimi_prefill_move> moves;
                std::vector<candidate> movable;
                size_t ssd_movable = 0;
                for (const candidate & item : candidates) {
                    if (item.source == KIMI_VRAM_PREFILL_TRANSIENT_SSD && ssd_movable++ >= ssd_limit) { continue; }
                    const int routes = token_routes[item.index];
                    const uint64_t bytes = expert_bytes(item.index);
                    moves.push_back({ cpu_cost(routes), gpu_cost(routes) + upload_cost(bytes), link_cost(bytes, routes) });
                    movable.push_back(item);
                }
                size_t best = 0;
                if (in_ring) {
                    // Every read is in flight at once: the list schedule over the reads' landing times (in request
                    // order), the same evaluator as decode's. A transient's upload is the worker's own time.
                    std::vector<kimi_place_expert> experts(source.size());
                    kimi_place_lanes lanes = {};
                    lanes.gpu_overhead = layer.gpu_overhead;
                    double landed = 0.0;
                    for (size_t i = 0; i < source.size(); ++i) {
                        auto & x = experts[i];
                        if (source[i].vram_slot >= 0) { x.resident = true; x.gpu = gpu_cost(token_routes[i]); continue; }
                        x.cpu = cpu_cost(token_routes[i]);
                        x.gpu = gpu_cost(token_routes[i]) + upload_cost(expert_bytes(i));
                        if (source[i].cache_hit < 0 && drive_bytes_per_us > 0.0) {
                            landed += double(expert_bytes(i)) / drive_bytes_per_us;
                            x.ready = landed;
                        }
                    }
                    lanes.reads_end = layer.drive;
                    std::vector<size_t> order;
                    for (const candidate & item : movable) { order.push_back(item.index); }
                    std::vector<uint8_t> none(source.size(), 0);
                    time_before = kimi_place_ring_time(lanes, experts.data(), experts.size(), none.data());
                    best = kimi_place_ring_best_prefix(lanes, experts.data(), experts.size(), order.data(), order.size(),
                        &time_after);
                } else {
                    time_before = kimi_prefill_time(layer, layer.cpu, layer.gpu, layer.link, layer.gpu_work);
                    best = kimi_prefill_best_prefix(layer, moves.data(), moves.size(), &time_after);
                }
                chosen.assign(movable.begin(), movable.begin() + best);
            } else {
                double cpu_backlog = layer.cpu, gpu_backlog = layer.gpu;
                time_before = std::max(cpu_backlog, gpu_backlog);
                size_t ssd_selected = 0;
                for (const candidate & item : candidates) {
                    if (item.source == KIMI_VRAM_PREFILL_TRANSIENT_SSD && ssd_selected >= ssd_limit) { continue; }
                    const int routes = token_routes[item.index];
                    const double old_makespan = std::max(cpu_backlog, gpu_backlog);
                    const double next_cpu = cpu_backlog - cpu_cost(routes);
                    const double next_gpu = gpu_backlog + gpu_cost(routes) + upload_cost(expert_bytes(item.index));
                    if (std::max(next_cpu, next_gpu) + std::max(50.0, old_makespan * 0.02) >= old_makespan) { continue; }
                    cpu_backlog = next_cpu; gpu_backlog = next_gpu;
                    ssd_selected += item.source == KIMI_VRAM_PREFILL_TRANSIENT_SSD;
                    chosen.push_back(item);
                }
                time_after = std::max(cpu_backlog, gpu_backlog);
            }
            size_t ram_selected = 0, ssd_selected = 0;
            uint64_t ram_selected_routes = 0, ssd_selected_routes = 0;
            uint64_t ram_selected_bytes = 0, ssd_selected_bytes = 0;
            double cpu_after = layer.cpu, gpu_after = layer.gpu, link_after = layer.link;
            for (const candidate & item : chosen) {
                const size_t index = item.index;
                const int routes = token_routes[index];
                const uint64_t bytes = expert_bytes(index);
                kimi_vram_prefill_transient selected = {};
                selected.request_index = int32_t(index);
                selected.source = item.source;
                if (item.source == KIMI_VRAM_PREFILL_TRANSIENT_RAM) {
                    for (int role = 0; role < 3; ++role) {
                        selected.role[role] = static_cast<const char *>(cache.data(source[index].cache_hit, role)) + source[index].role[role].data_offset;
                    }
                    ++ram_selected;
                    ram_selected_routes += uint64_t(routes);
                    ram_selected_bytes += bytes;
                } else {
                    source[index].vram_ssd_transient = true;
                    source[index].vram_ssd_staging = int(ssd_selected++);
                    ssd_selected_routes += uint64_t(routes);
                    ssd_selected_bytes += bytes;
                }
                prefill_transient.push_back(selected);
                source[index].vram_transient = true;
                cpu_after -= cpu_cost(routes);
                gpu_after += gpu_cost(routes) + upload_cost(bytes);
                link_after += link_cost(bytes, routes);
            }
            const bool gpu_work = layer.gpu_work || !chosen.empty();
            if (makespan) {
                bool ram_locked = false;
                for (const candidate & item : chosen) {
                    ram_locked = ram_locked || (item.source == KIMI_VRAM_PREFILL_TRANSIENT_RAM && ram_slot_locked &&
                        ram_slot_locked(source[item.index].cache_hit));
                }
                prefill_plan = { true, batch.ntokens, cpu_after, gpu_after, link_after, layer.drive, layer.fill,
                    gpu_work ? layer.gpu_overhead : 0.0, time_after, ram_locked };
                // Each lane's predicted start of its experts, in the order the lane takes them: the GPU worker
                // its transients in selection order, the CPU the rest in request order.
                double gpu_clock = 0.0, cpu_clock = 0.0;
                for (const auto & item : prefill_transient) {
                    const size_t index = size_t(item.request_index);
                    source[index].lane_start_us = gpu_clock;
                    gpu_clock += gpu_cost(token_routes[index]) + upload_cost(expert_bytes(index));
                }
                for (size_t i = 0; i < source.size(); ++i) {
                    if (source[i].vram_slot >= 0 || source[i].vram_transient) { continue; }
                    source[i].lane_start_us = cpu_clock;
                    cpu_clock += cpu_cost(token_routes[i]);
                }
            }
            std::fprintf(stderr,
                "[kimi_cpu_prefill_schedule] format=%s rule=%s ram_threshold=%d ssd_threshold=%d ssd_limit=%zu "
                "ram_candidates=%zu ram_selected=%zu ram_selected_routes=%llu ram_upload_bytes=%llu "
                "ssd_candidates=%zu ssd_selected=%zu ssd_selected_routes=%llu ssd_upload_bytes=%llu "
                "cpu_backlog_before_us=%.1f cpu_backlog_after_us=%.1f gpu_backlog_before_us=%.1f "
                "gpu_backlog_after_us=%.1f link_backlog_before_us=%.1f link_backlog_us=%.1f "
                "drive_us=%.1f fill_us=%.1f gpu_overhead_us=%.1f ring=%zu form=%s "
                "makespan_before_us=%.1f makespan_after_us=%.1f scratch_available=1 displacement_slots=0\n",
                mxfp4?"mxfp4":"q2",makespan?"makespan":"legacy",
                ram_threshold,ssd_threshold,ssd_limit,
                ram_candidates,ram_selected,(unsigned long long)ram_selected_routes,
                (unsigned long long)ram_selected_bytes,ssd_candidates,ssd_selected,
                (unsigned long long)ssd_selected_routes,(unsigned long long)ssd_selected_bytes,
                layer.cpu,cpu_after,layer.gpu,gpu_after,layer.link,link_after,
                layer.drive,layer.fill,gpu_work?layer.gpu_overhead:0.0,ring_slots,in_ring?"ring":"streamed",
                time_before,time_after);
        }

        if (cold_admission_ready) {
            std::vector<size_t> cold_candidates;
            for (size_t i = 0; i < source.size(); ++i) {
                if (source[i].vram_slot < 0 && source[i].cache_hit < 0 && !source[i].vram_transient &&
                        token_routes[i] >= 16) {
                    cold_candidates.push_back(i);
                }
            }
            std::stable_sort(cold_candidates.begin(), cold_candidates.end(), [&](size_t a, size_t b) {
                if (token_routes[a] != token_routes[b]) { return token_routes[a] > token_routes[b]; }
                return source[a].id < source[b].id;
            });
            batch.gpu_cold_admit_candidates = cold_candidates.size();
            const size_t selected = std::min(cold_candidates.size(), size_t(q2_prefill_cold_admit_per_layer));
            uint64_t selected_routes = 0, selected_bytes = 0;
            for (size_t position = 0; position < selected; ++position) {
                const size_t index = cold_candidates[position];
                source[index].vram_cold_admit = true;
                source[index].vram_cold_admit_routes = token_routes[index];
                selected_routes += uint64_t(token_routes[index]);
                for (int role = 0; role < 3; ++role) { selected_bytes += source[index].role[role].nbytes; }
            }
            std::fprintf(stderr,
                "[kimi_cpu_prefill_cold_admit_schedule] format=q2 min_routes=16 per_layer=%d "
                "candidates=%zu selected=%zu selected_routes=%llu selected_bytes=%llu zero_decode_heat=1 evict_existing=0\n",
                q2_prefill_cold_admit_per_layer,cold_candidates.size(),selected,
                (unsigned long long)selected_routes,(unsigned long long)selected_bytes);
        }
    }
    void bind_prefill_ssd_transient(size_t request_index, const void * const data[3],
            int (*stage)(void *, int32_t, int) = nullptr, void * context = nullptr) {
        for (auto & item : prefill_transient) {
            if (item.source != KIMI_VRAM_PREFILL_TRANSIENT_SSD ||
                    item.request_index != int32_t(request_index)) { continue; }
            for (int role = 0; role < 3; ++role) {
                if (!data[role] || item.role[role]) {
                    throw std::runtime_error("invalid SSD prefill staging binding");
                }
                item.role[role] = data[role];
            }
            item.stage = stage;
            item.stage_context = context;
            return;
        }
        throw std::runtime_error("SSD prefill staging request was not scheduled");
    }
    void admit_cold_prefill(size_t request_index, const void * const data[3],
            const sink_request & source, kimi_cpu_sink_prefill_batch & batch) {
        if (!source.vram_cold_admit) { return; }
        if (!context || !prefill_enabled || request_index >= requests.size()) {
            throw std::runtime_error("cold-prefill VRAM admission lost its prepared request");
        }
        char error[512] = {};
        int admitted = 0;
        const auto start = sink_clock::now();
        check(prefill_api->admit_cold(context, &requests[request_index], data[0], data[1], data[2],
                &admitted, error, sizeof(error)), error);
        const double elapsed = elapsed_us(start);
        if (!admitted) { return; }
        ++batch.gpu_cold_admit_experts;
        batch.gpu_cold_admit_routes += uint64_t(source.vram_cold_admit_routes);
        for (int role = 0; role < 3; ++role) {
            batch.gpu_cold_admit_upload_bytes += source.role[role].nbytes;
        }
        batch.gpu_cold_admit_upload_us += elapsed;
    }
    void launch_prefill(const void * quantized, size_t quantized_stride, const float * f32, size_t f32_stride,
            const int32_t * route_requests, const float * weights, int nslots, int ntokens,
            const kimi_cpu_sink_activation & activation) {
        if (!context || !prefill_enabled || (!hits && prefill_transient.empty())) { return; }
        const kimi_vram_activation a = {
            activation.kind, activation.situ_beta, activation.situ_linear_beta, activation.swiglu_limit,
        };
        const void * input = format_f32_input ? static_cast<const void *>(f32) : quantized;
        const size_t stride = format_f32_input ? f32_stride : quantized_stride;
        char error[512] = {};
        check(prefill_api->launch(context, input, stride, route_requests, weights, nslots, ntokens,
                prefill_transient.data(), int(prefill_transient.size()), &a, error, sizeof(error)), error);
        prefill_job_active = true;
    }
    void finish_prefill(float * output, size_t stride, kimi_cpu_sink_prefill_batch & batch) {
        if (!context || !prefill_enabled || !prefill_job_active) { return; }
        char error[512] = {};
        prefill_stats = {};
        check(prefill_api->finish(context, output, stride, &prefill_stats, error, sizeof(error)), error);
        batch.gpu_transient_experts = prefill_stats.ram_transient_experts;
        batch.gpu_transient_routes = prefill_stats.ram_transient_routes;
        batch.gpu_transient_upload_bytes = prefill_stats.ram_transient_upload_bytes;
        batch.gpu_transient_upload_us = prefill_stats.ram_transient_upload_us;
        batch.gpu_ssd_transient_experts = prefill_stats.ssd_transient_experts;
        batch.gpu_ssd_transient_routes = prefill_stats.ssd_transient_routes;
        batch.gpu_ssd_transient_upload_bytes = prefill_stats.ssd_transient_upload_bytes;
        batch.gpu_ssd_transient_upload_us = prefill_stats.ssd_transient_upload_us;
        batch.gpu_worker_us = prefill_stats.total_worker_us;
        batch.gpu_expert_us = prefill_stats.gpu_fit_us;
        batch.gpu_read_wait_us = prefill_stats.read_wait_us;
        hits = 0; prefill_job_active = false; prefill_transient.clear();
    }
    // The CPU consumer's time for one expert of the current grouped-prefill layer.
    void observe_prefill_cpu(int routes, double us) {
        if (prefill_plan.valid && routes > 0) { prefill_cpu_fit.add(double(routes), us); }
    }
    // After each grouped-prefill layer: read_us is the I/O time with reads in flight, layer_us the layer so far.
    void calibrate_prefill(double cpu_us, const kimi_cpu_sink_prefill_batch & batch, uint64_t read_bytes,
            double read_us, double layer_us) {
        if (!format_mxfp4_input || mxfp4_prefill_legacy_schedule || !prefill_plan.valid) { return; }
        const prefill_plan_state plan = prefill_plan;
        prefill_plan.valid = false;
        // The link term is the rate of pageable copies (their enqueue stages the whole copy, so the worker's time is
        // the copy's): the ring's, and the RAM cache's unless a RAM transient of the layer comes from a page-locked
        // slot (the decode registrar): its enqueue returns at once and the worker's time is only what the copy left
        // after one computation, not a rate. That layer's RAM transient time stays in the GPU lane, where the worker
        // spent it.
        const double link_bytes = double(batch.gpu_ssd_transient_upload_bytes) +
            (plan.ram_locked ? 0.0 : double(batch.gpu_transient_upload_bytes));
        const double link_us = batch.gpu_ssd_transient_upload_us + (plan.ram_locked ? 0.0 : batch.gpu_transient_upload_us);
        const double upload_us = batch.gpu_transient_upload_us + batch.gpu_ssd_transient_upload_us;
        const bool gpu_work = prefill_stats.gpu_fit_experts > 0.0;
        // The GPU lane as predicted: its experts and uploads. Outside them, the worker's time less its read waits (the
        // output's zeroing, the routes' grouping, the output's copy back) and the output's addition at finish.
        const double gpu_lane_us = prefill_stats.gpu_fit_us + upload_us;
        const double gpu_overhead_us = !gpu_work ? 0.0 : prefill_stats.output_add_us +
            std::max(0.0, prefill_stats.total_worker_us - prefill_stats.read_wait_us - gpu_lane_us);
        kimi_prefill_fit gpu_experts;
        gpu_experts.n = prefill_stats.gpu_fit_experts; gpu_experts.x = prefill_stats.gpu_fit_routes;
        gpu_experts.xx = prefill_stats.gpu_fit_routes_sq; gpu_experts.t = prefill_stats.gpu_fit_us;
        gpu_experts.xt = prefill_stats.gpu_fit_route_us;
        const prefill_sample sample = { prefill_cpu_fit, gpu_experts, link_bytes, link_us, double(read_bytes),
            read_us, gpu_overhead_us, gpu_work ? double(plan.ntokens) : 0.0 };
        if (prefill_samples.size() < prefill_window) { prefill_samples.push_back(sample); }
        else { prefill_samples[prefill_sample_next] = sample; }
        prefill_sample_next = (prefill_sample_next + 1) % prefill_window;
        kimi_prefill_fit cpu, gpu, drive; // drive: per layer, its read time (us) against its bytes (MB)
        double window_link_bytes = 0, window_link_us = 0, overhead_us = 0, gpu_tokens = 0;
        for (const auto & s : prefill_samples) {
            cpu.add(s.cpu); gpu.add(s.gpu);
            window_link_bytes += s.link_bytes; window_link_us += s.link_us;
            if (s.read_bytes > 0.0 && s.read_us > 0.0) { drive.add(s.read_bytes / 1e6, s.read_us); }
            overhead_us += s.gpu_overhead_us; gpu_tokens += s.gpu_tokens;
        }
        kimi_prefill_fit_line(cpu, mxfp4_prior.cpu_fixed_us, mxfp4_prior.cpu_route_us,
            mxfp4_cost.cpu_fixed_us, mxfp4_cost.cpu_route_us);
        kimi_prefill_fit_line(gpu, mxfp4_prior.gpu_fixed_us, mxfp4_prior.gpu_route_us,
            mxfp4_cost.gpu_fixed_us, mxfp4_cost.gpu_route_us);
        if (window_link_bytes > 0.0 && window_link_us > 0.0) { mxfp4_cost.link_gbps = window_link_bytes / window_link_us / 1000.0; }
        if (drive.n > 0.0) {
            // Without a start rate the prior is a line through the origin, which the fit scales to the samples' rate.
            double us_per_mb = 0.0;
            kimi_prefill_fit_line(drive, mxfp4_prior.drive_fixed_us,
                mxfp4_prior.drive_gbps > 0.0 ? 1e3 / mxfp4_prior.drive_gbps : 1.0, mxfp4_cost.drive_fixed_us, us_per_mb);
            if (us_per_mb > 0.0) { mxfp4_cost.drive_gbps = 1e3 / us_per_mb; }
        }
        if (gpu_tokens > 0.0) { mxfp4_cost.gpu_token_us = overhead_us / gpu_tokens; }
        ++cost_calibrations;
        // The terms by the names GGML_KIMI_CPU_PREFILL_MXFP4_COST takes, then this layer's prediction (made before it
        // ran, on the terms of then) against what it measured.
        std::fprintf(stderr,
            "[kimi_cpu_prefill_mxfp4_cost] calibrated=%llu tokens=%d samples=%zu cpu_fixed_us=%.3f cpu_route_us=%.3f "
            "gpu_fixed_us=%.3f gpu_route_us=%.3f link_gbps=%.3f drive_gbps=%.3f drive_fixed_us=%.3f gpu_token_us=%.3f "
            "predicted_us=%.1f measured_us=%.1f cpu_predicted_us=%.1f cpu_us=%.1f gpu_predicted_us=%.1f gpu_us=%.1f "
            "drive_predicted_us=%.1f drive_us=%.1f gpu_overhead_predicted_us=%.1f gpu_overhead_us=%.1f\n",
            (unsigned long long) cost_calibrations, plan.ntokens, prefill_samples.size(),
            mxfp4_cost.cpu_fixed_us, mxfp4_cost.cpu_route_us, mxfp4_cost.gpu_fixed_us, mxfp4_cost.gpu_route_us,
            mxfp4_cost.link_gbps, mxfp4_cost.drive_gbps, mxfp4_cost.drive_fixed_us, mxfp4_cost.gpu_token_us,
            plan.layer_us, layer_us, plan.cpu_us, cpu_us, plan.gpu_us, gpu_lane_us,
            plan.drive_us, read_us, plan.gpu_overhead_us, gpu_overhead_us);
    }
    void cancel_prefill() {
        if (!context || !prefill_enabled ||
                (!prefill_job_active && !prefill_api->cold_admission_active(context))) { return; }
        char error[512] = {};
        if (prefill_api->cancel(context, error, sizeof(error))) {
            std::fprintf(stderr, "[kimi_vram_prefill] failed to drain GPU work: %s; aborting\n", error);
            std::abort();
        }
        hits = 0; prefill_job_active = false; prefill_transient.clear();
    }
    // Moves the RAM leases of this layer's queued uploads out of the batch; collect_deferred releases them.
    void retain_deferred(sink_cache & cache, const std::vector<sink_request> & source) {
        for (size_t i = 0; i < source.size() && i < upload_queued.size(); ++i) {
            if (!upload_queued[i]) { continue; }
            const int index = source[i].cache_hit >= 0 ? source[i].cache_hit : source[i].cache_admit;
            const auto lease = std::find(ram_leases.begin(), ram_leases.end(), index);
            if (index < 0 || lease == ram_leases.end()) { throw std::runtime_error("deferred VRAM upload without a RAM lease"); }
            ram_leases.erase(lease);
            deferred.push_back({requests[i], source[i], index, cache.generation});
        }
    }
    static bool same_key(const kimi_vram_request & a, const kimi_vram_request & b) {
        if (a.layer != b.layer || a.expert != b.expert) { return false; }
        for (int r = 0; r < 3; ++r) {
            const auto & x = a.role[r];
            const auto & y = b.role[r];
            if (x.device != y.device || x.inode != y.inode || x.size != y.size || x.offset != y.offset || x.bytes != y.bytes ||
                    x.mtime_s != y.mtime_s || x.mtime_ns != y.mtime_ns || x.ctime_s != y.ctime_s || x.ctime_ns != y.ctime_ns) {
                return false;
            }
        }
        return true;
    }
    // Publishes completed deferred uploads, releases their RAM leases and, when exclusive, drops the RAM copy.
    void collect_deferred(sink_cache & cache, int32_t wait_layer) {
        if (!deferred_upload || !context) { return; }
        try {
            kimi_vram_request done[16] = {};
            int32_t aborted[16] = {};
            std::vector<kimi_vram_request> keys;
            std::vector<sink_request> sources;
            int count = 0;
            do {
                char error[512] = {};
                check(deferred_api->collect(context, wait_layer, done, aborted, 16, &count, error, sizeof(error)), error);
                wait_layer = -2;
                keys.clear(); sources.clear();
                for (int k = 0; k < count; ++k) {
                    const auto it = std::find_if(deferred.begin(), deferred.end(),
                        [&](const deferred_source & d) { return same_key(d.key, done[k]); });
                    if (it == deferred.end()) { throw std::runtime_error("GPU tier collected an upload without a CPU sink lease"); }
                    ++deferred_collected;
                    // A RAM cache released since the lease (pressure pause, geometry change) no longer holds it.
                    if (it->cache_generation != cache.generation) { ++deferred_stale; deferred.erase(it); continue; }
                    cache.release_lease(it->lease);
                    // An aborted upload's slot went with a reclaimed block: the RAM copy stays the only one.
                    if (aborted[k]) { ++deferred_aborted; }
                    else if (exclusive && decode && have_token) { keys.push_back(it->key); sources.push_back(it->source); }
                    deferred.erase(it);
                }
                if (!keys.empty()) { drop_collected(cache, keys, sources); }
            } while (count == 16);
        } catch (...) {
            abandon_deferred(cache);
            throw;
        }
    }
    void drop_collected(sink_cache & cache, const std::vector<kimi_vram_request> & keys, const std::vector<sink_request> & sources) {
        char error[512] = {};
        int ready[16] = {}; kimi_vram_ready_lease lease = {};
        int rc = residency->acquire(context, epoch, keys.data(), int(keys.size()), &lease, ready, error, sizeof(error));
        check(rc, error);
        rc = residency->validate(context, &lease, error, sizeof(error)); check(rc, error);
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!ready[i]) { continue; }
            ++ready_confirmations;
            const int dropped = cache.drop_duplicate(keys[i].layer, sources[i]);
            ram_drops += dropped > 0; ram_drop_blocked += dropped < 0;
        }
        rc = residency->release(context, &lease, error, sizeof(error)); check(rc, error);
    }
    // After a failed collect: let every copy finish (a failed tier has already drained), then return the leases.
    void abandon_deferred(sink_cache & cache) {
        char error[512] = {};
        if (context) { deferred_api->wait(context, error, sizeof(error)); }
        for (const auto & d : deferred) {
            if (d.cache_generation != cache.generation) { continue; }
            const auto & e = cache.entries.at(size_t(d.lease));
            if (e.leases && !e.reserved) { cache.release_lease(d.lease); }
        }
        deferred.clear();
    }
    // Called by the RAM cache before it unmaps its arena.
    void wait_deferred() {
        if (!deferred_upload || !context || deferred.empty()) { return; }
        char error[512] = {};
        check(deferred_api->wait(context, error, sizeof(error)), error);
    }
    void release_ram_batch(sink_cache & cache) {
        for (int index : ram_leases) { cache.release_lease(index); }
        ram_leases.clear();
    }
    void drop_ready(sink_cache & cache,const std::vector<sink_request> & source) {
        if (!exclusive || source.empty()) { return; }
        char error[512]={};
        int ready[16] = {}; kimi_vram_ready_lease lease = {};
        int rc = residency->acquire(context, epoch, requests.data(), int(requests.size()), &lease, ready, error, sizeof(error));
        check(rc, error);
        rc = residency->validate(context, &lease, error, sizeof(error)); check(rc, error);
        for (size_t i = 0; i < source.size(); ++i) {
            if (!ready[i]) { continue; }
            ++ready_confirmations;
            const int dropped = cache.drop_duplicate(requests[i].layer, source[i]);
            ram_drops += dropped > 0; ram_drop_blocked += dropped < 0;
        }
        rc = residency->release(context, &lease, error, sizeof(error)); check(rc, error);
    }
    void capture_locations(const kimi_cache_location_candidate * candidates,int count,int latent,int hidden) {
        if(!location_api)return;
        if(count<=0 || count>16)throw std::runtime_error("invalid VRAM native candidate count");
        char error[512]={};check(location_api->configure(context,latent,hidden,error,sizeof(error)),error);
        requests.resize(count);location_sources.assign(count,{});
        for(int i=0;i<count;++i){
            auto & source=location_sources[i];auto & request=requests[i];
            source.id=candidates[i].expert;source.weight=1;
            request.layer=candidates[i].layer;request.expert=source.id;request.weight=1;
            for(int role=0;role<3;++role){
                source.role[role]=candidates[i].role[role];const auto k=sink_cache::key(source.role[role]);
                request.role[role]={k.device,k.inode,k.size,k.offset,k.bytes,k.mtime_s,k.mtime_ns,k.ctime_s,k.ctime_ns};
            }
        }
    }
    void cancel() {
        if (!context) { return; }
        char error[512] = {};
        if (api->cancel(context, error, sizeof(error))) {
            std::fprintf(stderr, "[kimi_vram_tier] failed to drain GPU work: %s; aborting\n", error);
            std::abort();
        }
        ram_leases.clear(); // caller releases RAM only after this drain
        deferred.clear(); upload_queued.clear();
        prefill_open = false;
    }
};
