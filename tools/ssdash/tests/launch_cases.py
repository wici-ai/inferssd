"""Synthetic launcher inputs, independent of the new package."""

LINUX_CASES = [
    {"name": "base"},
    {"name": "early-resolve", "env": {"LAUNCH_RESOLVE": "1", "SERVER_BIN": "@case/no-server"}},
    {"name": "scope-invalid", "env": {"LAUNCH_SCOPE": "*.scope"}},
    {
        "name": "gpu-route",
        "settings": (
            "GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0\n"
            "CONTEXT=8192 # note\n"
            "GGML_KIMI_CPU_SINK_PREFETCH=4\n"
            "GGML_NEW_KNOB=7\n"
            "ARGS=--jinja --seed 1\n"
        ),
    },
    {
        "name": "environment-wins",
        "settings": "CONTEXT=8192\nGGML_KIMI_CPU_SINK_PREFETCH=4\n",
        "env": {"CONTEXT": "4096", "GGML_KIMI_CPU_SINK_PREFETCH": "3"},
    },
    *[
        {"name": "threshold-" + value, "env": {"GGML_KIMI_CPU_PREFILL_MAX_TOKENS": value}}
        for value in ("0", "08", "0500", "8192", "4294967295", "4294967296", "all")
    ],
    {
        "name": "threshold-both",
        "settings": "GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0\n",
        "env": {"GGML_KIMI_CPU_PREFILL": "1"},
    },
    {
        "name": "sink-off",
        "env": {"GGML_KIMI_CPU_SINK": "0", "GGML_KIMI_CPU_PREFILL_MAX_TOKENS": "0"},
    },
    {"name": "legacy-zero", "env": {"GGML_KIMI_CPU_PREFILL": "0"}},
    {"name": "legacy-invalid", "env": {"GGML_KIMI_CPU_PREFILL": "yes"}},
    {"name": "settings-typo", "settings": "CONTXT=1\n"},
    {"name": "settings-export", "settings": "# comment\nexport CONTEXT=1\n"},
    {"name": "settings-space", "settings": "CONTEXT = 1\n"},
    {
        "name": "settings-duplicate",
        "settings": "CONTEXT=100\nCONTEXT='2048' # last\nGGML_NOTE=a=b\n",
    },
    {
        "name": "settings-pending",
        "settings": (
            "# GGML_KIMI_CPU_PREFILL_MAX_TOKENS=0 # pending\nUBATCH=4096 # source=decided\n"
        ),
    },
    {
        "name": "empty-values",
        "settings": "CONTEXT=\nGGML_KIMI_CPU_SINK_PREFETCH=\n",
        "env": {
            "THREADS": "",
            "UBATCH": "",
            "LLAMA_SERVER_HOST": "",
            "PROMPT_CACHE_DIR": "",
            "GGML_KIMI_CPU_SINK_VRAM_BYTES": "",
        },
    },
    {
        "name": "inherited",
        "env": {
            "GGML_KIMI_CPU_SINK_COMPUTE_CPUS": "3,1",
            "LLAMA_ARG_SEED": "5",
            "UNRELATED": "kept",
        },
    },
    {"name": "pack-contradicts", "env": {"GGML_MOE_EXPERT_PACK": "/not-selected"}},
    {"name": "threads-contradict", "env": {"THREADS": "4", "GGML_KIMI_CPU_SINK_THREADS": "3"}},
    {"name": "threads-agree", "env": {"THREADS": "4", "GGML_KIMI_CPU_SINK_THREADS": "4"}},
    {"name": "reserve", "env": {"VRAM_RESERVE_MIB": "576"}},
    *[
        {"name": "no-" + value.lower(), "env": {"GGML_KIMI_CPU_SINK_VRAM_" + value: "0"}}
        for value in ("ELASTIC", "GLOBAL_POOL", "KEEP_ACROSS_REQUESTS")
    ],
    {"name": "margin-given", "env": {"GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB": "96"}},
    {"name": "margin-blocks", "env": {"GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS": "2"}},
    {"name": "uma", "host": {"gpu_mib": "[N/A]"}},
    {"name": "uma-headroom", "host": {"gpu_mib": "[N/A]"}, "env": {"MEMORY_HEADROOM_MIB": "8192"}},
    {
        "name": "uma-learned",
        "host": {"gpu_mib": "[N/A]"},
        "settings": (
            "GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB=6145\n"
            "CONTEXT=4608\n"
            "UBATCH=4608\n"
            "CTX_CHECKPOINTS=1\n"
            "GGML_KIMI_CPU_PREFILL_MAX_TOKENS=314\n"
        ),
    },
    {"name": "mxfp4", "type": 39},
    {"name": "pack", "env": {"EXPERT_PACK": "@case/model.pack"}},
    {
        "name": "alias-wins-pack",
        "env": {"EXPERT_ALIAS": "@case/selected.idx", "EXPERT_PACK": "@case/model.pack"},
    },
    {
        "name": "dense",
        "dense": "blk.0.attn_q.weight\tdrive\noutput.weight\tram\n",
        "host": {"uring": True},
    },
    {"name": "dense-no-uring", "dense": "blk.0.attn_q.weight\tdrive\n"},
    {"name": "dense-no-split", "dense": "blk.0.attn_odd.weight\tdrive\n"},
    {
        "name": "extra-arguments",
        "settings": "ARGS=--seed 7 --override-kv 'not shell quoted'\n",
        "extra": ["--temp", "0", "--chat-template-kwargs", '{"space": "one two"}'],
    },
    {"name": "busy", "host": {"gpu_apps": "123, other-process"}},
    {
        "name": "busy-reserve",
        "host": {"gpu_apps": "123, other-process"},
        "env": {"VRAM_RESERVE_MIB": "512"},
    },
    {"name": "port-busy", "host": {"port_busy": "LISTEN"}},
    {"name": "bad-port", "env": {"LLAMA_SERVER_PORT": "70000"}},
    {"name": "bad-threads", "env": {"THREADS": "0"}},
    {"name": "bad-context", "env": {"CONTEXT": "0"}},
    {"name": "bad-margin", "env": {"VRAM_ELASTIC_SAFETY_MIB": "63"}},
    {"name": "missing-server", "env": {"SERVER_BIN": "@case/missing"}},
    {"name": "missing-model", "env": {"MODEL": "@case/missing.gguf"}},
    {"name": "index-failure", "host": {"fail_metadata": "index"}},
    {"name": "facts-failure", "host": {"fail_metadata": "facts"}},
    {"name": "model-ambiguous", "ambiguous": True},
    {"name": "no-cache", "env": {"PROMPT_CACHE_DIR": ""}},
    {"name": "automatic-memory", "env": {"MEMORY_MAX": "", "MEMORY_RESERVE_MIB": "256"}},
    {
        "name": "insufficient-memory",
        "env": {"MEMORY_MAX": ""},
        "host": {"mem_available_kib": 1048576},
    },
    {"name": "cache-freshness", "repeat": 3},
]

MAC_CASES = [
    *[
        {"name": profile + "-" + mode, "profile": profile, "mode": mode}
        for profile in ("dsv", "muse")
        for mode in ("chat", "server")
    ],
    {
        "name": "mac-overrides",
        "env": {
            "SSD_LLAMA_MAC_EXPERT_CACHE_MIB": "24576",
            "SSD_LLAMA_THREADS": "4",
            "SSD_LLAMA_PREFILL_THREADS": "12",
            "LLAMA_DSV41_TOPK_REUSE": "1",
            "UNRELATED": "kept",
        },
        "extra": ["--", "--seed", "7"],
    },
    *[
        {"name": "mac-bad-" + value, "env": {"SSD_LLAMA_MAC_EXPERT_CACHE_MIB": value}}
        for value in ("", "-1", "1x", "18446744073709551616")
    ],
    {"name": "mac-alignment", "env": {"SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT": "3"}},
    {"name": "mac-policy", "env": {"SSD_LLAMA_MAC_EXPERT_CACHE_POLICY": "bad"}},
    {"name": "mac-pack", "env": {"SSD_LLAMA_MAC_EXPERT_PACK": "pack.bin"}},
    {"name": "mac-missing-model", "missing_model": True},
    {"name": "mac-second-shard", "second_shard": True},
    {"name": "mac-incomplete", "incomplete": True},
    {"name": "mac-split", "split": True},
    {"name": "mac-missing-binary", "missing_binary": True},
    {"name": "mac-missing-template", "missing_template": True},
    {"name": "mac-platform", "system": "Linux"},
    {"name": "mac-model-override", "extra": ["--model=other.gguf"]},
    {"name": "mac-remote-env", "env": {"LLAMA_ARG_HF_REPO": "example/model"}},
    {"name": "mac-index-failure", "failure": "index"},
    {"name": "mac-index-empty", "failure": "empty"},
    {"name": "mac-index-partial", "failure": "partial"},
    {"name": "mac-replace-first", "failure": "replace1"},
    {"name": "mac-replace-second", "failure": "replace2"},
    {"name": "mac-stdout-failure", "failure": "stdout"},
    {"name": "mac-sysctl-failure", "failure": "sysctl"},
]
