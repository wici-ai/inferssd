"""Additional original F numeric-boundary observations; no candidate imports."""
NUMERIC_CASES = [
    {"name": "reserve-leading-zero", "env": {"VRAM_RESERVE_MIB": "010"}},
    {"name": "port-leading-zero", "env": {"LLAMA_SERVER_PORT": "010"}},
    {"name": "headroom-leading-zero", "env": {"MEMORY_HEADROOM_MIB": "010", "MEMORY_MAX": ""}},
    {"name": "checkpoint-leading-zero", "env": {"CTX_CHECKPOINTS": "0008"}},
    {"name": "reserve-signed-overflow", "env": {"VRAM_RESERVE_MIB": "18446744073709551616"}},
    {"name": "context-signed-overflow", "env": {"CONTEXT": "18446744073709551616"}},
    {"name": "ubatch-signed-overflow", "env": {"UBATCH": "9223372036854775808"}},
    {"name": "reserve-expression", "env": {"VRAM_RESERVE_MIB": "1+1"}},
    {"name": "headroom-variable", "env": {"MEMORY_HEADROOM_MIB": "OTHER"}},
]
