#!/usr/bin/env python3
"""No model load or inference: compile real settings tests and check integration."""
from pathlib import Path
import json
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
def run(argv):
    print(json.dumps({"argv": argv}), flush=True)
    subprocess.run(argv, cwd=root, check=True)

with tempfile.TemporaryDirectory(prefix="resident-rotation-") as tmp:
    binary = str(Path(tmp) / "options")
    run(["clang++", "-std=c++17", "-UNDEBUG", "-Iengine/ggml/src", "-Iengine/src", "-Iengine/ggml/include",
         "engine/tests/ssd/test-resident-rotation-options.cpp",
         "engine/ggml/src/ggml-ssd/settings/settings.cpp", "-o", binary])
    run([binary])
    for source in ("engine/src/llama-kv-cache.cpp", "engine/src/llama-context.cpp"):
        run(["clang++", "-std=c++17", "-fsyntax-only", "-Iengine/include", "-Iengine/src",
             "-Iengine/ggml/include", "-Iengine/ggml/src", "-Iengine/common", "-Iengine/ggml/src/ggml-cpu", source])

context = (root / "engine/src/llama-context.cpp").read_text()
cache = (root / "engine/src/llama-kv-cache.cpp").read_text()
assert context.count("resident_rotation_requested(") == 1
assert context.count("rsi.design.active ") == 1
assert context.index("memory.reset(model.create_memory") < context.index("resident_rotation_requested(")
assert context.index("resident_rotation_requested(") < context.index("// init backends")
assert "cache->get_lid()->enable_resident_indexer_rotation()" in context
assert "if (resident_k_rot) { return resident_k_rot; }" in cache
assert "if (dst == resident_k_rot) { return; }" in cache
body = cache.split("bool llama_kv_cache::enable_resident_indexer_rotation()", 1)[1].split("ggml_tensor * llama_kv_cache::build_input_k_rot", 1)[0]
for guard in ("layers.empty()", "n_embd_head_k_all != 128", "!first->data", "GGML_BACKEND_BUFFER_USAGE_WEIGHTS", "ggml_backend_tensor_set", "resident_rot_attempted"):
    assert guard in body, guard
clear = cache.split("void llama_kv_cache::clear(bool data)", 1)[1].split("bool llama_kv_cache::seq_rm", 1)[0]
assert "resident_rot_buffer" not in clear
print("PASS real consumer wiring, once per context, separate constant ownership, upload bypass, conservative guards")

assert "ssd_runtime ? &ssd_runtime->settings : nullptr, model.arch" in context
assert "model.arch == LLM_ARCH_DEEPSEEK41 || model.arch == LLM_ARCH_MUSE_SPARK" in context
print("PASS frozen argv needs no new toggle; exact architecture scope, unknowns excluded")

# Prove the nominated architecture reaches the modified tensor and both indexer uses.
proof = {
    "engine/src/llama-model.cpp": ["case LLM_ARCH_DEEPSEEK41:", "res = new llama_kv_cache_dsv4("],
    "engine/src/llama-kv-cache-dsv4.cpp": [
        "hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;",
        "kv_lid = std::make_unique<llama_kv_cache>(", "return kv->build_input_k_rot(ctx);",
        "kv->set_input_k_rot(dst);"],
    "engine/src/llama-graph.cpp": [
        "inp->inp_lid.k_rot = mctx_cur->get_lid()->build_input_k_rot(ctx0);",
        "mctx->get_lid()->set_input_k_rot(inp_lid.k_rot);"],
    "engine/src/models/deepseek41.cpp": [
        "idx_q = llama_mul_mat_hadamard(ctx0, idx_q, idx_k_rot);",
        "idx_k = llama_mul_mat_hadamard(ctx0, idx_k, inp_dsv4->get_lid().k_rot);"],
}
for source, needles in proof.items():
    text = (root / source).read_text()
    for needle in needles:
        assert needle in text, (source, needle)
assert "model.arch == LLM_ARCH_DEEPSEEK41 && hparams.n_embd_head_k_full == hparams.indexer_head_size" in cache
assert "ggml_set_input(rot)" not in body
header = (root / "engine/src/llama-kv-cache.h").read_text()
assert header.index("ggml_context_ptr resident_rot_ctx") < header.index("ggml_backend_buffer_ptr resident_rot_buffer")
print("PASS final-source DeepSeek41 memory-to-indexer call chain, constant graph classification and ownership order")
