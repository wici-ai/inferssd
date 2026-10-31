# SPDX-License-Identifier: Apache-2.0
"""Host-only scope and actual selector tests; CUDA device helpers are test doubles.

No kernels, model loads or inference. Compile the selector bodies verbatim from
final source so guard/argument changes are exercised, not a Python reimplementation.
"""
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def body(path, name):
    text = (ROOT / path).read_text()
    start = text.index("bool " + name + "(")
    end = text.index("{", start)
    depth = 1
    pos = end + 1
    while depth:
        depth += (text[pos] == "{") - (text[pos] == "}")
        pos += 1
    return text[start:pos]


class ThinMmvf(unittest.TestCase):
    def compile_run(self, source):
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / "test.cpp"
            exe = Path(temp) / "test"
            src.write_text(source)
            subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "engine/src"),
                            "-I", str(ROOT / "engine/ggml/include"),
                            str(src), "-o", str(exe)], check=True)
            return subprocess.run([str(exe)], check=True, text=True, capture_output=True)

    def test_context_scope_and_receipts(self):
        result = self.compile_run(r'''#include "llama-mmvf-policy.h"
#include <cassert>
int main() {
    for (int i = 0; i <= LLM_ARCH_UNKNOWN; ++i) {
        const bool enabled = llama_configure_thin_mmvf(static_cast<llm_arch>(i));
        assert(enabled == (i == LLM_ARCH_MUSE_SPARK));
        ggml_tensor node{};
        node.op = GGML_OP_MUL_MAT;
        node.flags = GGML_TENSOR_FLAG_OUTPUT;
        llama_mark_thin_mmvf(&node, enabled);
        assert(bool(node.flags & GGML_TENSOR_FLAG_THIN_MMVF) == enabled);
        assert(node.flags & GGML_TENSOR_FLAG_OUTPUT);
        llama_mark_thin_mmvf(&node, false);
        assert(!(node.flags & GGML_TENSOR_FLAG_THIN_MMVF));
        for (auto op : {GGML_OP_MUL_MAT_ID, GGML_OP_FLASH_ATTN_EXT, GGML_OP_ADD}) {
            node.op = op;
            llama_mark_thin_mmvf(&node, true);
            assert(!(node.flags & GGML_TENSOR_FLAG_THIN_MMVF));
        }
    }
}
''')
        lines = result.stderr.splitlines()
        records = [json.loads(line.removeprefix("rsi.design.active "))
                   if hasattr(str, "removeprefix") else json.loads(line[18:]) for line in lines]
        self.assertEqual(len(records), 132)
        self.assertEqual(sum(row["enabled"] for row in records), 1)
        self.assertEqual([row["model"] for row in records if row["enabled"]], ["muse"])
        self.assertTrue(any(row["model"] == "deepseek" and not row["enabled"] for row in records))
        self.assertTrue(all(set(row) == {"consumer", "model", "enabled", "once_per_context"}
                            and row["once_per_context"] for row in records))
        context = (ROOT / "engine/src/llama-context.cpp").read_text()
        self.assertEqual(context.count("llama_configure_thin_mmvf(model.arch)"), 1)
        model = (ROOT / "engine/src/llama-model.cpp").read_text()
        self.assertIn("llama_mark_thin_mmvf(ggml_graph_node(graph, i), params.cparams.cuda_thin_mmvf)", model)
        backend = (ROOT / "engine/ggml/src/ggml-backend.cpp").read_text()
        self.assertIn("dst->flags = src->flags;", backend)
        cuda = (ROOT / "engine/ggml/src/ggml-cuda/ggml-cuda.cu").read_text()
        calls = re.findall(r"ggml_cuda_should_use_mmvf\([^;]+", cuda)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("warp_size" in call and "GGML_TENSOR_FLAG_THIN_MMVF" in call for call in calls))
        self.assertIn("!is_mul_mat_id &&", calls[0])
        self.assertIn("&tensor->flags, sizeof(tensor->flags)", cuda)

    def test_actual_selectors(self):
        helpers = r'''#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include "ggml.h"
#define MMVF_MAX_BATCH_SIZE 8
#define GGML_CUDA_CC_TURING 750
#define GGML_CUDA_CC_ADA_LOVELACE 890
#define GGML_CUDA_CC_IS_NVIDIA(cc) ((cc) < 10000)
#define GGML_CUDA_CC_IS_AMD(cc) ((cc) >= 10000)
#define GGML_CUDA_CC_IS_CDNA3(cc) ((cc) == 10003)
#define GGML_CUDA_CC_IS_CDNA2(cc) ((cc) == 10002)
#define GGML_CUDA_CC_IS_CDNA1(cc) ((cc) == 10001)
#define GGML_CUDA_CC_IS_RDNA3_0(cc) ((cc) == 11000)
#define GGML_CUDA_CC_IS_RDNA3(cc) ((cc) == 11000)
#define GGML_CUDA_CC_IS_RDNA4(cc) ((cc) == 12000)
bool ampere_mma_available(int cc) { return cc >= 800 && cc < 10000; }
bool fp32_mma_hardware_available(int) { return true; }
bool fp16_mma_hardware_available(int) { return true; }
bool bf16_mma_hardware_available(int) { return true; }
bool volta_mma_available(int cc) { return cc == 700; }
bool turing_mma_available(int cc) { return cc >= 750 && cc < 10000; }
bool amd_wmma_available(int cc) { return cc >= 11000; }
bool amd_mfma_available(int cc) { return cc >= 10000 && cc < 11000; }
int mmf_get_rows_per_block(int) { return 64; }
extern "C" size_t ggml_type_size(ggml_type type) { return type == GGML_TYPE_F32 ? 4 : 2; }
extern "C" bool ggml_is_quantized(ggml_type type) { return type == GGML_TYPE_Q4_K; }
'''
        source = helpers + body("engine/ggml/src/ggml-cuda/mmf.cu", "ggml_cuda_should_use_mmf")
        source += body("engine/ggml/src/ggml-cuda/mmvf.cu", "ggml_cuda_should_use_mmvf")
        source += r'''int main() {
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_BF16}) {
        int64_t ne[] = {128, 65, 1, 1};
        size_t nb[] = {2, 256, 16640, 16640};
        auto use = [&](int cc, int cols, bool enabled) {
            return ggml_cuda_should_use_mmvf(type, cc, 32, ne, nb, cols, enabled);
        };
        for (int cols = 1; cols <= 16; ++cols) {
            assert(use(1200, cols, false) == (cols == 1));
            assert(use(1200, cols, true) == (cols <= 8));
            // AMD behavior is identical with either flag value.
            assert(use(12000, cols, false) == use(12000, cols, true));
        }
        ne[1] = 64; // MMF eligible: preserve original MMVF decision.
        assert(!use(1200, 2, true));
        ne[1] = 513; ne[2] = 2; // Too large and multiple channels.
        assert(!use(1200, 2, true));
        ne[2] = 1; // Single channel/sample exception.
        assert(use(1200, 8, true));
        ne[0] = 127;
        assert(!use(1200, 2, true));
        ne[0] = 128; nb[0] = 4;
        assert(!use(1200, 2, true));
        nb[0] = 2;
        for (int dim = 1; dim < 4; ++dim) {
            nb[dim] += 2;
            assert(!use(1200, 2, true));
            nb[dim] -= 2;
        }
    }
    int64_t ne[] = {128, 65, 1, 1};
    size_t nb[] = {4, 512, 33280, 33280};
    for (int cols = 1; cols <= 16; ++cols) {
        assert(ggml_cuda_should_use_mmvf(GGML_TYPE_F32, 1200, 32, ne, nb, cols, true) ==
               ggml_cuda_should_use_mmvf(GGML_TYPE_F32, 1200, 32, ne, nb, cols, false));
    }
    assert(!ggml_cuda_should_use_mmvf(GGML_TYPE_Q4_K, 1200, 32, ne, nb, 2, true));
}
'''
        self.compile_run(source)


if __name__ == "__main__":
    unittest.main()
