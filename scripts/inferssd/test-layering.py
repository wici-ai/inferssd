#!/usr/bin/env python3
"""Positive and negative examples for the runtime boundaries we enforce."""
from pathlib import Path
import unittest

from runtime_boundaries import framework_issues, source_issues


class RuntimeBoundaries(unittest.TestCase):
    def test_pure_policy_and_model_metadata(self):
        self.assertEqual([], source_issues(
            Path("engine/ggml/src/ggml-ssd/place/runtime-policy.h"), "policy",
            '#include "layer-plan.h"\n#include <array>\ndouble feedback(double sample);'))
        self.assertEqual([], source_issues(
            Path("engine/src/ssd/model-adapter.h"), "model",
            '#include "llama-arch.h"\n#include "ggml.h"\nstruct expert_requirements {};'))

    def test_policy_cannot_reach_execution(self):
        path = Path("engine/ggml/src/ggml-ssd/place/runtime-policy.h")
        for code in ('#include "../io/reader.h"', '#include "cpu-sink-internal.h"',
                     '#include <thread>', 'struct kimi_cpu_sink * private_state;',
                     'cudaLaunchKernel(kernel);'):
            with self.subTest(code=code):
                self.assertTrue(source_issues(path, "policy", code))

    def test_model_adaptation_cannot_use_executor(self):
        self.assertTrue(source_issues(Path("engine/src/ssd/model-adapter.cpp"), "model",
                                      '#include "ggml-cpu/ssd/runtime.h"'))

    def test_backends_cannot_make_model_decisions(self):
        for backend in ("cpu", "cuda", "metal"):
            self.assertTrue(source_issues(
                Path(f"engine/ggml/src/ggml-{backend}/ssd/compute.cpp"), backend,
                'if (arch == LLM_ARCH_MUSE_SPARK) {}'))

    def test_metal_import_boundary(self):
        path = Path("engine/ggml/src/ggml-metal/ops.mm")
        self.assertEqual([], source_issues(path, "metal", '#import <Metal/Metal.h>'))
        self.assertTrue(source_issues(path, "metal", '#import "ggml-cuda/ssd/provider.h"'))
        self.assertTrue(source_issues(path.with_suffix(".metal"), "metal",
                                     '#include "ssd/model-adapter.h"'))

    def test_installed_framework_boundary(self):
        path = Path("framework/src/inferssd/launch/typed.py")
        self.assertEqual([], framework_issues(path, 'from inferssd.launch.resources import read_resources'))
        for code in ('from engine.src import model', 'import bench.ci.private_controller'):
            with self.subTest(code=code):
                self.assertTrue(framework_issues(path, code))


if __name__ == "__main__":
    unittest.main()
