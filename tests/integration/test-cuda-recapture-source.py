# SPDX-License-Identifier: Apache-2.0
"""Model-free binding checks for the context option and existing scheduler route."""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class RecaptureBinding(unittest.TestCase):
    def test_real_context_configuration_and_receipt(self):
        source = (ROOT / "engine/src/llama-context.cpp").read_text()
        self.assertEqual(source.count("llama_cuda_recapture_default(model.arch)"), 1)
        self.assertEqual(source.count("recapture_active |= configure(backend.get(), recapture.enabled)"), 1)
        self.assertEqual(source.count("rsi.design.active "), 1)
        self.assertIn('recapture.model, recapture_active ? "true" : "false"', source)
        self.assertLess(source.index("llama_cuda_recapture_default(model.arch)"),
                        source.index("ssd_runtime->bind_cpu(backend_cpu)"))
        self.assertIn("ggml_backend_sched_graph_compute_async(sched.get(), gf)", source)

    def test_backend_option_consumed_with_guards(self):
        source = (ROOT / "engine/ggml/src/ggml-cuda/ggml-cuda.cu").read_text()
        state = (ROOT / "engine/ggml/src/ggml-cuda/common.cuh").read_text()
        self.assertIn("bool stable_recapture_enabled = false;", state)
        self.assertIn("GGML_ASSERT(!ctx->stable_recapture_configured)", source)
        self.assertIn("ctx->stable_recapture_enabled = enabled;", source)
        self.assertIn('return (void *) ggml_backend_cuda_configure_stable_recapture;', source)
        self.assertIn("ggml_cuda_can_recapture(cuda_ctx->stable_recapture_enabled, all_vmm,", source)
        self.assertIn("graph->instance != nullptr, graph->replayed_since_capture, nodes_changed)", source)
        self.assertIn("all_vmm &= ggml_cuda_info().devices[device].vmm", source)
        self.assertIn("*nodes_changed = true;", source)
        self.assertIn("ggml_cuda_recapture_node_changed(", source)
        self.assertIn("replayed_since_capture = use_cuda_graph && !cuda_graph_update_required;", source)
        self.assertIn("ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, false, false, graph_key);", source)
        self.assertIn("graph_lookup.phase_reentry && graph_lookup.had_instance", source)

    def test_both_architectures_have_builders_and_scheduler_dispatch(self):
        model = (ROOT / "engine/src/llama-model.cpp").read_text()
        self.assertIn("std::make_unique<llm_build_deepseek41>(*this, params)", model)
        self.assertIn("std::make_unique<llm_build_muse_spark>(*this, params)", model)
        scheduler = (ROOT / "engine/ggml/src/ggml-backend.cpp").read_text()
        self.assertIn("ggml_backend_graph_compute_async", scheduler)


if __name__ == "__main__":
    unittest.main()
