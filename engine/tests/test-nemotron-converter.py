# SPDX-License-Identifier: MIT
"""Test the actual converter metadata method without loading torch or weights.

Extract only NemotronModel from the monolithic converter; replace its unrelated
base-class work and writer with test doubles. This is a metadata unit test, not
an end-to-end HF-to-GGUF conversion test.
"""
import ast
from pathlib import Path
import types
import unittest


class Recorder:
    def __init__(self):
        self.values = {}

    def __getattr__(self, name):
        return lambda value: self.values.__setitem__(name, value)


class TextModel:
    def set_gguf_parameters(self):
        pass

    def find_hparam(self, names):
        return next(self.hparams[name] for name in names if name in self.hparams)


def load_metadata_class():
    path = Path(__file__).resolve().parents[1] / "convert_hf_to_gguf.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NemotronModel")
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "TextModel": TextModel,
        "gguf": types.SimpleNamespace(
            MODEL_ARCH=types.SimpleNamespace(NEMOTRON="nemotron"),
            RopeScalingType=types.SimpleNamespace(NONE="none", LINEAR="linear")),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["NemotronModel"]


class NemotronMetadataTest(unittest.TestCase):
    def test_rope_dimensions(self):
        model_class = load_metadata_class()
        # (hidden_size, heads, explicit head_dim, partial_rotary_factor, expected n_rot)
        cases = [(4096, 32, None, 0.5, 64), (4096, 48, 128, 0.5, 64),
                 (4096, 48, 128, 1.0, 128), (64, 6, 16, 0.5, 8),
                 (4096, 32, 128, 0.5, 64)]
        for hidden, heads, head_dim, fraction, expected in cases:
            with self.subTest(hidden=hidden, heads=heads, head_dim=head_dim, fraction=fraction):
                model = model_class()
                model.hparams = {"hidden_size": hidden, "num_attention_heads": heads,
                                 "partial_rotary_factor": fraction, "vocab_size": 32,
                                 "layer_norm_eps": 1e-5}
                if head_dim is not None:
                    model.hparams["head_dim"] = head_dim
                model.gguf_writer = Recorder()
                model.set_gguf_parameters()
                values = model.gguf_writer.values
                self.assertEqual(values["add_rope_dimension_count"], expected)
                self.assertEqual(values["add_layer_norm_eps"], 1e-5)
                self.assertEqual(values["add_rope_scaling_type"], "none")


if __name__ == "__main__":
    unittest.main()
