"""Verify unchanged tensor selection and reject mismatched quantization bases."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

from tools.model.publication import file_hash
from tools.quantization.qwen_moe_mix import compose


def source(root, bits, *, different_sign=False):
    root.mkdir()
    expert = "model.language_model.layers.0.mlp.experts.down_proj"
    dense = "model.language_model.layers.0.linear_attn.in_proj_z.weight"
    dtype = np.uint16 if bits == 2 else np.uint32
    tensors = {
        expert + ".packed": np.full((2, 8, 16), bits, dtype),
        expert + ".scale": np.full((2, 8), bits, np.float16),
        dense + ".packed": np.full((8, 16), bits, dtype),
        dense + ".scale": np.full(8, bits, np.float16),
        f"orinfer.e8p_q{bits}.basis": np.zeros((256, 8), np.int8),
        "orinfer.rotation.k128.signs": np.ones(128, np.int8),
    }
    if different_sign:
        tensors["orinfer.rotation.k128.signs"][0] = -1
    metadata = {
        f"orin.layout.{n}": f"e8p-q{bits}-row-u{bits * 8}"
        if n.endswith(".packed")
        else "native_contiguous"
        for n in tensors
    }
    path = root / "weights.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    q = dict(
        quant_method=f"orinfer_e8p_q{bits}_int8",
        coefficients=[4, 1] if bits == 4 else [1],
        rotation="signed-block128",
        scale="fp16-output-row",
        compute_dtype="int8_quality",
        basis_tensor=f"orinfer.e8p_q{bits}.basis",
        layout="u32-low-coarse-high-residual" if bits == 4 else "u16-e8p",
        source="Qwen/test",
        source_revision="a" * 40,
    )
    (root / "config.json").write_text(
        json.dumps(dict(model_type="qwen3_5_moe", text_config={}, quantization_config=q))
    )
    (root / "progress.json").write_text(
        json.dumps(dict(complete=True, parameters=4096, contract="test" + str(bits)))
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(
                metadata={"orinfer.sha256": {path.name: file_hash(path)}},
                weight_map={n: path.name for n in tensors},
            )
        )
    )
    return expert, dense


class MixedMoeTests(unittest.TestCase):
    def test_single_selected_codes_and_scales_without_float_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert, dense = source(root / "q4", 4)
            source(root / "q2", 2)
            payload = compose(root / "q4", root / "q2", root / "mixed")
            data = load_file(str(root / "mixed/weights.safetensors"))
            self.assertEqual(data[expert + ".packed"].dtype, np.uint16)
            self.assertTrue(np.all(data[expert + ".packed"] == 2))
            self.assertTrue(np.all(data[expert + ".scale"] == 2))
            self.assertEqual(data[dense + ".packed"].dtype, np.uint32)
            self.assertTrue(np.all(data[dense + ".packed"] == 4))
            self.assertTrue(np.all(data[dense + ".scale"] == 4))
            self.assertNotIn("orinfer.e8p_q2.basis", data)
            self.assertEqual(payload, sum(t.nbytes for t in data.values()))
            q = json.loads((root / "mixed/config.json").read_text())["quantization_config"]
            self.assertEqual(q["experts"], "e8p_q2")
            self.assertEqual(q["quant_method"], "orinfer_e8p_q4_int8")

    def test_different_rotation_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source(root / "q4", 4)
            source(root / "q2", 2, different_sign=True)
            with self.assertRaisesRegex(ValueError, "Rotation basis differs"):
                compose(root / "q4", root / "q2", root / "mixed")
            self.assertFalse((root / "mixed").exists())

    def test_source_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source(root / "q4", 4)
            source(root / "q2", 2)
            with (root / "q2/weights.safetensors").open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                compose(root / "q4", root / "q2", root / "mixed")


if __name__ == "__main__":
    unittest.main()
