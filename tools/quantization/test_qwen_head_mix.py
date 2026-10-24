"""A separate Q4 head preserves trunk bytes and verifies its source basis."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

from tools.build.files import file_hash
from tools.quantization.qwen_head_mix import compose
from tools.quantization.test_qwen_moe_mix import source


def checkpoint(root, bits, *, different_sign=False):
    source(root, bits, different_sign=different_sign)
    config = json.loads((root / "config.json").read_text())
    config.update(
        model_type="qwen3_5",
        text_config=dict(tie_word_embeddings=False, hidden_size=128, vocab_size=8),
    )
    (root / "config.json").write_text(json.dumps(config))
    path = root / "weights.safetensors"
    values = load_file(str(path))
    values.update(
        {
            "lm_head.weight.packed": np.full((8, 16), bits, np.uint16 if bits == 2 else np.uint32),
            "lm_head.weight.scale": np.full(8, bits, np.float16),
        }
    )
    save_file(
        values,
        str(path),
        metadata={
            "orin.layout." + n: f"e8p-q{bits}-row-u{bits * 8}"
            if n.endswith(".packed")
            else "native_contiguous"
            for n in values
        },
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(
                metadata={"orinfer.sha256": {path.name: file_hash(path)}},
                weight_map={n: path.name for n in values},
            )
        )
    )


class HeadMixTests(unittest.TestCase):
    def test_only_head_codes_and_scale_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root / "q2", 2)
            checkpoint(root / "q4", 4)
            digest = file_hash(root / "q2/weights.safetensors")
            compose(root / "q2", root / "q4", root / "mixed")
            before = load_file(str(root / "q2/weights.safetensors"))
            after = load_file(str(root / "mixed/weights.safetensors"))
            self.assertEqual(set(before), set(after))
            for name in before:
                if name.startswith("lm_head."):
                    self.assertTrue(np.all(after[name] == 4))
                else:
                    np.testing.assert_array_equal(before[name], after[name])
            self.assertEqual(after["lm_head.weight.packed"].dtype, np.uint32)
            self.assertEqual(file_hash(root / "q2/weights.safetensors"), digest)
            q = json.loads((root / "mixed/config.json").read_text())["quantization_config"]
            self.assertEqual(q["output_head"], "e8p_q4")
            self.assertEqual(q["quant_method"], "orinfer_e8p_q2_int8")

    def test_changed_rotation_or_tied_model_is_rejected(self):
        for tied in [False, True]:
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint(root / "q2", 2)
                checkpoint(root / "q4", 4, different_sign=not tied)
                if tied:
                    config = json.loads((root / "q2/config.json").read_text())
                    config["text_config"]["tie_word_embeddings"] = True
                    (root / "q2/config.json").write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    compose(root / "q2", root / "q4", root / "mixed")
                self.assertFalse((root / "mixed").exists())
