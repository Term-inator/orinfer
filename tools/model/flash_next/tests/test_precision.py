"""Q4 publication reads original BF16 in bounded rows and shares QKV/Z signs."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from safetensors import safe_open

from tools.model.flash_next.precision import publish, selected, signs_for
from tools.model.flash_next.storage import Weights
from tools.quantization.row_q4 import fit


class PrecisionTests(unittest.TestCase):
    def test_bounded_original_conversion_and_verified_resume(self):
        rng = np.random.default_rng(31)
        original = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        bits = (rng.normal(size=(7, 128)).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
        values = (bits.astype(np.uint32) << 16).view(np.float32)
        checkpoint = SimpleNamespace(
            parts={original: [], "model.language_model.layers.1.ple.ple_embedding.weight": []},
            config={
                "quantization_config": {
                    "source": "modelscope:Qwen/test",
                    "source_revision": "a" * 40,
                }
            },
            shape=lambda name: [7, 128],
        )

        class Original:
            repo, revision = "modelscope:Qwen/test", "a" * 40
            reads = []

            def tensor(self, name):
                return "source.safetensors", 0, dict(dtype="BF16", shape=[7, 128])

            def rows(self, name, first, count):
                self.reads.append((first, count))
                return bits[first : first + count].tobytes()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = Weights(checkpoint, root / "model")
            source = Original()
            records = publish(weights, source, root / "build", rows=3)
            self.assertEqual(source.reads, [(0, 3), (3, 3), (6, 1)])
            prefix = "Q4_blk.0.attn_qkv"
            expected = fit(values, signs_for(prefix, 128))
            path = weights.directory / (prefix + ".safetensors")
            with safe_open(path, framework="np") as stream:
                np.testing.assert_array_equal(
                    stream.get_tensor(prefix + "_packed"), expected.packed.view(np.uint16)
                )
                np.testing.assert_array_equal(stream.get_tensor(prefix + "_scale"), expected.scales)
            source.reads.clear()
            resumed = Weights(checkpoint, root / "model")
            self.assertEqual(publish(resumed, source, root / "build", rows=1), records)
            self.assertEqual(source.reads, [])
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                old = stream.read(1)[0]
                stream.seek(-1, 2)
                stream.write(bytes([old ^ 1]))
            with self.assertRaisesRegex(ValueError, "receipt"):
                publish(Weights(checkpoint, root / "model"), source, root / "build")

    def test_precision_selection_and_paired_input_basis(self):
        self.assertIsNone(selected(None))
        self.assertIsNone(selected("blk.48.attn_q.weight"))
        self.assertIsNone(selected("blk.0.ssm_out.weight"))
        self.assertEqual(selected("output.weight"), "Q4_output")
        for layer in (0, 1, 2, 4, 46):
            np.testing.assert_array_equal(
                signs_for(f"Q4_blk.{layer}.attn_qkv", 2560),
                signs_for(f"Q4_blk.{layer}.attn_gate", 2560),
            )


if __name__ == "__main__":
    unittest.main()
