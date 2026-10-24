"""Selected promotions replace one bank and preserve unrelated encoded bytes."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from safetensors.numpy import load_file, save_file

from tools.build.files import file_hash
from tools.model.compact import write_shard
from tools.quantization.flash_next_aux import int8_rows
from tools.quantization.promotion import upgrade
from tools.quantization.vq import rotate


class PromotionTests(unittest.TestCase):
    def test_explicit_projection_reads_bf16_and_preserves_other_quantized_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, source, output = (root / n for n in ("bf16", "q2", "mixed"))
            original.mkdir()
            source.mkdir()
            name = "model.language_model.layers.3.self_attn.k_proj.weight"
            other = "model.language_model.layers.3.self_attn.v_proj.weight"
            values = np.random.default_rng(20261002).normal(size=(128, 128)).astype(np.float32)
            raw = (values.view(np.uint32) >> 16).astype(np.uint16)
            values = (raw.astype(np.uint32) << 16).view(np.float32)
            payload = root / "raw.bin"
            payload.write_bytes(raw.tobytes())
            write_shard(
                original / "weights.safetensors",
                [
                    dict(
                        name=name,
                        dtype="BF16",
                        shape=[128, 128],
                        segments=[(payload, 0, raw.nbytes)],
                    )
                ],
            )
            origin = dict(repo="Qwen/test", revision="a" * 40)
            (original / "source.json").write_text(json.dumps(origin))
            config = dict(text_config=dict(hidden_size=128, intermediate_size=128))
            (original / "config.json").write_text(json.dumps(config))
            (original / "model.safetensors.index.json").write_text(
                json.dumps(dict(weight_map={name: "weights.safetensors"}))
            )
            config["quantization_config"] = dict(
                source=origin["repo"], source_revision=origin["revision"]
            )
            (source / "config.json").write_text(json.dumps(config))
            signs = np.random.default_rng(3).choice(np.array([-1, 1], np.int8), 128)
            tensors = {
                name + ".packed": np.ones((128, 16), np.uint16),
                name + ".scale": np.ones(128, np.float16),
                other + ".packed": np.full((128, 16), 37, np.uint16),
                "orinfer.rotation.k128.signs": signs,
            }
            shard = source / "weights.safetensors"
            save_file(
                tensors,
                str(shard),
                metadata={
                    "orin.layout." + n: "e8p-q2-row-u16"
                    if n.endswith(".packed")
                    else "native_contiguous"
                    for n in tensors
                },
            )
            digest = file_hash(shard)
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    dict(
                        metadata={"orinfer.sha256": {shard.name: digest}},
                        weight_map={n: shard.name for n in tensors},
                    )
                )
            )
            (source / "progress.json").write_text(
                json.dumps(dict(complete=True, contract="fixture", shards={}, quantized={}))
            )
            upgrade(
                SimpleNamespace(source_dir=original, upgrade_from=source, output=output, rows=17),
                configuration=lambda c, **kwargs: None,
                signature=lambda c: c["text_config"],
                layers=[],
                layer_prefix="model.language_model.layers.",
                matrices=[name],
            )
            result = load_file(str(output / shard.name))
            code, scale = int8_rows(rotate(values, signs))
            np.testing.assert_array_equal(result[name + ".int8"], code)
            np.testing.assert_array_equal(result[name + ".scale"], scale)
            np.testing.assert_array_equal(result[other + ".packed"], tensors[other + ".packed"])
            self.assertNotIn(name + ".packed", result)
            self.assertEqual(file_hash(shard), digest)
            self.assertEqual(
                json.loads((output / "config.json").read_text())["quantization_config"][
                    "rotated_w8_matrices"
                ],
                [name],
            )
