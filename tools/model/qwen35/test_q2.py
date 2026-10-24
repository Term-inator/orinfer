"""Q2 config and lossless U16 publication contracts."""

import hashlib
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from tools.model.qwen35.build import configuration
from tools.model.qwen35.storage import Weights
from tools.model.qwen35.test_build import config


class Q2Tests(unittest.TestCase):
    def test_null_expert_precision_uses_the_global_codec(self):
        from tools.model.qwen35.build import Builder

        class ReachedWeights(Exception):
            pass

        candidate = config()
        candidate["quantization_config"].update(
            quant_method="orinfer_e8p_q2_int8",
            coefficients=[1],
            basis_tensor="orinfer.e8p_q2.basis",
            layout="u16-e8p",
            experts=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(candidate))
            (root / "progress.json").write_text(json.dumps(dict(complete=True)))
            builder = Builder.__new__(Builder)
            with (
                patch.dict(
                    "sys.modules", {"tools.model.qwen35.compiler": SimpleNamespace(Aot=object)}
                ),
                patch("tools.model.qwen35.build.Weights", side_effect=ReachedWeights),
                self.assertRaises(ReachedWeights),
            ):
                builder.__init__(
                    root,
                    root / "model",
                    root / "output",
                    root / "cache",
                    4096,
                    None,
                    prefill_rows=128,
                )
            self.assertEqual(builder.expert_codec, "e8p_q2")

    def test_routed_experts_use_their_own_codec_with_the_shared_e8p_table(self):
        from tools.model.qwen35 import moe

        for dense, expert in [("e8p_q4", "e8p_q2"), ("e8p_q2", "e8p_q4")]:
            expected = "e8p_q2_k128" if expert == "e8p_q2" else "e8p_q4"

            def routed(*args, **kwargs):
                self.assertEqual(kwargs["codec"], expected)
                return {"codec": kwargs["codec"]}

            names = ["up", "us", "down", "ds"]
            builder = SimpleNamespace(
                q4_codec=dense,
                expert_codec=expert,
                h=2048,
                f=512,
                capacity=128,
                text=dict(num_experts=256, num_experts_per_tok=8),
                book="book",
                lock=None,
                output=Path("unused"),
                aot=SimpleNamespace(identity={}),
                weights=SimpleNamespace(
                    extra={"book": np.zeros((256, 1), np.uint32)},
                    array=lambda name: np.zeros(1, np.uint16),
                    buffers={name: {"data": {"sha256": "0" * 64}} for name in names},
                ),
            )
            modules = {
                "torch": SimpleNamespace(
                    from_numpy=lambda array: SimpleNamespace(cuda=lambda: array)
                ),
                "tools.model.expert_tuning": SimpleNamespace(
                    MoeTuner=lambda *args, **kwargs: SimpleNamespace(routed=routed)
                ),
            }
            with (
                self.subTest(dense=dense, expert=expert),
                patch.dict("sys.modules", modules),
                patch("tools.model.qwen35.build.gpu_lock", return_value=nullcontext()),
            ):
                self.assertEqual(
                    moe.selection(builder, [("up", "us"), ("down", "ds")]), {"codec": expected}
                )

    def test_shared_expert_uses_dense_projection_codec(self):
        from tools.model.qwen35 import moe

        class ReachedProjection(Exception):
            pass

        def kernel(*args, **kwargs):
            return None

        modules = {
            "kernels.model.moe": SimpleNamespace(
                **{
                    name: kernel
                    for name in (
                        "router_projection",
                        "router_topk_warp",
                        "router_topk",
                        "expert_histogram_local",
                        "expert_histogram_offsets",
                        "expert_histogram_finish",
                        "expert_dispatch",
                    )
                }
            ),
            "kernels.model.rotation_a8": SimpleNamespace(),
            "kernels.operators.op30_activation_quantization": SimpleNamespace(_compile=kernel),
        }
        for codec, draft in (("e8p_q2", False), ("e8p_q4", False), ("e8p_q2", True)):
            with self.subTest(codec=codec, draft=draft):

                def projection(*args, **kwargs):
                    self.assertEqual(kwargs["kind"], "w8" if draft else codec)
                    raise ReachedProjection

                builder = SimpleNamespace(
                    h=2048,
                    f=512,
                    capacity=128,
                    q4_codec=codec,
                    lock=None,
                    text={"num_experts": 256, "num_experts_per_tok": 8},
                    weights=SimpleNamespace(combined=lambda *args: "router"),
                    moe_router_tuner=SimpleNamespace(router=kernel, topk=kernel),
                    row_choices=lambda *args: [],
                    projection=projection,
                )
                with (
                    patch.dict("sys.modules", modules),
                    patch.object(moe, "workspace"),
                    patch.object(moe, "banks", return_value=(("up", "scale"), ("down", "scale"))),
                    patch.object(moe, "emitter", return_value=kernel),
                    self.assertRaises(ReachedProjection),
                ):
                    moe.build(builder, "post", "layer.mlp", draft=draft)

    def test_packet_embedding_uses_the_checkpoint_codec(self):
        from tools.model.qwen35 import packet
        from tools.model.qwen35.parallel import TABLE

        def kernel(*args, **kwargs):
            return SimpleNamespace(buffer_map={})

        embedding_calls = []

        def embedding(*args, **kwargs):
            embedding_calls.append(kwargs)
            return kernel(*args, **kwargs)

        modules = {
            "kernels.model.qwen35": SimpleNamespace(prepare=kernel, embedding=kernel),
            "kernels.model.control": SimpleNamespace(advance=kernel),
            "kernels.model.e8p_q4": SimpleNamespace(embedding=embedding),
            "kernels.model.packet_io": SimpleNamespace(token_input=kernel, outputs=kernel),
            "kernels.model.speculation": SimpleNamespace(),
            "kernels.model.independent_rows": SimpleNamespace(
                independent_rows=lambda fn, *args, **kwargs: (fn, {})
            ),
            "kernels.model.rows": SimpleNamespace(static_rows=lambda fn: fn),
            "kernels.operators.op23_final_norm": SimpleNamespace(final_norm=kernel),
            "kernels.vision.features": SimpleNamespace(overlay=kernel),
        }
        emitted = []
        for codec in ("e8p_q2", "e8p_q4"):
            with self.subTest(codec=codec), patch.dict("sys.modules", modules):
                builder = SimpleNamespace(
                    h=5120,
                    v=248320,
                    context=4096,
                    standard=False,
                    embedding_q4=True,
                    q4_codec=codec,
                    audio=None,
                    vision=None,
                    mtp=None,
                    text={"layer_types": ["full_attention"]},
                    state_columns=[],
                    weights=SimpleNamespace(buffers={TABLE: {}}),
                    aot=SimpleNamespace(
                        kernels=[
                            {"name": "qwen_rows/" + role}
                            for role in (
                                "begin/prepare",
                                "begin/embedding",
                                "end/advance",
                                "gather/norm",
                            )
                        ],
                        compiler=SimpleNamespace(pointers=lambda old: {}),
                    ),
                    emit=lambda name, key, *args, **kwargs: emitted.append((name, key)),
                )
                packet.build(builder)
                self.assertEqual(embedding_calls[-1].get("codec"), codec)
        keys = [key for name, key in emitted if name == "packet/embedding"]
        self.assertEqual(len(set(keys)), 2)

    def test_config_pins_single_stage_codec_and_tied_embedding(self):
        candidate = config()
        candidate["quantization_config"].update(
            quant_method="orinfer_e8p_q2_int8",
            coefficients=[1],
            basis_tensor="orinfer.e8p_q2.basis",
            layout="u16-e8p",
            embedding="e8p_q2",
            vision="e8p_q2",
        )
        configuration(candidate, 4096)
        candidate["quantization_config"]["coefficients"] = [4, 1]
        with self.assertRaisesRegex(ValueError, "single-stage"):
            configuration(candidate, 4096)
        candidate["quantization_config"]["coefficients"] = [1]
        candidate["quantization_config"]["embedding"] = "e8p_q4"
        with self.assertRaises(ValueError):
            configuration(candidate, 4096)

    def test_dense_and_expert_banks_repack_without_expanding_q2(self):
        for shape in [(5, 32), (3, 5, 32)]:
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                codes = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
                scales = np.ones(shape[:-1], np.float16)
                shard = source / "layer.safetensors"
                save_file(
                    {"W.packed": codes, "W.scale": scales},
                    str(shard),
                    metadata={
                        "orin.layout.W.packed": "e8p-q2-row-u16",
                        "orin.layout.W.scale": "native_contiguous",
                    },
                )
                (source / "model.safetensors.index.json").write_text(
                    json.dumps(
                        dict(
                            metadata={
                                "orinfer.sha256": {
                                    shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()
                                }
                            },
                            weight_map={"W.packed": shard.name, "W.scale": shard.name},
                        )
                    )
                )
                weights = Weights(source, root / "model", packed_k=128)
                weights.immutable("W.packed")
                weights.immutable("W.scale")
                result = weights.array("W.packed")
                n, vectors = shape[-2:]
                restored = (
                    result.transpose(1, 0, 2).reshape(shape)
                    if len(shape) == 2
                    else result.transpose(0, 2, 1, 3).reshape(shape)
                )
                np.testing.assert_array_equal(restored, codes)
                np.testing.assert_array_equal(weights.array("W.scale"), scales)
                self.assertEqual(result.nbytes, codes.nbytes)
                self.assertEqual(weights.buffers["W.packed"]["dtype"], "u16")
                self.assertEqual(weights.buffers["W.packed"]["layout"], "e8p-q2-k128-u16")
                self.assertEqual(weights.finish(), codes.nbytes + scales.nbytes)


if __name__ == "__main__":
    unittest.main()
