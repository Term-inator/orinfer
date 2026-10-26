"""The native plan must cover every routed row with the selected expert tile."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tools.model.flash_next.native import Model


class PlanWeights(dict):
    def __missing__(self, name):
        def tensor(shape, dtype=torch.float16):
            return torch.empty(shape, dtype=dtype)

        if "_exps." in name:
            n, k = (1280, 2560) if "gate_up" in name else (2560, 640)
            value = dict(
                packed=tensor((512, k // 128, n, 16), torch.uint16),
                scale=tensor((512, n)),
                signs=tensor((k,), torch.int8),
                selector_table=tensor((1, 256, 1), torch.uint32),
                parity_xor=True,
            )
        elif "hc_" in name:
            shape = (
                (4, 2560)
                if "norm" in name
                else (320, 10240)
                if "down" in name
                else (10240, 320)
                if "up" in name
                else (4, 10240)
            )
            value = tensor(shape, torch.float32 if "norm" in name else torch.bfloat16)
        elif name.endswith(("ssm_a_log", "ssm_dt.bias", "ssm_norm.weight", "ssm_conv1d.weight")):
            shape = (10240, 4) if "conv1d" in name else (128,) if "norm" in name else (48,)
            value = tensor(shape, torch.float32)
        else:
            suffix = name.split(".", 2)[-1] if name.startswith("blk.") else name
            n, k = {
                "attn_qkv.weight": (10240, 2560),
                "attn_gate.weight": (6144, 2560),
                "ssm_alpha.weight": (48, 2560),
                "ssm_beta.weight": (48, 2560),
                "ssm_out.weight": (2560, 6144),
                "ffn_gate_inp.weight": (512, 2560),
                "ffn_gate_inp_shexp.weight": (1, 2560),
                "ffn_gate_shexp.weight": (640, 2560),
                "ffn_up_shexp.weight": (640, 2560),
                "ffn_down_shexp.weight": (2560, 640),
                "output.weight": (248320, 2560),
            }[suffix]
            value = (
                tensor((n, k), torch.bfloat16)
                if "gate_inp" in suffix
                else (tensor((n, k), torch.int8), tensor((n,)))
            )
        self[name] = value
        return value


class ExpertPipelineTests(unittest.TestCase):
    def test_native_route_tiles_match_expert_body_and_prefill_arena(self):
        empty, zeros = torch.empty, torch.zeros

        def cpu(fn):
            def allocate(*args, **kwargs):
                if str(kwargs.get("device", "")) == "cuda":
                    kwargs["device"] = "cpu"
                return fn(*args, **kwargs)

            return allocate

        with FakeTensorMode(), patch("torch.empty", cpu(empty)), patch("torch.zeros", cpu(zeros)):
            model = Model.__new__(Model)
            model.capacity, model.position, model.layer_ids = 4096, 0, (0,)
            model.is_mtp, model.draft_vocab, model.batch_body = False, None, False
            model.decode_w8_optimized, model.prefill_workspace_rows = True, 0
            model.plans, model.workspaces, model.weights = {}, {}, PlanWeights()
            model.position_gpu, model.lengths, model.position_out = (
                torch.empty(1, dtype=torch.int32) for _ in range(3)
            )
            model.states = {"0:conv": torch.empty(1, 3, 10240), "0:gdn": torch.empty(48, 128, 128)}
            model.kernel = lambda key, build: SimpleNamespace(build=build)
            for rows in (4096, 31, 1):
                plan = model.plan(rows)
                expert = next(
                    op
                    for op, label in zip(plan["ops"], plan["labels"])
                    if label.startswith("expert-gu-")
                )
                with patch(
                    "tools.model.flash_next.native.integer_vq_grouped",
                    side_effect=lambda *args, **kw: kw,
                ):
                    tile = expert[0].build()["block_m"]
                self.assertIn(f"offsets-{tile}", plan["labels"])
                self.assertIn(f"tiles-{rows}-{tile}", plan["labels"])
                assignments = rows * 10
                bound = (assignments + tile - 1) // tile + min(512, assignments)
                self.assertGreaterEqual(expert[1][8].numel(), bound)
                if rows >= 256:
                    self.assertGreaterEqual(expert[1][8].untyped_storage().nbytes(), bound * 4)


if __name__ == "__main__":
    unittest.main()
