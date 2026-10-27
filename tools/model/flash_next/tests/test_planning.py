"""Draft construction shares the head and isolates its actual causal state."""

import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from tools.model.flash_next.planning import Planner
from tools.model.flash_next.tests.test_config import fixture


def assembly(component):
    result = dict(config=fixture(component), buffers={}, logical={})
    prefix = "Mtp_" if component == "mtp" else ""

    def value(name, shape, dtype, suffix="value"):
        physical = f"{prefix}W_{name}_{suffix}"
        result["buffers"][physical] = dict(shape=list(shape), dtype=dtype)
        return physical

    def original(name, shape, dtype):
        result["logical"][name] = value(name, shape, dtype)

    def dense(name, n, k):
        result["logical"][name] = dict(
            weight=value(name, (n, k), "i8", "weight"),
            scale=value(name, (n,), "f16", "scale"),
        )

    if component == "text":
        dense("output.weight", 248320, 2560)
        original("output_hc_norm.weight", (4, 2560), "f32")
        return result
    for name, width in [("embedding", 2560), ("hidden", 10240)]:
        original(f"pre_fc_norm_{name}.weight", (width,), "f32")
        original(f"fc_{name}.weight", (2560, 2560), "bf16")
    for name in ["output_hc_", "blk.48.hc_attn_", "blk.48.hc_ffn_"]:
        original(name + "norm.weight", (4, 2560), "f32")
        original(name + "down.weight", (320, 10240), "bf16")
        original(name + "up.weight", (10240, 320), "bf16")
        if name != "output_hc_":
            original(name + "inject.weight", (4, 10240), "bf16")
    for name, n, k in [
        ("attn_q", 12288, 2560),
        ("attn_k", 512, 2560),
        ("attn_v", 512, 2560),
        ("attn_output", 2560, 6144),
        ("index_qk", 640, 2560),
        ("ffn_gate_shexp", 640, 2560),
        ("ffn_up_shexp", 640, 2560),
        ("ffn_down_shexp", 2560, 640),
    ]:
        dense(f"blk.48.{name}.weight", n, k)
    for name, width in [("attn_q", 256), ("attn_k", 256), ("index_q", 128), ("index_k", 128)]:
        original(f"blk.48.{name}_norm.weight", (width,), "f32")
    original("blk.48.ffn_gate_inp.weight", (512, 2560), "bf16")
    original("blk.48.ffn_gate_inp_shexp.weight", (1, 2560), "bf16")
    for name, n, k in [("gate_up", 1280, 2560), ("down", 2560, 640)]:
        logical = f"blk.48.ffn_{name}_exps.weight"
        result["logical"][logical] = dict(
            packed=value(logical, (512, k // 128, n, 16), "u16", "packed"),
            scale=value(logical, (512, n), "f16", "scale"),
            signs=value(logical, (k,), "i8", "signs"),
            selector_table=value(logical, (1, 256, 1), "u32", "selector_table"),
            parity_xor=True,
        )
    return result


class PlanningTests(unittest.TestCase):
    def setUp(self):
        # Fake CPU storages exercise the real arithmetic planner without a CUDA
        # driver or any physical tensor/weight allocation.
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("empty", "zeros"):
            allocate = getattr(torch, name)

            def cpu(*args, _allocate=allocate, **kwargs):
                if str(kwargs.get("device", "")) == "cuda":
                    kwargs["device"] = "cpu"
                return _allocate(*args, **kwargs)

            self.stack.enter_context(patch("torch." + name, cpu))

    def test_real_draft_plans_share_head_but_not_target_or_other_owner_state(self):
        target = Planner(assembly("text"), 4096, None)
        first = Planner(assembly("mtp"), 4096, None, target=target)
        second = Planner(assembly("mtp"), 4096, None, target=target)
        self.assertEqual(first.layer_ids, (48,))
        self.assertEqual(
            set(first.states),
            {"48:key", "48:value", "48:key_scale", "48:value_scale", "48:index", "48:pending"},
        )
        self.assertIs(first.weights["output.weight"], target.weights["output.weight"])
        for name, state in first.states.items():
            self.assertNotEqual(
                state.untyped_storage()._cdata, second.states[name].untyped_storage()._cdata
            )
        self.assertEqual(first.prefix_divisors["48:index"], 4)
        self.assertNotEqual(
            first.position_gpu.untyped_storage()._cdata,
            target.position_gpu.untyped_storage()._cdata,
        )
        self.assertIn("MtpLength", first.tensors)
        self.assertNotEqual(
            first.weights["output_hc_norm.weight"].untyped_storage()._cdata,
            target.weights["output_hc_norm.weight"].untyped_storage()._cdata,
        )
        for rows in (1, 2, 7, 512):
            plan = first.trace(rows)
            self.assertEqual(tuple(plan["condition"].shape), (rows, 4, 2560))
            self.assertEqual(tuple(plan["embedding"].shape), (rows, 2560))
            self.assertEqual(tuple(plan["output"].shape), (1, 248320))
            self.assertFalse(any(label.startswith(("gdn-", "ple-")) for label in plan["labels"]))
            head = next(
                args
                for (_, args), label in zip(plan["ops"], plan["labels"])
                if label.startswith("head:")
            )
            self.assertIs(head[1], target.weights["output.weight"][0])
            self.assertIs(head[2], target.weights["output.weight"][1])
            self.assertIsNone(target.plans.get((rows, 4096, False)))
        self.assertEqual(first.position, 0)
        self.assertEqual(target.plans, {})

    def test_identity_or_context_errors_precede_fake_allocation(self):
        target = Planner(assembly("text"), 4096, None)
        with patch("torch.empty", side_effect=AssertionError("unexpected allocation")):
            with self.assertRaisesRegex(ValueError, "matching target"):
                Planner(assembly("mtp"), 4096, None)
            with self.assertRaisesRegex(ValueError, "matching target"):
                Planner(assembly("mtp"), 8192, None, target=target)
            changed = copy.deepcopy(assembly("mtp"))
            changed["config"]["quantization_config"]["source_revision"] = "b" * 40
            with self.assertRaisesRegex(ValueError, "identities"):
                Planner(changed, 4096, None, target=target)
            with self.assertRaisesRegex(ValueError, "Only MTP"):
                Planner(assembly("text"), 4096, None, target=target)


if __name__ == "__main__":
    unittest.main()
