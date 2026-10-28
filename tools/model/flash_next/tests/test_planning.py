"""Draft construction shares the head and isolates its actual causal state."""

import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tools.model.flash_next.planning import Planner
from tools.model.flash_next.bindings import Bindings
from tools.model.flash_next.draft import DraftRows
from tools.model.flash_next.storage import Weights
from tools.model.flash_next.speculative import controls
from tools.model.compile import construct
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

    def test_visual_draft_shares_owner_coordinates_in_all_qsa_rotations(self):
        target = Planner(assembly("text"), 4096, None, vision=True)
        draft = Planner(assembly("mtp"), 4096, None, target=target)
        other = Planner(assembly("text"), 4096, None, vision=True)
        self.assertTrue(draft.has_vision)
        self.assertIs(draft.mrope_positions, target.mrope_positions)
        self.assertNotEqual(
            target.mrope_positions.untyped_storage()._cdata,
            other.mrope_positions.untyped_storage()._cdata,
        )
        # Exercise the target's QSA branch with the same shape-only fixture.
        # Layer 3 has target-owned KV/index state and the draft's QSA geometry.
        target.weights.update(
            {name.replace("blk.48.", "blk.3."): value for name, value in draft.weights.items()}
        )
        target.layer_ids = (3,)
        checked = set()
        for planner in (target, draft):
            plan = planner.trace(7)
            for (recipe, args), label in zip(plan["ops"], plan["labels"]):
                if label.startswith(("qsa-prepare-", "index-query-", "index-compress-")):
                    function, _ = construct(recipe.build, dynamic=True)
                    buffers = [
                        function.buffer_map[p] for p in function.params if p in function.buffer_map
                    ]
                    coordinates = next(b for b in buffers if b.name == "Coordinates")
                    self.assertIs(args[buffers.index(coordinates)], target.mrope_positions)
                    self.assertEqual(tuple(map(int, coordinates.shape)), (4096, 3))
                    for buffer, operand in zip(buffers, args):
                        self.assertEqual(
                            str(buffer.dtype), str(operand.dtype).removeprefix("torch.")
                        )
                    checked.add(label.split("-7")[0])
        self.assertEqual(checked, {"qsa-prepare", "index-query", "index-compress"})

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
        head = first.trace(128, all_logits=True)
        self.assertEqual(tuple(head["output"].shape), (128, 248320))

    def test_draft_row_emission_keeps_interfaces_and_affine_row_counts(self):
        target_assembly, draft_assembly = assembly("text"), assembly("mtp")
        target = Planner(target_assembly, 4096, None)
        draft = Planner(draft_assembly, 4096, None, target=target)
        emitted = {}

        class Compiler:
            def emit(self, old, name, pointers, rows, **kwargs):
                emitted[name] = dict(pointers=pointers, multiplier=kwargs.get("multiplier", 1))
                return dict(name=name), dict(name=name, capacity=rows)

        def artifact(binding, recipe, dynamic):
            primitive, _ = construct(recipe.build, dynamic=dynamic)
            return recipe.key, (primitive, None, None)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("tools.model.flash_next.bindings.source_identity", return_value="test"),
            patch.object(Bindings, "artifact", artifact),
        ):
            weights = Weights(SimpleNamespace(config=fixture()), Path(tmp))
            weights.buffers.update(target_assembly["buffers"] | draft_assembly["buffers"])
            weights.scopes.update({n: "weights" for n in weights.buffers})
            target_bindings = Bindings(target, weights, Path(tmp), Path(tmp) / "cache")
            target_bindings.compiler = Compiler()
            target_bindings.fixed("FlashDecode_Logits", "f32", [128, 248320])
            target_bindings.fixed("FlashDecode_Selected", "i32", [128, 2])
            child = Bindings(draft, weights, Path(tmp), Path(tmp) / "cache", shared=target_bindings)
            body, _ = DraftRows(child).build()
            self.assertIs(child.kernels, target_bindings.kernels)
            self.assertIs(child.buffers, target_bindings.buffers)
            self.assertEqual(len(child.groups["draft_rows"]["head"]), 9)
            self.assertEqual(len(child.groups["causal_rows"]["draft"]), 19)
            fc = [
                v
                for v in emitted.values()
                if any(
                    p["name"].endswith("W_fc_hidden.weight_value") for p in v["pointers"].values()
                )
            ]
            self.assertEqual(len(fc), 1)
            self.assertEqual(fc[0]["multiplier"], 4)
            experts = [
                v
                for v in emitted.values()
                if "Counts" in v["pointers"] and any(n in v["pointers"] for n in ("P", "W"))
            ]
            self.assertEqual(len(experts), 2)
            self.assertTrue(all(v["multiplier"] == 10 for v in experts))
            for v in experts:
                self.assertNotIn("PAddresses", v["pointers"])
                self.assertNotIn("WSAddresses", v["pointers"])
            private = [v for n, v in emitted.items() if n.startswith("causal_rows/draft/")]
            self.assertTrue(
                any(
                    v["pointers"].get("Step", {}).get("name") == "MtpPosition"
                    or any(p["name"] == "MtpPosition" for p in v["pointers"].values())
                    for v in private
                )
            )
            self.assertEqual(child.buffers["DraftRows_Hidden"]["shape"], [512, 4, 2560])
            self.assertEqual(child.scopes["State_48_pending"], "sequence")
            self.assertEqual(tuple(body["condition"].shape), (512, 4, 2560))
            self.assertFalse(any(n.startswith("Scratch") for n in child.buffers))
            for n in [1, 4096]:
                name = f"OriginalHidden{n}"
                target_bindings.fixed(name, "f16", [n, 10240])
                target_bindings.hidden[n] = dict(kind="buffer", name=name)
            mtp = controls(target_bindings)
            self.assertEqual(mtp["warm_plans"][0]["min_tokens"], 1)
            self.assertEqual(mtp["warm_plans"][0]["tokens"], 512)
            self.assertEqual(mtp["verification_plans"][0]["min_tokens"], 2)
            self.assertEqual(mtp["verification_plans"][0]["tokens"], 8)
            self.assertEqual(child.buffers["MtpHiddenRing"]["shape"], [4096, 10240])
            self.assertEqual(child.scopes["MtpHiddenRing"], "sequence")
            self.assertEqual(child.buffers["DraftLogits"]["shape"], [1, target.V])
            self.assertEqual(child.buffers["VerificationLogits"]["shape"], [8, target.V])
            for n in [1, 4096]:
                self.assertEqual(
                    emitted[f"mtp_capture_m{n}/body/k0"]["pointers"]["Hidden"]["name"],
                    f"OriginalHidden{n}",
                )
            self.assertEqual(
                emitted["causal_rows/warm/length"]["pointers"]["Length"]["name"], "MtpLength"
            )

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
