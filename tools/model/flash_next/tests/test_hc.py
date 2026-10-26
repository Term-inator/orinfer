"""Typed HC consumer contracts keep fused and separate SiLU semantics distinct."""

import copy
from types import SimpleNamespace
import unittest

from tools.model.flash_next.hc import inventory


class HcContractsTests(unittest.TestCase):
    def fixture(self, fused=False):
        weight = "W_blk.0.hc_attn_down.weight_value"
        pointers = {
            "target/layer0/k1": dict(A=dict(name="X"), W=dict(name=weight), C=dict(name="Down")),
            "target/layer0/k2": (
                dict(
                    A=dict(name="Down"),
                    W=dict(name=weight.replace("down.", "up.")),
                    C=dict(name="Y"),
                )
                if fused
                else dict(Down=dict(name="Down"), Activated=dict(name="Y"))
            ),
        }
        compiler = SimpleNamespace(
            pointers=lambda kernel: pointers[kernel["name"]],
            buffers={
                weight: dict(dtype="bf16", shape=[320, 10240]),
                "X": dict(dtype="f16", shape=[1024 * 10240]),
                "Down": dict(dtype="f16", shape=[1024 * 320]),
            },
        )
        package = dict(
            config_signature=dict(text=dict(hc_lowrank=320, hc_count=4, hidden_size=2560)),
            kernels=[dict(name=n) for n in pointers],
            dynamic_batch_kernels=[dict(name="target/layer0/k1", capacity=1024)],
        )
        return package, compiler, pointers

    def test_fused_and_separate_outputs_are_classified_from_consumers(self):
        for fused in (False, True):
            package, compiler, _ = self.fixture(fused)
            groups, dynamic, rank, streams, hidden = inventory(package, compiler)
            self.assertEqual(list(groups), [fused])
            self.assertEqual([rank, streams, hidden], [320, 4, 2560])
            self.assertEqual(dynamic[groups[fused][0]["name"]]["capacity"], 1024)

    def test_old_row_variants_do_not_duplicate_canonical_bindings(self):
        package, compiler, _ = self.fixture()
        package["kernels"].append(dict(name="target/layer0/k1/rows/1-1024"))
        groups, *_ = inventory(package, compiler)
        self.assertEqual(len(groups[False]), 1)

    def test_missing_or_ambiguous_consumer_is_rejected(self):
        for ambiguous in (False, True):
            package, compiler, pointers = self.fixture()
            if ambiguous:
                pointers["target/layer0/k3"] = dict(
                    A=dict(name="Down"),
                    W=dict(name="W_blk.0.hc_attn_up.weight_value"),
                )
                package["kernels"].append(dict(name="target/layer0/k3"))
            else:
                pointers["target/layer0/k2"]["Down"] = dict(name="Other")
            with self.assertRaisesRegex(ValueError, "ambiguous SiLU"):
                inventory(package, compiler)

    def test_weight_geometry_and_slices_are_checked(self):
        package, compiler, pointers = self.fixture()
        bad = copy.deepcopy(compiler.buffers)
        bad[next(iter(bad))]["shape"] = [320, 2560]
        with self.assertRaisesRegex(ValueError, "geometry"):
            inventory(package, SimpleNamespace(pointers=compiler.pointers, buffers=bad))
        pointers["target/layer0/k1"]["A"]["offset"] = 2
        with self.assertRaisesRegex(ValueError, "whole native"):
            inventory(package, compiler)

    def test_activation_dtype_and_physical_capacity_are_checked(self):
        for field, value in (("dtype", "bf16"), ("shape", [320])):
            package, compiler, _ = self.fixture()
            compiler.buffers["Down"][field] = value
            with self.assertRaisesRegex(ValueError, "physical FP16"):
                inventory(package, compiler)


if __name__ == "__main__":
    unittest.main()
