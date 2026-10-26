"""Typed HC consumer contracts keep fused and separate SiLU semantics distinct."""

import copy
from types import SimpleNamespace
import unittest

from tools.model.flash_next.hc import inventory, published_variants


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

    def test_multiple_down_layers_keep_the_weight_role(self):
        package, compiler, pointers = self.fixture()
        for source, target in (
            ("target/layer0/k1", "target/layer1/k1"),
            ("target/layer0/k2", "target/layer1/k2"),
        ):
            pointers[target] = copy.deepcopy(pointers[source])
            package["kernels"].append(dict(name=target))
        package["dynamic_batch_kernels"].append(dict(name="target/layer1/k1", capacity=1024))
        groups, *_ = inventory(package, compiler)
        self.assertEqual(len(groups[False]), 2)

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

    def test_injection_uses_four_output_checkpoint_contract(self):
        package, compiler, pointers = self.fixture()
        old = "W_blk.0.hc_attn_down.weight_value"
        new = old.replace("down.", "inject.")
        pointers["target/layer0/k1"]["W"]["name"] = new
        compiler.buffers[new] = dict(dtype="bf16", shape=[4, 10240])
        compiler.buffers["Down"]["shape"] = [1024 * 4]
        groups, *_ = inventory(package, compiler, "injection")
        self.assertEqual(len(groups[False]), 1)
        second = "target/layer1/k1"
        pointers[second] = copy.deepcopy(pointers["target/layer0/k1"])
        package["kernels"].append(dict(name=second))
        package["dynamic_batch_kernels"].append(dict(name=second, capacity=1024))
        groups, *_ = inventory(package, compiler, "injection")
        self.assertEqual(len(groups[False]), 2)
        self.assertEqual(pointers[groups[False][0]["name"]]["W"]["name"], new)
        compiler.buffers[new]["shape"] = [320, 10240]
        with self.assertRaisesRegex(ValueError, "geometry"):
            inventory(package, compiler, "injection")

    def test_injection_checks_output_capacity_and_stream_count(self):
        package, compiler, pointers = self.fixture()
        new = "W_blk.0.hc_attn_inject.weight_value"
        pointers["target/layer0/k1"]["W"]["name"] = new
        compiler.buffers[new] = dict(dtype="bf16", shape=[4, 10240])
        compiler.buffers["Down"]["shape"] = [1024 * 4 - 1]
        with self.assertRaisesRegex(ValueError, "physical FP16"):
            inventory(package, compiler, "injection")
        package["config_signature"]["text"]["hc_count"] = 3
        compiler.buffers[new]["shape"] = [3, 7680]
        compiler.buffers["Down"]["shape"] = [1024 * 3]
        with self.assertRaisesRegex(ValueError, "four stream gates"):
            inventory(package, compiler, "injection")

    def test_published_variants_keep_distinct_host_and_symbol_contracts(self):
        base = dict(
            name="hc/down",
            symbol="main",
            **{key: dict(sha256=key) for key in ("module", "source", "host_abi")},
        )
        duplicate = copy.deepcopy(base)
        duplicate["name"] += "/rows/1-8"
        changed = copy.deepcopy(duplicate)
        changed["name"] = "hc/down/rows/9-16"
        changed["host_abi"]["sha256"] = "other-host"
        symbol = copy.deepcopy(duplicate)
        symbol["name"] = "hc/down/rows/17-32"
        symbol["symbol"] = "other-symbol"
        compiler = SimpleNamespace(pointers=lambda kernel: dict(A="x", W="w", C="y"))
        self.assertEqual(
            published_variants(dict(kernels=[base, duplicate, changed, symbol]), compiler, base),
            [base, changed, symbol],
        )

    def test_published_variant_cannot_change_its_operand_views(self):
        base = dict(
            name="hc/down",
            symbol="main",
            **{key: dict(sha256=key) for key in ("module", "source", "host_abi")},
        )
        other = dict(base, name="hc/down/rows/1-8")
        compiler = SimpleNamespace(pointers=lambda kernel: dict(A=kernel["name"], W="w", C="y"))
        with self.assertRaisesRegex(ValueError, "operand views"):
            published_variants(dict(kernels=[base, other]), compiler, base)


if __name__ == "__main__":
    unittest.main()
