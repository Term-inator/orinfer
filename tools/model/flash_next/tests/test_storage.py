"""Native assembly preserves sliced weights and reference rounding contracts."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from tools.model.flash_next.storage import Weights, payload_hash
from tools.model.safetensors_source import local_header
from tools.quantization.e8p import basis


def piece(root, number, first, count, arrays):
    path = root / f"slice-{number}.safetensors"
    save_file(arrays, str(path))
    begin, header = local_header(path)
    return dict(
        path=path,
        begin=begin,
        header=header,
        keys={name: name for name in arrays},
        first=first,
        count=count,
    )


class NativeStorageTests(unittest.TestCase):
    def test_sliced_experts_and_dense_rows_keep_exact_payload_and_shared_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packed = np.arange(3 * 2 * 16, dtype=np.uint16).reshape(3, 1, 2, 16)
            scales = np.arange(6, dtype=np.float16).reshape(3, 2) + 1
            parts = [
                piece(
                    root,
                    i,
                    first,
                    count,
                    dict(
                        indices=packed[first : first + count],
                        scales=scales[first : first + count],
                        table=basis(),
                        signs=np.ones(128, np.int8),
                    ),
                )
                for i, (first, count) in enumerate(((0, 1), (1, 2)))
            ]
            dense = np.arange(15, dtype=np.int8).reshape(3, 5)
            dense_scale = np.array([0.25, 0.5, 1], np.float16)
            dense_parts = [
                piece(
                    root,
                    3 + i,
                    first,
                    count,
                    dict(
                        weight=dense[first : first + count],
                        scale=dense_scale[first : first + count],
                    ),
                )
                for i, (first, count) in enumerate(((0, 2), (2, 1)))
            ]
            source = SimpleNamespace(
                parts=dict(expert=parts, dense=dense_parts),
                shape=lambda name: (3, 2, 128) if name == "expert" else (3, 5),
            )
            weights = Weights(source, root / "native")
            weights.expert("expert", "blk.0.ffn_down_exps.weight")
            weights.expert("expert", "blk.1.ffn_down_exps.weight")
            self.assertEqual(len(weights.books), 1)
            first = weights.logical["blk.0.ffn_down_exps.weight"]
            second = weights.logical["blk.1.ffn_down_exps.weight"]
            self.assertEqual(first["table"], second["table"])
            with safe_open(
                weights.directory / weights.mapping[first["packed"]], framework="np"
            ) as stream:
                np.testing.assert_array_equal(stream.get_tensor(first["packed"]), packed)
                np.testing.assert_array_equal(stream.get_tensor(first["scale"]), scales)
            weights.dense("dense", "output.weight")
            for suffix, expected in (("weight", dense), ("scale", dense_scale)):
                name = weights.logical["output.weight"][suffix]
                path = weights.directory / weights.mapping[name]
                with safe_open(path, framework="np") as stream:
                    np.testing.assert_array_equal(stream.get_tensor(name), expected)
                begin, header = local_header(path)
                left, right = header[name]["data_offsets"]
                self.assertEqual(
                    weights.buffers[name]["data"]["sha256"],
                    payload_hash(path, begin + left, right - left),
                )

    def test_basis_disagreement_and_bad_k_major_layout_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = [
                piece(
                    root,
                    i,
                    i,
                    1,
                    dict(
                        indices=np.zeros((1, 1, 2, 16), np.uint16),
                        scales=np.ones((1, 2), np.float16),
                        table=basis(),
                        signs=np.full(128, 1 if i == 0 else -1, np.int8),
                    ),
                )
                for i in range(2)
            ]
            source = SimpleNamespace(parts=dict(expert=parts), shape=lambda _: (2, 2, 128))
            weights = Weights(source, root / "native")
            with self.assertRaisesRegex(ValueError, "disagree"):
                weights.expert("expert", "blk.0.ffn_down_exps.weight")
            parts[0]["header"]["indices"]["shape"] = [1, 2, 1, 16]
            with self.assertRaisesRegex(ValueError, "geometry"):
                weights.expert("expert", "blk.0.ffn_down_exps.weight")

    def test_zero_centered_qsa_norm_and_convolution_shape_match_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = np.array([-0.5, 0, 0.5, 1], np.float32)
            source = SimpleNamespace(
                parts=dict(norm=[]), shape=lambda _: (4,), tensor=lambda _: values.copy()
            )
            weights = Weights(source, root / "native")
            weights.original("norm", "blk.3.attn_q_norm.weight")
            name = weights.logical["blk.3.attn_q_norm.weight"]
            with safe_open(weights.directory / weights.mapping[name], framework="np") as stream:
                np.testing.assert_array_equal(stream.get_tensor(name), values + 1)
            source.shape = lambda _: (2, 1, 2)
            weights.original("norm", "blk.0.ssm_conv1d.weight")
            name = weights.logical["blk.0.ssm_conv1d.weight"]
            self.assertEqual(weights.buffers[name]["shape"], [2, 2])


if __name__ == "__main__":
    unittest.main()
