import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

import numpy as np
from safetensors.numpy import save_file

from tools.model.safetensors_source import Source, validate_header


class OriginalSourceTests(unittest.TestCase):
    def test_modelscope_reads_pinned_range_and_rejects_full_download(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "index.json"
            path.write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
            source = Source(path, repo="modelscope:Qwen/model", revision="a" * 40)

            class Response:
                status = 206
                headers = {"Content-Range": "bytes 16-19/100"}

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def read(self, count):
                    self.read_count = count
                    return b"data"

            response = Response()
            with patch("urllib.request.urlopen", return_value=response) as request:
                self.assertEqual(source.read("weights.safetensors", 16, 4), b"data")
                req = request.call_args.args[0]
                url = urllib.parse.urlparse(req.full_url)
                self.assertEqual(url.netloc, "modelscope.cn")
                self.assertEqual(url.path, "/api/v1/models/Qwen/model/repo")
                self.assertEqual(urllib.parse.parse_qs(url.query)["Revision"], ["a" * 40])
                self.assertEqual(req.get_header("Range"), "bytes=16-19")
                self.assertEqual(response.read_count, 5)
                response.status = 200
                with (
                    patch("time.sleep"),
                    self.assertRaisesRegex(ValueError, "exact requested range"),
                ):
                    source.read("weights.safetensors", 16, 4)
                # The body of a full-file HTTP 200 must never be consumed.
                self.assertEqual(response.read_count, 5)

    def test_remote_header_uses_two_bounded_ranges_and_reuses_metadata_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            save_file({"x": np.arange(32, dtype=np.int8)}, str(root / "weights.safetensors"))
            raw = (root / "weights.safetensors").read_bytes()
            index = root / "index.json"
            index.write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
            source = Source(index, repo="Qwen/model", revision="0" * 40, cache=root / "headers")
            with patch.object(
                source, "read", side_effect=lambda _, offset, count: raw[offset : offset + count]
            ) as read:
                begin, header = source.header("weights.safetensors")
                self.assertEqual(read.call_count, 2)
                self.assertEqual(begin + header["x"]["data_offsets"][1], len(raw))
                self.assertEqual(source.header("weights.safetensors"), (begin, header))
                self.assertEqual(read.call_count, 2)
            cached = Source(index, repo="Qwen/model", revision="0" * 40, cache=root / "headers")
            with patch.object(
                cached, "read", side_effect=AssertionError("Metadata cache should avoid HTTP")
            ):
                self.assertEqual(cached.header("weights.safetensors"), (begin, header))

    def test_local_official_reader_rejects_truncated_payload_before_row_access(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            save_file({"x": np.arange(32, dtype=np.int8)}, str(root / "weights.safetensors"))
            shard = root / "weights.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1])
            index = root / "index.json"
            index.write_text(json.dumps({"weight_map": {"x": shard.name}}))
            source = Source(index, directory=root)
            with self.assertRaisesRegex(ValueError, "Invalid safetensors file"):
                source.tensor("x")

    def test_local_reads_exact_expert_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = np.arange(3 * 5 * 8, dtype=np.uint16).reshape(3, 5, 8)
            save_file({"bank": raw}, str(root / "shard.safetensors"))
            (root / "index.json").write_text(
                json.dumps({"weight_map": {"bank": "shard.safetensors"}})
            )
            s = Source(root / "index.json", directory=root, cache=root / "headers")
            self.assertEqual(s.rows("bank", 1, 1), raw[1].tobytes())
            self.assertEqual(s.rows("bank", 0, 3), raw.tobytes())
            for start, count in ((-1, 1), (3, 1), (0, 4), (0, 0)):
                with self.assertRaises(ValueError):
                    s.rows("bank", start, count)
            with self.assertRaises(EOFError):
                s.read("shard.safetensors", 10000, 1)
            with self.assertRaises(ValueError):
                s.read("../escape", 0, 1)

    def test_reject_invalid_header_extents(self):
        for t in (
            {"dtype": "BF16", "shape": [3, 8], "data_offsets": [0, 47]},
            {"dtype": "BF16", "shape": [3, -8], "data_offsets": [0, 48]},
            {"dtype": "BF16", "shape": [3, 8], "data_offsets": [4, 52]},
            {"dtype": "BAD", "shape": [3, 8], "data_offsets": [0, 48]},
        ):
            with self.assertRaises(ValueError):
                validate_header({"bad": t})

    def test_remote_requires_immutable_revision_and_safe_index_paths(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "index.json"
            path.write_text(json.dumps({"weight_map": {"x": "../escape"}}))
            with self.assertRaises(ValueError):
                Source(path, repo="Qwen/model", revision="0" * 40)
            path.write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
            with self.assertRaises(ValueError):
                Source(path, repo="Qwen/model", revision="main")


if __name__ == "__main__":
    unittest.main()
