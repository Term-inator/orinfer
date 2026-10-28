"""Pinned vision-only downloads retain BF16 tensors and reject corrupted reuse."""

import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import urllib.parse
from unittest.mock import patch

from tools.vision.source import fetch_vision, fetch_processor


class VisionSourceTests(unittest.TestCase):
    def test_pinned_processor_download_and_cached_overrides(self):
        valid = dict(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            size=dict(longest_edge=16777216, shortest_edge=65536),
        )
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root)
            file = destination / "preprocessor_config.json"
            with patch(
                "urllib.request.urlopen",
                return_value=contextlib.closing(io.BytesIO(json.dumps(valid).encode())),
            ) as request:
                self.assertEqual(
                    fetch_processor("modelscope:org/model", "a" * 40, destination), file
                )
                url = urllib.parse.urlparse(request.call_args.args[0])
                self.assertEqual(url.netloc, "modelscope.cn")
                query = urllib.parse.parse_qs(url.query)
                self.assertEqual(query["Revision"], ["a" * 40])
                self.assertEqual(query["FilePath"], ["preprocessor_config.json"])
            self.assertEqual(fetch_processor("org/model", "a" * 40, destination), file)
            valid["resample"] = 1
            file.write_text(json.dumps(valid))
            with self.assertRaisesRegex(ValueError, "arithmetic"):
                fetch_processor("org/model", "a" * 40, destination)

    def test_bounded_original_ranges_and_checked_reuse(self):
        config = dict(
            model_type="qwen4_exp",
            quantization_config=dict(source="org/model", source_revision="a" * 40),
        )
        payloads = {"first": bytes(range(24)), "second": bytes(range(8))}
        entries = {
            "model.visual.a": ("first", 0, dict(dtype="BF16", shape=[2, 2], data_offsets=[0, 8])),
            "model.visual.b": ("first", 0, dict(dtype="BF16", shape=[2, 2], data_offsets=[16, 24])),
            "model.visual.c": ("second", 0, dict(dtype="BF16", shape=[2, 2], data_offsets=[0, 8])),
        }

        class Original:
            def __init__(self, *args, **kwargs):
                self.weight_map = {n: item[0] for n, item in entries.items()}

            def tensor(self, name):
                return entries[name]

            def read(self, file, start, count):
                return payloads[file][start : start + count]

        @contextlib.contextmanager
        def index(*args, **kwargs):
            import io

            yield io.BytesIO(
                json.dumps(
                    dict(
                        patch_size=16,
                        temporal_patch_size=2,
                        merge_size=2,
                        image_mean=[0.5] * 3,
                        image_std=[0.5] * 3,
                        size=dict(longest_edge=16777216, shortest_edge=65536),
                    )
                ).encode()
            )

        with (
            tempfile.TemporaryDirectory() as root,
            patch("tools.vision.source.Source", Original),
            patch("urllib.request.urlopen", index),
        ):
            path = fetch_vision(config, Path(root))
            raw = path.read_bytes()
            (size,) = struct.unpack("<Q", raw[:8])
            header = json.loads(raw[8 : 8 + size])
            data = raw[8 + size :]
            self.assertEqual(
                data, payloads["first"][:8] + payloads["first"][16:] + payloads["second"]
            )
            self.assertEqual(header["model.visual.c"]["data_offsets"], [16, 24])
            published_index = json.loads((Path(root) / "model.safetensors.index.json").read_text())
            self.assertEqual(
                published_index["weight_map"], {name: "model.safetensors" for name in entries}
            )
            self.assertEqual(fetch_vision(config, Path(root)), path)
            other = json.loads(json.dumps(config))
            other["quantization_config"]["source_revision"] = "b" * 40
            with self.assertRaisesRegex(ValueError, "pinned source"):
                fetch_vision(other, Path(root))
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "pinned source"):
                fetch_vision(config, Path(root))


if __name__ == "__main__":
    unittest.main()
