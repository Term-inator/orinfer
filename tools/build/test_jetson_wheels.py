"""Release downloads must preserve the pinned wheel identity and atomic publication."""

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.build.jetson_wheels import download_asset


class WheelDownloadTests(unittest.TestCase):
    def test_verified_wheel_is_published_and_cache_works_offline(self):
        payload = b"verified wheel payload"
        asset = dict(
            filename="torch.whl",
            size=len(payload),
            url="https://github.com/iMMIQ/pytorch/releases/download/test/torch.whl",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        response = io.BytesIO(payload)
        response.status = 200
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch("urllib.request.urlopen", return_value=response):
                path = download_asset(asset, output)
                self.assertEqual(path.read_bytes(), payload)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                self.assertEqual(download_asset(asset, output), path)

    def test_bad_payloads_and_partial_responses_preserve_previous_file(self):
        payload = b"expected"
        asset = dict(
            filename="torch.whl",
            size=len(payload),
            url="https://github.com/iMMIQ/pytorch/releases/download/test/torch.whl",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for status, data in [
            (206, payload),
            (200, b"short"),
            (200, b"tampered"),
            (200, payload + b"extra"),
        ]:
            with self.subTest(status=status, data=data):
                with tempfile.TemporaryDirectory() as folder:
                    output = Path(folder)
                    target = output / "torch.whl"
                    target.write_bytes(b"previous wheel")
                    response = io.BytesIO(data)
                    response.status = status
                    with patch("urllib.request.urlopen", return_value=response):
                        with self.assertRaises(ValueError):
                            download_asset(asset, output)
                    self.assertEqual(target.read_bytes(), b"previous wheel")
                    self.assertEqual(list(output.iterdir()), [target])
