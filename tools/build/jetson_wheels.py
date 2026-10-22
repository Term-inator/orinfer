"""Fetch hash-pinned Jetson wheels from our Torch build releases."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def download_asset(asset, output):
    filename = asset["filename"]
    if Path(filename).name != filename or not filename.endswith(".whl"):
        raise ValueError("Expected a wheel basename")
    target = output / filename
    if target.is_file():
        with target.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == asset["sha256"]:
                return target
    size = asset["size"]
    if size <= 0:
        raise ValueError("Expected a positive wheel size")
    request = urllib.request.Request(asset["url"])
    temporary = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise ValueError("Wheel download did not return a complete response")
            digest = hashlib.sha256()
            remaining = size
            with tempfile.NamedTemporaryFile(dir=output, delete=False) as destination:
                temporary = Path(destination.name)
                while remaining:
                    chunk = response.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Truncated wheel download")
                    digest.update(chunk)
                    destination.write(chunk)
                    remaining -= len(chunk)
                if response.read(1) or digest.hexdigest() != asset["sha256"]:
                    raise ValueError("Wheel payload does not match its pinned SHA256")
                destination.flush()
                os.fsync(destination.fileno())
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for asset in manifest["assets"]:
        print(download_asset(asset, args.output).name, flush=True)


if __name__ == "__main__":
    main()
