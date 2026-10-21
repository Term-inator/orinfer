"""Read-only machine snapshot; does not change clocks or power settings."""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {"unavailable": str(exc)}


def sample():
    gpu = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "meminfo": read("/proc/meminfo"),
        "nvidia_smi_xml": gpu.stdout,
        "background_containers": subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], check=True, capture_output=True, text=True
        ).stdout.splitlines(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(sample(), ensure_ascii=False, indent=2))
