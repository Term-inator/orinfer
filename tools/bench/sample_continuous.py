"""Host-only continuous read-only machine sampler (one sample/sec)."""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def read(p):
    try:
        return Path(p).read_text().strip()
    except (OSError, TypeError, ValueError) as e:
        return {"unavailable": str(e)}


def sample(phase):
    temp = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        temp[str(read(zone / "type"))] = read(zone / "temp")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,memory.free,utilization.gpu,temperature.gpu,"
            "power.draw,power.limit,clocks.current.sm,clocks.current.memory",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        phase=read(phase),
        nvidia_smi_csv=gpu.stdout,
        thermal_millicelsius=temp,
        meminfo=read("/proc/meminfo"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--phase", type=Path, required=True)
    ap.add_argument("--stop", type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = dict(
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        background_containers=subprocess.run(
            ["docker", "ps", "--format", "{{.Names}} {{.Image}}"], capture_output=True, text=True
        ).stdout.splitlines(),
    )
    args.output.with_suffix(".start.json").write_text(json.dumps(start, indent=2))
    with args.output.open("a", buffering=1) as f:
        while not args.stop.exists():
            f.write(json.dumps(sample(args.phase)) + "\n")
            time.sleep(1)


if __name__ == "__main__":
    main()
