"""Build and execute native CUDA resource tests under the shared GPU wrapper."""

import argparse
import json
from pathlib import Path
import subprocess


TESTS = (
    "cuda::weight_cache::tests::cold_weights_reload_without_changing_addresses_and_poison_failed_uploads",
    "cuda::weight_upload::gpu_tests::streamed_bytes_and_reader_error_cleanup",
    "cuda::weight_upload::tests::staging_reuse_preserves_changed_inputs_and_recovers_read_errors",
    "cuda::experts::gpu_tests::sparse_ranges_preserve_neighbors_tails_and_eviction",
    "cuda::experts::compact_gpu_tests::compact_slots_preserve_all_banks_tails_transients_and_reuse",
    "cuda::virtual_memory::tests::vmm_shared_prefix_cow_and_graph_replay",
    "cuda::virtual_memory::tests::vmm_graph_growth_reset_and_last_token",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build = subprocess.run(
        [
            "cargo",
            "test",
            "--locked",
            "--offline",
            "-p",
            "orinfer-engine",
            "--lib",
            "--no-run",
            "--message-format=json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (args.output / "build.jsonl").write_text(build.stdout)
    artifacts = [json.loads(line) for line in build.stdout.splitlines()]
    executables = [
        entry["executable"]
        for entry in artifacts
        if entry.get("reason") == "compiler-artifact"
        and entry.get("executable")
        and entry.get("profile", {}).get("test")
    ]
    if len(executables) != 1:
        raise ValueError(f"Expected one freshly resolved test executable: {executables}")
    for name in TESTS:
        result = subprocess.run(
            [executables[0], name, "--exact", "--ignored", "--nocapture"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout, flush=True)
        (args.output / (name.rsplit("::", 1)[1] + ".log")).write_text(result.stdout + result.stderr)
        if "1 passed" not in result.stdout:
            raise ValueError(f"Test did not execute: {name}")
    (args.output / "result.json").write_text(
        json.dumps(dict(passed=True, executable=executables[0], tests=TESTS), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
