"""Verify scalar FP32 QSA merge, short rows and changed-input graph replay."""

import argparse
from pathlib import Path

import torch

from kernels.model.qsa import sparse_merge
from kernels.model.rows import explicit_rows
from tools.model.compile import compile_function, construct
from tools.operators.common import configure, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    results = []
    for splits in (1, 8):
        function, passes = construct(lambda: sparse_merge(4096, splits), dynamic=True)
        kernel = compile_function(explicit_rows(function), pass_configs=passes)
        for rows in (1, 3, 17):
            maximum = torch.zeros(rows, 24, splits, device="cuda")
            denominator = torch.ones_like(maximum)
            parts = torch.randn(rows, 24, splits, 256, device="cuda")
            gate = torch.zeros(rows, 24, 256, device="cuda", dtype=torch.float16)
            output = torch.empty_like(gate)

            def expected():
                total = torch.zeros_like(parts[:, :, 0])
                for index in range(splits):
                    total += parts[:, :, index]
                return (total / splits / 2).half()

            def launch():
                kernel.adapter.func(
                    maximum,
                    denominator,
                    parts,
                    gate,
                    output,
                    rows,
                    stream=torch.cuda.current_stream().cuda_stream,
                )

            launch()
            torch.cuda.synchronize()
            if not torch.equal(output, expected()):
                raise AssertionError(f"QSA merge differs at {rows} rows, {splits} partitions")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            parts.copy_(torch.randn_like(parts))
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(output, expected()):
                raise AssertionError(
                    f"QSA merge replay differs at {rows} rows, {splits} partitions"
                )
            results.append(dict(rows=rows, splits=splits, exact=True, replay=True))
            print("QSA MERGE", rows, splits, "PASS", flush=True)
    write_json(args.output / "results.json", dict(complete=True, cases=results))


if __name__ == "__main__":
    main()
