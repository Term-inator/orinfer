"""Compile and execute real-width residual updates, tails and graph replay."""

import argparse
from pathlib import Path

import torch

from kernels.model.flash_next import residual_add
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
    for width in (3, 128, 2560, 10240):
        function, passes = construct(lambda: residual_add(4096, width), dynamic=True)
        kernel = compile_function(explicit_rows(function), pass_configs=passes)
        for rows in (1, 3, 17):
            x = torch.randn(rows, width, device="cuda", dtype=torch.float16)
            update = torch.randn_like(x)
            output = torch.empty_like(x)
            stream = torch.cuda.current_stream().cuda_stream
            kernel.adapter.func(x, update, output, rows, stream=stream)
            torch.cuda.synchronize()
            expected = (x.float() + update.float()).half()
            if not torch.equal(expected, output):
                raise AssertionError(f"Residual differs at {rows}x{width}")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel.adapter.func(
                    x, update, output, rows, stream=torch.cuda.current_stream().cuda_stream
                )
            x.copy_(torch.randn_like(x))
            expected = (x.float() + update.float()).half()
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(expected, output):
                raise AssertionError(f"Residual graph used stale input at {rows}x{width}")
            kernel.adapter.func(x, update, x, rows, stream=stream)
            torch.cuda.synchronize()
            if not torch.equal(expected, x):
                raise AssertionError(f"Aliased residual differs at {rows}x{width}")
            results.append(dict(rows=rows, width=width, exact=True, replay=True, alias=True))
            print("RESIDUAL", rows, width, "PASS", flush=True)
    write_json(args.output / "results.json", dict(complete=True, cases=results))


if __name__ == "__main__":
    main()
