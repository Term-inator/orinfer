"""Validate production GDN convolution under the fixed SM120 compiler policy."""

import argparse
from pathlib import Path
import torch
from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
from kernels.operators.op17_gdn_gated_norm import _compile_gated_norm
from tools.model.row_tuning import launch
from tools.operators.common import error
from tools.model.compile import primitive
from tools.operators.common import configure, export_kernel, write_json
from tools.operators.op08_gdn_conv_prep import MODES, checked_case, state_checks


def check_norm(output):
    records = []
    for heads in (16, 32, 48):
        for dtype in ("float16", "float32"):
            kernel = primitive(_compile_gated_norm, None, heads=heads, x_dtype=dtype)
            export_kernel(kernel, output / f"norm-h{heads}-{dtype}")
            for rows in (1, 3, 17, 128):
                x = torch.randn((rows, heads, 128), device="cuda", dtype=getattr(torch, dtype))
                z = torch.randn_like(x, dtype=torch.float16)
                w = torch.randn((128,), device="cuda", dtype=torch.float16)
                y = torch.full((rows + 1, heads, 128), 42, device="cuda", dtype=torch.float16)

                def run():
                    launch(kernel, x, z, w, y[:rows])

                def check():
                    torch.cuda.synchronize()
                    xf, zf = x.float(), z.float()
                    expected = (
                        (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)) * w.float()
                    ) * torch.nn.functional.silu(zf)
                    result = error(y[:rows], expected)
                    assert result["finite"] and result["relative_l2"] < 0.001, result
                    assert torch.all(y[rows:] == 42), "output tail overwritten"
                    return result

                run()
                initial = check()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                x.mul_(-0.375)
                z.add_(0.5)
                w.mul_(0.75)
                y[:rows].fill_(float("nan"))
                graph.replay()
                changed = check()
                records.append(
                    dict(
                        heads=heads, dtype=dtype, rows=rows, initial=initial, changed_replay=changed
                    )
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    mode = MODES["native_prefill"]
    weight = torch.randn((10240, 4), device="cuda", dtype=torch.float16) * 0.1
    production = primitive(gdn_conv_prep, 1, 128, tile_tokens=16, dynamic_tokens=True)
    export_kernel(production, args.output / "production")
    records = []
    for rows in (1, 3, 17, 128):
        x = torch.randn((1, rows, 10240), device="cuda", dtype=torch.float16)
        history = torch.randn((1, 3, 10240), device="cuda", dtype=torch.float16)
        lengths = torch.full((1,), rows, device="cuda", dtype=torch.int32)
        positions = torch.full((1,), 91, device="cuda", dtype=torch.int32)
        record = checked_case(production, x, weight, history, lengths, positions, mode, 1)
        records.append(dict(rows=rows, result=record))
    dynamic = gdn_conv_prep(tile_tokens=16, **mode)
    states = state_checks(dynamic, weight, mode)
    write_json(
        args.output / "result.json",
        dict(passed=True, checks=records, states=states, gated_norm=check_norm(args.output)),
    )


if __name__ == "__main__":
    main()
