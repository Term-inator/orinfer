"""Exact expert address indirection, tails and changed-slot graph replay."""

import argparse
from pathlib import Path

import torch

from kernels.model.e8p_decode import e8p_decode_grouped
from kernels.model.expert_addresses import indirect_experts
from tools.model.compile import construct, compile_function
from tools.operators.common import configure, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    records = []
    for m, n, k in ((1, 65, 640), (17, 65, 640), (9, 1280, 2560), (9, 2560, 640)):
        e = 3
        tiles = e * ((m + 7) // 8)
        fn, passes = construct(lambda: e8p_decode_grouped(e * m, e, tiles, n, k, parity_xor=True))
        direct = compile_function(fn, pass_configs=passes)
        indirect = compile_function(indirect_experts(fn), pass_configs=passes)
        x = torch.randint(-128, 128, (e * m, k), device="cuda", dtype=torch.int8)
        packed = torch.randint(0, 65536, (e, k // 128, n, 16), device="cuda", dtype=torch.int32).to(
            torch.uint16
        )
        book = torch.randint(1, 2**31, (1, 256, 1), device="cuda", dtype=torch.int64).to(
            torch.uint32
        )
        ws = (torch.rand((e, n), device="cuda") / 512).half()
        sa = (torch.rand(e * m, device="cuda") / 512).half()
        patch = torch.zeros(1, device="cuda", dtype=torch.uint16)
        counts = torch.full((e,), m, device="cuda", dtype=torch.int32)
        offsets = torch.arange(e, device="cuda", dtype=torch.int32) * m
        te = torch.arange(e, device="cuda", dtype=torch.int32).repeat_interleave((m + 7) // 8)
        tr = torch.arange(0, m, 8, device="cuda", dtype=torch.int32).repeat(e)
        tc = torch.tensor([tiles], device="cuda", dtype=torch.int32)
        expected = torch.empty((e * m, n), device="cuda", dtype=torch.float16)
        output = torch.full((e * m + 1, n), 91.0, device="cuda", dtype=torch.float16)
        # Independently allocated matrices with reversed physical placement.
        slots = [(packed[i].clone(), ws[i].clone()) for i in reversed(range(e))]
        pa = torch.tensor(
            [slots[e - 1 - i][0].data_ptr() for i in range(e)], device="cuda", dtype=torch.uint64
        )
        wa = torch.tensor(
            [slots[e - 1 - i][1].data_ptr() for i in range(e)], device="cuda", dtype=torch.uint64
        )

        def reference(direct=direct):
            direct.adapter.func(
                x,
                packed,
                book,
                patch,
                ws,
                sa,
                counts,
                offsets,
                te,
                tr,
                tc,
                expected,
                stream=torch.cuda.current_stream().cuda_stream,
            )

        def run(indirect=indirect):
            indirect.adapter.func(
                x,
                pa,
                book,
                patch,
                wa,
                sa,
                counts,
                offsets,
                te,
                tr,
                tc,
                output[: e * m],
                stream=torch.cuda.current_stream().cuda_stream,
            )

        reference()
        run()
        torch.cuda.synchronize()
        assert torch.equal(expected, output[: e * m])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for change in ("activation", "address", "weight", "counts"):
            if change == "activation":
                x.neg_()
            elif change == "address":
                # Same graph and table addresses, different expert->physical-slot mapping.
                pa.copy_(pa.flip(0))
                wa.copy_(wa.flip(0))
                packed.copy_(packed.flip(0))
                ws.copy_(ws.flip(0))
            elif change == "weight":
                slots[0][0].fill_(123)
                packed[0].fill_(123)
            else:
                counts[1] = 0
                pa[1] = 0
                wa[1] = 0
            expected.fill_(91)
            reference()
            output.fill_(91)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(expected, output[: e * m]), (m, n, k, change)
            assert bool((output[-1] == 91).all())
        records.append(
            dict(
                rows=m,
                outputs=n,
                inputs=k,
                exact=True,
                changed_input_address_weight_counts_graph=True,
            )
        )
        write_json(args.output / "results.json", dict(complete=False, cases=records))
        print(records[-1], flush=True)
        del graph, direct, indirect
    write_json(args.output / "results.json", dict(complete=True, cases=records))


if __name__ == "__main__":
    main()
