"""SM120 Q2 decoder, gather, projection tails and changed-input graph checks."""

import argparse
from pathlib import Path

import numpy as np
import torch

from kernels.model.e8p_q4 import embedding, projection, unpack
from tools.model.compile import primitive
from tools.operators.common import configure, write_json
from tools.quantization.e8p import basis
from tools.quantization.vq import e8p_decode, e8p_selector_table, rotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    rng = np.random.default_rng(20261002)
    book = torch.from_numpy(e8p_selector_table(basis())).cuda()
    results = []

    def launch(kernel, *values):
        kernel.adapter.func(*values, stream=torch.cuda.current_stream().cuda_stream)

    codes = np.arange(65536, dtype=np.uint16).reshape(4096, 16)
    packed = codes.reshape(4096, 1, 16).transpose(1, 0, 2).copy()
    weight = torch.from_numpy(packed).cuda()
    decoded = torch.empty((4096, 128), device="cuda", dtype=torch.int8)
    decoder = primitive(unpack, 4096, 128, packed_k=128, codec="e8p_q2")
    launch(decoder, weight, book, decoded.view(torch.uint32))
    expected = e8p_decode(codes, basis()).reshape(4096, 128)
    np.testing.assert_array_equal(decoded.cpu().numpy(), expected)
    results.append(dict(kind="exhaustive_decoder", codes=65536))

    signs = rng.choice(np.array([-1, 1], np.int8), 128)
    scale = np.full(4096, 0.01, np.float16)
    ids = torch.tensor([0, 37, 4095], device="cuda", dtype=torch.int32)
    output = torch.empty((3, 128), device="cuda", dtype=torch.float16)
    gather = primitive(embedding, 128, 4096, 128, codec="e8p_q2")
    launch(
        gather,
        ids,
        weight,
        book,
        torch.from_numpy(scale).cuda(),
        torch.from_numpy(signs).cuda(),
        output,
    )
    reference = rotate(expected[[0, 37, 4095]].astype(np.float32), np.ones(128, np.int8))
    reference *= signs * scale[0].astype(np.float32)
    np.testing.assert_allclose(output.cpu().numpy(), reference.astype(np.float16), rtol=0, atol=0)
    results.append(dict(kind="embedding", selected_rows=3))

    n, k = 65, 256
    raw = rng.integers(0, 65536, (n, k // 8), dtype=np.uint16)
    packed = raw.reshape(n, k // 128, 16).transpose(1, 0, 2).copy()
    weight = torch.from_numpy(packed).cuda()
    values = torch.from_numpy(e8p_decode(raw, basis()).reshape(n, k).astype(np.float32)).cuda()
    ws = torch.full((n,), 0.01, device="cuda", dtype=torch.float16)
    for algorithm, bm, bn in [("tensor_core", 8, 64), ("dp4a", 4, 16)]:
        kernel = primitive(
            projection,
            129,
            n,
            k,
            codec="e8p_q2",
            packed_k=128,
            algorithm=algorithm,
            block_m=bm,
            block_n=bn,
            dynamic_rows=True,
            output_dtype="float32",
            output_stride=n + 3,
        )
        for rows in [1, 2, 3, 8, 17, 129]:
            x = torch.randint(-128, 128, (rows, k), device="cuda", dtype=torch.int8)
            scales = torch.full((rows,), 0.001, device="cuda", dtype=torch.float16)
            output = torch.full((rows + 1, n + 3), 91.0, device="cuda")

            def run():
                activation = x.view(torch.int32) if algorithm == "dp4a" else x
                launch(kernel, activation, weight, book, ws, scales, output[:rows], rows)

            def check():
                ref = (x.float() @ values.T) * ws.float()[None] * scales.float()[:, None]
                torch.testing.assert_close(output[:rows, :n], ref, rtol=1e-5, atol=1e-7)
                assert bool(torch.all(output[rows:] == 91)) and bool(torch.all(output[:, n:] == 91))

            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.copy_(torch.randint_like(x, -127, 128))
            graph.replay()
            check()
            results.append(dict(kind="projection", algorithm=algorithm, rows=rows, replay=True))
    from kernels.model.moe import integer_experts

    for counts, tile in [([1, 0], 16), ([7, 8], 16), ([17, 16], 32)]:
        rows, experts, n, k, capacity = sum(counts), 2, 129, 128, 64
        tiles = (capacity + tile - 1) // tile + experts
        offsets = np.cumsum(counts, dtype=np.int32) - counts
        items = [(e, r) for e, count in enumerate(counts) for r in range(0, count, tile)]
        te, tr = np.zeros(tiles, np.int32), np.zeros(tiles, np.int32)
        te[: len(items)], tr[: len(items)] = zip(*items)
        routing = [
            torch.from_numpy(np.asarray(v, dtype=np.int32)).cuda()
            for v in (counts, offsets, te, tr, [len(items)])
        ]
        codes = rng.integers(0, 65536, (experts, n, k // 8), dtype=np.uint16)
        packed = codes.reshape(experts, n, k // 128, 16).transpose(0, 2, 1, 3).copy()
        bank = torch.from_numpy(packed).cuda()
        ws = torch.full((experts, n), 0.001, device="cuda", dtype=torch.float16)
        scales = torch.full((rows,), 0.002, device="cuda", dtype=torch.float16)
        x = torch.from_numpy(rng.integers(-128, 128, (rows, k), dtype=np.int8)).cuda()
        decoded = e8p_decode(codes, basis()).reshape(experts, n, k).astype(np.int64)
        for stages in (1, 2):
            kernel = primitive(
                integer_experts,
                capacity,
                n,
                k,
                experts,
                1,
                codec="e8p_q2",
                block_m=tile,
                stages=stages,
                dynamic_rows=True,
            )
            output = torch.full((rows + 1, n), 91, device="cuda", dtype=torch.float16)

            def run():
                launch(kernel, x, bank, book, ws, scales, *routing, output[:-1], rows)

            def check():
                activation = x.cpu().numpy().astype(np.int64)
                expected = np.empty((rows, n), np.float16)
                for e, count in enumerate(counts):
                    first, last = int(offsets[e]), int(offsets[e]) + count
                    dot = activation[first:last] @ decoded[e].T
                    expected[first:last] = (
                        dot.astype(np.float32) * np.float32(ws[0, 0].item())
                    ) * np.float32(scales[0].item())
                np.testing.assert_array_equal(output[:-1].cpu().numpy(), expected)
                assert bool(torch.all(output[-1] == 91))

            run()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            x.neg_()
            graph.replay()
            check()
            results.append(dict(kind="routed_experts", counts=counts, stages=stages, replay=True))
    write_json(args.output / "report.json", dict(passed=True, checks=results))
    print("PASS", len(results), flush=True)


if __name__ == "__main__":
    main()
