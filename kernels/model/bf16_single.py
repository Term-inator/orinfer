"""Exact single-token BF16 projection with cached inputs and direct MMA operands.

Grouped columns remain in the original row-major weights; no read cache or
expanded representation is created. FP32 accumulates in the original K16 order.
"""

import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.floating_core import mma_loop, cached_warp_loop, CACHED_WARP_PASSES
from kernels.model.rows import row_count


@orin_jit(pass_configs=CACHED_WARP_PASSES)
def bf16_single(
    N,
    K,
    groups=1,
    block_n=32,
    block_k=256,
    stages=3,
    *,
    rows=1,
    dynamic_rows=False,
    row_major_grid=False,
):
    if (
        any(type(x) is not int or x <= 0 for x in (N, K, groups, block_n, block_k, stages))
        or groups not in (1, 4)
        or block_n not in (16, 32)
        or block_k not in (128, 256)
        or stages not in (2, 3, 4)
        or N % block_n
        or K % block_k
    ):
        raise ValueError("Unsupported single-token BF16 projection geometry")
    bn, bk = block_n, block_k
    threads = bn * 2
    if type(rows) is not int or not 1 <= rows <= 1024:
        raise ValueError("Unsupported BF16 decode rows")
    if type(row_major_grid) is not bool:
        raise ValueError("Invalid request tile ordering")
    rows = row_count(rows, dynamic_rows)

    product = cached_warp_loop(rows, K, 1, bn, bk, stages)

    @T.prim_func
    def main(
        A: T.Tensor((rows, groups, K), T.float16),
        Weight: T.Tensor((N, groups * K // 2), T.uint32),
        Output: T.Tensor((groups, rows, N), T.float32),
    ):
        with T.Kernel(
            (N // bn) * rows if row_major_grid else N // bn,
            groups,
            1 if row_major_grid else rows,
            threads=threads,
        ) as (block, group, lane):
            bx = block // rows if row_major_grid else block
            row = block % rows if row_major_grid else lane
            tid = T.get_thread_binding()
            lane, warp = tid % 32, tid // 32
            acc = T.alloc_local((4,), T.float32)
            product(A, Weight, acc, row, bx, group)
            if lane % 4 == 0:
                for rr in T.unroll(2):
                    Output[group, row, bx * bn + warp * 16 + lane // 4 + rr * 8] = acc[rr * 2]

    return main.with_attr("global_symbol", "bf16_single")


@orin_jit
def bf16_rows(
    N, K, groups=4, *, rows=1024, block_m=16, block_n=64, dynamic_rows=True, weight_left=True
):
    """Share weights across token rows, retaining each group's K16 FP32 sum.

    Operand orientation is selectable; neither group reduction nor output
    rounding moves across this primitive's boundary.
    """
    if groups not in (1, 4) or N % block_n or K % 128 or block_m not in (16, 32, 64):
        raise ValueError("Invalid grouped BF16 row tile")
    rows = row_count(rows, dynamic_rows)

    project = mma_loop(
        rows,
        K,
        block_m,
        block_n,
        128,
        weight_left=weight_left,
        grouped_input=True,
        weight_first=True,
    )

    @T.prim_func
    def bf16_rows_kernel(
        A: T.Tensor((rows, groups, K), T.float16),
        Weight: T.Tensor((N, groups * K), T.bfloat16),
        Output: T.Tensor((groups, rows, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(rows, block_m), N // block_n, groups, threads=128) as (
            by,
            bx,
            group,
        ):
            acc = T.alloc_fragment(
                (block_n, block_m) if weight_left else (block_m, block_n), T.float32
            )
            project(A, Weight, acc, by, bx, group)
            for i, j in T.Parallel(block_n, block_m):
                if by * block_m + j < rows:
                    Output[group, by * block_m + j, bx * block_n + i] = (
                        acc[i, j] if weight_left else acc[j, i]
                    )

    return bf16_rows_kernel


@orin_jit(pass_configs=CACHED_WARP_PASSES)
def bf16_short(M, N, K, *, block_m=1, block_n=32, block_k=64, stages=3, dynamic_rows=False):
    """Short FP16 output projection with exact BF16 MMA K16 accumulation."""
    if (
        any(type(x) is not int or x <= 0 for x in (M, N, K))
        or block_m not in (1, 2, 3, 4, 8)
        or block_n not in (16, 32)
        or block_k not in (64, 128, 256)
        or stages not in (2, 3, 4)
        or N % block_n
        or K % block_k
    ):
        raise ValueError("Unsupported short BF16 projection geometry")
    rows = row_count(M, dynamic_rows)
    bm, bn = block_m, block_n
    product = cached_warp_loop(rows, K, bm, bn, block_k, stages)

    @T.prim_func
    def main(
        A: T.Tensor((rows, 1, K), T.float16),
        W: T.Tensor((N, K // 2), T.uint32),
        C: T.Tensor((rows, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(rows, bm), N // bn, threads=bn * 2) as (by, bx):
            tid = T.get_thread_binding()
            lane, warp = tid % 32, tid // 32
            acc = T.alloc_local((4,), T.float32)
            product(A, W, acc, by * bm, bx, 0)
            for rr in T.unroll(4):
                row = (lane % 4) * 2 + rr % 2
                column = warp * 16 + lane // 4 + (rr // 2) * 8
                if row < bm and by * bm + row < rows:
                    C[by * bm + row, bx * bn + column] = T.cast(acc[rr], T.float16)

    return main
