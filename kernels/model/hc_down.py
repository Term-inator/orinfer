"""Parallel HC stream sums with an ordered FP32 reduction inside one CTA."""

import tilelang.language as T
from kernels.model.floating_core import cached_warp_loop, CACHED_WARP_PASSES
from kernels.model.rows import row_count
from tools.operators.common import orin_jit


@orin_jit(pass_configs=CACHED_WARP_PASSES)
def hc_down_cached(
    M,
    N,
    K,
    *,
    block_m=1,
    block_n=16,
    block_k=128,
    stages=3,
    silu=False,
    streams=4,
    dynamic_rows=False,
):
    """Original BF16 weights; four independent K16 sums, then ordered addition.

    K is one stream's width. Inputs and weights have the original four-stream
    physical layout. Temporary operands and FP32 partials live in CTA shared
    memory; the published output keeps its original FP16/SiLU boundary.
    """
    if (
        any(type(n) is not int or n <= 0 for n in (M, N, K))
        or streams != 4
        or block_m not in (1, 2)
        or block_n not in (16, 32)
        or block_k not in (128, 256)
        or stages not in (2, 3, 4)
        or N % block_n
        or K % block_k
        or type(silu) is not bool
    ):
        raise ValueError("Invalid cached HC stream geometry")
    shared = streams * (block_m * K * 2 + stages * block_n * block_k * 2 + block_n * block_m * 4)
    if shared > 96 * 1024:
        raise ValueError("Cached HC operands exceed the shared memory budget")
    m = row_count(M, dynamic_rows)
    per = block_n * 2
    product = cached_warp_loop(m, K, block_m, block_n, block_k, stages, groups_per_block=streams)

    @T.prim_func
    def main(
        A: T.Tensor((m, streams, K), T.float16),
        W: T.Tensor((N, streams * K // 2), T.uint32),
        C: T.Tensor((m, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(m, block_m), N // block_n, threads=streams * per) as (block, bx):
            tx = T.get_thread_binding()
            group, tid = tx // per, tx % per
            lane, warp = tid % 32, tid // 32
            acc = T.alloc_local((4,), T.float32)
            partial = T.alloc_shared((streams, block_n, block_m), T.float32)
            product(A, W, acc, block * block_m, bx, 0)
            for rr in T.unroll(4):
                n = warp * 16 + lane // 4 + (rr // 2) * 8
                r = (lane % 4) * 2 + rr % 2
                if r < block_m:
                    partial[group, n, r] = acc[rr]
            T.sync_threads()
            total = T.alloc_fragment((block_m, block_n), T.float32)
            T.clear(total)
            for g in T.serial(streams):
                for r, n in T.Parallel(block_m, block_n):
                    total[r, n] += partial[g, n, r]
            for r, n in T.Parallel(block_m, block_n):
                if block * block_m + r < m:
                    rounded = T.cast(total[r, n], T.float16)
                    if silu:
                        value = T.cast(
                            T.cast(T.cast(rounded, T.float32) / streams, T.float16), T.float32
                        )
                        C[block * block_m + r, bx * block_n + n] = value / (1 + T.exp(-value))
                    else:
                        C[block * block_m + r, bx * block_n + n] = rounded

    return main
