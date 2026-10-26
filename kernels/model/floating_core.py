"""Shared floating MMA loop with explicit operand and reduction contracts.

Operand conversion and scheduling are compile-time choices. Independent group
accumulators retain their original K16 order; consumers own the ordered sum.
"""

import tilelang.language as T


BF16_SOURCE = r"""
__device__ __forceinline__ float orinfer_round_bf16(float x) {
    unsigned short h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return __uint_as_float(static_cast<unsigned>(h) << 16);
}
__device__ __forceinline__ void orinfer_store_bf16(void* p, float x) {
    unsigned short h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
    *static_cast<unsigned short*>(p) = h;
}
"""


def rounded_float(value, dtype):
    """Explicit stream rounding, without a vectorized BF16 alias round-trip."""
    if dtype == "bfloat16":
        return T.call_pure_extern("float32", "orinfer_round_bf16", value)
    return T.cast(T.cast(value, dtype), T.float32)


def mma_loop(
    m,
    k,
    bm,
    bn,
    bk,
    *,
    dtype="bfloat16",
    stages=2,
    weight_left=False,
    reduction_groups=1,
    grouped_input=False,
    weight_first=False,
    cast_input=False,
):
    @T.macro
    def product(A, W, acc, by, bx, group):
        a = T.alloc_shared((bm, bk), dtype)
        w = T.alloc_shared((bn, bk), dtype)
        if reduction_groups != 1:
            total = T.alloc_fragment((bn, bm) if weight_left else (bm, bn), T.float32)
            T.clear(total)
        for branch in T.serial(reduction_groups):
            T.clear(acc)
            for kg in T.Pipelined(T.ceildiv(k // reduction_groups, bk), num_stages=stages):
                offset = branch * (k // reduction_groups) + kg * bk
                if weight_first:
                    T.copy(W[bx * bn, group * k + offset], w)
                if cast_input:
                    for i, j in T.Parallel(bm, bk, coalesced_width=T.int32(1)):
                        a[i, j] = T.if_then_else(by * bm + i < m, A[by * bm + i, offset + j], 0.0)
                elif grouped_input:
                    for i, j in T.Parallel(bm, bk):
                        a[i, j] = 0
                        if by * bm + i < m:
                            a[i, j] = A[by * bm + i, group, offset + j]
                else:
                    T.copy(A[by * bm, offset], a)
                if not weight_first:
                    T.copy(W[bx * bn, group * k + offset], w)
                if weight_left:
                    T.gemm(w, a, acc, transpose_B=True)
                else:
                    T.gemm(a, w, acc, transpose_B=True)
            if reduction_groups != 1:
                for i, j in T.Parallel(bm, bn):
                    if weight_left:
                        total[j, i] += acc[j, i]
                    else:
                        total[i, j] += acc[i, j]
        if reduction_groups != 1:
            T.copy(total, acc)

    return product


CACHED_WARP_PASSES = {
    "tl.disable_thread_storage_sync": True,
    "tl.disable_shared_memory_reuse": True,
}


def cached_warp_loop(rows, K, tile, bn, bk, stages, *, groups_per_block=1):
    """Cache live BF16 rows and pipeline packed weights into exact warp MMA.

    A is token/group/K and W is row-major paired BF16. Every warp retains the
    original K16 FP32 accumulator; callers own the output/reduction boundary.
    Groups within a CTA share barriers but have disjoint operands and sums.
    """
    from kernels.model.bf16_mma import SOURCE

    per = bn * 2

    @T.macro
    def product(A, W, acc, row, bx, first_group):
        T.import_source(SOURCE)
        tx = T.get_thread_binding()
        group = tx // per
        tid = tx % per
        lane = tid % 32
        warp = tid // 32
        live = T.alloc_shared((groups_per_block, tile, K), T.bfloat16)
        w = T.alloc_shared((groups_per_block, stages, bn, bk // 2), T.uint32)
        v = T.alloc_local((4,), T.uint32)
        T.clear(acc)
        for initial in T.unroll(min(stages - 1, K // bk)):
            for it in T.unroll(bn * bk // (per * 8)):
                index = (tid + it * per) * 4
                n = index // (bk // 2)
                col = index % (bk // 2)
                T.ptx_cp_async(
                    T.access_ptr(w[group, initial, n, col ^ ((n % 8) * 4)], "w", 4),
                    T.access_ptr(
                        W[
                            bx * bn + n,
                            (first_group + group) * (K // 2) + initial * (bk // 2) + col,
                        ],
                        "r",
                        4,
                    ),
                    4,
                )
            T.ptx_commit_group()
        for g, r, j in T.Parallel(groups_per_block, tile, K):
            live[g, r, j] = 0
            if row + r < rows:
                live[g, r, j] = A[row + r, first_group + g, j]
        for step in T.serial(K // bk):
            slot = step % stages
            if stages == 4:
                if step + 2 < K // bk:
                    T.ptx_wait_group(2)
                elif step + 1 < K // bk:
                    T.ptx_wait_group(1)
                else:
                    T.ptx_wait_group(0)
            elif stages == 3:
                if step + 1 < K // bk:
                    T.ptx_wait_group(1)
                else:
                    T.ptx_wait_group(0)
            else:
                T.ptx_wait_group(0)
            T.sync_threads()
            if step + stages - 1 < K // bk:
                nxt = (step + stages - 1) % stages
                for it in T.unroll(bn * bk // (per * 8)):
                    index = (tid + it * per) * 4
                    n = index // (bk // 2)
                    col = index % (bk // 2)
                    T.ptx_cp_async(
                        T.access_ptr(w[group, nxt, n, col ^ ((n % 8) * 4)], "w", 4),
                        T.access_ptr(
                            W[
                                bx * bn + n,
                                (first_group + group) * (K // 2)
                                + (step + stages - 1) * (bk // 2)
                                + col,
                            ],
                            "r",
                            4,
                        ),
                        4,
                    )
                T.ptx_commit_group()
            for kk in T.unroll(bk // 16):
                b0 = T.call_pure_extern(
                    "uint32",
                    "hc_raw_pair",
                    T.access_ptr(
                        live[group, (lane // 4) % tile, step * bk + kk * 16 + (lane % 4) * 2],
                        "r",
                        2,
                    ),
                )
                b1 = T.call_pure_extern(
                    "uint32",
                    "hc_raw_pair",
                    T.access_ptr(
                        live[group, (lane // 4) % tile, step * bk + kk * 16 + (lane % 4) * 2 + 8],
                        "r",
                        2,
                    ),
                )
                for rr in T.unroll(4):
                    n = warp * 16 + lane // 4 + (rr % 2) * 8
                    col = kk * 8 + lane % 4 + (rr // 2) * 4
                    v[rr] = w[group, slot, n, col ^ ((n % 8) * 4)]
                T.evaluate(
                    T.call_extern(
                        "void",
                        "hc_mma_bf16",
                        T.access_ptr(acc[0], "rw", 4),
                        v[0],
                        v[1],
                        v[2],
                        v[3],
                        b0,
                        b1,
                    )
                )
            T.sync_threads()

    return product
