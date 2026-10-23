"""Qwen3 per-head RMSNorm/RoPE and an exact two-row last-position head."""

import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.rows import row_count


@orin_jit
def qk_rope(capacity, dynamic_rows=True):
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m, 4096), T.float16),
        QNorm: T.Tensor((128,), T.float16),
        KNorm: T.Tensor((128,), T.float16),
        Cos: T.Tensor((capacity, 128), T.float16),
        Sin: T.Tensor((capacity, 128), T.float16),
        Q: T.Tensor((m, 2048), T.float16),
        K: T.Tensor((m, 1024), T.float16),
        V: T.Tensor((m, 1024), T.float16),
    ):
        with T.Kernel(m, 32, threads=128) as (row, head):
            values = T.alloc_fragment((128,), T.float32)
            squares = T.alloc_fragment((128,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            normalized = T.alloc_shared((128,), T.float16)
            for d in T.Parallel(128):
                values[d] = T.cast(X[row, head * 128 + d], T.float32)
                squares[d] = values[d] * values[d]
            if head < 24:
                T.reduce_sum(squares, total, dim=0)
                for d in T.Parallel(128):
                    v = T.cast(values[d] * T.rsqrt(total[0] / 128 + 1e-6), T.float16)
                    normalized[d] = v * T.if_then_else(head < 16, QNorm[d], KNorm[d])
                T.sync_threads()
                for d in T.Parallel(128):
                    a = T.cast(normalized[d] * Cos[row, d], T.float16)
                    b = T.cast(
                        normalized[(d + 64) % 128]
                        * T.if_then_else(d < 64, -Sin[row, d], Sin[row, d]),
                        T.float16,
                    )
                    value = T.cast(T.cast(a, T.float32) + T.cast(b, T.float32), T.float16)
                    if head < 16:
                        Q[row, head * 128 + d] = value
                    else:
                        K[row, (head - 16) * 128 + d] = value
            else:
                for d in T.Parallel(128):
                    V[row, (head - 24) * 128 + d] = X[row, head * 128 + d]

    return main


@orin_jit
def classifier(capacity, dynamic_rows=True):
    m = row_count(capacity, dynamic_rows)

    @T.prim_func
    def main(
        X: T.Tensor((m, 1024), T.float16),
        W: T.Tensor((2, 1024), T.float16),
        Y: T.Tensor((2,), T.float32),
    ):
        with T.Kernel(2, threads=256) as label:
            product = T.alloc_fragment((1024,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for d in T.Parallel(1024):
                product[d] = T.cast(X[m - 1, d], T.float32) * T.cast(W[label, d], T.float32)
            T.reduce_sum(product, total, dim=0)
            Y[label] = total[0]

    return main
