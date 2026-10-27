"""DSv4.1 sqrt-softplus routing, correction bias and normalized top-6 weights."""

import tilelang.language as T

from tools.operators.ascend.vector_math import rsqrt_fp32


def route(rows, experts=384, selected=6, routed_scale=1.5):
    output_stride = (selected + 7) // 8 * 8

    @T.prim_func
    def main(
        logits: T.Tensor((rows, experts), "float32"),
        bias: T.Tensor((experts,), "float32"),
        indices: T.Tensor((rows, output_stride), "int32"),
        weights: T.Tensor((rows, output_stride), "float32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            x = T.alloc_ub((experts,), "float32")
            b = T.alloc_ub((experts,), "float32")
            score = T.alloc_ub((experts,), "float32")
            exponential = T.alloc_ub((experts,), "float32")
            tail = T.alloc_ub((experts,), "float32")
            correction = T.alloc_ub((experts,), "float32")
            root = T.alloc_ub((experts,), "float32")
            mask = T.alloc_ub((((experts + 255) // 256) * 32,), "uint8")
            corrected = T.alloc_ub((experts,), "float32")
            top = T.alloc_ub((2 * output_stride,), "float32")
            idx = T.alloc_ub((output_stride,), "float32")
            offsets = T.alloc_ub((output_stride,), "int32")
            unsigned_offsets = T.alloc_ub((output_stride,), "uint32")
            oi = T.alloc_ub((output_stride,), "int32")
            ow = T.alloc_ub((1, output_stride), "float32")
            lanes = T.alloc_ub((64,), "float32")
            padding = T.alloc_ub((32,), "uint8")
            zero = T.alloc_ub((1, output_stride), "float32")
            denominator = T.alloc_ub((1, 1), "float32")
            row = cid * 2 + vid
            if row < rows:
                T.copy(logits[row, :], x)
                T.copy(bias, b)
                T.tile.abs(exponential, x)
                T.tile.mul(exponential, exponential, -1.0)
                T.tile.exp(exponential, exponential)
                # Compensated log1p keeps negative-tail scores before selection.
                T.tile.add(tail, exponential, 1.0)
                T.tile.sub(correction, tail, 1.0)
                T.tile.sub(correction, exponential, correction)
                T.tile.div(correction, correction, tail)
                T.tile.ln(exponential, tail)
                T.tile.add(exponential, exponential, correction)
                T.tile.max(score, x, 0.0)
                T.tile.add(score, score, exponential)
                T.tile.max(score, score, 1e-30)
                rsqrt_fp32(score, root, correction)
                T.tile.mul(score, score, root)
                # Preserve tiny scores in the negative tail when 1+exp(x) rounds to 1.
                T.tile.mul(tail, x, 0.5)
                T.tile.exp(tail, tail)
                T.tile.compare(mask, x, -20.0, "LT")
                T.tile.select(score, mask, tail, score, "VSEL_TENSOR_TENSOR_MODE")
                T.tile.add(corrected, score, b)
                T.tile.topk(top, corrected, selected, experts)
                T.tile.gather_mask(idx, top, "P1010")
                for i in T.Parallel(output_stride - selected):
                    idx[selected + i] = 0.0
                T.tile.mul(idx, idx, 4.0)
                T.tile.cast(offsets, idx, "CAST_RINT", output_stride)
                T.reinterpretcast(unsigned_offsets, offsets, "uint32_t")
                T.tile.gather(ow, score, unsigned_offsets, 0)
                # Vector masking keeps every UB address 32-byte aligned and
                # avoids this compiler's scalar-store cast loss after aliasing.
                T.tile.createvecindex(lanes, 0.0)
                T.tile.fill(zero, 0.0)
                T.tile.compare(padding, lanes, float(selected), "LT")
                T.tile.select(ow, padding, ow, zero, "VSEL_TENSOR_TENSOR_MODE")
                T.reduce_sum(ow, denominator, dim=1)
                T.tile.div(ow, ow, denominator[0, 0])
                T.tile.mul(ow, ow, routed_scale)
                T.tile.div(idx, idx, 4.0)
                T.tile.cast(oi, idx, "CAST_RINT", output_stride)
                for i in T.Parallel(output_stride - selected):
                    oi[selected + i] = -1
                T.copy(oi, indices[row, :])
                T.copy(ow[0, :], weights[row, :])

    return main
