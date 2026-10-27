"""A2 cache/embedding row movement, Engram dequantization and complex RoPE."""

import math

import tilelang.language as T


def gather_rows(rows, table_rows, width, dtype="bfloat16", index_dtype="int64"):
    if table_rows > 2147483647:
        raise ValueError("A2 row table exceeds signed 32-bit addressing")
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        table: T.Tensor((table_rows, width), dtype),
        indices: T.Tensor((rows,), index_dtype),
        out: T.Tensor((rows, width), dtype),
        errors: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            a = T.alloc_ub((block,), dtype)
            zeros = T.alloc_ub((max(1, block // 2),), "int16")
            flag = T.alloc_ub((8,), "int32")
            row = cid // blocks * 2 + vid
            column = cid % blocks * block
            if row < rows:
                index = indices[row]
                if dtype == "int8":
                    T.reinterpretcast(a, zeros, "int8_t")
                    T.tile.fill(zeros, 0)
                else:
                    T.tile.fill(a, 0.0)
                if index >= 0 and index < table_rows:
                    T.copy(table[T.cast(index, "int32"), column : column + block], a)
                T.copy(a, out[row, column : column + block])
                if column == 0:
                    flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                    T.copy(flag[:1], errors[row : row + 1])

    return main


def scatter_rows(rows, table_rows, width, dtype="bfloat16", index_dtype="int64"):
    """Destination indices must be unique. -1 means no write; other OOBs flag an error."""
    if table_rows > 2147483647:
        raise ValueError("A2 row table exceeds signed 32-bit addressing")
    block = math.gcd(width, 256)
    blocks = width // block

    @T.prim_func
    def main(
        values: T.Tensor((rows, width), dtype),
        indices: T.Tensor((rows,), index_dtype),
        table: T.Tensor((table_rows, width), dtype),
        errors: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(T.ceildiv(rows, 2) * blocks, is_npu=True) as (cid, vid):
            a = T.alloc_ub((block,), dtype)
            flag = T.alloc_ub((8,), "int32")
            row = cid // blocks * 2 + vid
            column = cid % blocks * block
            if row < rows:
                index = indices[row]
                if index >= 0 and index < table_rows:
                    T.copy(values[row, column : column + block], a)
                    T.copy(a, table[T.cast(index, "int32"), column : column + block])
                if column == 0:
                    flag[0] = T.if_then_else(index < -1 or index >= table_rows, 1, 0)
                    T.copy(flag[:1], errors[row : row + 1])

    return main


def dequant_rows(rows, width, group=32):
    @T.prim_func
    def main(
        values: T.Tensor((rows, width), "int8"),
        scales: T.Tensor((rows, width // group), "float32"),
        out: T.Tensor((rows, width), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows, 2), is_npu=True) as (cid, vid):
            q = T.alloc_ub((width,), "int8")
            half_values = T.alloc_ub((width,), "float16")
            a = T.alloc_ub((width,), "float32")
            scale = T.alloc_ub((width // group,), "float32")
            y = T.alloc_ub((width,), "bfloat16")
            row = cid * 2 + vid
            if row < rows:
                T.copy(values[row, :], q)
                T.copy(scales[row, :], scale)
                T.tile.cast(half_values, q, "CAST_NONE", width)
                T.tile.cast(a, half_values, "CAST_NONE", width)
                for g in T.serial(width // group):
                    T.tile.mul(
                        a[g * group : (g + 1) * group], a[g * group : (g + 1) * group], scale[g]
                    )
                T.tile.cast(y, a, "CAST_RINT", width)
                T.copy(y, out[row, :])

    return main


def rope(rows, heads, rotary=64):
    """Adjacent complex pairs. Caller supplies the model's YaRN/theta cache."""
    half = rotary // 2

    @T.prim_func
    def main(
        x: T.Tensor((rows, heads, rotary), "bfloat16"),
        cosine: T.Tensor((rows, half), "float32"),
        sine: T.Tensor((rows, half), "float32"),
        out: T.Tensor((rows, heads, rotary), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(rows * heads, 2), is_npu=True) as (cid, vid):
            xb = T.alloc_ub((rotary,), "bfloat16")
            a = T.alloc_ub((rotary,), "float32")
            even = T.alloc_ub((half,), "float32")
            odd = T.alloc_ub((half,), "float32")
            co = T.alloc_ub((half,), "float32")
            si = T.alloc_ub((half,), "float32")
            product = T.alloc_ub((half,), "float32")
            result = T.alloc_ub((2, half), "float32")
            offsets = T.alloc_ub((rotary,), "uint32")
            y = T.alloc_ub((rotary,), "bfloat16")
            index = cid * 2 + vid
            row = index // heads
            head = index % heads
            if row < rows:
                T.copy(x[row, head, :], xb)
                T.copy(cosine[row, :], co)
                T.copy(sine[row, :], si)
                T.tile.cast(a, xb, "CAST_NONE", rotary)
                T.tile.gather_mask(even, a, "P0101")
                T.tile.gather_mask(odd, a, "P1010")
                T.tile.mul(result[0, :], even, co)
                T.tile.mul(product, odd, si)
                T.tile.sub(result[0, :], result[0, :], product)
                T.tile.mul(result[1, :], odd, co)
                T.tile.mul(product, even, si)
                T.tile.add(result[1, :], result[1, :], product)
                for i in T.Parallel(rotary):
                    offsets[i] = T.cast((i % 2 * half + i // 2) * 4, "uint32")
                T.tile.gather(a, result, offsets, 0)
                T.tile.cast(y, a, "CAST_RINT", rotary)
                T.copy(y, out[row, head, :])

    return main
