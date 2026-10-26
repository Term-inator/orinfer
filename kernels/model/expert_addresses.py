"""Address compressed experts through caller-owned tables; preserve arithmetic."""

import math

from tilelang import tvm
from tvm import tirx as tir

SOURCE = r"""
__device__ __forceinline__ void* orin_expert_pointer(unsigned long long address) {
    return reinterpret_cast<void*>(address);
}
"""


def indirect_experts(function, operands=("P", "WS")):
    """Replace expert-major weights by uint64[experts] tables of matrix addresses.

    Bind each matrix after the guarded TileExpert load. Original cp.async
    accesses, MMA, tensor indices within an expert and reductions stay intact.
    """
    selected = {b.data: b for b in function.buffer_map.values() if b.name in operands}
    if {b.name for b in selected.values()} != set(operands):
        raise ValueError("Expert address lowering requires every declared weight operand")
    tables, aliases = {}, {}
    for data, b in selected.items():
        if (
            b.dtype not in ("uint16", "float16")
            or len(b.shape) < 2
            or (
                b.strides
                and any(int(v) != math.prod(b.shape[i + 1 :]) for i, v in enumerate(b.strides))
            )
        ):
            raise ValueError("Expert addresses require contiguous expert-major Q2 weights/scales")
        tables[data] = tir.decl_buffer((b.shape[0],), "uint64", name=b.name + "Addresses")
        pointer = tir.Var(b.name + "_expert", tvm.ir.PointerType(tvm.ir.PrimType(b.dtype)))
        aliases[data] = tir.decl_buffer(b.shape[1:], b.dtype, name=b.name + "Matrix", data=pointer)

    indices = []

    def inspect(node):
        if isinstance(node, tir.BufferLoad) and node.buffer.data in selected:
            indices.append(node.indices[0])

    tir.stmt_functor.post_order_visit(function.body, inspect)
    if (
        not indices
        or not isinstance(indices[0], tir.Var)
        or any(not i.same_as(indices[0]) for i in indices)
    ):
        raise ValueError("Expert weights must share one guarded expert index")
    expert = indices[0]
    bindings = []
    for data, alias in aliases.items():
        bindings.extend(
            [
                tir.Bind(
                    alias.data,
                    tir.call_extern(
                        "handle", "orin_expert_pointer", tir.BufferLoad(tables[data], [expert])
                    ),
                ),
                tir.DeclBuffer(alias),
            ]
        )
    bound = 0

    def replace(node):
        nonlocal bound
        if isinstance(node, tir.BufferLoad) and node.buffer.data in selected:
            return tir.BufferLoad(aliases[node.buffer.data], node.indices[1:], node.predicate)
        if isinstance(node, tir.SeqStmt):
            sequence = []
            for stmt in node.seq:
                sequence.append(stmt)
                if isinstance(stmt, tir.Bind) and stmt.var.same_as(expert):
                    sequence.extend(bindings)
                    bound += 1
            return tir.SeqStmt(sequence)
        if isinstance(node, tir.SBlock):
            annotations = dict(node.annotations)
            if node.name_hint == "tilelang_root":
                annotations["pragma_import_c"] = (
                    str(annotations.get("pragma_import_c", "")) + SOURCE
                )
            return tir.SBlock(
                node.iter_vars,
                [v for v in node.reads if v.buffer.data not in selected],
                [v for v in node.writes if v.buffer.data not in selected],
                node.name_hint,
                node.body,
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                annotations,
            )
        return None

    body = tir.stmt_functor.ir_transform(function.body, None, replace)
    if bound != 1:
        raise ValueError("Expected one guarded expert binding")
    buffers = {p: tables.get(b.data, b) for p, b in function.buffer_map.items()}
    used = set()
    tir.stmt_functor.post_order_visit(
        body, lambda n: used.add(n) if isinstance(n, tir.Var) else None
    )
    if set(selected) & used:
        raise ValueError("Expert address lowering left a direct weight access")
    return tir.PrimFunc(function.params, body, function.ret_type, buffers, function.attrs)
