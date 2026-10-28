"""Compile a bounded TP8 V4.1 text execution plan for the native A2 runner.

The plan references original checkpoint tensors. It neither rewrites the
checkpoint nor expands permanent W4 banks to W8. Candidate filtering is exact
full coverage within the initial <=16384-position single-stream scope.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import math
import multiprocessing
from pathlib import Path
import shutil

import numpy as np
import torch

from tools.build.files import atomic_json, file_hash
from tools.operators.ascend import attention, compressor, dense, engram, indexer, memory
from tools.operators.ascend import model_ops, moe, routing, topk, vector
from tools.operators.ascend.build import compile_kernel, operand

DTYPE = {
    "bf16": ("bfloat16", 2),
    "i8": ("int8", 1),
    "i32": ("int32", 4),
    "i64": ("int64", 8),
    "f16": ("float16", 2),
    "f32": ("float32", 4),
    "u64": ("uint64", 8),
}


class Builder:
    def __init__(self, output, rank, capacity, kernels):
        self.output, self.rank, self.capacity = output, rank, capacity
        self.kernels = kernels
        self.buffers, self.by_name, self.types = [], {}, {}
        self.prefix = ""
        self.steps = []

    def buffer(self, name, shape, dtype, **source):
        size = math.prod(shape) * DTYPE[dtype][1]
        if name in self.by_name:
            assert (self.by_name[name]["bytes"], self.types[name]) == (size, dtype), name
            return name
        definition = {"name": name, "bytes": size, **source}
        self.buffers.append(definition)
        self.by_name[name], self.types[name] = definition, dtype
        return name

    def temp(self, name, shape, dtype="bf16"):
        return self.buffer(self.prefix + name, shape, dtype)

    def asset(self, name, array, dtype):
        path = self.output / "data" / (name + ".bin")
        if not path.exists():
            path.write_bytes(np.asarray(array).tobytes())
        return self.buffer(
            name,
            array.shape,
            dtype,
            data={"file": str(path.relative_to(self.output)), "sha256": file_hash(path)},
        )

    def ones(self, size):
        return self.asset(f"ones{size}", np.ones(size, dtype="<f4"), "f32")

    def matrix(self, name, dtype, rows, columns, first_row=0, first_column=0, float32=False):
        label = (
            f"w.{name}.{first_row}-{rows}.{first_column}-{columns}.{'f32' if float32 else dtype}"
        )
        return self.buffer(
            label,
            (rows, columns),
            "f32" if float32 else dtype,
            weight={
                "kind": "matrix",
                "name": name,
                "dtype": dtype,
                "first_row": first_row,
                "rows": rows,
                "first_column": first_column,
                "columns": columns,
                "float32": float32,
            },
        )

    def pad(self, name, shape):
        return self.buffer(
            "w." + name,
            shape,
            "f32",
            weight={"kind": "pad", "name": name, "elements": math.prod(shape)},
        )

    def kernel(self, module, name, arguments, tensors):
        specs = [operand(slot, dtype, shape, access) for slot, _, dtype, shape, access in tensors]
        key = json.dumps([module.__name__, name, arguments, specs], sort_keys=True)
        if key not in self.kernels:
            self.kernels[key] = {
                "module": module.__name__,
                "name": name,
                "arguments": arguments,
                "specs": specs,
                "placeholder": f"@kernel{len(self.kernels)}",
            }
        for _, buffer, dtype, shape, _ in tensors:
            assert (
                buffer in self.by_name
                and self.by_name[buffer]["bytes"] == math.prod(shape) * DTYPE[dtype][1]
            ), (name, buffer, shape)
        self.steps.append(
            {
                "op": "kernel",
                "manifest": self.kernels[key]["placeholder"],
                "buffers": [tensor[1] for tensor in tensors],
            }
        )

    def norm(self, x, weight, width, output="norm", rows=None):
        rows = self.rows if rows is None else rows
        out = self.temp(output, (rows, width))
        precise = self.temp(output + ".fp32", (rows, width), "f32")
        self.kernel(
            model_ops,
            "norm",
            {"rows": rows, "width": width},
            [
                ("x", x, "bf16", (rows, width), "read"),
                ("weight", weight, "f32", (width,), "read"),
                ("out", out, "bf16", (rows, width), "write"),
                ("precise", precise, "f32", (rows, width), "write"),
            ],
        )
        return out, precise

    def quant(self, x, width, output="quant", rows=None, fp16=False):
        rows = self.rows if rows is None else rows
        output = f"{output}.{width}.{rows}.{'fp16' if fp16 else 'fp32'}"
        out = self.temp(output, (rows, width), "i8")
        scales = self.temp(output + ".scale", (rows,), "f16" if fp16 else "f32")
        self.kernel(
            vector,
            "quantize",
            {"rows": rows, "width": width, "fp16_query_scale": fp16},
            [
                ("x", x, "bf16", (rows, width), "read"),
                ("out", out, "i8", (rows, width), "write"),
                ("scales", scales, "f16" if fp16 else "f32", (rows,), "write"),
            ],
        )
        return out, scales

    def norm_quant(self, x, weight, width, output="norm.quant", rows=None):
        rows = self.rows if rows is None else rows
        out = self.temp(output, (rows, width), "i8")
        scales = self.temp(output + ".scale", (rows,), "f32")
        self.kernel(
            model_ops,
            "norm_quant",
            {"rows": rows, "width": width},
            [
                ("x", x, "bf16", (rows, width), "read"),
                ("weight", weight, "f32", (width,), "read"),
                ("out", out, "i8", (rows, width), "write"),
                ("scales", scales, "f32", (rows,), "write"),
            ],
        )
        return out, scales

    def swiglu_quant(self, gate, up, width, output="swiglu.quant", rows=None):
        rows = self.rows if rows is None else rows
        out = self.temp(output, (rows, width), "i8")
        scales = self.temp(output + ".scale", (rows,), "f32")
        self.kernel(
            vector,
            "swiglu_quant",
            {"rows": rows, "width": width},
            [
                ("gate", gate, "bf16", (rows, width), "read"),
                ("up", up, "bf16", (rows, width), "read"),
                ("out", out, "i8", (rows, width), "write"),
                ("scales", scales, "f32", (rows,), "write"),
            ],
        )
        return out, scales

    def swiglu_packed_quant(self, packed, width, output="expert.act.quant", rows=None):
        rows = self.rows if rows is None else rows
        out = self.temp(output, (rows, width), "i8")
        scales = self.temp(output + ".scale", (rows,), "f32")
        self.kernel(
            model_ops,
            "swiglu_packed_quant",
            {"rows": rows, "width": width},
            [
                ("packed", packed, "bf16", (rows, 2 * width), "read"),
                ("out", out, "i8", (rows, width), "write"),
                ("scales", scales, "f32", (rows,), "write"),
            ],
        )
        return out, scales

    def projection(
        self,
        x,
        weight,
        n,
        k,
        output,
        integer=False,
        token_scale=None,
        channel_scale=None,
        rows=None,
    ):
        rows = self.rows if rows is None else rows
        out = self.temp(output, (rows, n))
        scratch = self.temp(
            f"cube.{integer}.{n}.{rows}", ((rows + 15) // 16 * 16, n), "i32" if integer else "f32"
        )
        self.kernel(
            dense,
            "projection",
            {
                "rows": rows,
                "output": n,
                "reduction": k,
                "integer": integer,
                "block_k": math.gcd(k, 512) if rows == 1 else 128,
            },
            [
                ("activation", x, "i8" if integer else "bf16", (rows, k), "read"),
                ("weight", weight, "i8" if integer else "bf16", (n, k), "read"),
                ("token_scale", token_scale or self.ones(rows), "f32", (rows,), "read"),
                ("channel_scale", channel_scale or self.ones(n), "f32", (n,), "read"),
                ("out", out, "bf16", (rows, n), "write"),
                (
                    "workspace",
                    scratch,
                    "i32" if integer else "f32",
                    ((rows + 15) // 16 * 16, n),
                    "scratch",
                ),
            ],
        )
        return out

    def w8(self, x, scales, prefix, n, k, output, first=0):
        weight = self.matrix(prefix + ".weight", "i8", n, k, first_row=first)
        scale = self.matrix(prefix + ".weight_scale", "f32", n, 1, first_row=first)
        return self.projection(x, weight, n, k, output, True, scales, scale)

    def fp32(self, x, weight, n, k, output, input_dtype="bf16", rows=None):
        rows = self.rows if rows is None else rows
        partials = self.temp(f"fp32.partial.{output}", (rows, n // 32, 8, 32), "f32")
        out = self.temp(output, (rows, n), "f32")
        self.kernel(
            dense,
            "fp32_partials",
            {"rows": rows, "output": n, "reduction": k, "activation_dtype": DTYPE[input_dtype][0]},
            [
                ("activation", x, input_dtype, (rows, k), "read"),
                ("weight", weight, "f32", (n, k), "read"),
                ("out", partials, "f32", (rows, n // 32, 8, 32), "write"),
            ],
        )
        self.kernel(
            dense,
            "fp32_reduce",
            {"rows": rows, "output": n},
            [
                ("partials", partials, "f32", (rows, n // 32, 8, 32), "read"),
                ("out", out, "f32", (rows, n), "write"),
            ],
        )
        return out

    def rope(self, x, heads, dimension, cosine, sine, output, inverse=False):
        out = self.temp(output, (self.rows, heads, dimension))
        self.kernel(
            model_ops,
            "partial_rope",
            {"rows": self.rows, "heads": heads, "dimension": dimension, "inverse": inverse},
            [
                ("x", x, "bf16", (self.rows, heads, dimension), "read"),
                ("cosine", cosine, "f32", (self.rows, 32), "read"),
                ("sine", sine, "f32", (self.rows, 32), "read"),
                ("out", out, "bf16", (self.rows, heads, dimension), "write"),
            ],
        )
        return out

    def scatter(self, x, ids, table, width, capacity, dtype="bf16"):
        flags = self.temp("scatter.flags", (self.rows,), "i32")
        self.kernel(
            memory,
            "scatter_rows",
            {
                "rows": self.rows,
                "table_rows": capacity,
                "width": width,
                "dtype": DTYPE[dtype][0],
                "index_dtype": "int32",
            },
            [
                ("values", x, dtype, (self.rows, width), "read"),
                ("indices", ids, "i32", (self.rows,), "read"),
                ("table", table, dtype, (capacity, width), "read_write"),
                ("errors", flags, "i32", (self.rows,), "write"),
            ],
        )

    def convert(self, x, width, source, destination, output):
        out = self.temp(output, (self.rows, width), destination)
        self.kernel(
            model_ops,
            "convert",
            {
                "rows": self.rows,
                "width": width,
                "source_dtype": DTYPE[source][0],
                "destination_dtype": DTYPE[destination][0],
            },
            [
                ("x", x, source, (self.rows, width), "read"),
                ("out", out, destination, (self.rows, width), "write"),
            ],
        )
        return out

    def select(self, scores, length, lengths, output="selected", rows=None):
        rows = self.rows if rows is None else rows
        sources = (length + 4095) // 4096
        candidates = self.temp(f"select.local.{length}.{rows}", (rows, sources, 1024), "f32")
        self.kernel(
            topk,
            "local",
            {"rows": rows, "length": length},
            [
                ("scores", scores, "f32", (rows, length), "read"),
                ("lengths", lengths, "i32", (rows,), "read"),
                ("out", candidates, "f32", (rows, sources, 1024), "write"),
            ],
        )
        while sources > 1:
            groups = (sources + 3) // 4
            merged = self.temp(
                f"select.merge.{length}.{rows}.{sources}", (rows, groups, 1024), "f32"
            )
            self.kernel(
                topk,
                "merge",
                {"rows": rows, "sources": sources},
                [
                    ("candidates", candidates, "f32", (rows, sources, 1024), "read"),
                    ("out", merged, "f32", (rows, groups, 1024), "write"),
                ],
            )
            candidates, sources = merged, groups
        indices = self.temp(output, (rows, 512), "i32")
        values = self.temp(output + ".values", (rows, 512), "f32")
        self.kernel(
            topk,
            "finalize",
            {"rows": rows},
            [
                ("candidates", candidates, "f32", (rows, 1024), "read"),
                ("indices", indices, "i32", (rows, 512), "write"),
                ("values", values, "f32", (rows, 512), "write"),
            ],
        )
        return indices, values

    def hc(self, current, incoming, layer, sublayer):
        rows = self.rows
        prefix = f"layers.{layer}.hc_{sublayer}"
        fn = self.pad(prefix + "_fn", (32, 20480))
        base = self.pad(prefix + "_base", (32,))
        scale = self.pad(prefix + "_scale", (8,))
        partials = self.temp("hc.partials", (rows, 8, 32), "f32")
        mixes = self.temp("hc.mixes", (rows, 32), "f32")
        next_mix = self.temp("pre.b" if incoming.endswith("pre.a") else "pre.a", (rows, 8), "f32")
        post = self.temp("hc.post", (rows, 8), "f32")
        combination = self.temp("hc.combination", (rows, 16), "f32")
        self.kernel(
            vector,
            "mhc_prenorm_partials",
            {"rows": rows, "width": 5120, "lanes": 8 if rows == 1 else 32},
            [
                ("residual", current, "bf16", (rows, 20480), "read"),
                ("fn", fn, "f32", (32, 20480), "read"),
                ("out", partials, "f32", (rows, 8, 32), "write"),
            ],
        )
        self.kernel(
            vector,
            "mhc_prenorm_reduce",
            {"rows": rows, "width": 5120},
            [
                ("partials", partials, "f32", (rows, 8, 32), "read"),
                ("out", mixes, "f32", (rows, 32), "write"),
            ],
        )
        self.kernel(
            vector,
            "mhc_split_sinkhorn",
            {"rows": rows},
            [
                ("mixes", mixes, "f32", (rows, 32), "read"),
                ("scale", scale, "f32", (8,), "read"),
                ("base", base, "f32", (32,), "read"),
                ("pre", next_mix, "f32", (rows, 8), "write"),
                ("post", post, "f32", (rows, 8), "write"),
                ("combination", combination, "f32", (rows, 16), "write"),
            ],
        )
        x = self.temp("collapsed", (rows, 5120))
        self.kernel(
            vector,
            "mhc_pre_apply",
            {"rows": rows, "width": 5120},
            [
                ("residual", current, "bf16", (rows, 4, 5120), "read"),
                ("mix", incoming, "f32", (rows, 8), "read"),
                ("out", x, "bf16", (rows, 5120), "write"),
            ],
        )
        return x, next_mix, post, combination

    def hc_post(self, x, residual, post, combination):
        out = self.temp(
            "residual.b" if residual.endswith("residual.a") else "residual.a", (self.rows, 4, 5120)
        )
        self.kernel(
            vector,
            "mhc_post",
            {"rows": self.rows, "width": 5120},
            [
                ("layer", x, "bf16", (self.rows, 5120), "read"),
                ("residual", residual, "bf16", (self.rows, 4, 5120), "read"),
                ("post", post, "f32", (self.rows, 8), "read"),
                ("combination", combination, "f32", (self.rows, 16), "read"),
                ("out", out, "bf16", (self.rows, 4, 5120), "write"),
            ],
        )
        return out

    def engram(self, hidden, layer):
        rows = self.rows
        prefix = f"layers.{layer}.engram"
        values = self.temp("engram.values", (rows * 24, 256), "i8")
        scales = self.temp("engram.scales", (rows * 24, 8), "f32")
        mask = self.temp("engram.mask", (rows,), "i8")
        self.steps.append(
            {"op": "lookup", "layer": layer, "values": values, "scales": scales, "mask": mask}
        )
        lookup = self.temp("engram.lookup", (rows, 6144))
        self.kernel(
            memory,
            "dequant_rows",
            {"rows": rows * 24, "width": 256},
            [
                ("values", values, "i8", (rows * 24, 256), "read"),
                ("scales", scales, "f32", (rows * 24, 8), "read"),
                ("out", lookup, "bf16", (rows * 24, 256), "write"),
            ],
        )
        projected = self.projection(
            lookup,
            self.matrix(prefix + ".wkv.weight", "bf16", 25600, 6144),
            25600,
            6144,
            "engram.projected",
        )
        key = self.temp("engram.key", (rows, 20480))
        value = self.temp("engram.value", (rows, 5120))
        self.kernel(
            model_ops,
            "split",
            {"rows": rows, "left": 20480, "right": 5120},
            [
                ("packed", projected, "bf16", (rows, 25600), "read"),
                ("first", key, "bf16", (rows, 20480), "write"),
                ("second", value, "bf16", (rows, 5120), "write"),
            ],
        )
        original = self.temp("engram.original", (rows, 4, 5120), "f32")
        self.kernel(
            engram,
            "rotate_blocks",
            {"rows": rows, "width": 5120, "block": 32},
            [
                ("hidden", hidden, "bf16", (rows, 4, 5120), "read"),
                ("rotation", "rotation", "f32", (32, 32), "read"),
                ("out", original, "f32", (rows, 4, 5120), "write"),
            ],
        )
        channel = self.buffer(
            "w." + prefix + ".channel",
            (4, 5120),
            "f32",
            weight={
                "kind": "channel_product",
                "left": prefix + ".q_weight",
                "right": prefix + ".k_weight",
            },
        )
        out = self.temp(
            "residual.b" if hidden.endswith("residual.a") else "residual.a", (rows, 4, 5120)
        )
        self.kernel(
            engram,
            "gate",
            {"rows": rows, "width": 5120},
            [
                ("hidden", hidden, "bf16", (rows, 4, 5120), "read"),
                ("original", original, "f32", (rows, 4, 5120), "read"),
                ("key", key, "bf16", (rows, 4, 5120), "read"),
                ("value", value, "bf16", (rows, 5120), "read"),
                ("channel_weight", channel, "f32", (4, 5120), "read"),
                ("token_mask", mask, "i8", (rows,), "read"),
                ("out", out, "bf16", (rows, 4, 5120), "write"),
            ],
        )
        return out

    def attention(self, x, layer, coordinates, reset):
        rows, capacity = self.rows, self.capacity
        prefix = f"layers.{layer}.attn"
        compressed = layer >= 2
        cosine = coordinates["long_cos" if compressed else "base_cos"]
        sine = coordinates["long_sin" if compressed else "base_sin"]
        quant, scales = self.quant(x, 5120)
        qa = self.w8(quant, scales, prefix + ".wq_a", 1280, 5120, "q.a")
        q8, qs = self.norm_quant(
            qa, self.matrix(prefix + ".q_norm.weight", "f32", 1280, 1), 1280, "q.norm.quant"
        )
        query = self.w8(q8, qs, prefix + ".wq_b", 4096, 1280, "q.b", first=self.rank * 4096)
        query = self.rope(query, 8, 512, cosine, sine, "q.rope")
        kv = self.w8(quant, scales, prefix + ".wkv", 512, 5120, "kv")
        kv, _ = self.norm(
            kv, self.matrix(prefix + ".kv_norm.weight", "f32", 512, 1), 512, "kv.normalized"
        )
        kv = self.rope(kv, 1, 512, cosine, sine, "kv.rope")
        swa = self.buffer(f"cache.swa.{layer}", (capacity, 512), "bf16")
        self.scatter(kv, coordinates["slots"], swa, 512, capacity)
        source = max((s for s in (2, 8, 14, 20) if s <= layer), default=2)
        ratio = 2 if source < 20 else 1
        compressed_capacity = capacity // ratio
        long_kv = self.buffer(f"cache.long.{source}", (compressed_capacity, 512), "bf16")
        key_cache = self.buffer(f"cache.key.{source}", (compressed_capacity, 128), "i8")
        key_scale = self.buffer(f"cache.key_scale.{source}", (compressed_capacity,), "f16")
        if layer in (2, 8, 14, 20):
            if ratio == 2:
                projected = []
                for kind in ("wkv", "wgate"):
                    weight = self.matrix(
                        prefix + ".compressor." + kind + ".weight", "f32", 512, 5120
                    )
                    projected.append(self.fp32(x, weight, 512, 5120, "compress." + kind))
                ring = self.buffer(f"cache.ring.{layer}", (2, 32, 1024), "f32")
                reset.append(ring)
                pooled = self.temp("compress.pooled", (rows, 512))
                errors = self.temp("compress.errors", (rows,), "i32")
                self.kernel(
                    compressor,
                    "pool",
                    {"tokens": rows, "pages": 2},
                    [
                        ("kv", projected[0], "f32", (rows, 512), "read"),
                        ("scores", projected[1], "f32", (rows, 512), "read"),
                        ("ring", ring, "f32", (2, 32, 1024), "read"),
                        ("metadata", coordinates["ring"], "i32", (rows, 8), "read"),
                        ("out", pooled, "bf16", (rows, 512), "write"),
                        ("errors", errors, "i32", (rows,), "write"),
                    ],
                )
                self.kernel(
                    compressor,
                    "commit",
                    {"tokens": rows, "pages": 2},
                    [
                        ("kv", projected[0], "f32", (rows, 512), "read"),
                        ("scores", projected[1], "f32", (rows, 512), "read"),
                        ("metadata", coordinates["ring"], "i32", (rows, 8), "read"),
                        ("ring", ring, "f32", (2, 32, 1024), "read_write"),
                        ("errors", errors, "i32", (rows,), "write"),
                    ],
                )
                source_cos, source_sin = coordinates["source_cos"], coordinates["source_sin"]
                slots = coordinates["c2_slots"]
            else:
                pooled = self.projection(
                    x,
                    self.matrix(prefix + ".compressor.wkv.weight", "bf16", 512, 5120),
                    512,
                    5120,
                    "compress.pooled",
                )
                source_cos, source_sin, slots = cosine, sine, coordinates["slots"]
            latent, _ = self.norm(
                pooled,
                self.matrix(prefix + ".compressor.norm.weight", "f32", 512, 1),
                512,
                "compress.normalized",
            )
            key = self.projection(
                latent,
                self.matrix(prefix + ".indexer.wk.weight", "bf16", 128, 512),
                128,
                512,
                "index.key",
            )
            key, _ = self.norm(
                key,
                self.matrix(prefix + ".indexer.k_norm.weight", "f32", 128, 1),
                128,
                "index.key.normalized",
            )
            key = self.rope(key, 1, 128, source_cos, source_sin, "index.key.rope")
            key8, key_scaling = self.quant(key, 128, output="index.key.quant")
            key_scaling = self.convert(key_scaling, 1, "f32", "f16", "index.key.scale")
            self.scatter(key8, slots, key_cache, 128, compressed_capacity, "i8")
            self.scatter(key_scaling, slots, key_scale, 1, compressed_capacity, "f16")
            latent = self.rope(latent, 1, 512, source_cos, source_sin, "compress.rope")
            self.scatter(latent, slots, long_kv, 512, compressed_capacity)
        selected = self.temp("selected", (rows, 512), "i32")
        if layer in (2, 8, 14, 20, 24, 28, 32, 36):
            iq = self.w8(q8, qs, prefix + ".indexer.wq_b", 4096, 1280, "index.query")
            iq = self.rope(iq, 32, 128, cosine, sine, "index.query.rope")
            iq8, iqs = self.quant(iq, 128, output="index.query.quant", rows=rows * 32, fp16=True)
            iw = self.projection(
                x,
                self.matrix(prefix + ".indexer.weights_proj.weight", "bf16", 32, 5120),
                32,
                5120,
                "index.weights",
            )
            scores = self.temp(
                f"index.scores.{compressed_capacity}", (rows, compressed_capacity), "f32"
            )
            self.kernel(
                indexer,
                "quantized_logits",
                {"rows": rows, "length": compressed_capacity, "table_rows": compressed_capacity},
                [
                    ("query", iq8, "i8", (rows, 32, 128), "read"),
                    ("query_scale", iqs, "f16", (rows, 32), "read"),
                    ("keys", key_cache, "i8", (compressed_capacity, 128), "read"),
                    ("key_scale", key_scale, "f16", (compressed_capacity,), "read"),
                    ("weights", iw, "bf16", (rows, 32), "read"),
                    (
                        "indices",
                        coordinates[f"c{ratio}_indices"],
                        "i32",
                        (rows, compressed_capacity),
                        "read",
                    ),
                    ("out", scores, "f32", (rows, compressed_capacity), "write"),
                ],
            )
            selected, _ = self.select(scores, compressed_capacity, coordinates[f"c{ratio}_lengths"])
        count = 640 if compressed else 128
        gathered = self.temp(f"attention.gathered.{count}", (rows, count, 512))
        mask = self.temp(f"attention.mask.{count}", (rows, count), "i32")
        self.kernel(
            model_ops,
            "gather_attention",
            {
                "rows": rows,
                "capacity": capacity,
                "compressed_capacity": compressed_capacity,
                "compressed": compressed,
            },
            [
                ("swa", swa, "bf16", (capacity, 512), "read"),
                ("long_kv", long_kv, "bf16", (compressed_capacity, 512), "read"),
                ("swa_indices", coordinates["swa_indices"], "i32", (rows, 128), "read"),
                ("long_indices", selected, "i32", (rows, 512), "read"),
                ("out", gathered, "bf16", (rows, count, 512), "write"),
                ("mask", mask, "i32", (rows, count), "write"),
            ],
        )
        attended = self.temp("attention.output", (rows, 8, 512))
        sink = self.matrix(prefix + ".attn_sink", "f32", 8, 1, first_row=self.rank * 8)
        self.kernel(
            attention,
            "sparse_attention",
            {"rows": rows, "heads": 8, "selected": count},
            [
                ("query", query, "bf16", (rows, 8, 512), "read"),
                ("gathered_kv", gathered, "bf16", (rows, count, 512), "read"),
                ("indices", mask, "i32", (rows, count), "read"),
                ("sink", sink, "f32", (8,), "read"),
                ("out", attended, "bf16", (rows, 8, 512), "write"),
            ],
        )
        attended = self.rope(attended, 8, 512, cosine, sine, "attention.inverse_rope", inverse=True)
        oa = self.projection(
            attended,
            self.matrix(prefix + ".wo_a.weight", "bf16", 1024, 4096, first_row=self.rank * 1024),
            1024,
            4096,
            "o.a",
        )
        output = self.projection(
            oa,
            self.matrix(prefix + ".wo_b.weight", "bf16", 5120, 1024, first_column=self.rank * 1024),
            5120,
            1024,
            "o.b",
        )
        reduced = self.temp("attention.reduced", (rows, 5120))
        self.steps.append(
            {
                "op": "reduce",
                "source": output,
                "destination": reduced,
                "elements": rows * 5120,
                "fp32": False,
            }
        )
        return reduced

    def ffn(self, x, precise, layer):
        rows = self.rows
        prefix = f"layers.{layer}.ffn"
        router = self.fp32(
            precise,
            self.matrix(prefix + ".gate.weight", "bf16", 384, 5120, float32=True),
            384,
            5120,
            "router.logits",
            "f32",
        )
        indices = self.temp("router.indices", (rows, 8), "i32")
        weights = self.temp("router.weights", (rows, 8), "f32")
        bias = self.matrix(prefix + ".gate.bias", "f32", 384, 1)
        self.kernel(
            routing,
            "route",
            {"rows": rows},
            [
                ("logits", router, "f32", (rows, 384), "read"),
                ("bias", bias, "f32", (384,), "read"),
                ("indices", indices, "i32", (rows, 8), "write"),
                ("weights", weights, "f32", (rows, 8), "write"),
            ],
        )
        quant, scale = self.quant(x, 5120)
        gate = self.w8(quant, scale, prefix + ".shared_experts.w1", 2304, 5120, "shared.gate")
        up = self.w8(quant, scale, prefix + ".shared_experts.w3", 2304, 5120, "shared.up")
        act8, act_scale = self.swiglu_quant(gate, up, 2304, "shared.act.quant")
        shared = self.w8(
            act8, act_scale, prefix + ".shared_experts.w2", 5120, 2304, "shared.output"
        )
        expert_rows = rows * 6
        gather = self.temp("expert.gather", (expert_rows,), "i32")
        restore = self.temp("expert.restore", (rows, 8), "i32")
        counts = self.temp(f"L{layer}.counts", (48,), "i64")
        route_errors = self.temp("route.errors", (rows + 1,), "i32")
        up_name, down_name = f"{self.prefix}L{layer}.up", f"{self.prefix}L{layer}.down"
        self.kernel(
            routing,
            "regroup",
            {
                "rows": rows,
                "experts": 384,
                "local": 48,
                "selected": 6,
                "base": self.rank * 48,
                "pad_last": False,
            },
            [
                ("indices", indices, "i32", (rows, 8), "read"),
                ("gather", gather, "i32", (expert_rows,), "write"),
                ("restore", restore, "i32", (rows, 8), "write"),
                ("counts", counts, "i64", (48,), "write"),
                ("errors", route_errors, "i32", (rows + 1,), "write"),
            ],
        )
        gathered = self.temp("expert.input", (expert_rows, 5120), "i8")
        gathered_scale = self.temp("expert.input.scale", (expert_rows,), "f32")
        error = self.temp("expert.gather.errors", (expert_rows,), "i32")
        for source, destination, width, dtype in [
            (quant, gathered, 5120, "i8"),
            (scale, gathered_scale, 1, "f32"),
        ]:
            self.kernel(
                memory,
                "gather_rows",
                {
                    "rows": expert_rows,
                    "table_rows": rows,
                    "width": width,
                    "dtype": DTYPE[dtype][0],
                    "index_dtype": "int32",
                },
                [
                    ("table", source, dtype, (rows, width), "read"),
                    ("indices", gather, "i32", (expert_rows,), "read"),
                    ("out", destination, dtype, (expert_rows, width), "write"),
                    ("errors", error, "i32", (expert_rows,), "write"),
                ],
            )
        experts = [
            f"{prefix}.experts.{expert}" for expert in range(self.rank * 48, (self.rank + 1) * 48)
        ]
        packed_output = self.temp("expert.packed", (expert_rows, 4608))
        expert_output = self.temp("expert.output", (expert_rows, 5120))
        workspace = self.buffer("w4.workspace", (256 << 20,), "i8")
        definitions = []
        for label, parts, k, n in [
            (up_name, ["w1", "w3"], 5120, 4608),
            (down_name, ["w2"], 2304, 5120),
        ]:
            planes = []
            for plane, shape, dtype in [
                ("packed", (48, k, n // 2), "i8"),
                ("scale", (48, 1, n), "u64"),
                ("assist", (48, n), "f32"),
            ]:
                planes.append(
                    self.buffer(
                        f"w.L{layer}.{parts}.{plane}",
                        shape,
                        dtype,
                        weight={
                            "kind": "w4",
                            "experts": experts,
                            "parts": parts,
                            "input": k,
                            "output": n,
                            "plane": plane,
                        },
                    )
                )
            definitions.append((label, k, n, planes))
        self.steps.append(
            {
                "op": "w4",
                "name": up_name,
                "rows": expert_rows,
                "input": 5120,
                "output": 4608,
                "experts": 48,
                "buffers": [gathered, *definitions[0][3], gathered_scale, packed_output],
                "workspace": workspace,
                "groups": counts,
            }
        )
        down8, down_scale = self.swiglu_packed_quant(
            packed_output, 2304, "expert.act.quant", rows=expert_rows
        )
        self.steps.append(
            {
                "op": "w4",
                "name": down_name,
                "rows": expert_rows,
                "input": 2304,
                "output": 5120,
                "experts": 48,
                "buffers": [down8, *definitions[1][3], down_scale, expert_output],
                "workspace": workspace,
                "groups": counts,
            }
        )
        routed = self.temp("expert.local", (rows, 5120), "f32")
        zeros = self.temp("zero.shared", (rows, 5120))
        errors = self.temp("expert.errors", (rows,), "i32")
        self.kernel(
            moe,
            "combine",
            {"rows": rows, "expert_rows": expert_rows, "width": 5120, "output_dtype": "float32"},
            [
                ("experts", expert_output, "bf16", (expert_rows, 5120), "read"),
                ("shared", zeros, "bf16", (rows, 5120), "read"),
                ("row_map", restore, "i32", (rows, 8), "read"),
                ("weights", weights, "f32", (rows, 8), "read"),
                ("out", routed, "f32", (rows, 5120), "write"),
                ("errors", errors, "i32", (rows,), "write"),
            ],
        )
        reduced = self.temp("expert.reduced", (rows, 5120), "f32")
        self.steps.append(
            {
                "op": "reduce",
                "source": routed,
                "destination": reduced,
                "elements": rows * 5120,
                "fp32": True,
            }
        )
        output = self.temp("ffn.output", (rows, 5120))
        self.kernel(
            model_ops,
            "add_shared",
            {"rows": rows, "width": 5120},
            [
                ("routed", reduced, "f32", (rows, 5120), "read"),
                ("shared", shared, "bf16", (rows, 5120), "read"),
                ("out", output, "bf16", (rows, 5120), "write"),
            ],
        )
        return output

    def packet(self, rows):
        self.rows, self.prefix, self.steps = rows, f"m{rows}.", []
        coordinates = {}
        for name in ("base_cos", "base_sin", "long_cos", "long_sin", "source_cos", "source_sin"):
            coordinates[name] = self.temp(name, (rows, 32), "f32")
        for name, shape in [
            ("slots", (rows,)),
            ("c2_slots", (rows,)),
            ("swa_indices", (rows, 128)),
            ("c1_indices", (rows, self.capacity)),
            ("c2_indices", (rows, self.capacity // 2)),
            ("c1_lengths", (rows,)),
            ("c2_lengths", (rows,)),
            ("ring", (rows, 8)),
        ]:
            coordinates[name] = self.temp(name, shape, "i32")
        incoming = coordinates["premix"] = self.temp("pre.a", (rows, 8), "f32")
        embedding = self.temp("embedding", (rows, 5120))
        self.steps.append({"op": "embed", "destination": embedding})
        current = self.temp("residual.a", (rows, 4, 5120))
        self.kernel(
            vector,
            "mhc_expand",
            {"rows": rows, "width": 5120},
            [
                ("x", embedding, "bf16", (rows, 5120), "read"),
                ("out", current, "bf16", (rows, 4, 5120), "write"),
            ],
        )
        reset = []
        for layer in range(40):
            if layer in (1, 14):
                current = self.engram(current, layer)
            for sublayer in ("attn", "ffn"):
                x, next_mix, post, combination = self.hc(current, incoming, layer, sublayer)
                x, precise = self.norm(
                    x,
                    self.matrix(f"layers.{layer}.{sublayer}_norm.weight", "f32", 5120, 1),
                    5120,
                    "norm." + sublayer,
                )
                output = (
                    self.attention(x, layer, coordinates, reset)
                    if sublayer == "attn"
                    else self.ffn(x, precise, layer)
                )
                current = self.hc_post(output, current, post, combination)
                incoming = next_mix
            self.steps.append({"op": "snapshot", "name": f"layer{layer:02}", "buffer": current})
        body, self.steps = self.steps, []
        collapsed = self.temp("final.collapsed", (rows, 5120))
        self.kernel(
            vector,
            "mhc_pre_apply",
            {"rows": rows, "width": 5120},
            [
                ("residual", current, "bf16", (rows, 4, 5120), "read"),
                ("mix", incoming, "f32", (rows, 8), "read"),
                ("out", collapsed, "bf16", (rows, 5120), "write"),
            ],
        )
        normalized, _ = self.norm(
            collapsed, self.matrix("norm.weight", "f32", 5120, 1), 5120, "final.norm"
        )
        last = self.buffer(
            self.prefix + "final.last",
            (1, 5120),
            "bf16",
            view={"parent": normalized, "offset": (rows - 1) * 5120 * 2},
        )
        logits = self.fp32(
            last,
            self.matrix("head.weight", "f32", 16160, 5120, first_row=self.rank * 16160),
            16160,
            5120,
            "logits",
            rows=1,
        )
        length = self.asset("vocab_length", np.array([16160], dtype="<i4"), "i32")
        indices, values = self.select(logits, 16160, length, "head.selected", rows=1)
        return {
            "rows": rows,
            "coordinates": coordinates,
            "body": body,
            "head": self.steps,
            "reset": reset,
            "best_value": values,
            "best_index": indices,
            "vocab_first": self.rank * 16160,
        }


def rope_tables(output, capacity):
    result = {}
    for name, base, yarn in [("base", 10000, False), ("long", 160000, True)]:
        frequencies = base ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
        extrapolated = 1.0 / frequencies
        if yarn:
            low = max(
                0, math.floor(64 * math.log(65536 / (32 * 2 * math.pi)) / (2 * math.log(base)))
            )
            high = min(63, math.ceil(64 * math.log(65536 / (2 * math.pi)) / (2 * math.log(base))))
            ramp = (
                (torch.arange(32).float() - low) / (high - low if high != low else 0.001)
            ).clamp(0, 1)
            interpolated = 1.0 / (16 * frequencies)
            frequencies = interpolated * ramp + extrapolated * (1 - ramp)
        else:
            frequencies = extrapolated
        phases = torch.arange(capacity).float()[:, None] * frequencies[None, :]
        for part, data in [("cos", phases.cos()), ("sin", phases.sin())]:
            path = output / "data" / f"{name}_{part}.bin"
            path.write_bytes(data.numpy().astype("<f4").tobytes())
            result[f"{name}_{part}"] = {
                "file": str(path.relative_to(output)),
                "sha256": file_hash(path),
            }
    return result


def compile_task(task_and_output):
    task, output = task_and_output
    torch.set_num_threads(2)
    output = Path(output)
    module = importlib.import_module(task["module"])
    function = getattr(module, task["name"])(**task["arguments"])
    path = compile_kernel(
        function,
        f"{task['module'].split('.')[-1]}.{task['name']}",
        task["specs"],
        task["arguments"],
        output / "cache",
    )
    return task["placeholder"], str(path.relative_to(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--engram-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=8448)
    parser.add_argument("--compile-workers", type=int, default=4)
    args = parser.parse_args()
    if not 128 <= args.capacity <= 16384 or args.capacity % 32:
        parser.error("capacity must be divisible by 32, within 128..16384")
    if not 1 <= args.compile_workers <= 8:
        parser.error("compile workers must be in 1..8")
    torch.set_num_threads(4)
    config = json.loads((args.model_dir / "config.json").read_text())["text_config"]
    for name, value in {
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
        "index_topk": 512,
        "candidate_block_size": 8,
        "candidate_topk_blocks": 2048,
        "hc_mult": 4,
        "q_lora_rank": 1280,
    }.items():
        if config[name] != value:
            raise ValueError(f"Unsupported V4.1 geometry: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data").mkdir(exist_ok=True)
    shutil.copyfile(args.engram_metadata, args.output / "data/engram.json")
    from safetensors import safe_open

    with safe_open(
        args.model_dir / "optional/quarot.safetensors", framework="pt", device="cpu"
    ) as archive:
        rotation = archive.get_tensor("global_rotation").float()
    block = rotation[:32, :32].contiguous()
    if not torch.equal(rotation, torch.block_diag(*([block] * 160))):
        raise ValueError("Expected repeated block32 Engram rotation")
    kernels, ranks = {}, []
    for rank in range(8):
        builder = Builder(args.output, rank, args.capacity, kernels)
        builder.asset("rotation", block.numpy().astype("<f4"), "f32")
        packets = [builder.packet(rows) for rows in (1, 16)]
        ranks.append({"rank": rank, "buffers": builder.buffers, "packets": packets})
        print(
            json.dumps(
                {
                    "rank": rank,
                    "device_bytes": sum(b["bytes"] for b in builder.buffers if "view" not in b),
                    "kernels": len(kernels),
                }
            ),
            flush=True,
        )

    def identity(name):
        return {"file": name, "sha256": file_hash(args.model_dir / name)}

    model = {
        "schema_version": 1,
        "config": identity("config.json"),
        "index": identity("quant_model_weights.safetensors.index.json"),
        "engram": {
            "file": "data/engram.json",
            "sha256": file_hash(args.output / "data/engram.json"),
        },
        "capacity": args.capacity,
        "layers": 40,
        "ranks": ranks,
        "rope": rope_tables(args.output, args.capacity),
    }
    tasks = list(kernels.values())
    manifests = {}
    with ProcessPoolExecutor(
        max_workers=args.compile_workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for placeholder, path in pool.map(
            compile_task, [(task, str(args.output)) for task in tasks]
        ):
            manifests[placeholder] = path
            print(
                json.dumps({"compiled": len(manifests), "total": len(tasks), "manifest": path}),
                flush=True,
            )
    for rank in ranks:
        for packet in rank["packets"]:
            for step in packet["body"] + packet["head"]:
                if step["op"] == "kernel":
                    step["manifest"] = manifests[step["manifest"]]
    atomic_json(args.output / "model.json", model)
    print(
        json.dumps({"model": str(args.output / "model.json"), "unique_kernels": len(kernels)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
