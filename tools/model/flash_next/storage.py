"""Bounded assembly of Flash checkpoint slices into native weight operands.

Expert banks retain their K128 layout. Joining slices copies bytes directly
between files; it never allocates a full bank on the host or the GPU.
MTP operands and shard names have a separate namespace for their own HC mixer
and codebooks; the shared target head is bound by the arithmetic planner.
"""

import hashlib
import math
from pathlib import Path
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from tools.build.files import write_json
from tools.model.compact import write_shard
from tools.model.flash_next.roles import role
from tools.model.safetensors_source import local_header
from tools.quantization.vq import (
    e8p_selector_supports_parity_xor,
    e8p_selector_table,
    e8p_sign_table,
)


DTYPES = {
    "F16": "f16",
    "BF16": "bf16",
    "F32": "f32",
    "I8": "i8",
    "U8": "u8",
    "U16": "u16",
    "U32": "u32",
}


def payload_hash(path, offset, size):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        while size:
            block = stream.read(min(size, 8 * 1024**2))
            if not block:
                raise EOFError(f"Truncated weight operand: {path}")
            digest.update(block)
            size -= len(block)
    return digest.hexdigest()


def segments(parts, key):
    result = []
    for part in parts:
        info = part["header"][part["keys"][key]]
        first, last = info["data_offsets"]
        result.append((part["path"], part["begin"] + first, last - first))
    return result


class Weights:
    def __init__(self, checkpoint, root):
        self.source, self.root = checkpoint, Path(root)
        self.sources = [checkpoint]
        self.prefix = (
            "Mtp_" if checkpoint.config["quantization_config"].get("component") == "mtp" else ""
        )
        self.directory = self.root / "cache/weights"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffers, self.scopes, self.mapping, self.logical = {}, {}, {}, {}
        self.books = {}

    def shard(self, filename, tensors):
        filename = self.prefix + filename
        if any(t["name"] in self.buffers for t in tensors):
            raise ValueError("Duplicate native operand")
        path = self.directory / filename
        layouts = {f"orin.layout.{t['name']}": "native_contiguous" for t in tensors}
        offsets, _ = write_shard(path, tensors, layouts)
        for tensor in tensors:
            name = tensor["name"]
            size = sum(segment[2] for segment in tensor["segments"])
            self.buffers[name] = dict(
                name=name,
                dtype=DTYPES[tensor["dtype"]],
                shape=list(tensor["shape"]),
                layout="native_contiguous",
                alignment=256,
                access="read",
                data=dict(tensor=name, sha256=payload_hash(path, offsets[name], size)),
            )
            self.scopes[name] = "weights"
            self.mapping[name] = filename

    def constants(self, filename, arrays):
        with tempfile.TemporaryDirectory(prefix=".constants-", dir=self.directory) as tmp:
            path = Path(tmp) / "values.safetensors"
            save_file(
                {name: np.ascontiguousarray(value) for name, value in arrays.items()}, str(path)
            )

            begin, header = local_header(path)
            self.shard(
                filename,
                [
                    dict(
                        name=name,
                        dtype=info["dtype"],
                        shape=info["shape"],
                        segments=[
                            (
                                path,
                                begin + info["data_offsets"][0],
                                info["data_offsets"][1] - info["data_offsets"][0],
                            )
                        ],
                    )
                    for name, info in header.items()
                    if name != "__metadata__"
                ],
            )

    def attach_mtp(self, draft):
        """Join physical arenas; each planner retains its own logical HC roles."""
        from tools.model.flash_next.config import validate_pair

        validate_pair(draft.source.config, self.source.config, 1)
        if "output.weight" in draft.logical:
            raise ValueError("MTP must share the target output head")
        if self.root.resolve() != draft.root.resolve():
            raise ValueError("MTP and target weights must share the publication root")
        if set(self.buffers) & set(draft.buffers) or set(self.mapping.values()) & set(
            draft.mapping.values()
        ):
            raise ValueError("MTP weight arena collides with the target")
        self.buffers.update(draft.buffers)
        self.scopes.update(draft.scopes)
        self.mapping.update(draft.mapping)
        self.sources.extend(draft.sources)
        return self.finish()

    def expert(self, original, name):
        parts = self.source.parts[original]
        experts, outputs, inputs = self.source.shape(original)
        if inputs % 128:
            raise ValueError("Expert inputs require complete K128 blocks")
        table, signs = None, None
        for part in parts:
            count = part["count"]
            for key, dtype, shape in (
                ("indices", "U16", [count, inputs // 128, outputs, 16]),
                ("scales", "F16", [count, outputs]),
            ):
                info = part["header"][part["keys"][key]]
                if info["dtype"] != dtype or info["shape"] != shape:
                    raise ValueError(f"Invalid expert slice geometry: {original}/{key}")
            with safe_open(part["path"], framework="np") as stream:
                current = stream.get_tensor(part["keys"]["table"])
                rotation = stream.get_tensor(part["keys"]["signs"])
            if table is None:
                table, signs = current, rotation
            elif not np.array_equal(table, current) or not np.array_equal(signs, rotation):
                raise ValueError(f"Expert slices disagree on their basis: {original}")
        prefix = f"{self.prefix}W_{name}"
        operands = {"packed": prefix + "_packed", "scale": prefix + "_scale"}
        self.shard(
            name + ".safetensors",
            [
                dict(
                    name=operands["packed"],
                    dtype="U16",
                    shape=[experts, inputs // 128, outputs, 16],
                    segments=segments(parts, "indices"),
                ),
                dict(
                    name=operands["scale"],
                    dtype="F16",
                    shape=[experts, outputs],
                    segments=segments(parts, "scales"),
                ),
            ],
        )
        key = hashlib.sha256(table.tobytes() + signs.tobytes()).hexdigest()
        if key not in self.books:
            book = f"{self.prefix}W_e8p_{len(self.books)}"
            operands_book = {
                suffix: book + "_" + suffix
                for suffix in ("table", "short_table", "selector_table", "signs")
            }
            selector = e8p_selector_table(table)
            self.constants(
                f"W_e8p_{len(self.books)}.safetensors",
                {
                    operands_book["table"]: table.view("<u4").reshape(1, 256, 2),
                    operands_book["short_table"]: e8p_sign_table(table)[None],
                    operands_book["selector_table"]: selector[None],
                    operands_book["signs"]: signs,
                },
            )
            self.books[key] = dict(
                **operands_book, parity_xor=e8p_selector_supports_parity_xor(selector)
            )
        self.logical[name] = dict(**operands, **self.books[key])

    def dense(self, original, name):
        parts = self.source.parts[original]
        outputs, inputs = self.source.shape(original)
        operands = {suffix: f"{self.prefix}W_{name}_{suffix}" for suffix in ("weight", "scale")}
        for part in parts:
            for key, dtype, shape in (
                ("weight", "I8", [part["count"], inputs]),
                ("scale", "F16", [part["count"]]),
            ):
                info = part["header"][part["keys"][key]]
                if info["dtype"] != dtype or info["shape"] != shape:
                    raise ValueError(f"Invalid dense slice geometry: {original}/{key}")
        self.shard(
            name + ".safetensors",
            [
                dict(name=operands[key], dtype=dtype, shape=shape, segments=segments(parts, key))
                for key, dtype, shape in (
                    ("weight", "I8", [outputs, inputs]),
                    ("scale", "F16", [outputs]),
                )
            ],
        )
        self.logical[name] = operands

    def original(self, original, name):
        parts = self.source.parts[original]
        shape = list(self.source.shape(original))
        operand = f"{self.prefix}W_{name}_value"
        floating = (
            "norm.weight" in name
            or "ple_norm_" in name
            or name.startswith("pre_fc_norm_")
            or name.endswith(("ssm_a_log", "ssm_dt.bias", "ssm_conv1d.weight"))
        )
        if len(shape) == 3 and shape[1] == 1:
            shape.pop(1)
        if ("hc_" in name and name.endswith("norm.weight")) or "ple_norm_" in name:
            shape = [4, 2560]
        if floating:
            values = self.source.tensor(original).astype(np.float32)
            if name.startswith("pre_fc_norm_") or name.endswith(
                (
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                    "index_q_norm.weight",
                    "index_k_norm.weight",
                )
            ):
                values += np.float32(1)
            self.constants(name + ".safetensors", {operand: values.reshape(shape)})
        else:
            dtypes = {p["header"][p["keys"][original]]["dtype"] for p in parts}
            if dtypes != {"BF16"}:
                raise ValueError(f"Expected original BF16 operand: {original}")
            self.shard(
                name + ".safetensors",
                [dict(name=operand, dtype="BF16", shape=shape, segments=segments(parts, original))],
            )
        self.logical[name] = operand

    def assemble(self):
        for original in sorted(self.source.parts):
            name = role(original)
            if name is None or name == "token_embd.weight":
                continue
            kind = self.source.kind(original)
            if kind == "e8p-expert":
                self.expert(original, name)
            elif kind == "int8-row":
                self.dense(original, name)
            elif kind == "original":
                self.original(original, name)
            else:
                raise ValueError(f"Unsupported native operand representation: {original}: {kind}")
            print("ASSEMBLED", name, flush=True)
        total = self.finish()
        return dict(
            config=self.source.config,
            buffers=self.buffers,
            scopes=self.scopes,
            logical=self.logical,
            mapping=dict(self.mapping),
            weight_bytes=total,
        )

    def finish(self):
        sizes = {"f16": 2, "bf16": 2, "f32": 4, "i8": 1, "u8": 1, "u16": 2, "u32": 4}
        total = sum(
            math.prod(b["shape"]) * sizes[b["dtype"]]
            for b in self.buffers.values()
            if b["access"] == "read"
        )
        write_json(
            self.directory / "model.safetensors.index.json",
            dict(metadata=dict(total_size=total), weight_map=self.mapping),
        )
        return total
