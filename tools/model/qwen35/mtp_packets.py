"""Preserve single-owner arithmetic while moving MTP operands into a fixed table."""

import hashlib
import math
from pathlib import Path
import re

from tools.build.files import source_hash, write_json

TABLE = "QwenMtpPacketPointers"


def build(b):
    if not b.mtp:
        return
    from kernels.model.independent_rows import indirect_parameters
    from kernels.model.packet_io import copy_rows
    from tools.model.compile import compile_function, instantiate
    from tools.model.qwen35.parallel import LANES

    root = Path(__file__).resolve().parents[3]
    b.aot.identity["mtp_packet_source"] = source_hash(
        root,
        [
            Path(__file__),
            root / "kernels/model/independent_rows.py",
            root / "kernels/model/packet_io.py",
            root / "tools/model/compile.py",
        ],
    )
    columns = sorted(name for name, scope in b.weights.scopes.items() if scope == "sequence")
    indices = {name: i for i, name in enumerate(columns)}
    b.weights.fixed(TABLE, "u64", [LANES * len(columns)])
    originals = {k["name"]: k for k in b.aot.kernels}
    raw = {}
    aliases = {}
    scalar_arguments = {}
    resource_changes = {}
    for name, recipe in list(b.aot.recipes.items()):
        if not name.startswith("qwen_rows/") or any(s in name for s in ("/vision/", "/audio/")):
            continue
        key, make, _, rows, dynamic, divisor, multiplier = recipe
        kernel = originals[name]
        pointers = b.aot.compiler.pointers(kernel)
        private = {role: arg["name"] for role, arg in pointers.items() if arg["name"] in indices}
        if not private:
            continue
        offsets = {role: pointers[role].get("offset", 0) for role in private}
        # Geometry and private column placement determine code; tensor values do not.
        identity = hashlib.sha256(
            repr((key, len(columns), sorted(private), sorted(offsets.items()))).encode()
        ).hexdigest()
        alias = "qwen_rows/packet_kernel/" + hashlib.sha256(name.encode()).hexdigest()

        def wrapped(key=key, make=make, private=private, offsets=offsets):
            if key not in raw:
                raw[key] = make()
            original = raw[key]
            fn, _ = indirect_parameters(original.prim_func, indices, private, offsets)
            return compile_function(
                fn, pass_configs=original.pass_configs, compile_flags=original.compile_flags
            )

        bindings = {role: arg for role, arg in pointers.items() if role not in private}
        bindings["Pointers"] = dict(kind="buffer", name=TABLE)
        b.aot.emit(
            alias,
            "mtp-packet-" + identity,
            wrapped,
            bindings,
            rows,
            dynamic=dynamic,
            row_divisor=divisor,
            row_multiplier=multiplier,
            constants={"column_" + role: indices[target] for role, target in private.items()},
        )
        replacement = b.aot.kernels[-1]
        if any(replacement[field] != kernel[field] for field in ("grid", "block", "cooperative")):
            raise ValueError(
                f"MTP indirection changes the original launch geometry: {name}: "
                + repr(
                    {
                        field: (kernel[field], replacement[field])
                        for field in ("grid", "block", "cooperative")
                        if kernel[field] != replacement[field]
                    }
                )
            )
        if replacement["shared_memory_bytes"] != kernel["shared_memory_bytes"]:
            resource_changes[name] = dict(
                original=kernel["shared_memory_bytes"], wrapped=replacement["shared_memory_bytes"]
            )
        original_host = b.aot.compiler.host(kernel)
        replacement_host = b.aot.compiler.host(replacement)
        scalar_positions = {
            arg["value"]: i
            for i, arg in enumerate(replacement_host["ordered_arguments"])
            if arg["ctype"] in ("ctypes.c_int32", "c_int32")
        }
        scalar_arguments[name] = []
        for i, arg in enumerate(original_host["ordered_arguments"]):
            if arg["ctype"] in ("ctypes.c_int32", "c_int32"):
                if arg["value"] not in scalar_positions:
                    raise ValueError(f"MTP indirection changes scalar ABI: {name}")
                scalar_arguments[name].append((i, scalar_positions[arg["value"]]))
        aliases[name] = alias

    copy_specs = [
        ("Token", "Input", 4),
        ("MtpToken", "MtpInput", 4),
        ("MtpLastHidden", "MtpCondition", b.h * 2),
        ("MtpCondition", "HeadHidden", b.h * 2),
        ("HeadLogits", "MtpLogits", b.v * 4),
        ("HeadLogits", "SequenceLogits", b.v * 4),
        ("HeadLogits", "Logits", b.v * 4),
    ]
    # Mixed packets also commit each ordinary GDN segment's convolution tail.
    for destination in columns:
        if re.fullmatch(r"L\d+_History", destination):
            buffer = b.weights.buffers[destination]
            if buffer["dtype"] != "f16":
                raise ValueError("Expected FP16 Qwen convolution history")
            copy_specs.append(("HistoryOut", destination, math.prod(buffer["shape"]) * 2))
    copies = []
    for source, destination, width in copy_specs:
        private = {
            role: name
            for role, name in [("Source", source), ("Destination", destination)]
            if name in indices
        }
        fn, inputs = indirect_parameters(instantiate(copy_rows, width), indices, private)
        bindings = {
            role: dict(kind="buffer", name=name)
            for role, name in [("Source", source), ("Destination", destination)]
            if role in inputs
        }
        bindings["Pointers"] = dict(kind="buffer", name=TABLE)
        name = f"qwen_rows/packet_copy/{source}/{destination}"
        identity = hashlib.sha256(str(fn).encode()).hexdigest()
        sizes = [
            math.prod(b.weights.buffers[n]["shape"])
            * {"i32": 4, "f16": 2, "f32": 4}[b.weights.buffers[n]["dtype"]]
            for n in (source, destination)
        ]
        capacity = max(2, min(size // width for size in sizes))
        b.aot.emit(
            name,
            "mtp-copy-" + identity,
            lambda fn=fn: compile_function(fn),
            bindings,
            capacity,
            constants={"column_" + role: indices[target] for role, target in private.items()},
        )
        copies.append(dict(source=source, destination=destination, kernel=name, row_bytes=width))
    write_json(b.aot.output / "mtp-packet-resources.json", resource_changes)
    b.mtp["packet_table"] = dict(
        buffer=TABLE,
        columns=columns,
        kernels=aliases,
        scalar_arguments=scalar_arguments,
        copies=copies,
    )
