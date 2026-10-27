"""Bound warp-owned expert launches by live routing rows, preserving arithmetic."""

import json
import math

from tools.build.files import write_json
from tools.model.build_cache import identity


def publish(model, package, compiler, output):
    from kernels.model.e8p_decode import e8p_decode_grouped
    from kernels.model.expert_addresses import indirect_experts
    from kernels.model.rows import explicit_rows
    from tools.model.compile import construct, compile_function

    config = json.loads((model / "config.json").read_text())
    if config["quantization_config"]["basis"] != "integer-e8p-spread29-v1":
        raise ValueError("Expert launch rebuilding requires the nonzero spread29 codec")
    top = config["text_config"]["num_experts_per_tok"]
    dynamics = {entry["name"]: entry for entry in package["dynamic_batch_kernels"]}
    compiled, records = {}, []
    for old in package["kernels"]:
        if old["name"] not in dynamics:
            continue
        pointers = compiler.pointers(old)
        if not {"PAddresses", "WSAddresses", "TileCount"} <= pointers.keys():
            continue
        if set(pointers) != {
            "A",
            "AS",
            "Book",
            "C",
            "Counts",
            "Offsets",
            "PAddresses",
            "TileCount",
            "TileExpert",
            "TileRow",
            "WSAddresses",
        } or any(v.get("offset", 0) for v in pointers.values()):
            raise ValueError("Expert launch rebuilding requires whole warp-owned operands")
        packed_name = pointers["PAddresses"]["name"].removesuffix("_addresses")
        weight = compiler.buffers[packed_name]
        experts, groups, n, lanes = weight["shape"]
        k = groups * 128
        capacity = dynamics[old["name"]]["capacity"]
        tiles = (capacity * top + 7) // 8 + min(experts, capacity * top)
        if (
            weight["dtype"] != "u16"
            or lanes != 16
            or old["block"] != [128, 1, 1]
            or old["grid"] != [tiles, (n + 63) // 64, 1]
        ):
            raise ValueError("Expert launch rebuilding requires the BM8/BN64/SK1 K128 contract")
        scale = compiler.buffers[pointers["WSAddresses"]["name"].removesuffix("_addresses")]
        if (
            not weight.get("data")
            or scale["dtype"] != "f16"
            or scale["shape"] != [experts, n]
            or not scale.get("data")
        ):
            raise ValueError(
                "Expert launch rebuilding requires immutable compressed banks and scales"
            )
        for role, dtype, elements in (
            ("A", "i8", capacity * top * k),
            ("AS", "f16", capacity * top),
            ("C", "f16", capacity * top * n),
            ("Counts", "i32", experts),
            ("Offsets", "i32", experts),
            ("TileExpert", "i32", tiles),
            ("TileRow", "i32", tiles),
            ("TileCount", "i32", 1),
            ("PAddresses", "u64", experts),
            ("WSAddresses", "u64", experts),
        ):
            buffer = compiler.buffers[pointers[role]["name"]]
            if buffer["dtype"] != dtype or math.prod(buffer["shape"]) < elements:
                raise ValueError("Expert launch operands differ from their physical row contract")
        shape = dict(rows=capacity * top, E=experts, tiles=tiles, N=n, K=k)
        key = identity(shape)
        if key not in compiled:
            fn, passes = construct(
                lambda: e8p_decode_grouped(**shape, parity_xor=True, dynamic_rows=True, top_k=top)
            )
            compiled[key] = compile_function(
                explicit_rows(indirect_experts(fn)), pass_configs=passes
            )
        rebuilt, launch = compiler.emit(
            old,
            old["name"],
            pointers,
            capacity,
            kernel=compiled[key],
            key="expert-live-" + key,
        )
        if (
            rebuilt["block"] != old["block"]
            or rebuilt["shared_memory_bytes"] != old["shared_memory_bytes"]
        ):
            raise ValueError("Expert launch rebuilding changed the certified compute tile")
        old.update(rebuilt)
        dynamics[old["name"]] = launch
        records.append(dict(name=old["name"], shape=shape))
    if not records:
        raise ValueError("No dynamic indirect expert projections selected")
    package["dynamic_batch_kernels"] = list(dynamics.values())
    write_json(output / "expert-launch.json", dict(projections=records))
    return [dict(kind="expert-grid", bindings=len(records))]
