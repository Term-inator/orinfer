"""Rebuild expert addressing while retaining certified arithmetic and row plans."""

import json
import shutil

from tools.build.files import file_hash, write_json
from tools.model.build_cache import BuildCache
from tools.model.compact import compact_package
from tools.model.host_compiler import Compiler
from tools.model.package import execution_identity, model_library
from tools.model.publication import atomic_model, clone_model, load_model


def rebuild(args):
    from kernels.model.e8p_decode import e8p_decode_grouped
    from kernels.model.expert_addresses import indirect_experts
    from tools.model.compile import construct, compile_function
    from tools.model.flash_next.build import source_identity, toolchain
    from tools.operators.common import export_kernel

    data, origin, package = load_model(args.model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expert address rebuilding requires Flash Next")
    config = json.loads((args.model / "config.json").read_text())
    if config["quantization_config"]["basis"] != "integer-e8p-spread29-v1":
        raise ValueError("Expert address rebuilding requires the nonzero spread29 codec")
    args.output.mkdir(parents=True, exist_ok=True)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    dynamic = {d["name"]: d for d in package["dynamic_batch_kernels"]}
    cache = BuildCache(args.cache)
    contract = dict(
        source=source_identity(),
        model=file_hash(args.model / "cache/model.json"),
        toolchain=toolchain(),
    )
    top_k = config["text_config"]["num_experts_per_tok"]
    bank_weights = {
        f"W_blk.{layer}.ffn_{role}_exps.weight_packed"
        for layer in range(config["text_config"]["num_hidden_layers"])
        for role in ("gate_up", "down")
    }
    records = []
    with atomic_model(args.model_output) as staging:
        root = clone_model(args.model, staging, origin)
        compiler = Compiler(origin, root, args.output, buffers)
        for old in package["kernels"]:
            pointers = compiler.pointers(old)
            packed = pointers.get("P")
            if packed is None or packed["name"] not in bank_weights:
                continue
            if set(pointers) != {
                "A",
                "AS",
                "Book",
                "C",
                "Counts",
                "Offsets",
                "P",
                "TileCount",
                "TileExpert",
                "TileRow",
                "WS",
            }:
                raise ValueError("Expert address rebuild needs the warp-owned INT8 projection ABI")
            weight = buffers[packed["name"]]
            experts, groups, n, lanes = weight["shape"]
            k = groups * 128
            rows = dynamic.get(old["name"], {}).get("capacity", 1)
            if lanes != 16 or old["block"] != [128, 1, 1] or old["grid"][1:] != [(n + 63) // 64, 1]:
                raise ValueError(
                    "Expert address rebuild must retain the BM8/BN64/SK1 K128 contract"
                )
            shape = dict(rows=rows * top_k, E=experts, tiles=old["grid"][0], N=n, K=k)
            key = dict(**contract, projection=shape)

            def build(folder):
                fn, passes = construct(lambda: e8p_decode_grouped(**shape, parity_xor=True))
                export_kernel(compile_function(indirect_experts(fn), pass_configs=passes), folder)

            artifact, hit = cache.get(key, build)
            for role in ("P", "WS"):
                original = pointers.pop(role)
                if original["kind"] != "buffer":
                    raise ValueError("Expert storage needs whole compressed bank operands")
                name = original["name"] + "_addresses"
                table = dict(
                    name=name,
                    dtype="u64",
                    shape=[experts],
                    layout="native_contiguous",
                    alignment=256,
                    access="read_write",
                )
                if name not in buffers:
                    buffers[name] = table
                    data["metadata"]["buffers"].append(table)
                    package["buffer_contracts"].append(table)
                    data["buffer_scopes"][name] = "workspace"
                pointers[role + "Addresses"] = dict(kind="buffer", name=name)
            rebuilt, _ = compiler.emit(
                old,
                old["name"],
                pointers,
                rows,
                artifact=artifact,
                key="expert-address-" + artifact.name,
                symbolic=False,
            )
            if rebuilt["grid"] != old["grid"] or rebuilt["block"] != old["block"]:
                raise ValueError("Expert address lowering changed the original launch geometry")
            old.update(rebuilt)
            records.append(dict(name=old["name"], shape=shape, cache_hit=hit))
        if not records:
            raise ValueError("No direct expert projections selected for address rebuilding")
        package["execution"] = execution_identity(
            model_library(), architecture="flash_next", compute_policy="int8_quality"
        )
        # The sealed library is regenerated from the resulting typed bindings.
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
    write_json(
        args.output / "results.json",
        dict(complete=True, contract=contract, execution_package=digest, projections=records),
    )
    print("EXPERT ADDRESSES COMPLETE", args.model_output, digest, flush=True)
