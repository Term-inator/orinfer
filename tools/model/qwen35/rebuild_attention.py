"""Rebuild short Qwen attention from a package's measured decode geometry.

The native package, large-prefill choices, weights and request buffer contract
are reused. Run through tools/operators/run.sh with the fixed compiler image.
"""

import argparse
from tools.build.target import CUDA_TARGET
import hashlib
import json
from pathlib import Path
import re
import shutil

from tools.build.files import source_hash, write_json
from tools.model.publication import atomic_model, clone_model, load_model
from tools.model.host_compiler import Compiler, private_roles

ROOT = Path(__file__).resolve().parents[3]


def decode_geometry(kernel, source, dimension, head_group=None):
    from tools.model.qwen35.rebuild_mtp import attention_geometry

    geometry = attention_geometry(kernel, source, dimension, head_group)
    if geometry["block_m"] != 16:
        raise ValueError("Rebuild requires the measured compact decode tile")
    geometry.pop("target_tiles")
    geometry["independent_queries"] = True
    return geometry


def short_alias(name):
    match = re.fullmatch(r"(.*/attention(?:_merge)?)/rows/(\d+)-(\d+)", name)
    if match and int(match[2]) <= 8:
        return None if int(match[3]) <= 8 else f"{match[1]}/rows/9-{match[3]}"
    return name


def rebuild(args):
    import tilelang
    import torch
    from kernels.model.attention_rows import causal_attention_int8, merge_attention_int8
    from tools.model.compile import instantiate
    from tools.model.private_rows import wrap_private
    from tools.model.qwen35.parallel import attention_pointers, factory
    from tools.model.compact import compact_package

    source = args.model.resolve(strict=True)
    data, origin, package = load_model(source)
    if data["architecture"] not in ("qwen3_5", "qwen3_asr"):
        raise ValueError("Expected a native Qwen text execution package")
    metadata = data["metadata"]
    verification = (metadata.get("mtp") or {}).get("verification_table")
    if (metadata.get("mtp") or {}).get("packet_table"):
        raise ValueError("Rebuild attention before adding the MTP operand table")
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    qh, kvh = text["num_attention_heads"], text["num_key_value_heads"]
    dimension = text.get("head_dim", text["hidden_size"] // qh)
    original = {k["name"]: k for k in package["kernels"]}
    sample = next(
        k
        for k in package["kernels"]
        if re.fullmatch(r"qwen_rows/state/layer\d+/attention", k["name"])
    )
    identity = source_hash(
        ROOT,
        [
            ROOT / p
            for p in (
                "kernels/model/attention_rows.py",
                "kernels/model/kv_int8.py",
                "kernels/model/rows.py",
                "kernels/model/independent_rows.py",
                "tools/model/compile.py",
                "tools/model/private_rows.py",
                "tools/model/host_compiler.py",
                "tools/model/qwen35/parallel.py",
                "tools/model/qwen35/rebuild_attention.py",
                "tools/build/compiler-requirements.txt",
            )
        ],
    )
    changed = []
    with atomic_model(args.target) as stage:
        operator = clone_model(source, stage, origin)
        buffers = {b["name"]: b for b in metadata["buffers"]}
        compiler = Compiler(origin, operator, args.output, buffers)
        saved_geometry = package["toolchain"].get("causal_attention")
        geometry = (
            json.loads(saved_geometry)
            if saved_geometry
            else decode_geometry(
                sample, compiler.read(sample["source"]).decode(), dimension, qh // kvh
            )
        )
        if geometry["head_group"] != qh // kvh:
            raise ValueError("Expected the measured complete GQA head group")
        compiler_identity = dict(
            tilelang=tilelang.__version__, cuda=torch.version.cuda, target=CUDA_TARGET["arch"]
        )
        identity = hashlib.sha256(
            json.dumps(
                dict(
                    source=identity,
                    geometry=geometry,
                    context=metadata["max_context"],
                    query_heads=qh,
                    kv_heads=kvh,
                    dimension=dimension,
                    compiler=compiler_identity,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        dynamic = {d["name"]: d for d in package["dynamic_batch_kernels"]}
        renamed = {name: short_alias(name) for name in original}
        package["kernels"] = [
            dict(k, name=renamed[k["name"]])
            for k in package["kernels"]
            if renamed[k["name"]] is not None
        ]
        package["dynamic_batch_kernels"] = [
            dict(d, name=renamed[d["name"]])
            for d in dynamic.values()
            if renamed[d["name"]] is not None
        ]
        package["plan_kernel_contracts"] = sorted(
            {renamed.get(n, n) for n in package["plan_kernel_contracts"] if renamed.get(n, n)}
        )
        columns = metadata["batch_layout"]["state_columns"]
        layers = len(metadata["batch_layout"]["layers"])
        # Append the new partial layout so retained GDN/convolution wrappers
        # keep their certified scratch offsets. Both attention paths share it.
        scratch = buffers["QwenRowScratch"]
        cursor = (scratch["shape"][0] + 255) // 256 * 256
        attention_offsets = {}
        for role, width in (("Partial", dimension), ("Maximum", 1), ("Denominator", 1)):
            attention_offsets[role] = cursor
            size = (qh // geometry["head_group"]) * 8 * 32 * geometry["block_m"] * width * 4
            cursor += (size + 255) // 256 * 256
        scratch["shape"][0] = cursor
        for old in list(original.values()):
            name = old["name"]
            normal = re.fullmatch(r"qwen_rows/state/layer(\d+)/(attention|attention_merge)", name)
            verified = re.fullmatch(
                r"qwen_rows/verify_state/layer(\d+)/attention\d+/(attention|attention_merge)", name
            )
            shared = re.fullmatch(
                r"qwen_rows/(?:layer\d+/private|mtp)/(attention|attention_merge)", name
            )
            if not normal and not shared and not verified:
                continue
            role = normal[2] if normal else verified[2] if verified else shared[1]
            if normal or verified:
                selected_columns = verification["columns"] if verified else columns
                if normal:
                    fn = factory(role, metadata["max_context"], geometry, text)
                    options = dict(batch_parameters=("independent_instances",))
                    table = "QwenRowPointers"
                else:
                    from types import SimpleNamespace
                    from tools.model.qwen35.causal_parallel import factories, symbolic_rows

                    b = SimpleNamespace(
                        qh=qh,
                        kvh=kvh,
                        hk=text["linear_num_key_heads"],
                        hv=text["linear_num_value_heads"],
                        qkv=(2 * text["linear_num_key_heads"] + text["linear_num_value_heads"])
                        * 128,
                        context=metadata["max_context"],
                        gdn_execution=metadata["gdn_execution"],
                    )
                    fn, variables = symbolic_rows(factories(b, geometry, {})[role]())
                    options = dict(
                        scalar_columns={v.name: ("SegmentLength", 8) for v in variables},
                        row_offset=verification["row_offset"],
                        row_capacity=metadata["chunk_tokens"],
                    )
                    table = verification["buffer"]
                source_text = compiler.read(old["source"]).decode()
                roles, offsets = private_roles(
                    source_text, selected_columns, layers * len(selected_columns)
                )
                offsets.update(attention_offsets)
                modes, pointers = {}, compiler.pointers(old)
                for buf in fn.buffer_map.values():
                    if buf.name in roles:
                        modes[buf.name] = roles[buf.name]
                    elif buf.name in pointers:
                        modes[buf.name] = ("row", pointers[buf.name]["name"])
                fn, bindings = wrap_private(
                    fn,
                    128,
                    modes,
                    selected_columns,
                    pointers,
                    table,
                    table_offset=int((normal or verified)[1]) * len(selected_columns) * 8,
                    row_stride=layers * len(selected_columns),
                    offsets=offsets,
                    **options,
                )
                emitted, launch = compiler.emit(
                    old,
                    name,
                    bindings,
                    128,
                    build=lambda fn=fn: fn,
                    key=f"causal-{'verify' if verified else 'owner'}-{role}-{identity}",
                )
                package["kernels"] = [k for k in package["kernels"] if k["name"] != name]
                package["dynamic_batch_kernels"] = [
                    k for k in package["dynamic_batch_kernels"] if k["name"] != name
                ]
            else:
                name += "/rows/1-8"
                bindings = compiler.pointers(old)
                if role == "attention":
                    prefix = old["name"].rsplit("/", 1)[0]
                    prepared = original[
                        prefix + ("/kv_prepare" if prefix.endswith("/mtp") else "/prepare")
                    ]
                    bindings = attention_pointers(bindings, compiler.pointers(prepared))
                    if geometry["stages"]:
                        bindings.update(
                            {"live_" + k.lower(): bindings[k] for k in ("K", "V", "KS", "VS")}
                        )
                options = {
                    k: v for k, v in geometry.items() if k not in ("block_n", "threads", "stages")
                }
                options.update(
                    query_heads=qh,
                    kv_heads=kvh,
                    dimension=dimension,
                    gated=data["architecture"] != "qwen3_asr",
                )
                fn = causal_attention_int8 if role == "attention" else merge_attention_int8
                if role == "attention":
                    options.update(
                        context=metadata["max_context"],
                        block_n=geometry["block_n"],
                        threads=geometry["threads"],
                        stages=geometry["stages"],
                    )
                emitted, launch = compiler.emit(
                    old,
                    name,
                    bindings,
                    8,
                    build=lambda fn=fn, options=options: instantiate(fn, 8, **options),
                    key=f"causal-shared-{role}-{identity}",
                )
                package["plan_kernel_contracts"].append(name)
            package["kernels"].append(emitted)
            package["dynamic_batch_kernels"].append(launch)
            changed.append(name)
        if not changed:
            raise ValueError("No Qwen attention bindings rebuilt")
        package["plan_kernel_contracts"] = sorted(set(package["plan_kernel_contracts"]))
        package["toolchain"] = dict(
            package["toolchain"],
            causal_attention=json.dumps(geometry, sort_keys=True),
            causal_attention_source=identity,
            causal_attention_compiler=json.dumps(compiler_identity, sort_keys=True),
        )
        metadata["toolchain"] = package["toolchain"]
        package["buffer_contracts"] = [
            {k: v for k, v in b.items() if k != "data"} for b in metadata["buffers"]
        ]
        digest = compact_package(operator, package, stage / "cache/packages")
        data["execution_package"] = digest
        write_json(stage / "cache/model.json", data)
        shutil.rmtree(operator)
    result = dict(
        model=str(args.target.resolve()),
        execution_package=digest,
        base_model=str(source),
        geometry=geometry,
        rebuilt=changed,
        weight_payloads_unchanged=True,
    )
    write_json(args.output / "result.json", result)
    print("COMPLETE", digest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    rebuild(parser.parse_args())


if __name__ == "__main__":
    main()
