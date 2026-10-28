"""Rebuild a native Flash Next execution package with resumable real-weight tuning.

Input is the supported schema-1 model and its pinned architecture contract, not
arbitrary bare weights. Stateful/float/expert portfolios retain that certified
contract; independent integer projections are constructed from typed operands.
"""

from tools.build.target import COMPILER_PASSES
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.model.build_cache import BuildCache, identity  # noqa: E402
from tools.model.flash_next.construction import (
    expert_matrices,
    fixed_extents,
    prefill_names,
    projection,
    select_projections,
    bounded_measurements,
)  # noqa: E402
from tools.model.host_compiler import Compiler  # noqa: E402
from tools.model.publication import ROOT, atomic_model, clone_model, load_model, source_path  # noqa: E402
from tools.build.files import file_hash, write_json, source_hash  # noqa: E402


def inventory(data, package, manifest, compiler):
    if data["architecture"] != "flash_next" or data["schema_version"] != 1:
        raise ValueError("Expected the current native Flash Next schema-1 model")
    if not manifest.get("row_execution"):
        raise ValueError("The native model must provide its shared row construction contract")
    fixed = fixed_extents(manifest)
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    groups = {}
    for kernel in package["kernels"]:
        result = projection(
            kernel,
            compiler.host(kernel),
            compiler.pointers(kernel),
            compiler.buffers,
            dynamic,
            fixed,
        )
        if result is None:
            continue
        spec, weights = result
        groups.setdefault(spec, []).append((kernel, weights))
    if not groups:
        raise ValueError("No supported integer construction contracts found")
    return groups, dynamic


def source_identity():
    # This pins both code generation and Rust's reconstruction of the input
    # architecture contract. Paths and mtimes never determine cache validity.
    paths = [ROOT / "Cargo.lock", ROOT / "Cargo.toml", ROOT / "rust-toolchain.toml"]
    paths.extend((ROOT / "crates").rglob("Cargo.toml"))
    paths.extend((ROOT / "configs").glob("*.json"))
    for directory, suffix in (
        ("kernels/model", "*.py"),
        ("tools/model", "*.py"),
        ("tools/operators", "*.py"),
        ("tools/build", "*.py"),
        ("crates/orinfer-models", "*.rs"),
        ("crates/orinfer-model-sdk", "*.rs"),
    ):
        paths.extend((ROOT / directory).rglob(suffix))
    return source_hash(ROOT, paths)


def weight_contract(weights, buffers):
    return {
        role: {k: buffers[operand["name"]][k] for k in ("dtype", "shape", "layout", "data")}
        for role, operand in sorted(weights.items())
    }


def read_weights(model, operands, buffers):
    import torch
    from safetensors import safe_open

    base = (model / "cache/weights").resolve(strict=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for role, operand in operands.items():
        buffer = buffers[operand["name"]]
        tensor = buffer["data"]["tensor"]
        with safe_open(source_path(base, index[tensor]), framework="pt", device="cpu") as stream:
            value = stream.get_tensor(tensor)
        raw = value.view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != buffer["data"]["sha256"]:
            raise ValueError(f"Real tuning weight identity mismatch: {tensor}")
        if list(value.shape) != buffer["shape"]:
            raise ValueError(f"Real tuning weight shape mismatch: {tensor}")
        expected = {
            "i8": torch.int8,
            "u16": torch.uint16,
            "u32": torch.uint32,
            "f16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if value.dtype != expected[buffer["dtype"]]:
            raise ValueError(f"Real tuning weight dtype mismatch: {tensor}")
        result[role] = value.cuda()
    return result


def toolchain():
    import tilelang
    import torch

    return dict(
        tilelang=tilelang.__version__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        capability=".".join(map(str, torch.cuda.get_device_capability())),
    )


def rebuild(args):
    started = time.monotonic()
    manifest = {}
    data, origin, package = load_model(args.model, plan=manifest)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    inspector = Compiler(origin, origin, args.output, buffers)
    groups, dynamic = inventory(data, package, manifest, inspector)
    bounded = bounded_measurements(manifest)
    prefill = prefill_names(manifest)
    prefill_groups = {
        s
        for s, items in groups.items()
        if s.dynamic and any(k["name"] in prefill for k, _ in items)
    }
    if args.prefill_only:
        groups = {s: items for s, items in groups.items() if s in prefill_groups}
        if not groups:
            raise ValueError("No dynamic prefill projection contracts found")
    groups = select_projections(groups, kinds=args.projection_kinds, shapes=args.projection_shape)
    partial = args.prefill_only or args.projection_kinds or args.projection_shape
    experts = expert_matrices(package, inspector)
    if partial:
        experts = {}
    report = dict(
        source_package=data["execution_package"],
        source=source_identity(),
        model=file_hash(args.model / "cache/model.json"),
        operators=[dict(**s.spec(), bindings=len(items)) for s, items in sorted(groups.items())],
        preserved_bindings=len(package["kernels"]) - sum(map(len, groups.values())),
        expert_geometries=[list(key) for key in sorted(experts)],
        prefill_only=args.prefill_only,
        projection_kinds=args.projection_kinds,
        projection_shapes=args.projection_shape,
        prefill_tuning_rows=args.prefill_tuning_rows,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "inventory.json", report)
    if args.plan_only:
        print(json.dumps(report, indent=2), flush=True)
        return

    from tools.model.projection_tuning import ProjectionTuner
    from tools.operators.common import export_kernel
    import torch

    contract = dict(**report, toolchain=toolchain(), seed=20261002)
    cache = BuildCache(args.cache)
    # A repeated command validates the completed output without rebuilding it.
    result_file = args.output / "results.json"
    if args.model_output.exists():
        output_data, _, _ = load_model(args.model_output)
        # The receipt is committed atomically with the model. Recover even if
        # interrupted after directory publication and before the external log.
        receipt = json.loads((args.model_output / "cache/build.json").read_text())
        if (
            receipt.get("contract") != identity(contract)
            or receipt.get("execution_package") != output_data["execution_package"]
        ):
            raise ValueError("Existing destination does not match this completed build")
        if not result_file.exists():
            write_json(
                result_file,
                dict(
                    complete=True,
                    contract=contract,
                    execution_package=output_data["execution_package"],
                    recovered_publication=True,
                ),
            )
        print("COMPLETE cached publication", output_data["execution_package"], flush=True)
        return

    tuning = ProjectionTuner(args.output, args.cache / "tuning")
    records = []
    certificates = []
    # Validate both legal expert tiles at actual N/K/E, without independently
    # changing the certified coupled route + up + down selection table.
    from tools.model.projection_tuning import make_prim, EXACT_PASSES, TARGET
    import tilelang

    for (count, n, k), operands in sorted(experts.items()):
        spec = dict(
            kind="expert",
            rows=4096,
            outputs=n,
            inputs=k,
            dtype="float16",
            dynamic=True,
            experts=count,
        )
        wc = weight_contract(operands, buffers)
        for bm in (8, 16):
            expert_contract = dict(
                source=contract["source"],
                toolchain=contract["toolchain"],
                seed=contract["seed"],
                spec=spec,
                weights=wc,
                block_m=bm,
            )

            def certify(folder):
                actual = read_weights(args.model, operands, buffers)
                config = dict(block_m=bm)
                candidate = tilelang.compile(
                    make_prim(spec, config),
                    out_idx=[],
                    execution_backend="nvrtc",
                    target=TARGET,
                    pass_configs={**COMPILER_PASSES, **(EXACT_PASSES or {})},
                )
                for live in (1, 17, 512, 4096):
                    ProjectionTuner._validate(candidate, spec, config, live, actual)
                export_kernel(candidate, folder)
                write_json(
                    folder / "validation.json", dict(config=config, checked_rows=[1, 17, 512, 4096])
                )
                del actual
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            _, hit = cache.get(expert_contract, certify)
            certificates.append(dict(spec=spec, block_m=bm, cache_hit=hit))
            print("EXPERT CHECKED", count, n, k, bm, "cache" if hit else "validated", flush=True)
    with atomic_model(args.model_output) as staging:
        root = clone_model(args.model, staging, origin)
        compiler = Compiler(origin, root, args.output, buffers)
        replacements, launches = {}, {}
        for number, (spec, items) in enumerate(sorted(groups.items()), 1):
            # Geometry is shared between layers; a deterministic real operand
            # set supplies values, scales and a checked identity to the tuner.
            representative, operands = min(items, key=lambda item: item[0]["name"])
            wc = weight_contract(operands, buffers)
            measured = (
                sorted({min(r, spec.rows) for r in args.prefill_tuning_rows})
                if spec in prefill_groups
                else None
            )

            def compile_extent(measurements):
                entry_contract = dict(
                    source=contract["source"],
                    toolchain=contract["toolchain"],
                    seed=contract["seed"],
                    spec=spec.spec(),
                    weights=wc,
                    measure_rows=measurements,
                )

                def construct(folder):
                    actual = read_weights(args.model, operands, buffers)
                    kernel, config = tuning.compile(
                        **spec.spec(),
                        weights=actual,
                        weight_identity=identity(wc),
                        measure_rows=measurements,
                    )
                    export_kernel(kernel, folder)
                    write_json(
                        folder / "validation.json",
                        dict(config=config, tuning=list(tuning.results.values())[-1]),
                    )
                    del actual
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                artifact, hit = cache.get(entry_contract, construct)
                return artifact, hit, identity(entry_contract)

            tick = time.monotonic()
            artifact, hit, key = compile_extent(measured)
            extents = {tuple(measured or []): (artifact, hit, key)}
            for old, _ in items:
                samples = (
                    bounded.get(old["name"])
                    if spec.dynamic and spec not in prefill_groups
                    else None
                )
                sample_key = tuple(samples or measured or [])
                if sample_key not in extents:
                    extents[sample_key] = compile_extent(list(sample_key))
                selected, _, selected_key = extents[sample_key]
                capacity = dynamic.get(old["name"], {}).get("capacity", spec.rows)
                item, launch = compiler.emit(
                    old,
                    old["name"],
                    compiler.pointers(old),
                    capacity,
                    key=selected_key,
                    artifact=selected,
                )
                replacements[old["name"]] = item
                if old["name"] in dynamic:
                    launches[old["name"]] = launch
            records.append(
                dict(
                    spec=spec.spec(),
                    key=key,
                    cache_hit=hit,
                    seconds=time.monotonic() - tick,
                    representative=representative["name"],
                    bounded_choices=[
                        dict(rows=list(rows), key=value[2])
                        for rows, value in extents.items()
                        if rows != tuple(measured or [])
                    ],
                )
            )
            write_json(
                args.output / "progress.json",
                dict(complete=False, finished=number, total=len(groups), operators=records),
            )
            print("BUILT", number, "/", len(groups), spec, "cache" if hit else "tuned", flush=True)
        package["kernels"] = [replacements.get(k["name"], k) for k in package["kernels"]]
        package["dynamic_batch_kernels"] = [
            launches.get(k["name"], k) for k in package["dynamic_batch_kernels"]
        ]
        from tools.model.row_convergence import coalesce_adjacent_selections
        from tools.model.compact import compact_package

        from tools.model.sampling import append as append_sampling

        append_sampling(data, package, root, args.output / "sampling")
        if not partial:
            from tools.model.auxiliary_tuning import publish as publish_auxiliary
            from tools.model.flash_next.hc import publish as publish_hc

            publish_hc(args.model, package, compiler, args.output / "hc", args.cache)
            publish_auxiliary(
                data,
                package,
                manifest,
                compiler,
                args.output / "auxiliary",
                cache=args.cache,
                model=args.model,
            )
            if (
                data["metadata"].get("vision")
                and data["metadata"]["vision"]["max_features"] < data["metadata"]["max_context"]
            ):
                from tools.model.vision_capacity import rebuild_overlays

                rebuild_overlays(data, package, manifest, compiler, data["metadata"]["max_context"])
        coalesce_adjacent_selections(package)
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
        write_json(
            staging / "cache/build.json",
            dict(contract=identity(contract), execution_package=digest),
        )
    result = dict(
        complete=True,
        contract=contract,
        execution_package=digest,
        seconds=time.monotonic() - started,
        operators=records,
        expert_certificates=certificates,
    )
    write_json(result_file, result)
    write_json(
        args.output / "progress.json", dict(complete=True, finished=len(groups), total=len(groups))
    )
    print("COMPLETE", digest, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Construct the initial native text package from our sliced checkpoint",
    )
    p.add_argument(
        "--source-index",
        type=Path,
        help="Original safetensors index for bounded BF16 Q4 reads at the checkpoint's pinned source",
    )
    p.add_argument(
        "--mtp-checkpoint",
        type=Path,
        help="Assemble the matching shared-head MTP checkpoint with an initial text model",
    )
    p.add_argument(
        "--vision-checkpoint",
        type=Path,
        help="Include the matching original BF16 vision checkpoint in initial construction",
    )
    p.add_argument("--context", type=int, default=4096)
    for name in ("model-output", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--cache", type=Path, default=ROOT / "artifacts/build-cache/flash-next")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--workspace-only",
        action="store_true",
        help="Share certified serial QSA/expert staging without changing kernel arithmetic",
    )
    p.add_argument(
        "--expert-storage-only",
        action="store_true",
        help="Rebuild compressed expert address tables with unchanged row/arithmetic contracts",
    )
    p.add_argument(
        "--auxiliary-only",
        nargs="+",
        choices=(
            "router",
            "histogram",
            "injection",
            "hc-down",
            "hc-up",
            "expert-grid",
            "ple",
            "qsa",
        ),
        help="Rebuild selected auxiliary families with unchanged model arithmetic",
    )
    p.add_argument(
        "--hc-only",
        action="store_true",
        help="Measure and rebuild HC down row selections with unchanged arithmetic",
    )
    p.add_argument(
        "--prefill-only",
        action="store_true",
        help="Rebuild only declared prefill projections; preserve other certified bindings",
    )
    p.add_argument("--prefill-tuning-rows", type=int, nargs="+", default=[512, 2048, 4096])
    p.add_argument(
        "--projection-kinds",
        nargs="+",
        choices=["q4", "w8", "swiglu"],
        help="Rebuild selected projection kinds; preserve other projections and auxiliary bindings",
    )
    p.add_argument(
        "--projection-shape",
        nargs=2,
        type=int,
        action="append",
        metavar=("N", "K"),
        help="Restrict rebuilds to these output/input dimensions; repeat for multiple geometries",
    )
    args = p.parse_args()
    if args.mtp_checkpoint and not args.checkpoint:
        p.error("mtp-checkpoint requires initial construction with --checkpoint")
    if args.vision_checkpoint and not args.checkpoint:
        p.error("vision-checkpoint requires initial construction with --checkpoint")
    if args.workspace_only:
        if (
            args.model is None
            or args.checkpoint
            or args.expert_storage_only
            or args.hc_only
            or args.plan_only
            or args.prefill_only
            or args.projection_kinds
            or args.projection_shape
            or args.auxiliary_only
        ):
            p.error("workspace-only requires --model and an independent workspace rebuild")
        from tools.model.flash_next.workspace import rebuild as rebuild_workspace

        rebuild_workspace(args)
        return
    if args.auxiliary_only:
        if (
            args.model is None
            or args.checkpoint
            or args.expert_storage_only
            or args.hc_only
            or args.plan_only
            or args.prefill_only
            or args.projection_kinds
            or args.projection_shape
        ):
            p.error("auxiliary-only requires --model and a complete auxiliary family rebuild")
        from tools.model.auxiliary_tuning import rebuild as rebuild_auxiliary

        args.families = args.auxiliary_only
        rebuild_auxiliary(args)
        return
    if args.hc_only:
        if (
            args.checkpoint
            or args.expert_storage_only
            or args.plan_only
            or args.prefill_only
            or args.projection_kinds
            or args.projection_shape
        ):
            p.error("hc-only requires --model and a complete HC down rebuild")
        from tools.model.flash_next.hc import rebuild as rebuild_hc

        rebuild_hc(args)
        return
    if args.expert_storage_only:
        if (
            args.checkpoint
            or args.plan_only
            or args.prefill_only
            or args.projection_kinds
            or args.projection_shape
        ):
            p.error("expert-storage-only requires --model and a complete storage rebuild")
        from tools.model.flash_next.expert_storage import rebuild as rebuild_expert_storage

        rebuild_expert_storage(args)
        return
    if args.checkpoint:
        if not args.source_index:
            p.error("--checkpoint requires --source-index")
        if args.prefill_only or args.projection_kinds or args.projection_shape:
            p.error("Initial construction includes all projections")
        from tools.model.flash_next.bootstrap import build

        build(args)
        return
    if args.model.resolve() == args.model_output.resolve():
        p.error("model-output must differ from the immutable input")
    if any(r < 1 for r in args.prefill_tuning_rows):
        p.error("prefill-tuning-rows must be positive")
    if any(n < 1 or k < 1 for n, k in args.projection_shape or []):
        p.error("projection-shape dimensions must be positive")
    rebuild(args)


if __name__ == "__main__":
    main()
