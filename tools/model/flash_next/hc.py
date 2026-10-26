"""Measure exact HC projection portfolios against the published cubins."""

import ctypes
import math
import re
import shutil
from types import SimpleNamespace

from tools.build.files import file_hash, write_json
from tools.model.build_cache import identity
from tools.model.host_compiler import Compiler
from tools.model.publication import atomic_model, clone_model, load_model
from tools.model.row_selection import choices
from tools.model.row_tuning import RowTuner, launch
from tools.operators.abi import evaluate


class Reference:
    """Execute the original published ABI on the current torch stream."""

    def __init__(self, compiler, kernel):
        from cuda.bindings import driver as cu

        self.cu, self.host = cu, compiler.host(kernel)
        self.adapter = SimpleNamespace(func=self.__call__)
        self.module = None
        self.module = self.checked(cu.cuModuleLoadData(compiler.read(kernel["module"])))[0]
        try:
            self.function = self.checked(
                cu.cuModuleGetFunction(self.module, kernel["symbol"].encode())
            )[0]
            self.checked(
                cu.cuFuncSetAttribute(
                    self.function,
                    cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    kernel["shared_memory_bytes"],
                )
            )
        except Exception:
            self.close()
            raise

    @staticmethod
    def checked(result):
        if int(result[0]) != 0:
            raise RuntimeError(f"Published HC reference CUDA failure: {result[0]}")
        return result[1:]

    def close(self):
        if self.module is not None:
            self.checked(self.cu.cuModuleUnload(self.module))
            self.module = None

    def __call__(self, x, w, out, rows, *, stream=None):
        import torch

        values = dict(A=x, W=w, C=out)
        arguments = []
        for argument in self.host["ordered_arguments"]:
            name = argument["value"]
            arguments.append(
                ctypes.c_uint64(values[name.removesuffix(".data_ptr()")].data_ptr())
                if argument["ctype"].endswith("c_void_p")
                else ctypes.c_int32(rows)
            )
        pointers = (ctypes.c_void_p * len(arguments))(
            *(ctypes.addressof(value) for value in arguments)
        )
        config = self.host["launch_expressions"]
        dimensions = dict(m=rows, rows=rows, M=rows)
        self.checked(
            self.cu.cuLaunchKernel(
                self.function,
                *(evaluate(config["gridDim" + axis], dimensions) for axis in "XYZ"),
                *(evaluate(config["blockDim" + axis], dimensions) for axis in "XYZ"),
                evaluate(config["sharedMemBytes"], dimensions),
                self.cu.CUstream(
                    torch.cuda.current_stream().cuda_stream if stream is None else stream
                ),
                ctypes.addressof(pointers),
                0,
            )
        )


def inventory(package, compiler, kind="down"):
    """Validate typed HC operands and down consumers against checkpoint geometry."""
    if kind not in ("down", "injection"):
        raise ValueError("Unknown HC projection kind")
    text = package["config_signature"]["text"]
    rank, streams, hidden = (text[key] for key in ("hc_lowrank", "hc_count", "hidden_size"))
    outputs = rank if kind == "down" else streams
    weight_role = "down" if kind == "down" else "inject"
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    kernels = [k for k in package["kernels"] if "/rows/" not in k["name"]]
    pointers = {k["name"]: compiler.pointers(k) for k in kernels}
    sections = {}
    for k in kernels:
        sections.setdefault(k["name"].rsplit("/", 1)[0], []).append(k)
    result = {}
    for old in kernels:
        p = pointers[old["name"]]
        weight = p.get("W", {}).get("name", "")
        if not re.fullmatch(
            rf"W_(?:blk\.\d+\.hc_(?:attn|ffn)_|output_hc_){weight_role}\.weight_value", weight
        ):
            continue
        if set(p) != {"A", "W", "C"} or old["name"] not in dynamic:
            continue  # Fixed-row partial/finish has its own arithmetic contract.
        descriptor = compiler.buffers[weight]
        if descriptor["dtype"] != "bf16" or descriptor["shape"] != [outputs, streams * hidden]:
            raise ValueError(f"HC {kind} weight differs from the checkpoint geometry")
        if any(v.get("offset", 0) for v in p.values()):
            raise ValueError(f"HC {kind} requires whole native operands")
        for role, width in (("A", streams * hidden), ("C", outputs)):
            operand = compiler.buffers[p[role]["name"]]
            if (
                operand["dtype"] != "f16"
                or math.prod(operand["shape"]) < dynamic[old["name"]]["capacity"] * width
            ):
                raise ValueError(f"HC {kind} activation differs from its physical FP16 row view")
        if kind == "injection":
            if streams != 4:
                raise ValueError("HC injection requires four stream gates")
            result.setdefault(False, []).append(old)
            continue
        consumers = sections[old["name"].rsplit("/", 1)[0]]
        separate = any(
            set(pointers[k["name"]]) == {"Down", "Activated"}
            and pointers[k["name"]]["Down"] == p["C"]
            for k in consumers
        )
        fused = any(
            pointers[k["name"]].get("A") == p["C"]
            and pointers[k["name"]].get("W", {}).get("name") == weight.replace("down.", "up.")
            for k in consumers
        )
        if separate == fused:
            raise ValueError("HC down has an ambiguous SiLU consumer contract")
        result.setdefault(fused, []).append(old)
    if not result:
        raise ValueError(f"No dynamic HC {kind} construction contracts found")
    return result, dynamic, rank, streams, hidden


def publish_kind(model, package, compiler, output, cache, kind):
    import torch
    from kernels.model.hyperconnection import hc_projection, hc_injection
    from tools.model.compile import primitive
    from tools.model.flash_next.build import read_weights, source_identity, toolchain

    groups, dynamics, rank, streams, hidden = inventory(package, compiler, kind)
    outputs = rank if kind == "down" else streams
    source = source_identity()
    machine = toolchain()
    output.mkdir(parents=True, exist_ok=True)
    replaced, added, launches, removed, records = {}, [], {}, set(), []
    # Every candidate keeps the sequential per-stream FP32 reductions and the
    # same MMA K16 traversal. Only row/output tiling changes.
    configs = [
        dict(block_m=m, block_n=n)
        for m, n in ((64, 128), (16, 16), (16, 32), (16, 64), (16, 128), (32, 32))
    ]
    if kind == "injection":
        configs = [dict(algorithm="published")] + [
            dict(algorithm="cached", block_m=m) for m in (1, 2, 3)
        ]
    for silu, items in sorted(groups.items()):
        representative = max(items, key=lambda k: dynamics[k["name"]]["capacity"])
        capacity = dynamics[representative["name"]]["capacity"]
        weight_name = compiler.pointers(representative)["W"]["name"]
        weight = compiler.buffers[weight_name]
        real = read_weights(
            model,
            {"W": dict(kind="buffer", name=weight_name)},
            compiler.buffers,
        )["W"]
        reference = Reference(compiler, representative)
        tuner = RowTuner(
            cache / "hc",
            dict(
                source=source,
                toolchain=machine,
                weight=weight["data"],
                reference=representative["module"],
            ),
            capacity=capacity,
        )

        def build(config):
            options = dict(config)
            algorithm = options.pop("algorithm", "matrix")
            if algorithm == "published":
                return reference
            if algorithm == "cached":
                return primitive(
                    hc_injection,
                    capacity,
                    outputs,
                    streams * hidden,
                    dynamic_rows=True,
                    dtype="float16",
                    **options,
                )
            return primitive(
                hc_projection,
                capacity,
                outputs,
                streams * hidden,
                dtype="float16",
                dynamic_rows=True,
                streams=streams,
                reduction_groups=streams if kind == "down" else 1,
                group_m=4,
                silu=silu,
                **options,
            )

        def inputs(rows):
            for scale in (1.0, 0.03125):
                x = (torch.randn(rows, streams * hidden, device="cuda") * scale).half()
                expected = torch.empty(rows, outputs, device="cuda", dtype=torch.float16)
                reference(x, real, expected, rows)

                def fixture(kernel, config):
                    storage = torch.full(
                        (rows * outputs + 31,), 91.0, device="cuda", dtype=torch.float16
                    )
                    out = storage[:-31].view(rows, outputs)

                    def run():
                        launch(kernel, x, real, out, rows)

                    def check():
                        torch.cuda.synchronize()
                        if not torch.equal(out, expected):
                            raise AssertionError(
                                f"HC {kind} arithmetic differs at {rows} rows: {config}"
                            )

                        if not bool((storage[-31:] == 91.0).all()):
                            raise AssertionError("HC projection wrote beyond its output")

                    def change():
                        x.neg_()
                        reference(x, real, expected, rows)
                        out.fill_(float("nan"))

                    return run, check, change

                yield fixture

        try:
            selected = tuner.select(
                dict(kind="hc-" + kind, silu=silu, rank=rank, streams=streams, hidden=hidden),
                configs,
                build,
                inputs,
                measure_rows=(
                    [
                        min(n, capacity)
                        for n in (
                            1,
                            2,
                            3,
                            4,
                            8,
                            17,
                            32,
                            64,
                            96,
                            128,
                            129,
                            144,
                            160,
                            176,
                            192,
                            224,
                            256,
                            257,
                            384,
                            512,
                            capacity,
                        )
                    ]
                    if kind == "injection"
                    else None
                ),
            )
            selected_kernels = {}
            for old in items:
                rows = dynamics[old["name"]]["capacity"]
                for suffix, config in choices(selected, rows):
                    key = (
                        "hc-"
                        + kind
                        + "-"
                        + identity(dict(silu=silu, capacity=capacity, config=config))[:20]
                    )
                    if key not in selected_kernels:
                        selected_kernels[key] = (
                            None if config.get("algorithm") == "published" else build(config)
                        )
                    item, dynamic = compiler.emit(
                        old,
                        old["name"] + suffix,
                        compiler.pointers(old),
                        rows,
                        key=key,
                        kernel=selected_kernels[key],
                    )
                    if suffix:
                        added.append(item)
                    else:
                        replaced[old["name"]] = item
                    launches[item["name"]] = dynamic
                removed.update(
                    k["name"]
                    for k in package["kernels"]
                    if k["name"].startswith(old["name"] + "/rows/")
                )
            records.append(dict(kind=kind, silu=silu, bindings=len(items), selection=selected))
        finally:
            torch.cuda.synchronize()
            reference.close()
    package["kernels"] = [
        replaced.get(k["name"], k) for k in package["kernels"] if k["name"] not in removed
    ] + added
    package["dynamic_batch_kernels"] = [
        launches.pop(k["name"], k)
        for k in package["dynamic_batch_kernels"]
        if k["name"] not in removed
    ] + list(launches.values())
    package["plan_kernel_contracts"] = sorted(
        (set(package["plan_kernel_contracts"]) - removed)
        | set(replaced)
        | {k["name"] for k in added}
    )
    write_json(output / "hc-selection.json", dict(source=source, records=records))
    return records


def publish(model, package, compiler, output, cache, *, kinds=("down", "injection")):
    records = []
    for kind in kinds:
        records.extend(publish_kind(model, package, compiler, output / kind, cache, kind))
    write_json(output / "hc-selection.json", dict(records=records))
    return records


def rebuild(args):
    from tools.model.compact import compact_package

    data, origin, package = load_model(args.model)
    if data["architecture"] != "flash_next":
        raise ValueError("HC rebuilding requires Flash Next")
    args.output.mkdir(parents=True, exist_ok=True)
    with atomic_model(args.model_output) as staging:
        root = clone_model(args.model, staging, origin)
        compiler = Compiler(
            origin, root, args.output, {b["name"]: b for b in data["metadata"]["buffers"]}
        )
        records = publish(args.model, package, compiler, args.output, args.cache)
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
    write_json(
        args.output / "results.json",
        dict(
            complete=True,
            source_model=file_hash(args.model / "cache/model.json"),
            execution_package=digest,
            groups=len(records),
        ),
    )
    print("HC COMPLETE", args.model_output, digest, flush=True)
