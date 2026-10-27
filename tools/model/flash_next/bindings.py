"""Bind shape-only Flash arithmetic to native arenas and the exported host ABI."""

import math

import numpy as np
import torch

from tools.model.build_cache import BuildCache
from tools.model.compile import compile_function, construct
from tools.model.host_compiler import Compiler
from tools.model.flash_next.build import source_identity
from tools.model.publication import link_or_copy
from tools.operators.abi import parse_host
from tools.operators.common import export_kernel


DTYPES = {
    getattr(torch, name): dtype
    for name, dtype in dict(
        float16="f16",
        bfloat16="bf16",
        float32="f32",
        int8="i8",
        uint8="u8",
        uint16="u16",
        uint32="u32",
        int32="i32",
    ).items()
}


class Bindings:
    def __init__(self, planner, weights, output, cache, *, shared=None):
        if shared is not None and (
            not planner.is_mtp
            or weights is not shared.weights
            or planner.mode is not shared.planner.mode
            or planner.weights["output.weight"] is not shared.planner.weights["output.weight"]
        ):
            raise ValueError("Draft bindings must share their target arena and head")
        self.planner, self.weights, self.output = planner, weights, output
        self.buffers, self.scopes = weights.buffers, weights.scopes
        self.names = dict(planner.storage_names)
        self.keep = shared.keep if shared is not None else []
        self.keep.extend(planner.tensors.values())
        self.recipes = {}
        self.kernels, self.dynamic, self.groups = (
            (shared.kernels, shared.dynamic, shared.groups) if shared is not None else ([], [], {})
        )
        self.cache = BuildCache(cache)
        self.identity = dict(source=source_identity(), context=planner.capacity)
        self.compiler = (
            shared.compiler
            if shared is not None
            else Compiler(weights.root / "cache", weights.root / "cache", output, self.buffers)
        )
        for name, tensor in planner.tensors.items():
            if name not in self.buffers:
                self.fixed(
                    name,
                    DTYPES[tensor.dtype],
                    list(tensor.shape),
                    "workspace" if name in ("PositionOut", "MtpPositionOut") else "sequence",
                )

    def fixed(self, name, dtype, shape, scope="workspace"):
        if name in self.buffers:
            old = self.buffers[name]
            if old["dtype"] != dtype or math.prod(old["shape"]) != math.prod(shape):
                raise ValueError(f"Inconsistent arena alias: {name}")
            return name
        self.buffers[name] = dict(
            name=name,
            dtype=dtype,
            shape=list(shape),
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        self.scopes[name] = scope
        return name

    def emit(self, name, build, pointers, rows, *, key, dynamic=True, construction_dynamic=None):
        """Publish a transformed primitive through the same cache and host ABI."""
        recipe = self.planner.kernel(key, build)
        artifact_key, (_, _, artifact) = self.artifact(
            recipe, dynamic if construction_dynamic is None else construction_dynamic
        )
        kernel, launch = self.compiler.emit(
            dict(name=name), name, pointers, rows, key=artifact_key, artifact=artifact
        )
        self.kernels.append(kernel)
        if dynamic:
            self.dynamic.append(launch)
        return kernel

    def buffer(self, tensor, name=None, scope="workspace"):
        storage = tensor.untyped_storage()
        identity = storage._cdata
        if identity not in self.names:
            prefix = "DraftScratch" if self.planner.is_mtp else "Scratch"
            name = name or f"{prefix}{len(self.names)}"
            self.fixed(
                name, DTYPES[tensor.dtype], [storage.nbytes() // tensor.element_size()], scope
            )
            self.names[identity] = name
            self.keep.append(tensor)
        name = self.names[identity]
        offset = tensor.storage_offset() * tensor.element_size()
        return (
            dict(kind="buffer_slice", name=name, offset=offset)
            if offset
            else dict(kind="buffer", name=name)
        )

    def zero(self, tensor, name):
        identity = tensor.untyped_storage()._cdata
        if identity not in self.names:
            types = {torch.uint8: np.uint8, torch.uint16: np.uint16}
            if name not in self.buffers:
                self.weights.constants(
                    name + ".safetensors",
                    {name: np.zeros(tuple(tensor.shape), types[tensor.dtype])},
                )
            self.names[identity] = name
            self.keep.append(tensor)

    def artifact(self, recipe, dynamic):
        key = recipe.key + ("-dynamic" if dynamic else "-fixed")
        expert = recipe.key.startswith(("expert-gu-", "expert-down-"))
        if expert and self.planner.is_mtp:
            key += "-resident"
        if key not in self.recipes:
            function, passes = construct(recipe.build, dynamic=dynamic)
            if expert and not self.planner.is_mtp:
                from kernels.model.expert_addresses import indirect_experts

                function = indirect_experts(function)
            # Preserve every free row scalar in the exported launch ABI.
            from kernels.model.rows import explicit_rows

            free = [p for p in function.params if str(p.dtype) == "int32"]
            if not free:
                from tvm import tirx as tir

                free = [
                    p
                    for p in tir.analysis.undefined_vars(function.body, list(function.params))
                    if str(p.dtype) == "int32" and p.name in ("m", "rows", "tokens", "batch")
                ]
            if free:
                function = explicit_rows(function)

            def build(folder):
                kernel = compile_function(function, pass_configs=passes)
                export_kernel(kernel, folder)

            artifact, hit = self.cache.get(dict(**self.identity, recipe=key), build)
            destination = self.output / "aot" / key
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("kernel.cubin", "kernel.cu", "host.txt"):
                link_or_copy(artifact / filename, destination / filename)
            host = parse_host((artifact / "host.txt").read_text())[0]
            self.recipes[key] = function, host, artifact
            print("FLASH AOT", key, "cached" if hit else "compiled", flush=True)
        return key, self.recipes[key]

    def bind(self, program, section, label, args, rows, *, dynamic, name_override=None):
        operations = self.groups.setdefault(program, {}).setdefault(section, [])
        if label == "gdn-history-copy":
            source, destination = (self.buffer(t) for t in args)
            if source["kind"] != "buffer" or destination["kind"] != "buffer":
                raise ValueError("History commits require full native operands")
            operation = dict(
                kind="copy",
                source=source["name"],
                destination=destination["name"],
                bytes=args[0].numel() * args[0].element_size(),
            )
        else:
            key = label.rsplit(":", 1)[-1]
            recipe = self.planner.kernels[key]
            artifact_key, (function, host, artifact) = self.artifact(recipe, dynamic)
            pointers = {}
            values = [function.buffer_map[p] for p in function.params if p in function.buffer_map]
            if len(values) != len(args):
                raise ValueError(
                    f"Arithmetic/operand count mismatch: {label}: {len(values)} != {len(args)}"
                )
            for buffer, tensor in zip(values, args):
                operand = self.buffer(tensor)
                if buffer.name in ("PAddresses", "WSAddresses"):
                    if operand["kind"] != "buffer":
                        raise ValueError("Expert address operands must be complete banks")
                    storage = self.buffers[operand["name"]]
                    table = operand["name"] + "_addresses"
                    self.fixed(table, "u64", [storage["shape"][0]])
                    operand = dict(kind="buffer", name=table)
                pointers[buffer.name] = operand
            slot = sum(
                op["kind"] == "kernel" and op["name"].rsplit("/", 1)[-1].startswith("k")
                for op in operations
            )
            name = name_override or f"{program}/{section}/k{slot}"
            # A dynamic expert projection counts routed assignments, ten per row.
            multiplier = (
                10
                if key.startswith(("expert-gu-", "expert-down-", "rotate-a8-ffn-"))
                else self.planner.C
                if label.startswith("mtp-fc-hidden:")
                else 1
            )
            kernel, launch = self.compiler.emit(
                dict(name=name),
                name,
                pointers,
                rows,
                key=artifact_key,
                artifact=artifact,
                multiplier=multiplier,
            )
            self.kernels.append(kernel)
            if dynamic:
                self.dynamic.append(launch)
            operation = dict(kind="kernel", name=name)
        operations.append(operation)
        return operation

    def trace(self, rows):
        from kernels.model.control import advance
        from kernels.model.flash_control import last_hidden, selection
        from tools.model.flash_next.roles import role

        plan = self.planner.trace(rows)
        self.buffer(plan["index_scores"], f"TargetIndexScoresM{rows}")
        for key, name in (
            ("embedding", f"M{rows}_Embedding"),
            ("ple_embedding", f"M{rows}_Ple"),
            ("output", "Logits"),
            ("token", "Selected"),
        ):
            self.buffer(plan[key], name, "sequence" if key in ("output", "token") else "workspace")
        for (_, args), label in zip(plan["ops"], plan["labels"]):
            if label == "gdn-history-copy":
                self.buffer(args[0], f"M{rows}_HistoryOut")
            if label.startswith("dense-quant-"):
                self.zero(args[1], f"ZeroMask{args[1].numel()}")
            if label.startswith(("expert-gu-", "expert-down-")):
                self.zero(args[3], "E8PPatchZero")
            if label.startswith("ple-conv-"):
                tensor = args[-2]
                name = "PleConvHalf"
                if name not in self.buffers:
                    original = next(
                        n for n in self.weights.source.parts if role(n) == "blk.1.ple_conv1d.weight"
                    )
                    self.weights.constants(
                        name + ".safetensors",
                        {name: self.weights.source.tensor(original).astype(np.float16).squeeze(1)},
                    )
                self.names[tensor.untyped_storage()._cdata] = name
                self.keep.append(tensor)
        program = "decode" if rows == 1 else f"prefill_m{rows}"
        head = f"head_m{rows}"
        section = "begin"
        for index, ((_, args), label) in enumerate(zip(plan["ops"], plan["labels"])):
            if index == plan["body_count"]:
                break
            layers = []
            for tensor in args:
                name = self.names.get(tensor.untyped_storage()._cdata, "")
                if name.startswith("W_blk."):
                    layers.append(int(name.split(".")[1]))
            if layers:
                section = f"layer{min(layers)}"
            self.bind(program, section, label, args, rows, dynamic=rows > 1)
        key = f"advance-{rows}"
        self.planner.kernel(key, lambda: advance(rows))
        self.bind(program, "end", key, (self.planner.position_gpu,), rows, dynamic=rows > 1)
        for index, ((_, args), label) in enumerate(
            zip(plan["ops"][plan["body_count"] :], plan["labels"][plan["body_count"] :])
        ):
            if rows > 1 and index == 4:
                with self.planner.mode:
                    full = args[0].as_strided(
                        (rows, self.planner.H), (self.planner.H, 1), storage_offset=0
                    )
                    gathered = torch.empty((1, self.planner.H), dtype=torch.float16, device="cuda")
                key = f"logical-gather-{rows}"
                self.planner.kernel(key, lambda: last_hidden(rows, self.planner.H))
                self.bind(
                    head,
                    "body",
                    key,
                    (full, gathered),
                    rows,
                    dynamic=True,
                    name_override=f"{head}/body/logical_gather",
                )
                args = (gathered, *args[1:])
            live_rows = rows if index < 4 else 1
            self.bind(head, "body", label, args, live_rows, dynamic=live_rows > 1)
        for (_, args), prefix in zip(plan["greedy_ops"], ("greedy-partials", "greedy-merge")):
            self.bind(head, "body", f"{prefix}-1-{self.planner.V}", args, 1, dynamic=False)
        with self.planner.mode:
            token = torch.empty(1, device="cuda", dtype=torch.int32)
            status = torch.empty_like(token)
        self.buffer(token, "Token", "sequence")
        self.buffer(status, "Status", "sequence")
        self.planner.kernel("selection", selection)
        self.bind(head, "body", "selection", (plan["token"], token, status), 1, dynamic=False)
        return plan
