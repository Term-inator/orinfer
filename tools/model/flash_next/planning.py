"""Shape-only Flash arithmetic planning for native AOT construction.

The reference supplies arithmetic and operand views. Fake tensors preserve
storage aliases and byte offsets without loading model weights onto CUDA.
An MTP planner takes its matching target planner, shares its output operands,
and gives layer 48 its own QSA state and position controls.
"""

from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tools.model.flash_next.native import Model
from tools.model.flash_next.config import validate, validate_pair


@dataclass(frozen=True)
class Recipe:
    key: str
    build: object


class Planner(Model):
    def __init__(self, assembly, context, output, *, target=None):
        if type(context) is not int or not 4096 <= context <= 262144:
            raise ValueError("Native Flash construction needs context in 4096..262144")
        self.config = assembly["config"]
        validate(self.config, context)
        self.is_mtp = self.config["quantization_config"].get("component") == "mtp"
        if self.is_mtp:
            if target is None or target.is_mtp or target.capacity != context:
                raise ValueError("MTP planning requires a matching target planner")
            validate_pair(self.config, target.config, context)
        elif target is not None:
            raise ValueError("Only MTP planning shares a target")
        self.capacity, self.output = context, output
        self.draft_vocab, self.decode_w8_optimized = None, True
        self.layer_ids = (48,) if self.is_mtp else tuple(range(48))
        self.weights, self.states, self.plans, self.kernels = {}, {}, {}, {}
        self.workspaces, self.prefix_divisors = {}, {}
        self.prefill_workspace_rows, self.position = 0, 0
        self.batch_body = False
        self.storage_names, self.tensors = {}, {}
        self.mode = target.mode if self.is_mtp else FakeTensorMode()
        types = {
            name: getattr(torch, dtype)
            for name, dtype in dict(
                f16="float16",
                bf16="bfloat16",
                f32="float32",
                i8="int8",
                u16="uint16",
                u32="uint32",
                i32="int32",
            ).items()
        }

        def tensor(name, shape, dtype):
            value = torch.empty(shape, dtype=types[dtype], device="cuda")
            self.storage_names[value.untyped_storage()._cdata] = name
            self.tensors[name] = value
            return value

        with self.mode:
            for name, buffer in assembly["buffers"].items():
                tensor(name, buffer["shape"], buffer["dtype"])
            for name, operands in assembly["logical"].items():
                if isinstance(operands, str):
                    self.weights[name] = self.tensors[operands]
                elif set(operands) == {"weight", "scale"}:
                    self.weights[name] = tuple(
                        self.tensors[operands[s]] for s in ("weight", "scale")
                    )
                else:
                    self.weights[name] = {
                        key: value if key == "parity_xor" else self.tensors[value]
                        for key, value in operands.items()
                    }
            if self.is_mtp:
                if "output.weight" in self.weights:
                    raise ValueError("MTP must share the target output head")
                self.weights["output.weight"] = target.weights["output.weight"]
                for value in self.weights["output.weight"]:
                    identity = value.untyped_storage()._cdata
                    name = target.storage_names[identity]
                    self.tensors[name] = value
                    self.storage_names[identity] = name
            for layer in self.layer_ids:
                if not self.is_mtp and layer % 4 != 3:
                    shapes = dict(conv=((1, 3, 10240), "f16"), gdn=((48, 128, 128), "f32"))
                else:
                    shapes = {suffix: ((context, 2, 256), "i8") for suffix in ("key", "value")}
                    shapes.update(
                        {suffix + "_scale": ((context, 2, 4), "f16") for suffix in ("key", "value")}
                    )
                    shapes.update(
                        index=(((context + 3) // 4, 128), "f16"), pending=((4, 128), "f16")
                    )
                for suffix, (shape, dtype) in shapes.items():
                    name = f"{layer}:{suffix}"
                    self.states[name] = tensor(f"State_{layer}_{suffix}", shape, dtype)
                    if (self.is_mtp or layer % 4 == 3) and suffix != "pending":
                        self.prefix_divisors[name] = 4 if suffix == "index" else 1
            if not self.is_mtp:
                self.states["ple"] = tensor("State_ple", (9, self.C * self.H), "f16")
            prefix = "Mtp" if self.is_mtp else ""
            self.position_gpu = tensor(prefix + "Position", (1,), "i32")
            self.lengths = tensor(prefix + "Length", (1,), "i32")
            self.position_out = tensor(prefix + "PositionOut", (1,), "i32")

    def kernel(self, key, build):
        if key not in self.kernels:
            self.kernels[key] = Recipe(key, build)
        return self.kernels[key]

    def trace(self, rows, *, verify=False, all_logits=False):
        with self.mode:
            self.position = self.capacity - rows
            try:
                return self.plan(rows, verify=verify, all_logits=all_logits)
            finally:
                self.position = 0
