"""Construct shared token rows and owner-addressed causal state from one recipe."""

import copy
import re

from tools.model.compile import construct
from tools.model.private_rows import PrivateRows


PRIVATE = (
    "gdn-conv-",
    "gdn-sequence-",
    "ple-conv-",
    "ple-history-",
    "qsa-prepare-",
    "qsa-kv-store-",
    "index-query-",
    "index-compress-",
    "index-pending-",
    "index-scores-",
    "index-hist-",
    "index-choose-",
    "index-counts-",
    "index-offsets-",
    "index-scatter-",
    "qsa-sparse-",
    "qsa-merge-",
)
INTERFACES = {
    "gdn-conv": (0,),
    "gdn-sequence": (3, 4, 7),
    "ple-conv": (0, 1, 4),
    "ple-history": (0,),
    "qsa-prepare": (0, 1, 2),
    "index-query": (0,),
    "index-compress": (0,),
    "index-pending": (0,),
    "qsa-merge": (4,),
}
TABLE, SCRATCH = "FlashBatchPointers", "FlashPrivateScratch"


def kind(label):
    return next((p[:-1] for p in PRIVATE if label.startswith(p)), None)


def sections(plan, names):
    layer, stage = 0, "pre"
    for index, ((_, args), label) in enumerate(zip(plan["ops"], plan["labels"])):
        previous = layer
        layers = [
            int(name.split(".")[1])
            for tensor in args
            if (name := names.get(tensor.untyped_storage()._cdata, "")).startswith("W_blk.")
        ]
        if layers:
            layer = min(layers)
        if label.startswith("dense-quant-") and args[0] is plan["ple_embedding"]:
            layer = 1
        if index == 1 or layer != previous:
            stage = "ple" if layer == 1 else "pre"
        private = kind(label)
        if index >= plan["body_count"]:
            section = "head"
        elif index == 0:
            section = "begin"
        elif private:
            section = f"layer{layer}/" + (
                "ple_private" if private.startswith("ple-") else "private"
            )
            stage = "pre" if private.startswith("ple-") else "post"
        elif label == "gdn-history-copy":
            section = f"layer{layer}/copy"
        else:
            section = f"layer{layer}/" + ("pre" if label.startswith("index-qk:") else stage)
        yield section, label, args


class Rows:
    def __init__(self, bindings):
        self.b = bindings
        self.p = bindings.planner
        self.roles, self.strides = {}, {}
        self.reference = self.p.trace(1)
        self.entries = list(sections(self.reference, self.b.names))
        for key, role in (
            ("embedding", "M1_Embedding"),
            ("ple_embedding", "M1_Ple"),
            ("residual", "Residual"),
            ("output", "Logits"),
            ("token", "Selected"),
        ):
            self.interface(self.reference[key], role, 1)
        for _, label, args in self.entries:
            for index in INTERFACES.get(kind(label), ()):
                name = self.b.buffer(args[index])["name"]
                self.interface(args[index], name, 1)

    def interface(self, tensor, role, rows):
        identity = tensor.untyped_storage()._cdata
        size = tensor.numel() * tensor.element_size() // rows
        old = self.roles.setdefault(identity, role)
        if old != role or self.strides.setdefault(role, size) != size:
            raise ValueError("Inconsistent Flash row interface")

    def aliases(self, plan, rows):
        result = {}
        for key, role in (
            ("embedding", "M1_Embedding"),
            ("ple_embedding", "M1_Ple"),
            ("residual", "Residual"),
            ("output", "Logits"),
            ("token", "Selected"),
        ):
            result[plan[key].untyped_storage()._cdata] = role
        reference = [(kind(label), args) for _, label, args in self.entries if kind(label)]
        current = [
            (kind(label), args) for _, label, args in sections(plan, self.b.names) if kind(label)
        ]
        if [k for k, _ in reference] != [k for k, _ in current]:
            raise ValueError("Shared/private Flash arithmetic order differs")
        for (key, old), (_, args) in zip(reference, current):
            for index in INTERFACES.get(key, ()):
                role = self.roles[old[index].untyped_storage()._cdata]
                tensor = args[index]
                if tensor.numel() * tensor.element_size() // rows != self.strides[role]:
                    raise ValueError("Shared Flash interface stride differs")
                result[tensor.untyped_storage()._cdata] = role
        return result

    def prepare(self, plan):
        for _, label, args in sections(plan, self.b.names):
            if label.startswith("dense-quant-"):
                self.b.zero(args[1], f"ZeroMask{args[1].numel()}")
            elif label.startswith(("expert-gu-", "expert-down-")):
                self.b.zero(args[3], "E8PPatchZero")
            elif label.startswith("ple-conv-"):
                self.b.names[args[-2].untyped_storage()._cdata] = "PleConvHalf"
                self.b.keep.append(args[-2])

    def shared(self):
        storage_names = {}
        self.p.batch_body = True
        try:
            plan = self.p.trace(1024)
            head = self.p.trace(128, all_logits=True)
        finally:
            self.p.batch_body = False
        for selected, rows in ((plan, 1024), (head, 128)):
            self.prepare(selected)
            aliases = self.aliases(selected, rows)
            for _, _, args in sections(selected, self.b.names):
                for tensor in args:
                    identity = tensor.untyped_storage()._cdata
                    old = self.b.names.get(identity)
                    if old and self.b.scopes[old] in ("weights", "sequence"):
                        continue
                    if identity in storage_names:
                        continue
                    role = aliases.get(identity)
                    if role:
                        name = "FlashDecode_" + role
                        count = (128 if role in ("Logits", "Selected") else 1024) * self.strides[
                            role
                        ]
                    else:
                        name = f"FlashRows{rows}_Scratch{len(storage_names)}"
                        count = tensor.untyped_storage().nbytes()
                    if name not in self.b.buffers:
                        from tools.model.flash_next.bindings import DTYPES

                        self.b.fixed(name, DTYPES[tensor.dtype], [count // tensor.element_size()])
                    self.b.names[identity] = name
                    storage_names[identity] = name
                    self.b.keep.append(tensor)
            for section, label, args in sections(selected, self.b.names):
                if (
                    kind(label)
                    or label == "gdn-history-copy"
                    or (section == "head") != (rows == 128)
                ):
                    continue
                self.b.bind("flash_decode", section, label, args, rows, dynamic=True)
        self.b.buffer(head["token"], "FlashDecode_Selected")
        for recipe, args in head["greedy_ops"]:
            self.b.bind("flash_decode", "head", recipe.key, args, 128, dynamic=True)
        # One complete bounded dynamic implementation is the canonical row choice.
        for kernel in list(self.b.kernels):
            if not kernel["name"].startswith("flash_decode/"):
                continue
            launch = next(d for d in self.b.dynamic if d["name"] == kernel["name"])
            suffix = f"/rows/1-{launch['capacity']}"
            variant, dynamic = copy.deepcopy(kernel), copy.deepcopy(launch)
            variant["name"] += suffix
            dynamic["name"] += suffix
            self.b.kernels.append(variant)
            self.b.dynamic.append(dynamic)
        self.b.fixed("FlashMixedResidual", "f16", [1024, 10240])
        self.b.fixed("FlashMixedHeadResidual", "f16", [128, 10240])
        return plan

    def private(self):
        row_buffers = {
            self.b.buffer(t)["name"]
            for t in self.b.keep
            if t.untyped_storage()._cdata in self.roles
        }
        private = PrivateRows(
            self.b.weights,
            row_buffers,
            SCRATCH,
            lambda name: re.sub(r"^State_\d+_", "State_", name),
            128,
            48,
        )
        work = []
        for section, label, args in self.entries:
            if not kind(label):
                continue
            recipe = self.p.kernels[label.rsplit(":", 1)[-1]]
            fn, _ = construct(recipe.build)
            pointers = {
                fn.buffer_map[p].name: self.b.buffer(t)
                for p, t in zip((p for p in fn.params if p in fn.buffer_map), args)
            }
            modes, pointers = private.bind(fn, pointers)
            work.append((section, label, recipe, modes, pointers))
        offsets, size, columns = private.layout(first=("M1_HistoryOut",))
        self.b.fixed(SCRATCH, "u8", [size], "sequence")
        self.b.fixed(TABLE, "u64", [128 * (48 * len(columns) + 10)])
        self.columns = columns
        slots = {}
        for section, label, recipe, modes, pointers in work:
            layer = int(section.split("/")[0][5:])
            program = (
                "flash_decode_ple"
                if kind(label).startswith("ple-")
                else f"flash_decode_state/layer{layer}"
            )
            slot = slots.get(program, 0)
            slots[program] = slot + 1

            def build(recipe=recipe, modes=modes, pointers=pointers, layer=layer):
                return private.wrap(recipe.build(), modes, pointers, layer, TABLE)[0]

            # Determine the surviving ABI without compiling or allocating CUDA weights.
            fn, _ = construct(recipe.build)
            _, bound = private.wrap(fn, modes, pointers, layer, TABLE)
            for argument in bound.values():
                if argument["name"] in row_buffers:
                    argument["name"] = "FlashDecode_" + self.roles_for_name(argument["name"])
            self.b.emit(
                f"{program}/k{slot}",
                build,
                bound,
                128,
                key="private-" + recipe.key,
                construction_dynamic=False,
            )
            if kind(label) == "gdn-conv":
                from kernels.model.independent_rows import history_commit

                history = self.b.buffer(args_history(self.reference))["name"]
                if offsets[history] != 0:
                    raise ValueError("Private convolution history must start its scratch arena")
                self.b.emit(
                    f"{program}/k{slot + 1}",
                    lambda: history_commit(
                        len(columns),
                        columns.index(SCRATCH),
                        columns.index("State_conv"),
                        48 * len(columns),
                    ),
                    {
                        "Pointers": dict(
                            kind="buffer_slice", name=TABLE, offset=layer * len(columns) * 8
                        )
                    },
                    128,
                    key="private-history-commit",
                )
                slots[program] += 1
        from kernels.model import decode_packet

        stride = 48 * len(columns)
        self.b.emit(
            "flash_decode/packet/inputs",
            lambda: decode_packet.inputs(stride, self.p.capacity, 0, False),
            {
                "Pointers": pointer(TABLE),
                "Embedding": pointer("FlashDecode_M1_Embedding"),
                "Ple": pointer("FlashDecode_M1_Ple"),
            },
            128,
            key="packet-inputs",
        )
        self.b.emit(
            "flash_decode/packet/end",
            lambda: decode_packet.end(stride, 512, False),
            {"Pointers": pointer(TABLE), "Hidden": pointer("FlashDecode_Residual")},
            128,
            key="packet-end",
        )
        self.b.emit(
            "flash_decode/packet/outputs",
            lambda: decode_packet.outputs(stride, self.p.V),
            {
                "Pointers": pointer(TABLE),
                "Logits": pointer("FlashDecode_Logits"),
                "Selected": pointer("FlashDecode_Selected"),
            },
            128,
            key="packet-outputs",
        )

    def roles_for_name(self, name):
        roles = {role for identity, role in self.roles.items() if self.b.names[identity] == name}
        if len(roles) != 1:
            raise ValueError("Ambiguous single-row interface name")
        return roles.pop()

    def causal(self):
        plan = self.p.trace(512)
        self.prepare(plan)
        self.b.names[plan["index_scores"].untyped_storage()._cdata] = "TargetIndexScoresM4096"
        aliases = self.aliases(plan, 512)
        for identity, role in aliases.items():
            self.b.names[identity] = "FlashDecode_" + role
        for section, label, args in sections(plan, self.b.names):
            if not kind(label):
                continue
            section = section.replace("ple_private", "ple")
            for tensor in args:
                self.b.buffer(tensor)
            self.b.bind("causal_rows/target", section, label, args, 512, dynamic=True)

    def layout(self):
        return dict(
            hidden=2560,
            layers=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)],
            row_strides=self.strides,
            profiles={"512": "flash_recurrent"},
            segment_tokens=512,
            small_mixed_shapes=[],
            state_columns=self.columns,
            preparation_programs={},
        )


def pointer(name):
    return dict(kind="buffer", name=name)


def args_history(plan):
    return next(
        args[0]
        for (_, args), label in zip(plan["ops"], plan["labels"])
        if label == "gdn-history-copy"
    )
