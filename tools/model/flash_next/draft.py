"""Shared MTP token rows around one owner-local QSA segment.

The draft uses its own resident expert banks and HC mixer, and the target's
single output head. Chronological controls are bound by the publication step.
"""

import copy
import math

from tools.model.flash_next.bindings import DTYPES
from tools.model.flash_next.rows import kind


def sections(plan):
    stage = "pre"
    for index, ((recipe, args), label) in enumerate(zip(plan["ops"], plan["labels"])):
        if index >= plan["body_count"]:
            section = "head"
        elif label.startswith("mtp-"):
            section = "begin"
        elif kind(label):
            section, stage = "private", "post"
        else:
            section = "pre" if label.startswith("index-qk:") else stage
        yield section, recipe, label, args


class DraftRows:
    def __init__(self, bindings):
        self.b, self.p = bindings, bindings.planner
        if not self.p.is_mtp:
            raise ValueError("Draft rows require an MTP planner")

    def alias(self, tensor, name, *, rows=None):
        if tensor.storage_offset() or not tensor.is_contiguous():
            raise ValueError("Draft interfaces require whole contiguous storage")
        if rows is not None:
            if tensor.shape[0] != rows:
                raise ValueError("Draft interface is not token-row contiguous")
            self.b.fixed(name, DTYPES[tensor.dtype], list(tensor.shape))
        else:
            buffer = self.b.buffers.get(name)
            if (
                buffer is None
                or buffer["dtype"] != DTYPES[tensor.dtype]
                or math.prod(buffer["shape"]) < tensor.numel()
            ):
                raise ValueError("Invalid shared draft output arena")
        self.b.names[tensor.untyped_storage()._cdata] = name
        self.b.keep.append(tensor)

    def build(self):
        old = self.p.batch_body
        self.p.batch_body = True
        try:
            body = self.p.trace(512)
            head = self.p.trace(128, all_logits=True)
        finally:
            self.p.batch_body = old
        for plan, rows in [(body, 512), (head, 128)]:
            self.alias(plan["residual"], "DraftRows_Hidden", rows=512 if rows == 512 else None)
            if rows == 512:
                for key, name in [("embedding", "Embedding"), ("condition", "Condition")]:
                    self.alias(plan[key], "DraftRows_" + name, rows=512)
                self.b.buffer(plan["index_scores"], "DraftIndexScoresM512")
                interfaces = {
                    "qsa-prepare": [(0, "QGate"), (1, "Key"), (2, "Value")],
                    "index-query": [(0, "IndexQK")],
                    "qsa-merge": [(4, "Attention")],
                }
                for section, _, label, args in sections(plan):
                    for index, role in interfaces.get(kind(label), []):
                        self.alias(args[index], "DraftRows_" + role, rows=512)
            else:
                self.alias(plan["output"], "FlashDecode_Logits")
                self.alias(plan["token"], "FlashDecode_Selected")
            for section, recipe, label, args in sections(plan):
                if label.startswith("dense-quant-"):
                    self.b.zero(args[1], f"ZeroMask{args[1].numel()}")
                elif label.startswith(("expert-gu-", "expert-down-")):
                    self.b.zero(args[3], "E8PPatchZero")
                if (section == "head") != (rows == 128):
                    continue
                if section == "private":
                    self.b.bind("causal_rows", "draft", label, args, 512, dynamic=True)
                else:
                    self.b.bind("draft_rows", section, label, args, rows, dynamic=True)
            if rows == 128:
                for recipe, args in plan["greedy_ops"]:
                    self.b.bind("draft_rows", "head", recipe.key, args, 128, dynamic=True)
        canonical = [k for k in self.b.kernels if k["name"].startswith("draft_rows/")]
        dynamic = {d["name"]: d for d in self.b.dynamic}
        for kernel in canonical:
            launch = dynamic[kernel["name"]]
            variant, selected = copy.deepcopy(kernel), copy.deepcopy(launch)
            suffix = f"/rows/1-{launch['capacity']}"
            variant["name"] += suffix
            selected["name"] += suffix
            self.b.kernels.append(variant)
            self.b.dynamic.append(selected)
        return body, head
