"""Add fixed MTP operand tables while retaining a native Qwen package's measured compute."""

import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import re
import types

from tools.build.files import write_json
from tools.model.publication import atomic_model, clone_model, load_model, AssetReader


def attention_geometry(kernel, source, dimension, head_group=None):
    call = re.search(
        r"= attn_split_count\([^,]+,\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,[^;]*)?\);", source
    )
    independent = call is None
    if independent:
        split = re.search(r"split_count\[0\] = (\d+);", source)
        if split is None or head_group is None:
            raise ValueError("Expected the registered Qwen attention split geometry")
        group, block_m, target, splits = (
            head_group,
            16,
            16,
            int(split[1]),
        )
    else:
        group, block_m, target, splits = map(int, call.groups())
    if target != 16:
        raise ValueError("Unsupported attention split target")
    result = dict(
        block_m=block_m,
        target_tiles=target,
        max_splits=splits,
        head_group=group,
        independent_queries=independent,
    )
    tiles = {int(n) for n in re.findall(r"\btile \* (32|64)\b", source)}
    if len(tiles) != 1:
        raise ValueError("Expected one registered attention KV tile")
    result.update(block_n=tiles.pop(), threads=kernel["block"][0], stages=int("packed_k" in source))
    return result


def chunk_geometry(kernel, source):
    """Recover the registered WY chunk and value tile without retuning."""
    loop = re.search(r"for \(int chunk = 0; chunk < \(\(m \+ (\d+)\) >> (\d+)\)", source)
    if loop is None or int(loop[1]) + 1 != 1 << int(loop[2]):
        raise ValueError("Expected the registered GDN chunk loop")
    grid = kernel["grid"][0]
    if type(grid) is not int or grid not in (2, 4, 8):
        raise ValueError("Expected the registered GDN value tile")
    return dict(C=int(loop[1]) + 1, BV=128 // grid, threads=kernel["block"][0])


class ExistingWeights:
    def __init__(self, metadata, scopes):
        self.buffers = {b["name"]: copy.deepcopy(b) for b in metadata["buffers"]}
        self.scopes = dict(scopes)

    def fixed(self, name, dtype, shape, scope="workspace"):
        if name not in self.buffers:
            self.buffers[name] = dict(
                name=name,
                dtype=dtype,
                shape=shape,
                layout="native_contiguous",
                alignment=256,
                access="read_write",
            )
            self.scopes[name] = scope
        elif self.buffers[name]["dtype"] != dtype or self.buffers[name]["shape"] != shape:
            raise ValueError(f"MTP reconstruction changes the physical buffer {name}")
        return name

    def immutable(self, name, *, cast=None, shape=None):
        target = name if cast is None else f"Runtime.{name}.{cast}"
        if target not in self.buffers:
            raise ValueError(f"Missing registered immutable tensor {target}")
        return target

    def combined(self, name, inputs):
        if name not in self.buffers:
            raise ValueError(f"Missing registered joined tensor {name}")
        return name

    def constant(self, name, value):
        if name not in self.buffers:
            raise ValueError(f"Missing registered constant {name}")
        return name


def recipes(b, package, origin):
    """Reconstruct private arithmetic only; matrix implementations stay unchanged."""
    from tools.model.compile import primitive
    from kernels.model.gdn_sequence import _gdn_sequence
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.model.attention_rows import causal_attention_int8, merge_attention_int8
    from tools.model.qwen35 import mtp, gdn, moe
    from tools.operators.abi import evaluate

    originals = {k["name"]: k for k in package["kernels"]}
    dynamic = {d["name"]: d for d in package["dynamic_batch_kernels"]}
    reader = AssetReader(origin)
    aot = b.aot
    actual_emit = aot.emit

    def record(
        name, key, make, bindings, rows=4096, *, dynamic=True, row_divisor=1, row_multiplier=1
    ):
        if name in originals:
            aot.recipes[name] = (key, make, bindings, rows, dynamic, row_divisor, row_multiplier)
        return dict(kind="kernel", name=name)

    aot.emit = record
    b.projection = lambda *args, **kwargs: None
    b.rotation = lambda *args, **kwargs: None
    b.norm_rotation = lambda *args, **kwargs: None
    b.row_tuner = types.SimpleNamespace(
        gdn_sequence=lambda: dict(configs=[dict(token_unroll=1)], default="0", selections=[]),
        gdn_ab=lambda: dict(configs=[{}], default="0", selections=[]),
    )

    def convolution(self, name, bindings):
        prefix = "qwen_rows/" + name
        for actual, kernel in originals.items():
            if actual != prefix and not actual.startswith(prefix + "/rows/"):
                continue
            host = aot.compiler.host(kernel)
            expression = host["launch_expressions"]["gridDimX"]

            def grid(rows):
                return evaluate(expression, dict(m=rows, M=rows, rows=rows, batch=rows))

            first = grid(1)
            tile = next((n - 1 for n in range(2, 129) if grid(n) != first), None)
            if tile is None:
                raise ValueError("Unrecognized causal convolution token tile")

            def make(tile=tile):
                return primitive(
                    gdn_conv_prep,
                    1,
                    self.capacity,
                    tile_tokens=tile,
                    dynamic_tokens=True,
                    key_heads=self.hk,
                    value_heads=self.hv,
                )

            record(
                actual,
                "retained-conv-" + kernel["module"]["sha256"],
                make,
                bindings,
                dynamic.get(actual, {}).get("capacity", 4096),
            )

    b.convolution = types.MethodType(convolution, b)

    def attention(self, name, **kwargs):
        prefix = "qwen_rows/" + name
        for actual, kernel in originals.items():
            if actual not in (prefix, prefix + "_merge") and not actual.startswith(
                (prefix + "/rows/", prefix + "_merge/rows/")
            ):
                continue
            merge = actual.startswith(prefix + "_merge")
            reference = originals[actual.replace(prefix + "_merge", prefix, 1)] if merge else kernel
            config = attention_geometry(
                reference,
                reader.read(reference["source"]).decode(),
                self.dimension,
                self.qh // self.kvh,
            )
            if merge:
                config = {
                    k: v for k, v in config.items() if k not in ("block_n", "threads", "stages")
                }
            fn = merge_attention_int8 if merge else causal_attention_int8
            config.update(
                query_heads=self.qh, kv_heads=self.kvh, dimension=self.dimension, gated=True
            )
            if not merge:
                config["context"] = self.context
            capacity = dynamic.get(actual, {}).get("capacity", kwargs.get("rows", 4096))

            def make(fn=fn, config=config, capacity=capacity):
                return primitive(
                    fn, min(capacity, 8) if config["independent_queries"] else capacity, **config
                )

            record(actual, "retained-attention-" + kernel["module"]["sha256"], make, {}, capacity)

    b.attention = types.MethodType(attention, b)
    saved_prefill, saved_moe = gdn.prefill, moe.build
    try:
        # The published recurrent construction is retained, including its arithmetic.
        # Source-only chunked-prefill candidates do not participate in this rebuild.
        gdn.prefill = lambda *args, **kwargs: None
        # Routed and shared FFN operands are session-owned. Retain their
        # measured kernels rather than tuning or rebuilding them here.
        moe.build = lambda *args, **kwargs: None
        b.forward()
        b.head()
        mtp.build(b)
        if b.vision:
            from kernels.vision.features import overlay

            b.emit(
                "begin/overlay",
                f"vision-overlay-{b.context}",
                lambda: primitive(
                    overlay, 4096, b.h, b.context, b.vision["max_features"], dynamic_rows=True
                ),
                dict(
                    Embedding="Hidden", Features="Features", Index="FeatureIndex", Position="Step"
                ),
            )
    finally:
        gdn.prefill, moe.build = saved_prefill, saved_moe
        aot.emit = actual_emit
    # Retain stateful prefill/materialization bodies used by mixed packets.
    # Their shared preparation and all matrix kernels stay in the original bank.
    for name, kernel in originals.items():
        if name.endswith("/private/recurrent_materialize"):
            from kernels.model.gdn_deferred import materialize

            capacity = b.gdn_execution["decode"]["capacity"]

            def make(capacity=capacity):
                return primitive(materialize, capacity, key_heads=b.hk, value_heads=b.hv)

            aot.recipes[name] = (
                "retained-materialize-" + kernel["module"]["sha256"],
                make,
                {},
                b.capacity,
                False,
                1,
                1,
            )
        elif name.endswith("/private/recurrent_chunk"):
            from kernels.model.gdn_chunk import recurrent

            config = chunk_geometry(kernel, reader.read(kernel["source"]).decode())

            def make(config=config):
                return primitive(
                    recurrent, b.capacity, HK=b.hk, HV=b.hv, dynamic_rows=True, **config
                )

            aot.recipes[name] = (
                "retained-chunk-" + kernel["module"]["sha256"],
                make,
                {},
                dynamic[name]["capacity"],
                True,
                1,
                1,
            )
    # Add selected sequential recurrence aliases with their original grid/ABI.
    for name, kernel in originals.items():
        if re.fullmatch(r"qwen_rows/layer\d+/private/recurrent(?:/rows/\d+(?:-\d+)?)?", name):

            def make():
                return primitive(
                    _gdn_sequence,
                    b.capacity,
                    in_place=True,
                    value_tile=16,
                    threads=32,
                    prefetch_qk=True,
                    dynamic_rows=True,
                    key_heads=b.hk,
                    value_heads=b.hv,
                    token_unroll=1,
                )

            aot.recipes[name] = (
                "retained-recurrent-" + kernel["module"]["sha256"],
                make,
                {},
                dynamic.get(name, {}).get("capacity", 4096),
                True,
                1,
                1,
            )


def rebuild(args):
    from tools.model.qwen35.build import Builder, configuration
    from tools.model.qwen35.compiler import Aot
    from tools.model.host_compiler import Compiler
    from tools.model.qwen35 import mtp_packets
    from tools.model.compact import compact_package

    manifest = {}
    data, origin, package = load_model(args.model, plan=manifest)
    if data["architecture"] != "qwen3_5" or manifest.get("mtp", {}).get("packet_table"):
        raise ValueError("Expected a native Qwen MTP package without an operand table")
    config = json.loads((args.model / "config.json").read_text())
    text = configuration(config, manifest["max_context"])
    if config["model_type"] not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError("Incremental MTP rebuild requires dense or MoE Qwen")
    with atomic_model(args.target) as stage:
        operator = clone_model(args.model, stage, origin)
        b = Builder.__new__(Builder)
        b.config, b.text = config, text
        b.standard, b.moe, b.lock = False, config["model_type"] == "qwen3_5_moe", None
        b.capacity = manifest["chunk_tokens"]
        b.head_rows = min(512, b.capacity)
        b.q4_codec = (
            "e8p_q2"
            if config["quantization_config"]["quant_method"] == "orinfer_e8p_q2_int8"
            else "e8p_q4"
        )
        b.context = manifest["max_context"]
        b.h, b.f, b.v = text["hidden_size"], text["intermediate_size"], text["vocab_size"]
        b.hk, b.hv = text["linear_num_key_heads"], text["linear_num_value_heads"]
        b.qh, b.kvh, b.dimension = (
            text["num_attention_heads"],
            text["num_key_value_heads"],
            text["head_dim"],
        )
        b.qkv, b.mixdim = (2 * b.hk + b.hv) * 128, b.hv * 128
        b.qproj, b.kv_width = 2 * b.qh * b.dimension, b.kvh * b.dimension
        b.fullx = b.qproj + 2 * b.kv_width
        b.embedding_q4 = config["quantization_config"].get("embedding") in ("e8p_q4", "e8p_q2")
        b.tied = text.get("tie_word_embeddings", False)
        b.head_weight = "model.language_model.embed_tokens.weight" if b.tied else "lm_head.weight"
        b.weights = ExistingWeights(manifest, data["buffer_scopes"])
        b.gdn_execution = copy.deepcopy(manifest["gdn_execution"])
        b.vision, b.audio = manifest.get("vision"), None
        b.reset = []
        b.kv = dict(manifest["kv_cache"]["buffers"])
        b.aot = Aot(operator, args.output, args.cache, b.weights.buffers, compile_lock=nullcontext)
        b.aot.identity["source_execution_package"] = data["execution_package"]
        b.aot.compiler = Compiler(origin, operator, args.output, b.weights.buffers)
        b.aot.kernels = copy.deepcopy(package["kernels"])
        b.aot.dynamic = copy.deepcopy(package["dynamic_batch_kernels"])
        embedding = next(k for k in package["kernels"] if k["name"] == "qwen_rows/begin/embedding")
        embedding_bindings = b.aot.compiler.pointers(embedding)
        b.signs = {b.h: embedding_bindings["Signs"]["name"]} if b.embedding_q4 else {}
        b.book = embedding_bindings["Book"]["name"] if b.embedding_q4 else None
        recipes(b, package, origin)
        b.mtp = copy.deepcopy(data["metadata"]["mtp"])
        mtp_packets.build(b)
        data["metadata"]["mtp"] = b.mtp
        data["metadata"]["buffers"] = list(b.weights.buffers.values())
        data["buffer_scopes"] = b.weights.scopes
        package["kernels"], package["dynamic_batch_kernels"] = b.aot.kernels, b.aot.dynamic
        package["buffer_contracts"] = [
            {k: v for k, v in buffer.items() if k != "data"}
            for buffer in b.weights.buffers.values()
        ]
        package["toolchain"]["mtp_packet_source"] = b.aot.identity["mtp_packet_source"]
        package["toolchain"]["mtp_packet_reference"] = b.aot.identity["source_execution_package"]
        data["execution_package"] = compact_package(operator, package, stage / "cache/packages")
        write_json(stage / "cache/model.json", data)
    write_json(
        args.output / "result.json",
        dict(model=str(args.target), execution_package=data["execution_package"], complete=True),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("artifacts/build-cache/qwen35-mtp"))
    rebuild(p.parse_args())


if __name__ == "__main__":
    main()
