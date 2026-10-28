"""Native Flash bindings for the shared original-weight ViT encoder."""

import json
import math

import numpy as np

from tools.build.files import file_hash
from tools.model.safetensors_source import Source
from tools.model.vision_encoder import build as build_encoder
from tools.vision.config import validate_vision
from tools.vision.source import fetch_processor

PATCHES = 32768


def overlay(b, name, embedding, rows, *, mtp=False):
    from kernels.vision.features import overlay as feature_overlay

    b.emit(
        name,
        lambda: feature_overlay(rows, b.planner.H, b.planner.capacity, b.planner.capacity),
        {
            k: dict(kind="buffer", name=n)
            for k, n in dict(
                Embedding=embedding,
                Features="Features",
                Index="MtpFeatureIndex" if mtp else "FeatureIndex",
                Position="MtpPosition" if mtp else "Position",
            ).items()
        },
        rows,
        key=f"vision-overlay-{rows}-{mtp}",
        dynamic=rows > 1,
    )


def assemble(weights, checkpoint):
    """Convert one bounded original BF16 matrix at a time to the FP16 stream."""
    identity = json.loads((checkpoint / "source.json").read_text())
    quant = weights.source.config["quantization_config"]
    if (
        identity["repo"] != quant["source"]
        or identity["revision"] != quant["source_revision"]
        or identity["sha256"] != file_hash(checkpoint / "model.safetensors")
    ):
        raise ValueError("Vision checkpoint differs from the target's pinned source")
    config = json.loads((checkpoint / "config.json").read_text())
    if config["vision_config"] != weights.source.config["vision_config"]:
        raise ValueError("Vision checkpoint configuration differs")
    validate_vision(config["vision_config"], 2560)
    fetch_processor(identity["repo"], identity["revision"], checkpoint)
    source = Source(checkpoint / "model.safetensors.index.json", directory=checkpoint)
    if len(source.weight_map) != 333 or any(
        not name.startswith("model.visual.") for name in source.weight_map
    ):
        raise ValueError("Flash vision requires the complete original 333-tensor encoder")
    for original in sorted(source.weight_map):
        _, _, item = source.tensor(original)
        if item["dtype"] != "BF16":
            raise ValueError("Vision source must retain original BF16 weights")
        shape = item["shape"]
        raw = source.rows(original, 0, shape[0])
        array = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        array = array.astype(np.float16).reshape(shape)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite FP16 vision weight: {original}")
        if original == "model.visual.patch_embed.proj.weight":
            array = array.reshape(1152, 1536)
        name = "V_" + original.removeprefix("model.visual.")
        weights.constants(name + ".safetensors", {name: array})
        weights.visual[original] = name
        weights.visual_parameters += math.prod(shape)
    return dict(source=identity, parameters=weights.visual_parameters)


def build(b):
    from kernels.vision import encoder as e

    v, features = b.planner.config["vision_config"], b.planner.capacity
    validate_vision(v, b.planner.H)
    patches = min(PATCHES, features * 4)
    h, f, out = v["hidden_size"], v["intermediate_size"], v["out_hidden_size"]
    for name, shape, dtype in [
        ("VPixels", [patches, 1536], "f16"),
        ("VGrid", [2], "i32"),
        ("VLength", [1], "i32"),
        ("VHidden", [patches, h], "f16"),
        ("VNorm", [patches, h], "f16"),
        ("VQKV", [patches, 3 * h], "f16"),
        ("VQ", [patches, h], "f16"),
        ("VK", [patches, h], "f16"),
        ("VV", [patches, h], "f16"),
        ("VAttention", [patches, h], "f16"),
        ("VProjection", [patches, h], "f16"),
        ("VFFN", [patches, f], "f16"),
        ("VMerged", [patches // 4, 4 * h], "f16"),
        ("VOutput", [patches // 4, out], "f16"),
    ]:
        b.fixed(name, dtype, shape)

    def weight(name, shape=None):
        physical = b.weights.visual["model.visual." + name]
        if shape is not None and b.buffers[physical]["shape"] != shape:
            raise ValueError(f"Vision weight geometry differs: {name}")
        return physical

    def emit(section, role, key, fn, args, *, divisor=1):
        operations = b.groups.setdefault("vision", {}).setdefault(section, [])
        name = f"vision/{section}/k{len(operations)}"
        b.emit(
            name,
            fn,
            {k: dict(kind="buffer", name=n) for k, n in args.items()},
            patches // divisor,
            key="vision-" + key,
            row_divisor=divisor,
        )
        operations.append(dict(kind="kernel", name=name))

    def dense(
        section, role, n, k, inp, dest, prefix, activation="none", divisor=1, separate_bias=False
    ):
        emit(
            section,
            role,
            f"linear-{n}-{k}-{activation}-{separate_bias}",
            lambda: e.linear(n, k, activation, separate_bias=separate_bias),
            dict(
                X=inp, W=weight(prefix + ".weight", [n, k]), Bias=weight(prefix + ".bias"), Y=dest
            ),
            divisor=divisor,
        )

    def norm(section, role, inp, dest, prefix):
        emit(
            section,
            role,
            f"norm-{h}",
            lambda: e.layer_norm(h),
            dict(X=inp, W=weight(prefix + ".weight"), B=weight(prefix + ".bias"), Y=dest),
        )

    build_encoder(
        v, emit, dense, norm, weight, kernel=lambda fn, *args, **kwargs: fn(*args, **kwargs)
    )
    return dict(
        dtype="f16",
        hidden=out,
        patch_size=v["patch_size"],
        temporal_patch_size=v["temporal_patch_size"],
        merge_size=v["spatial_merge_size"],
        max_patches=patches,
        max_features=features,
        pixels="VPixels",
        grid="VGrid",
        length="VLength",
        output="VOutput",
        features="Features",
        feature_index="FeatureIndex",
        mrope_positions="MRopePositions",
        plans=[dict(patches=patches, min_patches=4, program="vision")],
        **{
            k: b.planner.config[k]
            for k in ["image_token_id", "vision_start_token_id", "vision_end_token_id"]
        },
    )
