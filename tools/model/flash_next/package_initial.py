"""Validate and atomically publish an initial native Flash text execution package."""

import math
import json
import shutil

from tools.build.files import file_hash, write_json
from tools.build.target import CUDA_TARGET
from tools.model.compact import compact_cpu, compact_package, compact_weights
from tools.model.package import execution_identity, model_library
from tools.model.publication import atomic_model


def publish(bindings, layout, destination, output, engine):
    from tools.model.flash_next.build import toolchain
    from tools.model.sampling import append as append_sampling

    b, root = bindings, bindings.weights.root
    context = b.planner.capacity
    for name, dtype, shape, scope in (
        ("Input", "i32", [4096], "sequence"),
        ("LastIndex", "i32", [1], "sequence"),
        ("TargetIndexScoresM16", "f32", [context // 4, 16], "workspace"),
        ("TargetIndexScoresM128", "f32", [context // 4, 128], "workspace"),
    ):
        b.fixed(name, dtype, shape, scope)
    for name in ("M1_Embedding", "M1_Ple", "M4096_Embedding", "M4096_Ple", "Logits", "Selected"):
        b.scopes[name] = "sequence"
    reset = sorted(
        name
        for name in b.buffers
        if name.startswith("State_") or name in ("Position", "Token", "Status")
    )
    kv, divisors = {}, {}
    for layer in range(3, 48, 4):
        for suffix, stride in (
            ("key", 512),
            ("value", 512),
            ("key_scale", 16),
            ("value_scale", 16),
            ("index", 64),
        ):
            kv[f"State_{layer}_{suffix}"] = stride
        divisors[f"State_{layer}_index"] = 4
    scores = {f"TargetIndexScoresM{rows}": rows for rows in (1, 16, 128, 4096)}
    implicit = (
        set(reset)
        | set(scores)
        | {
            "Input",
            "LastIndex",
            "Length",
            "PositionOut",
            "M1_Embedding",
            "M1_Ple",
            "M4096_Embedding",
            "M4096_Ple",
            "Logits",
            "Selected",
            "FlashMixedResidual",
            "FlashMixedHeadResidual",
            "FlashPrivateScratch",
            "FlashBatchPointers",
        }
    )
    used = implicit | {a["name"] for k in b.kernels for a in k["args"] if "name" in a}
    b.weights.buffers = b.buffers = {n: v for n, v in b.buffers.items() if n in used}
    b.weights.scopes = b.scopes = {n: v for n, v in b.scopes.items() if n in used}
    b.weights.mapping = {n: f for n, f in b.weights.mapping.items() if n in b.buffers}
    total = b.weights.finish()
    configuration = b.weights.source.config
    identity = {key: str(value) for key, value in toolchain().items()}
    metadata = dict(
        schema_version=2,
        target=CUDA_TARGET["arch"],
        model="Qwen3.8-Flash-Next",
        chunk_tokens=4096,
        max_context=context,
        vocab=b.planner.V,
        toolchain=identity,
        buffers=list(b.buffers.values()),
        reset_buffers=reset,
        input="Input",
        token="Token",
        status="Status",
        logits="Logits",
        position="Position",
        weight_bytes=total,
        weight_parameters=sum(
            math.prod(b.weights.source.shape(n))
            for n in b.weights.source.parts
            if ".ple_embedding." not in n
        ),
        weight_scope="E8P Q2 experts, signed-block128 Q4 selected projections, W8 sensitive projections, BF16 HC; CPU embedding/PLE separate",
        input_assets=json.loads((root / "inputs-asset.json").read_text()),
        kv_cache=dict(
            direct_prefill=True,
            demand_mapping=True,
            buffers=kv,
            prefix_divisors=divisors,
            prefill_workspace=scores,
            growth={},
        ),
        batch_profiles=[2, 4, 8, 16, 32, 64, 128],
        batch_layout=layout,
        prefill_batch_profiles=[dict(tokens=1024, kind="flash_mixed_grouped")],
    )
    package = dict(
        schema_version=1,
        runtime_abi=1,
        target=CUDA_TARGET["arch"],
        architecture="flash_next",
        compute_policy="int8_quality",
        config_signature=dict(
            text=configuration["text_config"], quantization=configuration["quantization_config"]
        ),
        prefill_profiles=[dict(tokens=4096, min_tokens=1, kind="flash_dynamic_recurrent")],
        batch_profiles=metadata["batch_profiles"],
        prefill_batch_profiles=metadata["prefill_batch_profiles"],
        greedy_sampling=True,
        kernels=b.kernels,
        dynamic_batch_kernels=b.dynamic,
        buffer_contracts=[
            {k: v for k, v in value.items() if k != "data"} for value in b.buffers.values()
        ],
        toolchain=identity,
        execution=execution_identity(model_library(), engine, "flash_next", "int8_quality"),
        plan_kernel_contracts=[k["name"] for k in b.kernels],
    )
    data = dict(
        schema_version=1,
        architecture="flash_next",
        compute_policy="int8_quality",
        buffer_scopes=b.scopes,
        frontend_assets={
            name: file_hash(root / name)
            for name in ("tokenizer.json", "chat_template.jinja", "generation_config.json")
        },
        metadata=metadata,
    )
    append_sampling(data, package, root / "cache", output / "sampling")
    digest = compact_package(root / "cache", package, root / "cache/packages")
    data["execution_package"] = digest
    write_json(output / "publication.json", dict(execution_package=digest, weight_bytes=total))
    with atomic_model(destination, engine) as staging:
        staging.mkdir()
        for path in root.iterdir():
            if path.is_file() and path.name not in ("assembly.json", "inputs-asset.json"):
                shutil.copyfile(path, staging / path.name)
        compact_cpu(root, staging, data, 2 * 1024**3)
        compact_weights(root, staging, data, 2 * 1024**3)
        shutil.copytree(root / "cache/packages" / digest, staging / "cache/packages" / digest)
        write_json(staging / "cache/model.json", data)
    return data
