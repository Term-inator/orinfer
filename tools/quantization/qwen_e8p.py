"""Resumable E8P Q2A8/Q4A8 encoding of original dense Qwen and ASR weights.

Each source shard is consumed independently. GPU ownership is released between
bounded row blocks, and completed blocks are checksummed. Tied embeddings and
output heads share one Q4 bank. Vision matrices also use Q4; sensitive
projections, positional tables and small parameters retain W8 or BF16.
"""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import time

import numpy as np
from safetensors.numpy import save_file

from tools.model.compact import write_shard
from tools.build.files import file_hash
from tools.build.files import atomic_json as write_json
from tools.model.safetensors_source import Source, local_header
from tools.quantization.flash_next_aux import int8_rows
from tools.build.gpu import gpu_lock


def rotated_w8_matrices(config, matrices=None):
    """Validate explicit dense text projections in the existing rotated input basis."""
    selected = (
        config.get("quantization_config", {}).get("rotated_w8_matrices", [])
        if matrices is None
        else matrices
    )
    if not isinstance(selected, list) or any(not isinstance(n, str) for n in selected):
        raise ValueError("rotated_w8_matrices must be a list of projection names")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate rotated W8 projection")
    text = config.get("text_config", config)
    for name in selected:
        match = re.fullmatch(
            r"model\.language_model\.layers\.(\d+)\.(self_attn\.[qkv]_proj|linear_attn\.in_proj_(qkv|z)|mlp\.(gate|up|down)_proj)\.weight",
            name,
        )
        if config.get("model_type") != "qwen3_5" or not match:
            raise ValueError(f"Unsupported rotated W8 projection: {name}")
        layer = int(match[1])
        if str(layer) != match[1] or layer >= text["num_hidden_layers"]:
            raise ValueError(f"Rotated W8 layer out of range: {name}")
        role = match[2]
        kind = text["layer_types"][layer]
        if (role.startswith("self_attn.") and kind != "full_attention") or (
            role.startswith("linear_attn.") and kind != "linear_attention"
        ):
            raise ValueError(f"Rotated W8 projection differs from layer kind: {name}")
    return selected


def rotated_w8_layers(config):
    if config["model_type"] == "qwen3":
        from tools.model.qwen_reranker.config import ROTATED_W8_LAYERS

        return list(ROTATED_W8_LAYERS)
    if config["model_type"] == "mistral3":
        from tools.model.nemotron_embed.config import ROTATED_W8_LAYERS

        return list(ROTATED_W8_LAYERS)
    if config["model_type"] != "qwen3_asr":
        return []
    contract = json.loads(
        (Path(__file__).parents[2] / "configs/architecture-contract.json").read_text()
    )
    return contract["families"]["qwen3_asr"]["int8_quality"]["rotated_w8_layers"]


def policy(name, info, *, protected_layers=()):
    shape = info["shape"]
    if name.startswith(("model.audio_tower.", "model.multi_modal_projector.")):
        return "original"
    expert_bank = len(shape) == 3 and name.endswith(
        (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
    )
    if (
        info["dtype"] != "BF16"
        or (len(shape) != 2 and not expert_bank)
        or shape[-1] % 128
        or min(shape) < 128
        or name == "model.visual.pos_embed.weight"
        or "norm" in name
        or name.endswith(".mlp.gate.weight")
    ):
        return "original"
    if name.startswith("mtp.") or name.endswith(
        (
            "linear_attn.out_proj.weight",
            "self_attn.o_proj.weight",
            ".attention.o_proj.weight",
            "multi_modal_projector.linear_2.weight",
        )
    ):
        return "w8"
    if any(
        name.startswith(
            (
                f"model.language_model.layers.{layer}.",
                f"language_model.layers.{layer}.",
                f"model.layers.{layer}.",
            )
        )
        for layer in protected_layers
    ):
        return "w8-rotated"
    return "e8p-q4"


def text_configuration(config):
    if config.get("model_type") == "qwen3":
        from tools.model.qwen_reranker.config import configuration

        return configuration(config)
    if config.get("model_type") == "mistral3":
        from tools.model.nemotron_embed.config import configuration

        return configuration(config)
    if config.get("model_type") not in ("qwen3_5", "qwen3_5_moe", "qwen3_asr") or config.get(
        "quantization_config"
    ):
        raise ValueError("Expected original floating Qwen3.5 checkpoint")
    t = config["text_config"]
    if config["model_type"] == "qwen3_asr":
        t["attn_output_gate"] = False
        expected = dict(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
        )
        if (
            any(t.get(k) != v for k, v in expected.items())
            or t.get("layer_types") != ["full_attention"] * 28
        ):
            raise ValueError("Unsupported Qwen3 ASR decoder geometry")
        audio = config.get("audio_config", {})
        if any(
            audio.get(k) != v
            for k, v in dict(
                d_model=1024,
                encoder_layers=24,
                encoder_ffn_dim=4096,
                num_mel_bins=128,
                output_dim=2048,
            ).items()
        ):
            raise ValueError("Unsupported Qwen3 ASR audio geometry")
    elif config["model_type"] == "qwen3_5_moe":
        expected = dict(
            hidden_size=2048,
            num_hidden_layers=40,
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
        )
        if any(t.get(k) != v or type(t.get(k)) is not int for k, v in expected.items()):
            raise ValueError("Unsupported Qwen MoE geometry")
        # Canonical workspace width; the expert/shared widths remain explicit.
        t["intermediate_size"] = t["shared_expert_intermediate_size"]
    elif (t.get("hidden_size"), t.get("intermediate_size"), t.get("num_hidden_layers")) not in (
        (2048, 6144, 24),
        (2560, 9216, 32),
        (4096, 12288, 32),
        (5120, 17408, 64),
    ):
        raise ValueError("Unsupported dense Qwen geometry")
    # HF composite configs may keep tying on the parent instead of text_config.
    # Persist the effective value so construction and the native adapter agree.
    tied = t.setdefault("tie_word_embeddings", config.get("tie_word_embeddings", False))
    if type(tied) is not bool:
        raise ValueError("Invalid embedding/head tying configuration")
    return t


def canonical_name(name):
    # Official MoE checkpoints may place the same vision encoder at the root.
    return "model." + name if name.startswith("visual.") else name


def encode(args):
    source_dir, out = args.source_dir.resolve(), args.output.resolve()
    bits = args.bits
    codec = f"e8p_q{bits}"
    kind_name = f"e8p-q{bits}"
    if source_dir == out:
        raise ValueError("Source and encoded directories must differ")
    config = json.loads((source_dir / "config.json").read_text())
    t = text_configuration(config)
    protected_layers = rotated_w8_layers(config)
    model_policy = {
        "mistral3": "nemotron35-embed-e8p-q4a8-v1",
        "qwen3": "qwen3-reranker-e8p-q4a8-v1",
    }.get(config["model_type"], f"qwen35-e8p-q{bits}a8-v3")
    origin = json.loads((source_dir / "source.json").read_text())
    source = Source(source_dir / "model.safetensors.index.json", directory=source_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = out / ".payloads"
    payloads.mkdir(exist_ok=True)
    implementation = hashlib.sha256()
    for name in (
        "qwen_e8p.py",
        "e8p_q4_gpu.py",
        "e8p_q4.py",
        "e8p_gpu.py",
        "e8p.py",
        "vq.py",
        "rotation.py",
    ):
        implementation.update(Path(__file__).with_name(name).read_bytes())
    implementation.update((Path(__file__).parents[2] / "kernels/model/e8p_encode.py").read_bytes())
    contract = json.dumps(
        dict(
            source=origin,
            source_index=file_hash(source_dir / "model.safetensors.index.json"),
            encoder=implementation.hexdigest(),
            seed=20261002,
            coefficients=[4, 1] if bits == 4 else [1],
            search="beam8" if bits == 4 else "nearest",
            rows=args.rows,
            policy=model_policy,
            **(dict(rotated_w8_layers=protected_layers) if protected_layers else {}),
        ),
        sort_keys=True,
    )
    progress = out / "progress.json"
    state = (
        json.loads(progress.read_text())
        if progress.exists()
        else dict(
            contract=contract,
            complete=False,
            shards={},
            quantized={},
            parameters=0,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    )
    if state["contract"] != contract:
        raise ValueError("Quantization identity changed")
    write_json(progress, state)
    vision = config.get("vision_config", {})
    widths = sorted(
        {
            k
            for k in (
                t["hidden_size"],
                t["intermediate_size"],
                2 * t["hidden_size"],
                t.get("num_attention_heads", 0) * t.get("head_dim", 0),
                vision.get("hidden_size", 0),
                vision.get("intermediate_size", 0),
                4 * vision.get("hidden_size", 0),
            )
            if k > 0 and k % 128 == 0
        }
    )
    signs = {
        k: np.random.default_rng(20261002 + k).choice(np.array([-1, 1], np.int8), k) for k in widths
    }
    encoder = None
    from tools.operators.common import configure

    if bits == 4:
        from tools.quantization.e8p_q4_gpu import Encoder
    else:
        from tools.quantization.e8p_gpu import Encoder
    from tools.quantization.vq import rotate

    shards = sorted(set(source.weight_map.values()))
    if args.shard and args.shard not in shards:
        raise ValueError("Unknown source shard")
    for shard in shards:
        if args.shard and shard != args.shard:
            continue
        if shard in state["shards"]:
            if file_hash(out / shard) != state["shards"][shard]["sha256"]:
                raise ValueError("Completed shard changed")
            continue
        path = source_dir / shard
        begin, header = local_header(path)
        entries, layouts, garbage = [], {}, []
        parameters = 0
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            name = canonical_name(name)
            shape = info["shape"]
            parameters += math.prod(shape)
            kind = policy(name, info, protected_layers=protected_layers)
            if kind == "e8p-q4":
                kind = kind_name
            first, last = info["data_offsets"]
            if kind == "original":
                segments = [(path, begin + first, last - first)]
                dtype = info["dtype"]
                if (
                    config["model_type"] == "qwen3_asr"
                    and name.startswith(("model.audio_tower.", "model.multi_modal_projector."))
                    and dtype == "BF16"
                ):
                    values = np.memmap(
                        path, mode="r", offset=begin + first, dtype="<u2", shape=shape
                    )
                    fp16 = ((np.asarray(values).astype(np.uint32) << 16).view(np.float32)).astype(
                        np.float16
                    )
                    if not np.isfinite(fp16).all():
                        raise ValueError(f"Audio FP16 conversion overflow: {name}")
                    payload = payloads / (hashlib.sha256(name.encode()).hexdigest() + ".fp16")
                    fp16.tofile(payload)
                    segments = [(payload, 0, payload.stat().st_size)]
                    garbage.append(payload)
                    dtype = "F16"
                    del values, fp16
                layouts["orin.layout." + name] = "native_contiguous"
                entries.append(
                    dict(
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        segments=segments,
                    )
                )
                continue
            n, k = math.prod(shape[:-1]), shape[-1]
            key = hashlib.sha256(name.encode()).hexdigest()
            pc, sc, checkpoint = (payloads / (key + s) for s in (".codes", ".scale", ".json"))
            saved = (
                json.loads(checkpoint.read_text())
                if checkpoint.exists()
                else dict(
                    name=name,
                    shape=shape,
                    kind=kind,
                    rows=0,
                    blocks=[],
                    compute_s=0.0,
                    sse=0.0,
                    norm=0.0,
                )
            )
            if (saved["name"], saved["shape"], saved["kind"]) != (name, shape, kind):
                raise ValueError("Matrix checkpoint changed")
            codes_per_row = k * bits // 8 if kind == kind_name else k
            if saved["rows"]:
                with pc.open("rb") as pf, sc.open("rb") as sf:
                    for block in saved["blocks"]:
                        pf.seek(block["first"] * codes_per_row)
                        sf.seek(block["first"] * 2)
                        data = pf.read(block["rows"] * codes_per_row) + sf.read(block["rows"] * 2)
                        if hashlib.sha256(data).hexdigest() != block["sha256"]:
                            raise ValueError("Encoded block changed")
            raw = np.memmap(path, mode="r", offset=begin + first, dtype="<u2", shape=(n, k))
            with (
                pc.open("r+b" if pc.exists() else "w+b") as pf,
                sc.open("r+b" if sc.exists() else "w+b") as sf,
            ):
                pf.truncate(saved["rows"] * codes_per_row)
                sf.truncate(saved["rows"] * 2)
                pf.seek(0, 2)
                sf.seek(0, 2)
                for row in range(saved["rows"], n, args.rows):
                    rows = min(args.rows, n - row)
                    floating = (np.asarray(raw[row : row + rows]).astype(np.uint32) << 16).view(
                        np.float32
                    )
                    started = time.monotonic()
                    if kind == kind_name:
                        with gpu_lock(args.gpu_lock):
                            if encoder is None:
                                configure()
                                encoder = Encoder()
                            fitted = encoder.fit(floating, signs[k])
                            if bits == 4:
                                fitted = encoder.refine(floating, fitted)
                        data, scales = (
                            (fitted.packed if bits == 4 else fitted.indices).tobytes(),
                            fitted.scales.tobytes(),
                        )
                        reference = rotate(floating, signs[k])
                        error = fitted.dequantize() - reference
                    else:
                        reference = rotate(floating, signs[k]) if kind == "w8-rotated" else floating
                        codes, scale = int8_rows(reference)
                        data, scales = codes.tobytes(), scale.tobytes()
                        error = (
                            codes.astype(np.float32) * scale.astype(np.float32)[:, None] - reference
                        )
                    pf.write(data)
                    sf.write(scales)
                    pf.flush()
                    sf.flush()
                    os.fsync(pf.fileno())
                    os.fsync(sf.fileno())
                    saved["rows"] = row + rows
                    saved["compute_s"] += time.monotonic() - started
                    saved["sse"] += float(np.square(error).sum(dtype=np.float64))
                    saved["norm"] += float(np.square(reference).sum(dtype=np.float64))
                    saved["blocks"].append(
                        dict(first=row, rows=rows, sha256=hashlib.sha256(data + scales).hexdigest())
                    )
                    write_json(checkpoint, saved)
                    state["current"] = dict(
                        name=name, rows=saved["rows"], total_rows=n, compute_s=saved["compute_s"]
                    )
                    write_json(progress, state)
                    if row % (args.rows * 16) == 0:
                        print(
                            "ROWS",
                            name,
                            saved["rows"],
                            n,
                            "seconds",
                            round(saved["compute_s"], 2),
                            flush=True,
                        )
            del raw
            packed_name = name + (".packed" if kind == kind_name else ".int8")
            entries.extend(
                [
                    dict(
                        name=packed_name,
                        dtype=("U32" if bits == 4 else "U16") if kind == kind_name else "I8",
                        shape=[*shape[:-1], k // 8] if kind == kind_name else shape,
                        segments=[(pc, 0, pc.stat().st_size)],
                    ),
                    dict(
                        name=name + ".scale",
                        dtype="F16",
                        shape=shape[:-1],
                        segments=[(sc, 0, sc.stat().st_size)],
                    ),
                ]
            )
            layouts["orin.layout." + packed_name] = (
                f"e8p-q{bits}-row-u{bits * 8}" if kind == kind_name else "native_contiguous"
            )
            layouts["orin.layout." + name + ".scale"] = "native_contiguous"
            state["quantized"][name] = {k: v for k, v in saved.items() if k != "blocks"}
            garbage.extend([pc, sc, checkpoint])
            print(
                "FITTED",
                name,
                kind,
                "relative_mse",
                saved["sse"] / max(saved["norm"], 1e-30),
                flush=True,
            )
        temporary = out / (shard + ".partial")
        write_shard(temporary, entries, dict(contract=contract, **layouts))
        temporary.replace(out / shard)
        state["shards"][shard] = dict(
            sha256=file_hash(out / shard), parameters=parameters, names=[e["name"] for e in entries]
        )
        state.pop("current", None)
        write_json(progress, state)
        for path in garbage:
            path.unlink()
        print("SHARD", len(state["shards"]), len(shards), shard, flush=True)
    if set(state["shards"]) != set(shards):
        return
    from tools.quantization.e8p import basis

    common = {f"orinfer.rotation.k{k}.signs": s for k, s in signs.items()}
    common[f"orinfer.{codec}.basis"] = basis()
    save_file(common, str(out / "quantization.safetensors"))
    mapping = {n: f for f, r in state["shards"].items() for n in r["names"]}
    mapping.update({n: "quantization.safetensors" for n in common})
    hashes = {f: r["sha256"] for f, r in state["shards"].items()}
    hashes["quantization.safetensors"] = file_hash(out / "quantization.safetensors")
    sizes = sum((out / f).stat().st_size - local_header(out / f)[0] for f in hashes)
    write_json(
        out / "model.safetensors.index.json",
        dict(metadata={"total_size": sizes, "orinfer.sha256": hashes}, weight_map=mapping),
    )
    for path in source_dir.iterdir():
        if (
            path.is_file()
            and path.suffix not in (".safetensors",)
            and path.name
            in (
                "source.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "merges.txt",
                "vocab.json",
                "preprocessor_config.json",
                "processor_config.json",
                "LICENSE",
                "README.md",
                "config_sentence_transformers.json",
                "sentence_bert_config.json",
                "modules.json",
            )
        ):
            shutil.copyfile(path, out / path.name)
    config["quantization_config"] = dict(
        quant_method=f"orinfer_{codec}_int8",
        version=1,
        coefficients=[4, 1] if bits == 4 else [1],
        rotation="signed-block128",
        scale="fp16-output-row",
        compute_dtype="int8_quality",
        basis_tensor=f"orinfer.{codec}.basis",
        layout="u32-low-coarse-high-residual" if bits == 4 else "u16-e8p",
        policy=model_policy,
        embedding=codec,
        vision=None if config["model_type"] in ("qwen3_asr", "qwen3") else codec,
        **(
            dict(audio="float16", rotated_w8_layers=protected_layers)
            if config["model_type"] == "qwen3_asr"
            else {}
        ),
        source=origin["repo"],
        source_revision=origin["revision"],
    )
    write_json(out / "config.json", config)
    state.update(
        complete=True,
        parameters=sum(r["parameters"] for r in state["shards"].values()),
        weight_bytes=sizes,
    )
    write_json(progress, state)
    print("COMPLETE", sizes, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard")
    p.add_argument("--bits", type=int, choices=(2, 4), default=4)
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--gpu-lock", type=Path, default=Path("artifacts/gpu-experiment.lock"))
    args = p.parse_args()
    if not 1 <= args.rows <= 1024:
        p.error("Row block must be in 1..1024")
    encode(args)


if __name__ == "__main__":
    main()
