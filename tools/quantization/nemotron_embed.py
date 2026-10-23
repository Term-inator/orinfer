"""Encode Nemotron-3.5-Embed-1B with the shared two-stage E8P Q4A8 encoder.

Text and vision matrices and embeddings use one compressed bank. Attention
output projections and the final multimodal projection retain row-scaled W8;
patch convolution, norms and small parameters retain their original precision.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from tools.build.files import file_hash, write_json
from tools.model.compact import write_shard
from tools.model.nemotron_embed.config import ROTATED_W8_LAYERS, configuration, signature
from tools.model.publication import staged_directory, link_or_copy
from tools.model.safetensors_source import Source, local_header
from tools.quantization.flash_next_aux import int8_rows
from tools.quantization.qwen_e8p import encode
from tools.quantization.vq import rotate


def upgrade(args):
    """Retain existing E8P values and re-read only protected matrices from BF16."""
    original, source = args.source_dir.resolve(strict=True), args.upgrade_from.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    original_config = json.loads((original / "config.json").read_text())
    configuration(config, quantized=True)
    configuration(original_config)
    origin = json.loads((original / "source.json").read_text())
    if signature(config) != signature(original_config) or (
        config["quantization_config"].get("source"),
        config["quantization_config"].get("source_revision"),
    ) != (origin["repo"], origin["revision"]):
        raise ValueError("BF16 source differs from the encoded model identity")
    state = json.loads((source / "progress.json").read_text())
    if not state.get("complete"):
        raise ValueError("Source E8P conversion is incomplete")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()):
        if file_hash(source / filename) != index["metadata"]["orinfer.sha256"][filename]:
            raise ValueError("Encoded source shard checksum changed")
    bf16 = Source(original / "model.safetensors.index.json", directory=original)
    selected = [
        name
        for name in bf16.weight_map
        if any(name.startswith(f"language_model.layers.{layer}.") for layer in ROTATED_W8_LAYERS)
        and name.endswith(
            (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            )
        )
    ]
    if len(selected) != len(ROTATED_W8_LAYERS) * 6:
        raise ValueError("Protected matrices are incomplete")
    from safetensors import safe_open

    with safe_open(
        str(source / index["weight_map"]["orinfer.rotation.k2048.signs"]), framework="numpy"
    ) as common:
        signs = {k: common.get_tensor(f"orinfer.rotation.k{k}.signs") for k in (2048, 6144)}
    contract = json.dumps(
        dict(
            base_conversion=state["contract"],
            base_index=file_hash(source / "model.safetensors.index.json"),
            source=origin,
            precision="rotated-w8",
            layers=ROTATED_W8_LAYERS,
            implementation=file_hash(Path(__file__)),
        ),
        sort_keys=True,
    )
    with staged_directory(args.output) as out:
        out.mkdir()
        payloads = out / ".payloads"
        payloads.mkdir()
        replacement, statistics = {}, {}
        for number, name in enumerate(sorted(selected)):
            _, _, info = bf16.tensor(name)
            if info["dtype"] != "BF16" or len(info["shape"]) != 2:
                raise ValueError("Protected weights must be original BF16 matrices")
            n, k = info["shape"]
            codes, scales = payloads / f"{number}.codes", payloads / f"{number}.scale"
            digest, sse, norm = hashlib.sha256(), 0.0, 0.0
            with codes.open("wb") as pc, scales.open("wb") as ps:
                for first in range(0, n, args.rows):
                    count = min(args.rows, n - first)
                    raw = bf16.rows(name, first, count)
                    digest.update(raw)
                    values = (
                        (np.frombuffer(raw, "<u2").astype(np.uint32) << 16)
                        .view(np.float32)
                        .reshape(count, k)
                    )
                    reference = rotate(values, signs[k])
                    code, scale = int8_rows(reference)
                    pc.write(code.tobytes())
                    ps.write(scale.tobytes())
                    error = code.astype(np.float32) * scale.astype(np.float32)[:, None] - reference
                    sse += float(np.square(error).sum(dtype=np.float64))
                    norm += float(np.square(reference).sum(dtype=np.float64))
            replacement[name + ".int8"] = dict(
                name=name + ".int8",
                dtype="I8",
                shape=[n, k],
                segments=[(codes, 0, codes.stat().st_size)],
            )
            replacement[name + ".scale"] = dict(
                name=name + ".scale",
                dtype="F16",
                shape=[n],
                segments=[(scales, 0, scales.stat().st_size)],
            )
            statistics[name] = dict(
                name=name,
                shape=[n, k],
                kind="w8-rotated",
                rows=n,
                sse=sse,
                norm=norm,
                source_sha256=digest.hexdigest(),
            )
            print("PROMOTED", name, "relative_mse", sse / max(norm, 1e-30), flush=True)
        mapping, hashes, shards = {}, {}, copy.deepcopy(state["shards"])
        for filename in sorted(set(index["weight_map"].values())):
            path = source / filename
            begin, header = local_header(path)
            affected = [
                name for name in selected if index["weight_map"].get(name + ".packed") == filename
            ]
            if not affected:
                link_or_copy(path, out / filename)
                names = [n for n in header if n != "__metadata__"]
            else:
                entries = []
                metadata = dict(header.get("__metadata__", {}))
                removed = {name + suffix for name in affected for suffix in (".packed", ".scale")}
                for name, info in header.items():
                    if name == "__metadata__" or name in removed:
                        continue
                    first, last = info["data_offsets"]
                    entries.append(
                        dict(
                            name=name,
                            dtype=info["dtype"],
                            shape=info["shape"],
                            segments=[(path, begin + first, last - first)],
                        )
                    )
                for name in affected:
                    metadata.pop("orin.layout." + name + ".packed", None)
                    for suffix in (".int8", ".scale"):
                        entry = replacement[name + suffix]
                        entries.append(entry)
                        metadata["orin.layout." + entry["name"]] = "native_contiguous"
                metadata["contract"] = contract
                write_shard(out / filename, entries, metadata)
                names = [e["name"] for e in entries]
            mapping.update({n: filename for n in names})
            hashes[filename] = file_hash(out / filename)
            if filename in shards:
                shards[filename].update(sha256=hashes[filename], names=names)
        for path in payloads.iterdir():
            path.unlink()
        payloads.rmdir()
        size = sum((out / f).stat().st_size - local_header(out / f)[0] for f in hashes)
        for path in source.iterdir():
            if (
                path.is_file()
                and path.suffix != ".safetensors"
                and path.name
                not in ("config.json", "progress.json", "model.safetensors.index.json")
            ):
                shutil.copyfile(path, out / path.name)
        config["quantization_config"]["rotated_w8_layers"] = list(ROTATED_W8_LAYERS)
        state.update(contract=contract, complete=True, weight_bytes=size, shards=shards)
        state["quantized"].update(statistics)
        write_json(out / "config.json", config)
        write_json(out / "progress.json", state)
        write_json(
            out / "model.safetensors.index.json",
            dict(metadata={"total_size": size, "orinfer.sha256": hashes}, weight_map=mapping),
        )
        print("COMPLETE", size, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--upgrade-from",
        type=Path,
        help="Retain a completed E8P bank and promote the protected BF16 matrices",
    )
    p.add_argument("--shard")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--gpu-lock", type=Path, default=Path("artifacts/gpu-experiment.lock"))
    args = p.parse_args()
    if not 1 <= args.rows <= 1024:
        p.error("Row block must be in 1..1024")
    if args.upgrade_from:
        upgrade(args)
    else:
        configuration(json.loads((args.source_dir / "config.json").read_text()))
        encode(args)


if __name__ == "__main__":
    main()
