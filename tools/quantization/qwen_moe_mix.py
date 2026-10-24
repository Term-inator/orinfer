"""Compose Q4 dense weights with Q2 routed experts from matching E8P checkpoints.

Tensor payloads are copied unchanged and streamed into new shards. Both inputs
must be complete, hash-verified conversions of the same original checkpoint.
No GPU quantization, duplicate expert bank, or float reconstruction is needed.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from tools.model.compact import write_shard
from tools.model.publication import file_hash, staged_directory, write_json
from tools.model.safetensors_source import local_header


def is_expert(name):
    return name.startswith("model.language_model.layers.") and ".mlp.experts." in name


class Checkpoint:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.config = json.loads((root / "config.json").read_text())
        self.state = json.loads((root / "progress.json").read_text())
        if not self.state.get("complete"):
            raise ValueError("Both source conversions must be complete")
        self.index = json.loads((root / "model.safetensors.index.json").read_text())
        self.headers = {}
        for filename in sorted(set(self.index["weight_map"].values())):
            path = self.root / filename
            if path.resolve().parent != self.root:
                raise ValueError("Expected local checkpoint shards")
            if file_hash(path) != self.index["metadata"]["orinfer.sha256"][filename]:
                raise ValueError(f"Source shard identity changed: {filename}")
            self.headers[filename] = local_header(path)

    def tensor(self, name):
        filename = self.index["weight_map"][name]
        begin, header = self.headers[filename]
        info = header[name]
        first, last = info["data_offsets"]
        return dict(
            name=name,
            dtype=info["dtype"],
            shape=info["shape"],
            layout=header.get("__metadata__", {}).get(f"orin.layout.{name}", "native_contiguous"),
            segments=[(self.root / filename, begin + first, last - first)],
        )

    def small_payload(self, name):
        tensor = self.tensor(name)
        path, begin, size = tensor["segments"][0]
        if size > 65536:
            raise ValueError("Expected bounded codebook/rotation tensor")
        with path.open("rb") as stream:
            stream.seek(begin)
            return stream.read(size)


def compose(q4_root, q2_root, output):
    q4, q2 = Checkpoint(q4_root), Checkpoint(q2_root)
    for source, bits in [(q4, 4), (q2, 2)]:
        q = source.config["quantization_config"]
        if (
            source.config["model_type"] != "qwen3_5_moe"
            or any(
                q.get(k) != v
                for k, v in dict(
                    quant_method=f"orinfer_e8p_q{bits}_int8",
                    coefficients=[4, 1] if bits == 4 else [1],
                    rotation="signed-block128",
                    scale="fp16-output-row",
                    compute_dtype="int8_quality",
                    basis_tensor=f"orinfer.e8p_q{bits}.basis",
                    layout="u32-low-coarse-high-residual" if bits == 4 else "u16-e8p",
                ).items()
            )
            or q.get("experts") is not None
        ):
            raise ValueError("Expected uniform Q4 and Q2 MoE source checkpoints")
    if {k: v for k, v in q4.config.items() if k != "quantization_config"} != {
        k: v for k, v in q2.config.items() if k != "quantization_config"
    }:
        raise ValueError("Source model configurations differ")
    for field in ["source", "source_revision"]:
        if q4.config["quantization_config"][field] != q2.config["quantization_config"][field]:
            raise ValueError("Original checkpoint identity differs")
    if q4.state["parameters"] != q2.state["parameters"]:
        raise ValueError("Parameter counts differ")
    names4 = set(q4.index["weight_map"])
    names2 = set(q2.index["weight_map"])
    if names4 - {"orinfer.e8p_q4.basis"} != names2 - {"orinfer.e8p_q2.basis"}:
        raise ValueError("Tensor inventories differ")
    if q4.small_payload("orinfer.e8p_q4.basis") != q2.small_payload("orinfer.e8p_q2.basis"):
        raise ValueError("E8P basis differs")
    for name in sorted(n for n in names4 if n.startswith("orinfer.rotation.")):
        if q4.small_payload(name) != q2.small_payload(name):
            raise ValueError("Rotation basis differs")
    contract = json.dumps(
        dict(
            policy="qwen35-dense-q4-experts-q2",
            q4_contract=q4.state["contract"],
            q2_contract=q2.state["contract"],
            q4_index=file_hash(q4.root / "model.safetensors.index.json"),
            q2_index=file_hash(q2.root / "model.safetensors.index.json"),
        ),
        sort_keys=True,
    )
    groups = defaultdict(list)
    for name in sorted(names4):
        source = q2 if is_expert(name) else q4
        tensor = source.tensor(name)
        if is_expert(name):
            other = q4.tensor(name)
            if tensor["shape"] != other["shape"]:
                raise ValueError("Expert geometry differs")
            if name.endswith(".packed") and (
                tensor["dtype"] != "U16"
                or other["dtype"] != "U32"
                or tensor["layout"] != "e8p-q2-row-u16"
            ):
                raise ValueError("Expected single-stage U16 expert bank")
        groups[q4.index["weight_map"][name]].append(tensor)
    with staged_directory(output) as staging:
        staging.mkdir()
        mapping = {}
        hashes = {}
        payload = 0
        for filename, tensors in sorted(groups.items()):
            metadata = {f"orin.layout.{t['name']}": t["layout"] for t in tensors}
            _, digest = write_shard(staging / filename, tensors, metadata)
            hashes[filename] = digest
            mapping.update({t["name"]: filename for t in tensors})
            payload += sum(sum(segment[2] for segment in t["segments"]) for t in tensors)
            print("SHARD", filename, flush=True)
        for path in q4.root.iterdir():
            if (
                path.is_file()
                and path.suffix != ".safetensors"
                and path.name
                not in ("progress.json", "config.json", "model.safetensors.index.json")
            ):
                shutil.copyfile(path, staging / path.name)
        config = q4.config
        config["quantization_config"].update(experts="e8p_q2", policy="qwen35-dense-q4-experts-q2")
        write_json(staging / "config.json", config)
        write_json(
            staging / "model.safetensors.index.json",
            dict(metadata={"total_size": payload, "orinfer.sha256": hashes}, weight_map=mapping),
        )
        write_json(
            staging / "progress.json",
            dict(
                contract=contract,
                complete=True,
                parameters=q4.state["parameters"],
                payload_bytes=payload,
                source_sha256=hashlib.sha256(contract.encode()).hexdigest(),
            ),
        )
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["q4", "q2", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    print("COMPLETE", compose(a.q4, a.q2, a.output), flush=True)


if __name__ == "__main__":
    main()
