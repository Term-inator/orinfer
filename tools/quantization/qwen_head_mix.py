"""Replace an untied dense Qwen Q2 output head with the matching Q4 bank."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from tools.build.files import file_hash, write_json
from tools.model.compact import write_shard
from tools.model.publication import link_or_copy, staged_directory
from tools.quantization.qwen_moe_mix import Checkpoint


def compose(q2_root, q4_root, output):
    q2, q4 = Checkpoint(q2_root), Checkpoint(q4_root)
    for source, bits in ((q2, 2), (q4, 4)):
        config = source.config
        quant = config["quantization_config"]
        if (
            config["model_type"] != "qwen3_5"
            or config["text_config"].get(
                "tie_word_embeddings", config.get("tie_word_embeddings", False)
            )
            or quant.get("output_head") is not None
            or any(
                quant.get(k) != v
                for k, v in dict(
                    quant_method=f"orinfer_e8p_q{bits}_int8",
                    coefficients=[1] if bits == 2 else [4, 1],
                    rotation="signed-block128",
                    scale="fp16-output-row",
                    compute_dtype="int8_quality",
                    basis_tensor=f"orinfer.e8p_q{bits}.basis",
                    layout="u16-e8p" if bits == 2 else "u32-low-coarse-high-residual",
                ).items()
            )
        ):
            raise ValueError("Expected dense untied Q2 and Q4 source conversions")
    if {k: v for k, v in q2.config.items() if k != "quantization_config"} != {
        k: v for k, v in q4.config.items() if k != "quantization_config"
    }:
        raise ValueError("Source model configurations differ")
    for field in ("source", "source_revision"):
        if q2.config["quantization_config"][field] != q4.config["quantization_config"][field]:
            raise ValueError("Original checkpoint identity differs")
    if q2.state["parameters"] != q4.state["parameters"]:
        raise ValueError("Parameter counts differ")
    if q2.small_payload("orinfer.e8p_q2.basis") != q4.small_payload("orinfer.e8p_q4.basis"):
        raise ValueError("E8P basis differs")
    for name in (n for n in q2.index["weight_map"] if n.startswith("orinfer.rotation.")):
        if q2.small_payload(name) != q4.small_payload(name):
            raise ValueError("Rotation basis differs")
    selected = {"lm_head.weight.packed", "lm_head.weight.scale"}
    text = q2.config["text_config"]
    for name in selected:
        before, after = q2.tensor(name), q4.tensor(name)
        expected = (
            [text["vocab_size"], text["hidden_size"] // 8]
            if name.endswith(".packed")
            else [text["vocab_size"]]
        )
        if (
            before["shape"] != expected
            or after["shape"] != expected
            or (
                name.endswith(".packed")
                and (
                    before["dtype"] != "U16"
                    or after["dtype"] != "U32"
                    or before["layout"] != "e8p-q2-row-u16"
                    or after["layout"] != "e8p-q4-row-u32"
                )
            )
            or (name.endswith(".scale") and (before["dtype"] != "F16" or after["dtype"] != "F16"))
        ):
            raise ValueError("Output head encoding or geometry differs")
    contract = json.dumps(
        dict(
            policy="qwen35-q2-head-q4",
            q2_contract=q2.state["contract"],
            q4_contract=q4.state["contract"],
            q2_index=file_hash(q2.root / "model.safetensors.index.json"),
            q4_index=file_hash(q4.root / "model.safetensors.index.json"),
            implementation=file_hash(Path(__file__)),
        ),
        sort_keys=True,
    )
    with staged_directory(output) as staging:
        staging.mkdir()
        state = copy.deepcopy(q2.state)
        hashes, payload = {}, 0
        mapping = q2.index["weight_map"]
        for filename, (_, header) in q2.headers.items():
            names = [n for n in header if n != "__metadata__"]
            tensors = [(q4 if n in selected else q2).tensor(n) for n in names]
            if selected.intersection(names):
                write_shard(
                    staging / filename,
                    tensors,
                    {f"orin.layout.{t['name']}": t["layout"] for t in tensors},
                )
                hashes[filename] = file_hash(staging / filename)
            else:
                link_or_copy(q2.root / filename, staging / filename)
                hashes[filename] = q2.index["metadata"]["orinfer.sha256"][filename]
            payload += sum(sum(s[2] for s in t["segments"]) for t in tensors)
            if filename in state.get("shards", {}):
                state["shards"][filename].update(sha256=hashes[filename], names=names)
        for path in q2.root.iterdir():
            if (
                path.is_file()
                and path.suffix != ".safetensors"
                and path.name
                not in ("config.json", "progress.json", "model.safetensors.index.json")
            ):
                shutil.copyfile(path, staging / path.name)
        config = copy.deepcopy(q2.config)
        config["quantization_config"].update(output_head="e8p_q4", policy="qwen35-q2-head-q4")
        state.update(contract=contract, complete=True, weight_bytes=payload)
        if "lm_head.weight" in q4.state.get("quantized", {}):
            state["quantized"]["lm_head.weight"] = q4.state["quantized"]["lm_head.weight"]
        write_json(staging / "config.json", config)
        write_json(staging / "progress.json", state)
        write_json(
            staging / "model.safetensors.index.json",
            dict(metadata={"total_size": payload, "orinfer.sha256": hashes}, weight_map=mapping),
        )
    return dict(weight_bytes=payload, contract_sha256=hashlib.sha256(contract.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("q2", "q4", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compose(args.q2, args.q4, args.output)), flush=True)


if __name__ == "__main__":
    main()
