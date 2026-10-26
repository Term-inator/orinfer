"""Build Flash's selected Q4 operands directly from bounded original BF16 rows."""

import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from tools.build.files import write_json
from tools.model.flash_next.roles import role
from tools.quantization.row_q4 import fit


def selected(name):
    if name is None:
        return None
    if name == "output.weight":
        return "Q4_output"
    if name.startswith("blk."):
        layer = int(name.split(".")[1])
        suffix = name.split(".", 2)[2]
        if layer < 48 and suffix in (
            ("attn_q.weight",) if layer % 4 == 3 else ("attn_gate.weight", "attn_qkv.weight")
        ):
            return "Q4_" + name.removesuffix(".weight")
    return None


def signs_for(prefix, width):
    # QKV/Z share their input basis; no random state depends on matrix order.
    basis = prefix.replace(".attn_qkv", ".attn_gate")
    seed = int.from_bytes(hashlib.sha256(f"20261002:{basis}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed).integers(0, 2, width, dtype=np.int8) * 2 - 1


def publish(weights, source, output, *, rows=1024):
    if type(rows) is not int or rows < 1:
        raise ValueError("Q4 conversion needs a positive row bound")
    expected = weights.source.config["quantization_config"]
    if source.repo != expected["source"] or source.revision != expected["source_revision"]:
        raise ValueError("Q4 original source differs from the quantized checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for original in sorted(weights.source.parts):
        prefix = selected(role(original))
        if prefix is None:
            continue
        _, _, info = source.tensor(original)
        shape = weights.source.shape(original)
        if info["dtype"] != "BF16" or info["shape"] != list(shape) or len(shape) != 2:
            raise ValueError(f"Q4 requires matching original BF16 geometry: {original}")
        n, k = shape
        if k % 128:
            raise ValueError("Q4 requires complete signed block128 inputs")
        signs = signs_for(prefix, k)
        contract = dict(
            source=source.repo,
            revision=source.revision,
            tensor=original,
            shape=list(shape),
            codec="signed-block128-row-q4-v1",
            seed=20261002,
        )
        receipt = output / (prefix + ".json")
        filename = prefix + ".safetensors"
        # Reuse only a matching receipt and a completely hashed native operand.
        if receipt.exists():
            from tools.model.safetensors_source import local_header
            from tools.model.flash_next.storage import payload_hash, DTYPES

            saved = json.loads(receipt.read_text())
            if saved["contract"] != contract:
                raise ValueError("Existing Q4 artifact has a different original identity")
            path = weights.directory / filename
            begin, header = local_header(path)
            for name, descriptor in saved["buffers"].items():
                first, last = header[name]["data_offsets"]
                if (
                    payload_hash(path, begin + first, last - first) != descriptor["data"]["sha256"]
                    or DTYPES[header[name]["dtype"]] != descriptor["dtype"]
                    or header[name]["shape"] != descriptor["shape"]
                ):
                    raise ValueError("Existing Q4 artifact differs from its receipt")
                weights.buffers[name] = descriptor
                weights.scopes[name] = "weights"
                weights.mapping[name] = filename
            records.append(saved)
            print("Q4 ORIGINAL", prefix, "cached", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix=".q4-", dir=output) as tmp:
            packed = np.memmap(Path(tmp) / "packed", mode="w+", dtype=np.uint16, shape=(n, k // 4))
            scales = np.empty(n, np.float16)
            for first in range(0, n, rows):
                count = min(rows, n - first)
                raw = source.rows(original, first, count)
                values = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
                encoded = fit(values.reshape(count, k), signs)
                packed[first : first + count] = encoded.packed.view(np.uint16)
                scales[first : first + count] = encoded.scales
                del raw, values, encoded
            arrays = {prefix + "_packed": packed, prefix + "_scale": scales}
            if not prefix.endswith(".attn_qkv"):
                arrays[prefix + "_signs"] = signs
            weights.constants(filename, arrays)
            del arrays, packed
        record = dict(
            contract=contract,
            buffers={
                name: weights.buffers[name]
                for name in (prefix + "_packed", prefix + "_scale", prefix + "_signs")
                if name in weights.buffers
            },
            parameters=math.prod(shape),
        )
        write_json(receipt, record)
        records.append(record)
        print("Q4 ORIGINAL", prefix, "converted", flush=True)
    return records
