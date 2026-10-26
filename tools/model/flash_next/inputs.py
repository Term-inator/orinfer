"""Publish bounded native embedding/PLE lookups from the quantized checkpoint."""

import hashlib
import json

from safetensors import safe_open

from tools.build.files import file_hash, write_json
from tools.model.flash_next.ple import PleLookup
from tools.model.publication import link_or_copy


def publish(checkpoint, root):
    directory = root / "cache/cpu"
    directory.mkdir(parents=True, exist_ok=True)
    ple = PleLookup(checkpoint)
    hashes = json.loads((checkpoint.directory / "model.safetensors.index.json").read_text())[
        "metadata"
    ]["orinfer.sha256"]
    assets = dict(
        version=1,
        vocab=int(checkpoint.config["text_config"]["vocab_size"]),
        eos=ple.eos,
        multipliers=ple.multipliers,
        sizes=ple.sizes,
        offsets=ple.offsets,
        books=[],
        embedding=[],
        ple=[],
    )
    books, files = {}, {}
    for label, names in (
        ("embedding", ["model.language_model.embed_tokens.weight"]),
        ("ple", ple.names),
    ):
        first = 0
        for original in names:
            for part in checkpoint.parts[original]:
                if part["kind"] not in ("e8p-embedding", "int8-row"):
                    raise ValueError(f"Unsupported native CPU table: {original}")
                source = part["path"]
                relative = str(source.relative_to(checkpoint.directory))
                if source not in files:
                    destination = directory / (
                        hashlib.sha256(relative.encode()).hexdigest()[:16] + "-" + source.name
                    )
                    link_or_copy(source, destination)
                    files[source] = str(destination.relative_to(root))
                key = "rows" if part["kind"] == "e8p-embedding" else "weight"
                info = part["header"][part["keys"][key]]
                record = dict(
                    file=files[source],
                    sha256=hashes[relative],
                    offset=part["begin"] + info["data_offsets"][0],
                    first=first,
                    rows=part["count"],
                )
                if key == "rows":
                    with safe_open(source, framework="np") as stream:
                        table = stream.get_tensor(part["keys"]["table"])
                        signs = stream.get_tensor(part["keys"]["signs"])
                    identity = hashlib.sha256(table.tobytes() + signs.tobytes()).hexdigest()
                    if identity not in books:
                        books[identity] = len(books)
                        assets["books"].append(
                            dict(
                                table=table.flatten().astype(int).tolist(),
                                signs=signs.astype(int).tolist(),
                            )
                        )
                    record["book"] = books[identity]
                else:
                    info = part["header"][part["keys"]["scale"]]
                    record["scale_offset"] = part["begin"] + info["data_offsets"][0]
                assets[label].append(record)
                first += part["count"]
    destination = directory / "inputs.json"
    write_json(destination, assets)
    return dict(file="cache/cpu/inputs.json", sha256=file_hash(destination))
