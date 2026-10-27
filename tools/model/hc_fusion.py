"""Replace decode HC combine bindings with the fused combine-norm kernel.

The fused TileLang kernel is bit-identical to the separate hc_combine +
hc_norm chain (same fragment statements, same stream-dtype rounding). The
Rust adapter removes the now-redundant norm operation when it registers the
fused symbol; this tool only constructs checked operand bindings.
"""

import argparse
import json
from pathlib import Path

from tools.model.host_compiler import Compiler
from tools.model.publication import load_model

H = 2560
STREAMS = 4
FUSED_KEY = f"hc-combine-norm-m1-h{H}-s{STREAMS}"


def _role_set(compiler, kernel):
    return set(compiler.pointers(kernel))


def decode_pairs(manifest, package, compiler):
    """Adjacent (combine, norm) kernel names inside the single-row decode program."""
    contracts = {k["name"]: k for k in package["kernels"]}
    pairs = []
    for program, operations in manifest["programs"].items():
        if program != "decode":
            continue
        names = [o["name"] for o in operations if o["kind"] == "kernel"]
        for first, second in zip(names, names[1:]):
            combine = contracts.get(first)
            norm = contracts.get(second)
            if (
                combine is not None
                and norm is not None
                and _role_set(compiler, combine) == {"Block", "Inject", "Output", "Residual"}
                and len(combine["args"]) == 4
                and _role_set(compiler, norm) == {"Normed", "Residual", "Weight"}
                and len(norm["args"]) == 3
            ):
                pairs.append((first, second))
    return pairs


def publish(data, package, manifest, compiler, output):
    from kernels.model import hyperconnection

    output.mkdir(parents=True, exist_ok=True)
    pairs = decode_pairs(manifest, package, compiler)
    if not pairs:
        raise ValueError("No decode HC combine/norm pairs found to fuse")
    replacements = {}
    for combine_name, norm_name in pairs:
        combine = next(k for k in package["kernels"] if k["name"] == combine_name)
        norm = next(k for k in package["kernels"] if k["name"] == norm_name)
        c = compiler.pointers(combine)
        n = compiler.pointers(norm)
        bindings = {
            "Block": c["Block"],
            "Residual": c["Residual"],
            "Inject": c["Inject"],
            "Weight": n["Weight"],
            "Output": c["Output"],
            "Normed": n["Normed"],
        }
        item, launch = compiler.emit(
            combine,
            combine_name,
            bindings,
            1,
            build=lambda: hyperconnection.hc_combine_norm.__wrapped__(
                1, H, streams=STREAMS, dtype="float16"
            ),
            key=FUSED_KEY,
            symbolic=False,
        )
        replacements[combine_name] = item
    write = output / "hc-combine-norm.json"
    write.write_text(
        json.dumps(
            dict(fused=[dict(combine=a, norm=b) for a, b in pairs], symbol=replacements[pairs[0][0]]["symbol"]),
            indent=1,
        )
    )
    print(f"HC combine-norm fused {len(pairs)} decode pairs ->", write, flush=True)
    return replacements


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest = {}
    data, origin, package = load_model(Path(args.model), plan=manifest)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    package_dir = Path(args.model) / f"cache/packages/{data['execution_package']}"
    compiler = Compiler(package_dir, package_dir, Path(args.output), buffers)
    publish(data, package, manifest, compiler, Path(args.output))


if __name__ == "__main__":
    main()
