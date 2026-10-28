"""Rebuild the five GDN recurrence modules with narrow persistent-state storage.

The deployed package binds 216 kernels to five compiled modules: the static
decode recurrence, the dynamic in-place prefill recurrence, the compact
verification recurrence, the rank-one commit replay, and the address-table
decode wrapper that runs the single-token recurrence for up to 128 owners.
This tool recompiles those modules from the certified construction with float16
state storage while compute stays FP32, rebinds every kernel under its original
name and geometry, and moves the state arena contracts to f16 on both the
package and model sides. Kernel names, host ABIs, grids, blocks, and buffer
bindings are preserved, so the Rust binding library, packet programs, and graph
branches stay valid. The row wrapper keeps the certified recurrence, roles,
arena offsets, and launch geometry; TileLang lowers its fragments in the
default thread layout, so its reduction order differs from the retired builder.
"""

import argparse
import re
from pathlib import Path
import shutil

from tools.model.host_compiler import Compiler
from tools.model.publication import atomic_model, clone_model, load_model
from tools.build.files import write_json

STATE_BUFFER = re.compile(r"^State_(\d+)_gdn$")
SEQUENCE_ROLES = {"Q", "K", "V", "G", "Beta", "State", "Out"}
PREFIX_ROLES = SEQUENCE_ROLES | {"Prefix"}
COMMIT_ROLES = {"K", "G", "Update", "Accepted", "State"}
WRAPPER_ROLES = {"G", "Beta", "Out", "Pointers"}


def state_buffer_name(kernel):
    for arg in kernel["args"]:
        if arg.get("kind") == "buffer" and STATE_BUFFER.match(arg.get("name", "")):
            return arg["name"]
    return None


def classify(kernel, pointers, dynamics):
    """Recover the certified constructor from binding geometry, or reject."""
    roles = set(pointers)
    dynamic = kernel["name"] in dynamics
    grid_x, threads = kernel["grid"][0], kernel["block"][0]
    value_tile = 128 // grid_x
    if roles == WRAPPER_ROLES:
        # The wrapper's shared-memory budget depends on the reduction lowering,
        # so only its launch geometry and owner capacity are certified here.
        if (
            not dynamic
            or dynamics[kernel["name"]]["capacity"] != 128
            or (grid_x, kernel["grid"][1], threads) != (512, 48, 128)
        ):
            raise ValueError("Unexpected decode GDN row-wrapper geometry")
        return "rows", 128, None
    if roles == SEQUENCE_ROLES:
        if not dynamic:
            if (value_tile, threads) != (32, 128) or kernel["shared_memory_bytes"]:
                raise ValueError("Unexpected decode GDN geometry")
            return (
                "decode",
                1,
                dict(tokens=1, in_place=True, value_tile=value_tile, threads=threads),
            )
        if (value_tile, threads) != (16, 32) or not kernel["shared_memory_bytes"]:
            raise ValueError("Unexpected prefill GDN geometry")
        return (
            "prefill",
            dynamics[kernel["name"]]["capacity"],
            dict(
                tokens=4096,
                in_place=True,
                value_tile=value_tile,
                threads=threads,
                prefetch_qk=True,
                token_unroll=4,
                dynamic_rows=True,
            ),
        )
    if roles == PREFIX_ROLES and dynamic:
        capacity = dynamics[kernel["name"]]["capacity"]
        if (value_tile, threads) != (32, 128) or capacity != 8 or kernel["shared_memory_bytes"]:
            raise ValueError("Unexpected verification GDN geometry")
        return (
            "verify",
            capacity,
            dict(tokens=8, compact=True, value_tile=value_tile, threads=threads, dynamic_rows=True),
        )
    if roles == COMMIT_ROLES and dynamic:
        capacity = dynamics[kernel["name"]]["capacity"]
        if (value_tile, threads) != (32, 128) or capacity != 8 or kernel["shared_memory_bytes"]:
            raise ValueError("Unexpected commit GDN geometry")
        return "restore", capacity, None
    return None


def direct_build(kind, config, state_dtype):
    from kernels.model.gdn_sequence import _gdn_sequence, gdn_commit

    if kind == "restore":
        return lambda: gdn_commit.__wrapped__(
            8, value_tile=32, dynamic_rows=True, state_dtype=state_dtype
        )
    return lambda: _gdn_sequence(**config, state_dtype=state_dtype)


def wrapper_build(state_dtype, columns, stride, offsets):
    """The address-table decode wrapper over the single-token recurrence."""

    def build():
        from kernels.model.gdn_sequence import _gdn_sequence
        from kernels.model.independent_rows import independent_rows

        fn = _gdn_sequence(1, in_place=True, value_tile=32, threads=128, state_dtype=state_dtype)
        roles = {
            "Q": ("private", "FlashPrivateScratch"),
            "K": ("private", "FlashPrivateScratch"),
            "V": ("private", "FlashPrivateScratch"),
            "State": ("private", "State_gdn"),
            "G": ("row", "G"),
            "Beta": ("row", "Beta"),
            "Out": ("row", "Out"),
        }
        wrapped, _ = independent_rows(
            fn,
            128,
            roles,
            {name: index for index, name in enumerate(columns)},
            row_stride=stride,
            dynamic=True,
            offsets=offsets,
        )
        return wrapped

    return build


def certified_offsets(source):
    """Recover the wrapper's certified private-arena byte offsets."""
    offsets = {}
    for role in ("Q", "K", "V"):
        matches = re.findall(
            r"\bvoid\* " + role + r"_row = orin_row_pointer\(\([^;]+?\+ \(uint64_t\)(\d+)\)\);",
            source,
        )
        if len(matches) != 1:
            raise ValueError("Missing certified GDN row-wrapper scratch offset")
        offsets[role] = int(matches[0])
    return offsets


def publish(data, package, compiler, output, state_dtype, wrapper_only=False, direct_only=False):
    dynamics = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    layout = data["metadata"]["batch_layout"]
    columns, stride = layout["state_columns"], len(layout["state_columns"]) * len(layout["layers"])
    replaced, launches = {}, {}
    counts = {"decode": 0, "prefill": 0, "verify": 0, "restore": 0, "rows": 0}
    state_refs = set()
    # The prefix pass keys only the prefetch recurrence; the other modules keep
    # the plain pass set so their codegen matches the certified f32 package.
    keys = {
        "decode": "gdn-state-decode-f16",
        "prefill": "causal-gdn-prefill-f16",
        "verify": "gdn-state-verify-f16",
        "restore": "gdn-state-restore-f16",
        "rows": "gdn-state-rows-f16",
    }
    for old in package["kernels"]:
        state = state_buffer_name(old)
        if old["name"].startswith("flash_decode_state/"):
            from tools.model.host_compiler import private_roles

            roles, _ = private_roles(compiler.read(old["source"]).decode(), columns, stride)
            if roles.get("State", ("", ""))[1] != "State_gdn":
                state = None
                continue
        elif state is None:
            continue
        pointers = compiler.pointers(old)
        match = classify(old, pointers, dynamics)
        if match is None:
            raise ValueError(f"Uncertified GDN state kernel: {old['name']}")
        kind, rows, config = match
        if wrapper_only and kind != "rows":
            continue
        if direct_only and kind == "rows":
            continue
        if kind == "rows":
            build = wrapper_build(
                state_dtype,
                columns,
                stride,
                certified_offsets(compiler.read(old["source"]).decode()),
            )
        else:
            if "State" not in pointers:
                raise ValueError(f"GDN state kernel without a state pointer: {old['name']}")
            build = direct_build(kind, config, state_dtype)
        item, launch = compiler.emit(
            old,
            old["name"],
            pointers,
            rows,
            build=build,
            key=keys[kind],
            symbolic=False,
        )
        replaced[old["name"]] = item
        if old["name"] in dynamics:
            launches[old["name"]] = launch
        counts[kind] += 1
        if state is not None:
            state_refs.add(state)
    expected = {"decode": 36, "prefill": 72, "verify": 36, "restore": 36, "rows": 36}
    if wrapper_only:
        expected = {k: 0 for k in expected} | {"rows": 36}
    if direct_only:
        expected["rows"] = 0
    if counts != expected:
        raise ValueError(f"Incomplete GDN module coverage: {counts}")
    if wrapper_only:
        write_json(
            output / "published.json",
            dict(state_dtype=state_dtype, wrapper_only=True, counts=counts),
        )
        package["kernels"] = [replaced.get(k["name"], k) for k in package["kernels"]]
        package["dynamic_batch_kernels"] = [
            launches.get(k["name"], k) for k in package["dynamic_batch_kernels"]
        ]
        return counts

    def narrow(buffer):
        if STATE_BUFFER.match(buffer["name"]):
            if buffer["dtype"] != "f32" or buffer["shape"] != [786432]:
                raise ValueError(f"Unexpected GDN state contract: {buffer['name']}")
            buffer["dtype"] = "f16"
            return True
        return False

    if direct_only:
        if not state_refs:
            raise ValueError("Direct-only rebuild found no GDN state kernels")
    else:
        patched = sum(map(narrow, package["buffer_contracts"]))
        if patched != len(state_refs) or patched != sum(map(narrow, data["metadata"]["buffers"])):
            raise ValueError("GDN state contracts disagree between package and model")
    package["kernels"] = [replaced.get(k["name"], k) for k in package["kernels"]]
    package["dynamic_batch_kernels"] = [
        launches.get(k["name"], k) for k in package["dynamic_batch_kernels"]
    ]
    write_json(
        output / "published.json",
        dict(
            state_dtype=state_dtype,
            bindings=sum(counts.values()),
            counts=counts,
            state_buffers=sorted(state_refs),
        ),
    )
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "model-output", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--state-dtype",
        choices=("float16", "wrapper-f32", "direct-f32"),
        default="float16",
    )
    args = p.parse_args()
    from tools.operators.common import configure
    from tools.model.compact import compact_package

    configure()
    data, origin, package = load_model(args.model)
    with atomic_model(args.model_output) as staging:
        root = clone_model(args.model, staging, origin)
        compiler = Compiler(
            origin, root, args.output, {b["name"]: b for b in data["metadata"]["buffers"]}
        )
        mode = args.state_dtype
        publish(
            data,
            package,
            compiler,
            args.output,
            "float16" if mode == "float16" else "float32",
            wrapper_only=mode == "wrapper-f32",
            direct_only=mode == "direct-f32",
        )
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
    print("COMPLETE", digest, flush=True)


if __name__ == "__main__":
    main()
