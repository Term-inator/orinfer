"""Preserve shared QSA staging storage across prefill and row programs."""

import json
import math

from tools.build.files import write_json
from tools.model.host_compiler import Compiler
from tools.model.publication import atomic_model, clone_model, commit_package, load_model

PARTIAL = ("Out", "Max", "Den")
PRODUCER = {"Query", "K", "V", "KS", "VS", "Selected", "Position", *PARTIAL}
CONSUMER = {"Gate", "Output", *PARTIAL}


def _buffers(op, kernels):
    if op["kind"] == "kernel":
        return {
            a["name"]
            for a in kernels[op["name"]]["args"]
            if a["kind"] in ("buffer", "buffer_slice")
        }
    if op["kind"] == "copy":
        return {op["source"], op["destination"]}
    if op["kind"] == "zero":
        return {op["destination"]}
    raise ValueError("Unknown workspace operation")


def _side_buffers(manifest, kernels):
    """Use the same complete, unambiguous slot boundaries as the executor."""
    slots = {}
    for slot, variants in manifest.get("row_execution", {}).get("variants", {}).items():
        for variant in variants:
            for name in variant["kernels"]:
                if slots.setdefault(name, slot) != slot:
                    raise ValueError("Ambiguous row kernel slot")
    regions = {tuple(r) for rs in manifest.get("graph_branches", {}).values() for r in rs}
    result = set()
    for ops in manifest["programs"].values():
        positions, ambiguous = {}, set()
        for i, op in enumerate(ops):
            if op["kind"] != "kernel":
                continue
            slot = slots.get(op["name"], op["name"])
            if slot in positions:
                first, last = positions[slot]
                if last + 1 != i:
                    ambiguous.add(slot)
                positions[slot] = first, i
            else:
                positions[slot] = i, i
        for fork, start, last, join in regions:
            names = (fork, start, last, join)
            if any(n in ambiguous or n not in positions for n in names):
                continue
            f, s, e, j = (
                positions[fork][0],
                positions[start][0],
                positions[last][1],
                positions[join][0],
            )
            if not f < s <= e or e + 1 != j:
                raise ValueError("Invalid workspace fork/join contract")
            for op in ops[s : e + 1]:
                result.update(_buffers(op, kernels))
    return result


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def share(data, package, manifest, compiler):
    """Merge only complete, serial QSA producer/consumer staging lifetimes.

    The shape planner reuses its prefill pool for rows >=256. QSA Partial/Max/Den
    are not row interfaces or frame state: each adjacent pair produces/consumes
    them on the main stream. Match both exact exported ABIs and full storage
    geometry; reject any extra use, frame lifetime or side-stream consumer.
    """
    if data["architecture"] != "flash_next" or manifest.get("mtp") is not None:
        raise ValueError("Workspace sharing requires the current Flash target-only contract")
    kernels = {k["name"]: k for k in package["kernels"]}
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    prefill = {
        name
        for plan in manifest["prefill_plans"]
        for name in (plan["prefill_program"], *plan.get("stages", []))
    }
    rows = set(manifest["row_execution"]["programs"]) - prefill
    forbidden = _side_buffers(manifest, kernels)
    forbidden.update(manifest.get("reset_buffers", []))
    forbidden.update(n for p in manifest["prefill_plans"] for n in p.get("live_buffers", []))
    pointers = {}
    for name, kernel in kernels.items():
        host = compiler.host(kernel)
        roles = {
            a["value"].removesuffix(".data_ptr()")
            for a in host["ordered_arguments"]
            if a["ctype"] == "ctypes.c_void_p" and a["value"].endswith(".data_ptr()")
        }
        if roles == PRODUCER or roles == CONSUMER:
            pointers[name] = compiler.pointers(kernel)
    groups, chains, pairs = {}, {}, []
    for program, ops in manifest["programs"].items():
        for i, (first, second) in enumerate(zip(ops, ops[1:])):
            if first["kind"] != "kernel" or second["kind"] != "kernel":
                continue
            a, b = pointers.get(first["name"], {}), pointers.get(second["name"], {})
            if set(a) != PRODUCER or set(b) != CONSUMER:
                continue
            if any(a[r] != b[r] or a[r].get("offset", 0) for r in PARTIAL):
                continue
            triple = tuple(a[r]["name"] for r in PARTIAL)
            if len(set(triple)) != 3:
                continue
            signature = [
                [kernels[op["name"]][field] for field in ("module", "source", "host_abi", "symbol")]
                for op in (first, second)
            ]
            signature.append([{k: v for k, v in buffers[n].items() if k != "name"} for n in triple])
            key = json.dumps(signature, sort_keys=True)
            groups.setdefault(key, set()).add(triple)
            chains.setdefault(triple, set()).add(program)
            pairs.append((triple, program, i))
    groups = {
        key: triples
        for key, triples in groups.items()
        if len(triples) > 1 and any(chains[t] & prefill for t in triples)
    }
    selected = {triple for triples in groups.values() for triple in triples}
    covered = {}
    for triple, program, i in pairs:
        if triple not in selected:
            continue
        if program not in prefill | rows:
            raise ValueError("QSA staging lifetime outside declared prefill/row programs")
        for n in triple:
            v = buffers[n]
            if data["buffer_scopes"][n] != "workspace" or v["dtype"] != "f32" or v.get("data"):
                raise ValueError("QSA staging must be mutable FP32 workspace")
            if n in forbidden:
                raise ValueError("QSA staging survives a frame or enters a parallel branch")
            covered.setdefault(n, set()).update(((program, i), (program, i + 1)))
    for program, ops in manifest["programs"].items():
        for i, op in enumerate(ops):
            for name in _buffers(op, kernels) & covered.keys():
                if (program, i) not in covered[name]:
                    raise ValueError("QSA staging has a use outside its producer/consumer pair")
    used = {
        k
        for ops in manifest["programs"].values()
        for op in ops
        if op["kind"] == "kernel"
        for k in [op["name"]]
    }
    if any(
        k["name"] not in used and any(a.get("name") in covered for a in k["args"])
        for k in package["kernels"]
    ):
        raise ValueError("QSA staging has an unclassified binding")
    aliases = {}
    for triples in groups.values():
        original = [t for t in triples if chains[t] & prefill]
        if len(original) != 1:
            raise ValueError("QSA staging needs one original prefill storage")
        canonical = original[0]
        for triple in triples:
            if triple != canonical:
                if not chains[triple] <= rows:
                    raise ValueError("QSA storage belongs to another prefill program")
                aliases.update(zip(triple, canonical))
    for kernel in package["kernels"]:
        for argument in kernel["args"]:
            if argument["kind"] in ("buffer", "buffer_slice"):
                argument["name"] = aliases.get(argument["name"], argument["name"])
    saved = sum(math.prod(buffers[n]["shape"]) * 4 for n in aliases)
    package["buffer_contracts"] = [
        b for b in package["buffer_contracts"] if b["name"] not in aliases
    ]
    data["metadata"]["buffers"] = [
        b for b in data["metadata"]["buffers"] if b["name"] not in aliases
    ]
    for name in aliases:
        del data["buffer_scopes"][name]
    if set(aliases) & (set(_strings(data)) | set(_strings(package))):
        raise ValueError("Removed QSA storage has additional metadata references")
    return dict(aliases=aliases, saved_workspace_bytes=saved)


def rebuild(args):
    manifest = {}
    data, origin, package = load_model(args.model, plan=manifest)
    compiler = Compiler(
        origin, origin, args.output, {b["name"]: b for b in data["metadata"]["buffers"]}
    )
    report = share(data, package, manifest, compiler)
    with atomic_model(args.model_output) as staging:
        operator = clone_model(args.model, staging, origin)
        digest = commit_package(staging, operator, data, package)
    write_json(
        args.output / "results.json", dict(complete=True, execution_package=digest, **report)
    )
    print("COMPLETE", digest, report, flush=True)
