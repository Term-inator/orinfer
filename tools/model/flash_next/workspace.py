"""Preserve shared QSA and expert staging across prefill and row programs."""

from bisect import bisect_right
import json
import math

from tools.build.files import write_json
from tools.model.host_compiler import Compiler
from tools.model.publication import atomic_model, clone_model, commit_package, load_model

PARTIAL = ("Out", "Max", "Den")
PRODUCER = {"Query", "K", "V", "KS", "VS", "Selected", "Position", *PARTIAL}
CONSUMER = {"Gate", "Output", *PARTIAL}

DISPATCH = {"A", "D", "DS", "IDs", "RelativeSlot", "RowOffsets", "Scale", "SlotMap"}
EXPERT = {
    "A",
    "AS",
    "Book",
    "C",
    "Counts",
    "Offsets",
    "PAddresses",
    "TileCount",
    "TileExpert",
    "TileRow",
    "WSAddresses",
}
QUANT = {"Q", "S", "Signs", "X"}
COMBINE = {"Expert", "Output", "Prob", "Shared", "SharedGate", "SlotMap"}
# Producer roles, consumer roles, linked views, dtypes, adjacency requirement.
# Expert down output survives independent shared-FFN work up to the combine.
LIFETIMES = (
    (PRODUCER, CONSUMER, tuple((r, r) for r in PARTIAL), ("f32",) * 3, True),
    (DISPATCH, EXPERT, (("D", "A"),), ("i8",), True),
    (EXPERT, QUANT, (("C", "X"),), ("f16",), True),
    (EXPERT, COMBINE, (("C", "Expert"),), ("f16",), False),
)


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
    """Merge certified serial QSA and routed-expert staging lifetimes.

    Match exact exported primitives and complete storage geometry. Every use
    must belong to a producer/consumer interval in one declared program, with
    no frame, side-stream or overlapping alias lifetime.
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
    pointers, arguments = {}, {}
    for name, kernel in kernels.items():
        host = compiler.host(kernel)
        roles = {
            a["value"].removesuffix(".data_ptr()")
            for a in host["ordered_arguments"]
            if a["ctype"] == "ctypes.c_void_p" and a["value"].endswith(".data_ptr()")
        }
        if any(roles in (producer, consumer) for producer, consumer, *_ in LIFETIMES):
            pointers[name] = compiler.pointers(kernel)
            arguments[name] = host["ordered_arguments"]
    groups, chains, pairs = {}, {}, []
    for program, ops in manifest["programs"].items():
        uses = {}
        for i, op in enumerate(ops):
            for name in _buffers(op, kernels):
                uses.setdefault(name, []).append(i)
        for i, first in enumerate(ops):
            if first["kind"] != "kernel":
                continue
            a = pointers.get(first["name"], {})
            for pattern, (producer, consumer, links, dtypes, adjacent) in enumerate(LIFETIMES):
                if set(a) != producer:
                    continue
                views = [a[left] for left, _ in links]
                if any(v.get("offset", 0) for v in views):
                    continue
                triple = tuple(v["name"] for v in views)
                if len(set(triple)) != len(triple):
                    continue
                positions = [bisect_right(uses[n], i) for n in triple]
                if any(position == len(uses[n]) for n, position in zip(triple, positions)):
                    continue
                next_uses = [uses[n][position] for n, position in zip(triple, positions)]
                j = next_uses[0]
                if any(index != j for index in next_uses) or (adjacent and j != i + 1):
                    continue
                second = ops[j]
                b = pointers.get(second.get("name"), {}) if second["kind"] == "kernel" else {}
                if set(b) != consumer or any(a[left] != b[right] for left, right in links):
                    continue
                signature = [pattern]
                signature.extend(
                    [kernels[op["name"]][field] for field in ("module", "source", "symbol")]
                    for op in (first, second)
                )
                # Row defaults and launch expressions remain on each binding;
                # storage equivalence needs the same code and ordered ABI.
                signature.append([arguments[op["name"]] for op in (first, second)])
                signature.append(
                    [{k: v for k, v in buffers[n].items() if k != "name"} for n in triple]
                )
                key = json.dumps(signature, sort_keys=True)
                groups.setdefault(key, set()).add(triple)
                chains.setdefault(triple, set()).add(program)
                pairs.append((triple, program, i, j, dtypes))
    groups = {
        key: triples
        for key, triples in groups.items()
        if len(triples) > 1 and any(chains[t] & prefill for t in triples)
    }
    selected = {triple for triples in groups.values() for triple in triples}
    covered = {}
    for triple, program, i, j, dtypes in pairs:
        if triple not in selected:
            continue
        if program not in prefill | rows:
            raise ValueError("Staging lifetime outside declared prefill/row programs")
        for n, dtype in zip(triple, dtypes):
            v = buffers[n]
            if (
                data["buffer_scopes"][n] != "workspace"
                or v["dtype"] != dtype
                or v["access"] != "read_write"
                or v.get("data")
            ):
                raise ValueError("Staging must be mutable workspace with the primitive's dtype")
            if n in forbidden:
                raise ValueError("Staging survives a frame or enters a parallel branch")
            covered.setdefault(n, set()).update(((program, i), (program, j)))
    for program, ops in manifest["programs"].items():
        for i, op in enumerate(ops):
            for name in _buffers(op, kernels) & covered.keys():
                if (program, i) not in covered[name]:
                    raise ValueError("Staging has a use outside its producer/consumer pair")
    used = {
        op["name"] for ops in manifest["programs"].values() for op in ops if op["kind"] == "kernel"
    }
    # A selected row variant occupies the same logical slot. Its complete
    # pointer-role map and ordered ABI must preserve the certified lifetime.
    for slot, variants in manifest["row_execution"].get("variants", {}).items():
        if slot not in used:
            continue
        for variant in variants:
            names = variant["kernels"]
            if (
                len(names) == 1
                and slot in pointers
                and names[0] in pointers
                and pointers[slot] == pointers[names[0]]
                and arguments[slot] == arguments[names[0]]
            ):
                used.add(names[0])
    if any(
        k["name"] not in used and any(a.get("name") in covered for a in k["args"])
        for k in package["kernels"]
    ):
        raise ValueError("Staging has an unclassified binding")
    aliases = {}
    for triples in groups.values():
        original = [t for t in triples if chains[t] & prefill]
        if len(original) != 1:
            raise ValueError("Staging needs one original prefill storage")
        canonical = original[0]
        for triple in triples:
            if triple != canonical:
                if not chains[triple] <= rows:
                    raise ValueError("Staging belongs to another prefill program")
                for name, target in zip(triple, canonical):
                    if aliases.setdefault(name, target) != target:
                        raise ValueError("Conflicting staging aliases")
    intervals = {}
    for triple, program, i, j, _ in pairs:
        if triple in selected:
            for name in triple:
                target = aliases.get(name, name)
                intervals.setdefault((program, target), set()).add((i, j, name))
    for values in intervals.values():
        end, previous = -1, None
        for i, j, name in sorted(values):
            if i <= end and name != previous:
                raise ValueError("Shared staging lifetimes overlap")
            end, previous = max(end, j), name
    for kernel in package["kernels"]:
        for argument in kernel["args"]:
            if argument["kind"] in ("buffer", "buffer_slice"):
                argument["name"] = aliases.get(argument["name"], argument["name"])
    widths = {"i8": 1, "f16": 2, "f32": 4}
    saved = sum(math.prod(buffers[n]["shape"]) * widths[buffers[n]["dtype"]] for n in aliases)
    package["buffer_contracts"] = [
        b for b in package["buffer_contracts"] if b["name"] not in aliases
    ]
    data["metadata"]["buffers"] = [
        b for b in data["metadata"]["buffers"] if b["name"] not in aliases
    ]
    for name in aliases:
        del data["buffer_scopes"][name]
    if set(aliases) & (set(_strings(data)) | set(_strings(package))):
        raise ValueError("Removed staging has additional metadata references")
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
