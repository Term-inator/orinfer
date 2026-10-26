"""Lower selected Flash projections and their shared signed input transforms."""

import copy
import re

from tools.model.flash_next.kernel_policy import projection_policies


def operand(argument):
    return argument["name"], argument.get("offset", 0)


def lower(bindings):
    from kernels.model.q4_projection import q4_gemv, q4_projection
    from kernels.model.rotation_a8 import rotate_activation_a8

    b = bindings
    policies = projection_policies(b.buffers)
    original = list(b.kernels)
    dynamic = {d["name"]: d for d in b.dynamic}
    quantizers = {}
    pointers = {k["name"]: b.compiler.pointers(k) for k in original}
    for kernel in original:
        name = kernel["name"]
        if "/rows/" in name:
            continue
        p = pointers[name]
        if set(p) == {"Q", "S", "X"}:
            section, slot = name.rsplit("/", 1)
            if not re.fullmatch(r"k\d+", slot):
                continue
            quantizers.setdefault((section, operand(p["Q"]), operand(p["S"])), []).append(
                (int(slot[1:]), p["X"])
            )
    b.kernels, b.dynamic = [], []
    transforms, replaced = {}, {}
    for old in original:
        name = old["name"]
        if "/rows/" in name:
            continue
        p = pointers[name]
        policy = policies.get(p.get("Weight", {}).get("name"))
        if policy is None:
            b.kernels.append(old)
            if name in dynamic:
                b.dynamic.append(dynamic[name])
            continue
        if set(p) != {"A", "Weight", "WeightScale", "TokenScale", "Output"}:
            raise ValueError("Selected Q4 projection has an unexpected arithmetic ABI")
        section, slot = name.rsplit("/", 1)
        producers = quantizers.get((section, operand(p["A"]), operand(p["TokenScale"])), [])
        producers = [(i, x) for i, x in producers if i < int(slot[1:])]
        if not producers:
            raise ValueError(f"Q4 input has no preceding quantizer: {name}")
        _, x = max(producers, key=lambda item: item[0])
        symbolic = name in dynamic
        rows = dynamic[name]["capacity"] if symbolic else 1
        identity = section, operand(x), policy.signs, rows, symbolic
        if identity not in transforms:
            prefix = f"FlashQ4_{rows}_{policy.inputs}"
            a, scale = prefix + "_Activation", prefix + "_Scale"
            b.fixed(a, "i8", [rows, policy.inputs])
            b.fixed(scale, "f16", [rows])
            b.emit(
                name + "/q4_rotation",
                lambda: rotate_activation_a8(rows, policy.inputs),
                dict(X=x, Signs=pointer(policy.signs), Q=pointer(a), S=pointer(scale)),
                rows,
                key=f"q4-rotation-{rows}-{policy.inputs}",
                dynamic=symbolic,
            )
            transforms[identity] = pointer(a), pointer(scale)
        a, scale = transforms[identity]
        output_dtype = {"f16": "float16", "f32": "float32"}[b.buffers[p["Output"]["name"]]["dtype"]]

        def build(rows=rows, policy=policy, output_dtype=output_dtype, symbolic=symbolic):
            if symbolic:
                return q4_projection(
                    rows, policy.outputs, policy.inputs, output_dtype=output_dtype, block_m=16
                )
            return q4_gemv(rows, policy.outputs, policy.inputs, output_dtype=output_dtype)

        replaced[name] = b.emit(
            name,
            build,
            dict(A=a, AS=scale, W=pointer(policy.packed), S=pointer(policy.scale), Out=p["Output"]),
            rows,
            key=f"q4-{rows}-{policy.outputs}-{policy.inputs}-{output_dtype}",
            dynamic=symbolic,
        )
    launches = {d["name"]: d for d in b.dynamic}
    canonical = {k["name"]: k for k in b.kernels}
    for old in original:
        if "/rows/" not in old["name"]:
            continue
        slot = old["name"].split("/rows/", 1)[0]
        selected = copy.deepcopy(canonical[slot])
        selected["name"] = old["name"]
        launch = copy.deepcopy(launches[slot])
        launch["name"] = old["name"]
        b.kernels.append(selected)
        b.dynamic.append(launch)
    return dict(projections=len(replaced), input_transforms=len(transforms))


def pointer(name):
    return dict(kind="buffer", name=name)
