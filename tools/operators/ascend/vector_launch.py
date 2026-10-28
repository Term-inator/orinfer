"""Bounded AIV launch lowering for the pinned A2 compiler's paired Vector frame."""

import re


def lower_independent_vector(source):
    """Keep the original arithmetic and pair indices; omit the unused Cube task.

    The pinned code generator emits MIX_AIC_1_2 even for independent UB-only
    kernels. Its Vector block index already spans twice the logical grid.
    AIV-only launches therefore double the host grid and derive the pair lane
    from that physical index. Unrecognized or coupled programs keep their
    original launch. No cross-block synchronization is introduced.
    """
    mixed = "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)"
    if source.count(mixed) != 1:
        return source
    if any(x in source for x in ("ASCEND_IS_AIC", "CrossCore", "SyncAll", "GlobalBarrier")):
        return source
    if len(re.findall(r"\bfftsAddr\b", source)) != 4:
        return source
    if source.count("AscendC::GetSubBlockIdx()") != 1:
        return source
    block = re.search(r"auto (\w+) = AscendC::GetBlockIdx\(\);", source)
    if block is None or f"{block[1]} = {block[1]} / 2;" not in source:
        return source
    launches = list(re.finditer(r"\w+<<<(\d+), nullptr, stream>>>", source))
    if len(launches) != 1 or not 0 < int(launches[0][1]) <= 32767:
        return source
    unused = ("ascend_l0a", "ascend_l0b", "ascend_l1", "ascend_l0c")
    if any(len(re.findall(r"\b" + name + r"\b", source)) != 2 for name in unused):
        return source
    lines = source.splitlines(keepends=True)
    # Each removed line contains only the compiler's unused allocation setup.
    removed = [line for line in lines if any(re.search(r"\b" + n + r"\b", line) for n in unused)]
    if any(
        re.fullmatch(r"\s*(?:AscendC::TBuf<[^>]+> \w+;\s*)?(?:pipe.InitBuffer\([^;]+\);\s*)?", line)
        is None
        for line in removed
    ):
        return source
    lowered = "".join(line for line in lines if line not in removed)
    lowered = lowered.replace(mixed, "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)")
    lowered = lowered.replace("AscendC::GetSubBlockIdx()", "(AscendC::GetBlockIdx() % 2)")
    return re.sub(
        r"(\w+)<<<(\d+), nullptr, stream>>>",
        lambda m: f"{m[1]}<<<{int(m[2]) * 2}, nullptr, stream>>>",
        lowered,
    )
