"""Emit the shared chunkwise prefill recurrence around request-owned FP32 state."""

from kernels.model import gdn_chunk
from tools.model.compile import primitive
from tools.model.gdn_tuning import chunked


def prefill(b, section, layer):
    from tools.build.gpu import gpu_lock

    w, hv = b.weights, b.hv
    if "GdnChunkW" not in w.buffers:
        for name, dtype, shape in [
            ("GdnChunkW", "f16", [hv, b.capacity, 128]),
            ("GdnChunkU", "f16", [hv, b.capacity, 128]),
            ("GdnChunkD", "f16", [hv, b.capacity, 32]),
            ("GdnChunkG", "f32", [hv, b.capacity, 2]),
        ]:
            w.fixed(name, dtype, shape)
    with gpu_lock(b.lock):
        selection = chunked(b.row_tuner)
    config = selection["configs"][int(selection["default"])]
    b.gdn_execution["prefill"] = "chunked"
    common = dict(Q="Q", K="K", W="GdnChunkW", U="GdnChunkU", D="GdnChunkD", GP="GdnChunkG")
    identity = f"{b.hk}-{hv}-{config}"
    b.emit(
        section + "/recurrent_prepare",
        "gdn-wy-prepare-" + identity,
        lambda: primitive(
            gdn_chunk.prepare, b.capacity, HK=b.hk, HV=hv, C=config["C"], dynamic_rows=True
        ),
        dict(common, V="V", G="g", Beta="Beta"),
    )
    b.emit(
        section + "/recurrent_chunk",
        "gdn-wy-recurrent-" + identity,
        lambda: primitive(
            gdn_chunk.recurrent, b.capacity, HK=b.hk, HV=hv, **config, dynamic_rows=True
        ),
        dict(common, State=f"L{layer}_State", Out="Y"),
    )
