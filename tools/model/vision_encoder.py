"""Shared checkpoint-exact ViT traversal; callers bind storage and projections."""

import math

from tools.model.compile import primitive


def build(v, emit, dense, norm, weight, *, kernel=primitive):
    from kernels.vision import encoder as e

    h, f, out = v["hidden_size"], v["intermediate_size"], v["out_hidden_size"]
    dense("begin", "patch", h, 1536, "VPixels", "VNorm", "patch_embed.proj", separate_bias=True)
    emit(
        "begin",
        "position",
        f"position-{h}",
        lambda: kernel(e.position, h, math.isqrt(v["num_position_embeddings"])),
        dict(X="VNorm", W=weight("pos_embed.weight"), Grid="VGrid", Length="VLength", Y="VHidden"),
    )
    for i in range(v["depth"]):
        s = f"layer{i}"
        p = f"blocks.{i}"
        norm(s, "norm1", "VHidden", "VNorm", p + ".norm1")
        dense(s, "qkv", 3 * h, h, "VNorm", "VQKV", p + ".attn.qkv")
        emit(
            s,
            "rope",
            f"rope-{h}-{v['num_heads']}",
            lambda: kernel(e.qkv_rope, h, v["num_heads"]),
            dict(X="VQKV", Grid="VGrid", Length="VLength", Q="VQ", K="VK", V="VV"),
        )
        emit(
            s,
            "attention",
            f"attention-{h}-{v['num_heads']}",
            lambda: kernel(e.attention, h, v["num_heads"]),
            dict(Q="VQ", K="VK", V="VV", Length="VLength", Y="VAttention"),
        )
        dense(s, "projection", h, h, "VAttention", "VProjection", p + ".attn.proj")
        emit(
            s,
            "residual1",
            f"add-{h}",
            lambda: kernel(e.add, h),
            dict(X="VProjection", Residual="VHidden", Y="VHidden"),
        )
        norm(s, "norm2", "VHidden", "VNorm", p + ".norm2")
        dense(s, "fc1", f, h, "VNorm", "VFFN", p + ".mlp.linear_fc1", "gelu_tanh")
        dense(s, "fc2", h, f, "VFFN", "VProjection", p + ".mlp.linear_fc2")
        emit(
            s,
            "residual2",
            f"add-{h}",
            lambda: kernel(e.add, h),
            dict(X="VProjection", Residual="VHidden", Y="VHidden"),
        )
    norm("end", "norm", "VHidden", "VNorm", "merger.norm")
    dense("end", "merge1", 4 * h, 4 * h, "VNorm", "VMerged", "merger.linear_fc1", "gelu", divisor=4)
    dense("end", "merge2", out, 4 * h, "VMerged", "VOutput", "merger.linear_fc2", divisor=4)
