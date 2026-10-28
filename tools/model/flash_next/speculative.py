"""Native Flash speculation: raw HC capture and owner-local prefix commit."""

from kernels.model import causal_rows, control, flash_control
from kernels.model.gdn_sequence import gdn_commit
from kernels.model.speculation import gather_target_hidden, select_prefix
from kernels.model.speculative_factors import save_gdn_factors
from tools.model.flash_next.bindings import DTYPES
from tools.model.flash_next.rows import args_history, kind, pointer, sections


def controls(b):
    width, ring = b.planner.C * b.planner.H, 4096
    for name, dtype, shape in [
        ("MtpHiddenRing", "f16", [ring, width]),
        ("MtpInput", "i32", [512]),
        ("MtpToken", "i32", [1]),
        ("MtpStatus", "i32", [1]),
        ("DraftLogits", "f32", [1, b.planner.V]),
        ("DraftPendingSaved", "f16", [4, 128]),
        ("DraftM1_Embedding", "f16", [1, 2560]),
        ("DraftM1_Condition", "f16", [1, width]),
        ("DraftM512_Embedding", "f16", [512, 2560]),
        ("DraftM512_Condition", "f16", [512, width]),
        ("M8_Embedding", "f16", [8, 2560]),
        ("M8_Ple", "f16", [8, 2560]),
        ("VerificationTokens", "i32", [8]),
        ("VerificationStatus", "i32", [8]),
        ("VerificationLogits", "f32", [8, b.planner.V]),
        ("VerificationPairs", "i32", [8, 2]),
        ("AcceptedInputs", "i32", [1]),
    ]:
        b.fixed(name, dtype, shape, "sequence")
    if b.planner.has_vision:
        from tools.model.flash_next.vision import overlay

        overlay(b, "causal_rows/warm/overlay", "DraftM512_Embedding", 512, mtp=True)
        overlay(b, "mtp_verify/begin/k0", "M8_Embedding", 8)
    for name, build, pointers, rows in [
        (
            "gather",
            lambda: gather_target_hidden(512, width, ring, ring=True),
            {
                "Target": pointer("MtpHiddenRing"),
                "Step": pointer("MtpPosition"),
                "Out": pointer("DraftM512_Condition"),
            },
            512,
        ),
        ("length", lambda: causal_rows.length(512), {"Length": pointer("MtpLength")}, 512),
        (
            "embedding",
            lambda: causal_rows.copy(512, 2560),
            {
                "Source": pointer("DraftM512_Embedding"),
                "Destination": pointer("DraftRows_Embedding"),
            },
            512,
        ),
        (
            "condition",
            lambda: causal_rows.copy(512, width),
            {
                "Source": pointer("DraftM512_Condition"),
                "Destination": pointer("DraftRows_Condition"),
            },
            512,
        ),
        ("advance", lambda: control.advance(512), {"Step": pointer("MtpPosition")}, 512),
        (
            "capture",
            lambda: flash_control.last_hidden(512, width),
            {"Hidden": pointer("DraftRows_Hidden"), "Condition": pointer("DraftM1_Condition")},
            512,
        ),
    ]:
        b.emit("causal_rows/warm/" + name, build, pointers, rows, key="mtp-warm-" + name)
    b.emit(
        "draft_rows/selections",
        flash_control.selection,
        {
            "Pair": pointer("FlashDecode_Selected"),
            "Token": pointer("MtpToken"),
            "Status": pointer("MtpStatus"),
        },
        1,
        key="mtp-selection",
        dynamic=False,
    )
    for rows in [1, 4096]:
        if rows not in b.hidden:
            raise ValueError("Missing frozen target HC capture binding")
        b.emit(
            f"mtp_capture_m{rows}/body/k0",
            lambda rows=rows: flash_control.capture(rows, width, ring),
            {
                "Hidden": b.hidden[rows],
                "Position": pointer("Position"),
                "Ring": pointer("MtpHiddenRing"),
            },
            rows,
            key=f"mtp-capture-{rows}",
            dynamic=rows > 1,
        )
    for family, rows in [("target", 512), ("verify", 8)]:
        b.emit(
            f"causal_rows/{family}/capture",
            lambda rows=rows: flash_control.capture(rows, width, ring),
            {
                "Hidden": pointer("FlashDecode_Residual"),
                "Position": pointer("Position"),
                "Ring": pointer("MtpHiddenRing"),
            },
            rows,
            key=f"mtp-capture-rows-{rows}",
        )
    b.emit(
        "causal_rows/verify/advance",
        lambda: control.advance(8),
        {"Step": pointer("Position")},
        8,
        key="mtp-verify-advance",
    )
    b.emit(
        "causal_rows/verify/selections",
        lambda: flash_control.selections(8),
        {
            "Pair": pointer("VerificationPairs"),
            "Tokens": pointer("VerificationTokens"),
            "Status": pointer("VerificationStatus"),
        },
        8,
        key="mtp-verify-selections",
    )
    return dict(
        position="MtpPosition",
        input="MtpInput",
        token="MtpToken",
        status="MtpStatus",
        verification_tokens="VerificationTokens",
        verification_status="VerificationStatus",
        draft_logits="DraftLogits",
        verification_logits="VerificationLogits",
        feature_index="MtpFeatureIndex" if b.planner.has_vision else None,
        accepted_inputs="AcceptedInputs",
        target_length="PositionOut",
        draft_program="mtp_draft",
        hidden_ring="MtpHiddenRing",
        commit_always=True,
        draft_snapshot_program="mtp_snapshot",
        draft_restore_program="mtp_restore_draft",
        default_verification_tokens=4,
        warm_plans=[dict(tokens=512, min_tokens=1, program="mtp_warm", head_program="mtp_head")],
        capture_plans=[dict(tokens=n, program=f"mtp_capture_m{n}") for n in [1, 4096]],
        verification_plans=[
            dict(
                tokens=8,
                min_tokens=2,
                program="mtp_verify",
                restore_program="mtp_restore",
                capture_program="mtp_capture",
            )
        ],
        batch_verification_tokens=list(range(2, 9)),
        batch_draft_profiles=[2, 4, 8, 16, 32, 64, 128],
        batch_commit=True,
        batch_warm=True,
        batch_warm_capacity=512,
        shared_verification_state=True,
    )


def verification(rows):
    b, p = rows.b, rows.p
    old = p.batch_body
    p.batch_body = True
    try:
        plan = p.trace(8, verify=True)
    finally:
        p.batch_body = old
    rows.prepare(plan)
    b.buffer(plan["index_scores"], "VerifyIndexScoresM8")
    for identity, role in rows.aliases(plan, 8).items():
        b.names[identity] = "FlashDecode_" + role
    history = args_history(plan)
    b.fixed("M8_HistoryOut", DTYPES[history.dtype], list(history.shape))
    b.names[history.untyped_storage()._cdata] = "M8_HistoryOut"
    saved = {}
    for role, tensor in plan["prefix_states"].items():
        name = "VerifyState_" + role.replace(":", "_")
        b.fixed(name, DTYPES[tensor.dtype], list(tensor.shape), "sequence")
        b.names[tensor.untyped_storage()._cdata] = name
        saved[role] = name
    for role, factors in plan["prefix_updates"].items():
        for tag, tensor in zip(["k", "g", "update"], factors):
            name = "VerifyState_" + role.replace(":", "_") + "_" + tag
            b.fixed(name, DTYPES[tensor.dtype], list(tensor.shape), "sequence")
            b.names[tensor.untyped_storage()._cdata] = name

    def emit(program, section, build, pointers, key):
        operations = b.groups.setdefault(program, {}).setdefault(section, [])
        name = f"{program}/{section}/k{len(operations)}"
        b.emit(name, build, pointers, 8, key=key)
        operations.append(dict(kind="kernel", name=name))

    captured = set()
    for section, label, args in sections(plan, b.names):
        layer = int(section.split("/")[0][5:]) if section.startswith("layer") else None
        if label == "gdn-update-save":
            if layer not in captured:
                key, decay, _ = plan["prefix_updates"][f"{layer}:gdn"]
                # The recurrence consumes K/G from these exact operand views;
                # save both with one copy kernel and the same real-row stride.
                recurrence = next(
                    a
                    for (_, a), tag in zip(plan["ops"], plan["labels"])
                    if tag.startswith("gdn-sequence-") and a[5] is p.states[f"{layer}:gdn"]
                )
                pointers = dict(
                    K=b.buffer(recurrence[1]),
                    G=b.buffer(recurrence[3]),
                    SavedK=b.buffer(key),
                    SavedG=b.buffer(decay),
                )
                emit(
                    "causal_rows/verify",
                    f"layer{layer}/private",
                    lambda: save_gdn_factors(8),
                    pointers,
                    "mtp-save-gdn-factors",
                )
                captured.add(layer)
            continue
        if label.startswith("prefix-history-"):
            section = "layer1/ple" if args[1].numel() == 9 * p.C * p.H else f"layer{layer}/private"
        elif label.startswith("prefix-pending-"):
            section = f"layer{layer}/private"
        elif kind(label):
            section = section.replace("ple_private", "ple")
        else:
            continue
        b.bind("causal_rows/verify", section, label, args, 8, dynamic=True)
    for layer in p.layer_ids:
        section = f"layer{layer}"
        if layer % 4 == 3:
            sources = [(f"{layer}:pending", p.states[f"{layer}:pending"])]
        else:
            sources = [(f"{layer}:conv", p.states[f"{layer}:conv"])]
        if layer == 1:
            sources.append(("ple", p.states["ple"]))
        for role, state in sources:
            emit(
                "causal_rows/restore",
                section,
                lambda state=state: select_prefix(8, state.numel(), dtype="float16"),
                {
                    "Prefix": pointer(saved[role]),
                    "Count": pointer("AcceptedInputs"),
                    "State": b.buffer(state),
                },
                f"mtp-restore-prefix-{state.numel()}",
            )
        if layer % 4 != 3:
            k, g, update = plan["prefix_updates"][f"{layer}:gdn"]
            emit(
                "causal_rows/restore",
                section,
                lambda: gdn_commit(8),
                dict(
                    K=b.buffer(k),
                    G=b.buffer(g),
                    Update=b.buffer(update),
                    Accepted=pointer("AcceptedInputs"),
                    State=b.buffer(p.states[f"{layer}:gdn"]),
                ),
                "mtp-gdn-commit",
            )
    return plan
