"""Initial native text construction from the project's sliced Flash checkpoint."""

import json
from pathlib import Path
import shutil

from tools.build.files import file_hash, write_json
from tools.model.flash_next.checkpoint import Checkpoint
from tools.model.flash_next.config import validate, validate_pair
from tools.model.flash_next.storage import Weights, payload_hash
from tools.model.safetensors_source import Source, local_header


def restore(weights, assembly):
    """Verify a completed bounded assembly before reusing its immutable banks."""
    weights.buffers = assembly["buffers"]
    weights.scopes = assembly["scopes"]
    weights.logical = assembly["logical"]
    weights.mapping = dict(assembly["mapping"])
    headers = {}
    for name, descriptor in weights.buffers.items():
        path = weights.directory / weights.mapping[name]
        if path not in headers:
            headers[path] = local_header(path)
        begin, header = headers[path]
        item = header[name]
        first, last = item["data_offsets"]
        if (
            item["shape"] != descriptor["shape"]
            or payload_hash(path, begin + first, last - first) != descriptor["data"]["sha256"]
        ):
            raise ValueError(f"Cached initial assembly identity mismatch: {name}")


def build(args):
    from tools.model.flash_next.build import source_identity

    checkpoint = Checkpoint(args.checkpoint, verify_hashes=not args.plan_only)
    validate(checkpoint.config, args.context)
    draft_checkpoint = None
    if args.mtp_checkpoint:
        draft_checkpoint = Checkpoint(args.mtp_checkpoint, verify_hashes=not args.plan_only)
        validate_pair(draft_checkpoint.config, checkpoint.config, args.context)
    if args.context % 4 or args.context < 4096:
        raise ValueError(
            "Initial native text construction requires a multiple-of-four context >=4096"
        )
    configuration = checkpoint.config["quantization_config"]
    if configuration.get("component", "text") != "text":
        raise ValueError("Initial construction requires the target text checkpoint")
    contract = dict(
        checkpoint=file_hash(args.checkpoint / "model.safetensors.index.json"),
        config=file_hash(args.checkpoint / "config.json"),
        context=args.context,
        original_index=file_hash(args.source_index),
        source=source_identity(),
        mtp=(
            dict(
                checkpoint=file_hash(args.mtp_checkpoint / "model.safetensors.index.json"),
                config=file_hash(args.mtp_checkpoint / "config.json"),
            )
            if draft_checkpoint
            else None
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "inventory.json",
        dict(
            initial_text=True,
            contract=contract,
            original=configuration["source"],
            revision=configuration["source_revision"],
        ),
    )
    if args.plan_only:
        return
    from tools.model.flash_next.bindings import Bindings
    from tools.model.flash_next.compute import lower
    from tools.model.flash_next.inputs import publish as publish_inputs
    from tools.model.flash_next.package_initial import publish
    from tools.model.flash_next.planning import Planner
    from tools.model.flash_next.precision import publish as publish_q4
    from tools.model.flash_next.rows import Rows
    from tools.model.frontend import (
        prepare_chat_template,
        prepare_generation_config,
        prepare_tokenizer,
    )
    from tools.operators.common import configure

    configure()
    root = args.output / "model"
    weights = Weights(checkpoint, root)
    receipt = root / "assembly.json"
    assembly = json.loads(receipt.read_text()) if receipt.exists() else None
    if assembly and assembly.get("contract") == contract:
        restore(weights, assembly)
    else:
        assembly = dict(weights.assemble(), contract=contract)
        write_json(receipt, assembly)
    draft_assembly = None
    if draft_checkpoint:
        draft_weights = Weights(draft_checkpoint, root)
        receipt = root / "draft-assembly.json"
        draft_assembly = json.loads(receipt.read_text()) if receipt.exists() else None
        if draft_assembly and draft_assembly.get("contract") == contract:
            restore(draft_weights, draft_assembly)
        else:
            draft_assembly = dict(draft_weights.assemble(), contract=contract)
            write_json(receipt, draft_assembly)
        # Keep each immutable shape assembly separate from the joined arena.
        weights.buffers = dict(weights.buffers)
        weights.scopes = dict(weights.scopes)
        weights.attach_mtp(draft_weights)
    source = Source(
        args.source_index,
        repo=configuration["source"],
        revision=configuration["source_revision"],
        cache=args.cache / "source-headers",
    )
    publish_q4(weights, source, args.output / "q4", rows=4096)
    for name in ("config.json", "tokenizer_config.json", "LICENSE"):
        if (args.checkpoint / name).is_file():
            shutil.copyfile(args.checkpoint / name, root / name)
    prepare_tokenizer(args.checkpoint, root / "tokenizer.json")
    prepare_chat_template(args.checkpoint, root / "chat_template.jinja")
    prepare_generation_config(args.checkpoint, root / "generation_config.json")
    write_json(root / "inputs-asset.json", publish_inputs(checkpoint, root))
    planner = Planner(assembly, args.context, args.output)
    bindings = Bindings(planner, weights, args.output, args.cache)
    for rows in (4096, 1):
        bindings.trace(rows)
    row_programs = Rows(bindings, mtp=draft_checkpoint is not None)
    for method in (row_programs.shared, row_programs.private, row_programs.causal):
        method()
        print("INITIAL ROWS", method.__name__, "COMPLETE", flush=True)
    mtp = None
    if draft_checkpoint:
        from tools.model.flash_next.draft import DraftRows
        from tools.model.flash_next.speculative import controls, verification

        draft_planner = Planner(draft_assembly, args.context, args.output, target=planner)
        draft_bindings = Bindings(draft_planner, weights, args.output, args.cache, shared=bindings)
        DraftRows(draft_bindings).build()
        mtp = controls(bindings)
        verification(row_programs)
    precision = lower(bindings)
    write_json(
        args.output / "initial-bindings.json",
        dict(
            buffers=bindings.buffers,
            scopes=bindings.scopes,
            kernels=bindings.kernels,
            dynamic=bindings.dynamic,
            batch_layout=row_programs.layout(),
            precision=precision,
            mtp=mtp,
        ),
    )
    data = publish(
        bindings,
        row_programs.layout(),
        args.model_output,
        args.output,
        Path(__file__).resolve().parents[3] / "target/release/orinfer",
        mtp=mtp,
    )
    write_json(
        args.output / "results.json",
        dict(
            complete=True,
            contract=contract,
            model=str(args.model_output),
            execution_package=data["execution_package"],
            precision=precision,
            quality_validated=False,
            performance_validated=False,
        ),
    )
    print("INITIAL NATIVE COMPLETE", args.model_output, flush=True)
