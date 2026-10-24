"""Promote explicit dense Qwen E8P projections from original BF16 to rotated W8."""

import argparse
import json
from pathlib import Path

from tools.model.package import config_signature
from tools.quantization.promotion import upgrade
from tools.quantization.qwen_e8p import rotated_w8_matrices, text_configuration


def configuration(config, *, quantized=False):
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Projection promotion requires a dense Qwen checkpoint")
    if quantized:
        from tools.model.qwen35.build import configuration as encoded_configuration

        encoded_configuration(config, 4096)
    else:
        text_configuration(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "upgrade-from", "output", "matrices"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.rows <= 1024:
        parser.error("rows must be in 1..1024")
    config = json.loads((args.upgrade_from / "config.json").read_text())
    matrices = rotated_w8_matrices(config, json.loads(args.matrices.read_text()))
    upgrade(
        args,
        configuration=configuration,
        signature=config_signature,
        layers=[],
        layer_prefix="model.language_model.layers.",
        matrices=matrices,
    )


if __name__ == "__main__":
    main()
