"""Fixed deployment target shared with the Rust model SDK."""

import json
from pathlib import Path

TARGET = json.loads(
    (Path(__file__).resolve().parents[2] / "configs/deployment-target.json").read_text()
)
COMPUTE_CAPABILITY = tuple(TARGET["compute_capability"])
CUDA_TARGET = {"kind": "cuda", "arch": TARGET["cuda_arch"]}
ELF_MACHINE = TARGET["elf_machine"]

# The native launch ABI carries raw pointers and scalar dimensions.
COMPILER_PASSES = {
    "tl.disable_tma_lower": True,
    "tl.disable_warp_specialized": True,
    "tl.disable_shared_memory_reuse": True,
    "tl.disable_vectorize_256": True,
}
