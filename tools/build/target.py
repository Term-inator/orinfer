"""Fixed offline compilation target for the RTX 5070 Ti deployment."""

COMPUTE_CAPABILITY = (12, 0)
CUDA_TARGET = {"kind": "cuda", "arch": "sm_120"}
