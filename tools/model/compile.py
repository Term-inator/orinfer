"""Shared native AOT compilation with explicit runtime row scalars."""

import inspect

from tools.build.target import COMPILER_PASSES, CUDA_TARGET


def instantiate(function, *args, **kwargs):
    return inspect.unwrap(getattr(function, "func", function))(*args, **kwargs)


def compile_function(function, *, pass_configs=None, **options):
    import tilelang
    from tools.operators.common import validate_kernel_resources

    kernel = tilelang.compile(
        function,
        out_idx=[],
        execution_backend="nvrtc",
        target=CUDA_TARGET,
        pass_configs={**COMPILER_PASSES, **(pass_configs or {})},
        **options,
    )
    return validate_kernel_resources(kernel)


def primitive(function, *args, compiler_options=None, **kwargs):
    from kernels.model.rows import explicit_rows

    fn = instantiate(function, *args, **kwargs)
    if kwargs.get("dynamic_rows"):
        fn = explicit_rows(fn)
    passes = (
        {
            "tl.disable_thread_storage_sync": True,
            "tl.disable_shared_memory_reuse": True,
        }
        if kwargs.get("prefetch_qk")
        else {}
    )
    passes.update(getattr(function, "pass_configs", None) or {})
    passes.update(compiler_options or {})
    return compile_function(fn, pass_configs=passes)
