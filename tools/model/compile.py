"""Shared native AOT compilation with explicit runtime row scalars."""

import inspect
from contextvars import ContextVar

from tools.build.target import COMPILER_PASSES, CUDA_TARGET


_CONSTRUCTION = ContextVar("orinfer_primitive_construction", default=None)


def construction_call(function, args, kwargs, passes):
    """Instantiate nested JIT factories during shape-only AOT construction."""
    active = _CONSTRUCTION.get()
    if active is None:
        return None
    function = inspect.unwrap(getattr(function, "func", function))
    dynamic, records = active
    arguments = inspect.signature(function).bind(*args, **kwargs)
    parameters = inspect.signature(function).parameters
    if dynamic:
        for name in ("dynamic_rows", "dynamic_tokens"):
            if name in parameters:
                arguments.arguments[name] = True
    result = function(*arguments.args, **arguments.kwargs)
    records.append(dict(passes))
    return result


def construct(factory, *, dynamic=False):
    """Obtain a PrimFunc and its compiler passes without creating a CUDA module."""
    if type(dynamic) is not bool:
        raise ValueError("Invalid dynamic construction policy")
    records = []
    token = _CONSTRUCTION.set((dynamic, records))
    try:
        function = factory()
    finally:
        _CONSTRUCTION.reset(token)
    if not records or not hasattr(function, "buffer_map"):
        raise ValueError("Construction factory must return a TileLang PrimFunc")
    passes = {key: value for record in records for key, value in record.items()}
    return function, passes


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
