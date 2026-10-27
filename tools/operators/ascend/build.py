"""Offline A2 AOT compilation with a checked, pointer-table C ABI for Rust."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import threading

import tilelang
from tilelang.jit.adapter.libgen import LibraryGenerator

from tools.build.files import atomic_json, file_hash, source_hash

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [p for p in Path(__file__).parent.glob("*.py") if p.name != "validate.py"]
LOWER_LOCK = threading.RLock()


def lower_a2(function, configs):
    # The pinned compiler probes torch.npu.is_available() when choosing the
    # storage rewrite. In a device-free compiler container that enables an
    # unsafe generic reuse pass, despite the explicit Ascend target. Select the
    # NPU rewrite directly; GPU visibility must not alter the generated program.
    with LOWER_LOCK:
        rewrite = tilelang.transform.AscendStorageRewrite
        tilelang.transform.AscendStorageRewrite = lambda **_: rewrite(is_npu=True)
        try:
            with tilelang.tvm.transform.PassContext(opt_level=3, config=configs):
                return tilelang.engine.lower(function, target="ascendc", platform="A2")
        finally:
            tilelang.transform.AscendStorageRewrite = rewrite


def compile_kernel(function, name, operands, parameters, cache):
    """Static tensors only: reject implicit compiler workspaces/scalar arguments."""
    compiler = {
        "tilelang": tilelang.__version__,
        "bisheng": subprocess.check_output(["bisheng", "--version"], text=True).splitlines()[0],
        "cann_home": os.environ["ASCEND_HOME_PATH"],
        "storage_rewrite": "explicit-npu-v1",
    }
    module = Path(__file__).with_name(name.split(".")[0] + ".py")
    # Production names identify the source module. Include its inlined math
    # and the shared compiler/file helpers; unrelated operators have independent
    # cache identities. Legacy validation names retain the conservative closure.
    sources = (
        [module, Path(__file__), Path(__file__).with_name("vector_math.py")]
        if "." in name and module in SOURCES
        else SOURCES
    )
    identity = {
        "source_sha256": source_hash(ROOT, sources + [ROOT / "tools/build/files.py"]),
        "compiler": compiler,
        "target": {
            "host_arch": platform.machine(),
            "backend": "ascend",
            "device_arch": "ascend910b",
        },
        "op": name,
        "operands": operands,
        "parameters": parameters,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / key
    if output.exists():
        manifest = json.loads((output / "kernel.json").read_text())
        if any(manifest[k] != v for k, v in identity.items()):
            raise ValueError("AOT cache identity differs")
        if file_hash(output / manifest["library"]["file"]) != manifest["library"]["sha256"]:
            raise ValueError("AOT cache library hash differs")
        return output / "kernel.json"
    configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }
    artifact = lower_a2(function, configs)
    source = artifact.kernel_source
    prototype = re.search(r'extern "C" void call\(([^)]*)\)', source)
    if prototype is None:
        raise ValueError("TileLang did not emit an AOT call wrapper")
    actual = [p.strip() for p in prototype[1].split(",")]
    expected = [f"uint8_t* {p['name']}_handle" for p in operands] + ["aclrtStream stream"]
    if actual != expected:
        raise ValueError(f"Generated ABI differs: {actual!r} != {expected!r}")
    count = len(operands)
    arguments = ", ".join(f"static_cast<uint8_t*>(args[{i}])" for i in range(count))
    source += f"""
extern "C" const char* orinfer_ascend_kernel_spec_v1() {{
    return {json.dumps(json.dumps(identity, sort_keys=True))};
}}
extern "C" int32_t orinfer_ascend_kernel_v1(void* const* args, uint64_t count, aclrtStream stream) {{
    if (!args || count != {count} || !stream) return ACL_ERROR_INVALID_PARAM;
    for (uint64_t i = 0; i < count; ++i) if (!args[i]) return ACL_ERROR_INVALID_PARAM;
    call({arguments}, stream);
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}}
"""
    generator = LibraryGenerator(target="ascendc", platform="A2")
    generator.update_lib_code(source)
    generator.compile_lib(timeout=120)
    with tempfile.TemporaryDirectory(prefix="aot-", dir=cache) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        shutil.copyfile(generator.get_lib_path(), stage / "kernel.so")
        (stage / "kernel.cpp").write_text(source)
        manifest = {
            "schema_version": 1,
            **identity,
            "library": {"file": "kernel.so", "sha256": file_hash(stage / "kernel.so")},
        }
        atomic_json(stage / "kernel.json", manifest)
        stage.rename(output)
    return output / "kernel.json"


def operand(name, dtype, shape, access="read"):
    return {"name": name, "dtype": dtype, "shape": list(shape), "access": access}
