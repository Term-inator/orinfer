"""Check offline module references in the CUDA uv environment without using a GPU."""

import importlib
from pathlib import Path


def source_modules(root):
    modules = []
    for directory in ("kernels", "tools"):
        for current, directories, files in (root / directory).walk():
            directories[:] = [
                name for name in directories if not name.startswith(".") and name != "__pycache__"
            ]
            modules.extend(
                current / name
                for name in files
                if name.endswith(".py") and not name.startswith("test_") and name != "__init__.py"
            )
    return sorted(modules)


def main():
    root = Path(__file__).resolve().parents[2]
    modules = source_modules(root)
    failures = []
    for path in modules:
        name = ".".join(path.relative_to(root).with_suffix("").parts)
        try:
            importlib.import_module(name)
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Imported {len(modules)} offline modules")


if __name__ == "__main__":
    main()
