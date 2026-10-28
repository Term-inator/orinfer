"""Repository import checks exclude environments and inspect private source modules."""

from pathlib import Path
import tempfile
import unittest

from tools.build.check_imports import source_modules


class SourceModulesTests(unittest.TestCase):
    def test_nested_environments_are_not_repository_modules(self):
        files = [
            "kernels/model/_core.py",
            "tools/model/build.py",
            "tools/environments/cuda/.venv/lib/python3.12/site-packages/vendor.py",
            "tools/.cache/generated.py",
            "tools/__pycache__/temporary.py",
            "tools/model/test_build.py",
            "tools/model/__init__.py",
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("raise AssertionError('discovery does not execute source')\n")
            self.assertEqual(
                [str(path.relative_to(root)) for path in source_modules(root)], files[:2]
            )
