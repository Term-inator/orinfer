"""Shared row bindings preserve planner storage and distinct row interfaces."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class WorkspaceTests(unittest.TestCase):
    def test_shared_row_constructor_preserves_storage_but_keeps_interfaces_distinct(self):
        from tools.model.flash_next.rows import Rows

        class Tensor:
            dtype = "f32"

            def __init__(self, identity):
                self.storage = SimpleNamespace(_cdata=identity, nbytes=lambda: 64)

            def untyped_storage(self):
                return self.storage

            def element_size(self):
                return 4

        tensor, token = Tensor(1), Tensor(2)
        plan = dict(
            ops=[(None, [tensor])], labels=["work"], body_count=1, token=token, greedy_ops=[]
        )
        for interface in (False, True):
            b = SimpleNamespace(
                names={1: "OriginalScratch"},
                scopes={"OriginalScratch": "workspace"},
                buffers={"OriginalScratch": {}},
                keep=[],
                kernels=[],
            )
            b.fixed = lambda name, *args, **kwargs: (
                b.scopes.setdefault(name, "workspace"),
                b.buffers.setdefault(name, {}),
            )[1]
            b.bind = lambda *args, **kwargs: None
            b.buffer = lambda tensor, name: b.names.setdefault(tensor.storage._cdata, name)
            rows = object.__new__(Rows)
            rows.b, rows.p = b, SimpleNamespace(trace=lambda *args, **kwargs: plan)
            rows.strides = {"Residual": 16}
            rows.prepare = lambda plan: None
            rows.aliases = lambda *args: {1: "Residual"} if interface else {}
            with patch.dict(
                sys.modules,
                {"tools.model.flash_next.bindings": SimpleNamespace(DTYPES={"f32": "f32"})},
            ):
                rows.shared()
            self.assertEqual(b.names[1], "FlashDecode_Residual" if interface else "OriginalScratch")


if __name__ == "__main__":
    unittest.main()
