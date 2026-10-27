"""QSA storage sharing requires complete serial, nonpersistent lifetimes."""

import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.model.flash_next.workspace import CONSUMER, PARTIAL, PRODUCER, share


class Compiler:
    def __init__(self, roles):
        self.roles = roles

    def host(self, kernel):
        return dict(
            ordered_arguments=[
                dict(ctype="ctypes.c_void_p", value=r + ".data_ptr()")
                for r in self.roles[kernel["name"]]
            ]
        )

    def pointers(self, kernel):
        return dict(zip(self.roles[kernel["name"]], kernel["args"]))


def fixture():
    buffers, kernels, roles = [], [], {}
    for prefix in ("raw", "rows"):
        names = {r: prefix + r for r in PARTIAL}
        buffers.extend(
            dict(
                name=n,
                dtype="f32",
                shape=[16],
                layout="native_contiguous",
                alignment=256,
                access="read_write",
            )
            for n in names.values()
        )
        for stage, keys in (("producer", PRODUCER), ("consumer", CONSUMER)):
            name = prefix + stage
            order = sorted(keys)
            roles[name] = order
            kernels.append(
                dict(
                    name=name,
                    symbol=stage,
                    module=dict(sha256=stage),
                    source=dict(sha256=stage),
                    host_abi=dict(sha256=stage),
                    args=[dict(kind="buffer", name=names.get(r, r)) for r in order],
                )
            )
    data = dict(
        architecture="flash_next",
        metadata=dict(buffers=copy.deepcopy(buffers)),
        buffer_scopes={b["name"]: "workspace" for b in buffers},
    )
    package = dict(kernels=kernels, buffer_contracts=copy.deepcopy(buffers))
    manifest = dict(
        programs={
            p: [dict(kind="kernel", name=prefix + s) for s in ("producer", "consumer")]
            for p, prefix in (("prefill", "raw"), ("row/private", "rows"))
        },
        prefill_plans=[dict(prefill_program="prefill", live_buffers=[])],
        row_execution=dict(programs=["row/private"], variants={}),
        graph_branches={},
    )
    return data, package, manifest, Compiler(roles)


class WorkspaceTests(unittest.TestCase):
    def test_complete_serial_pairs_share_storage_and_remove_obsolete_names(self):
        d, p, m, c = fixture()
        result = share(d, p, m, c)
        self.assertEqual(result["aliases"], {"rows" + r: "raw" + r for r in PARTIAL})
        self.assertEqual(result["saved_workspace_bytes"], 192)
        self.assertEqual(len(d["metadata"]["buffers"]), 3)
        self.assertEqual(len(p["buffer_contracts"]), 3)
        self.assertEqual(set(d["buffer_scopes"]), {"raw" + r for r in PARTIAL})

    def test_frame_and_request_state_cannot_be_shared(self):
        for mode in ("frame", "private", "mtp"):
            d, p, m, c = fixture()
            if mode == "frame":
                m["prefill_plans"][0]["live_buffers"] = ["rawOut"]
            elif mode == "private":
                d["buffer_scopes"]["rowsOut"] = "sequence"
            else:
                m["mtp"] = {}
            with self.assertRaises(ValueError):
                share(d, p, m, c)

    def test_parallel_consumers_and_additional_uses_cannot_be_shared(self):
        d, p, m, c = fixture()
        for name in ("fork", "join"):
            p["kernels"].append(dict(name=name, args=[]))
            c.roles[name] = []
        m["programs"]["prefill"] = [
            dict(kind="kernel", name="fork"),
            *m["programs"]["prefill"],
            dict(kind="kernel", name="join"),
        ]
        m["graph_branches"] = dict(prefill=[["fork", "rawproducer", "rawconsumer", "join"]])
        with self.assertRaisesRegex(ValueError, "parallel"):
            share(d, p, m, c)
        d, p, m, c = fixture()
        m["programs"]["row/private"].append(dict(kind="zero", destination="rowsOut", bytes=64))
        with self.assertRaisesRegex(ValueError, "outside"):
            share(d, p, m, c)

    def test_mismatched_geometry_and_unknown_metadata_do_not_alias(self):
        d, p, m, c = fixture()
        next(b for b in d["metadata"]["buffers"] if b["name"] == "rowsOut")["shape"] = [32]
        self.assertEqual(share(d, p, m, c)["aliases"], {})
        d, p, m, c = fixture()
        d["metadata"]["additional"] = "rowsOut"
        with self.assertRaisesRegex(ValueError, "metadata"):
            share(d, p, m, c)

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
