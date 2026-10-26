"""Primitive extraction preserves nested factory policy and scoped state."""

from types import SimpleNamespace
import unittest

from tools.model.compile import construct, construction_call


class ConstructionTests(unittest.TestCase):
    def test_nested_factories_bind_dynamic_positional_arguments_and_keep_passes(self):
        calls = []

        def raw(rows, dynamic_rows=False):
            calls.append((rows, dynamic_rows))
            return SimpleNamespace(buffer_map={})

        def inner(rows, dynamic_rows=False):
            return construction_call(raw, (rows, dynamic_rows), {}, {"inner": True})

        def outer():
            return construction_call(inner, (4096, False), {}, {"outer": True})

        function, passes = construct(outer, dynamic=True)
        self.assertEqual(function.buffer_map, {})
        self.assertEqual(calls, [(4096, True)])
        self.assertEqual(passes, dict(inner=True, outer=True))
        self.assertIsNone(construction_call(raw, (1,), {}, {}))
        self.assertEqual(calls, [(4096, True)])

    def test_exception_restores_normal_compilation_and_fixed_policy(self):
        def fail():
            raise RuntimeError("invalid factory")

        with self.assertRaisesRegex(RuntimeError, "invalid factory"):
            construct(fail, dynamic=True)
        self.assertIsNone(construction_call(fail, (), {}, {}))

        def raw(dynamic_rows=False):
            self.assertFalse(dynamic_rows)
            return SimpleNamespace(buffer_map={})

        construct(lambda: construction_call(raw, (), {}, {}))
        with self.assertRaisesRegex(ValueError, "PrimFunc"):
            construct(lambda: None)
        with self.assertRaisesRegex(ValueError, "policy"):
            construct(lambda: None, dynamic=1)


if __name__ == "__main__":
    unittest.main()
