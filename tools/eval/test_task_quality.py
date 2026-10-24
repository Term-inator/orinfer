"""Protect task scoring against false passes and incomplete comparisons."""

import copy
import unittest

from task_quality import build_suite, check_answer, score_results


def response(ident, text, finish="stop"):
    return dict(
        custom_id=ident,
        error=None,
        response=dict(
            status_code=200,
            body=dict(
                choices=[dict(message=dict(content=text), finish_reason=finish)],
                usage=dict(prompt_tokens=10, completion_tokens=3),
            ),
        ),
    )


class TaskQualityTests(unittest.TestCase):
    def test_json_boolean_is_not_an_integer(self):
        rule = dict(kind="json_equal", expected=dict(enabled=True, count=3))
        self.assertEqual(check_answer(rule, '{"enabled":1,"count":3}')["status"], "fail")
        self.assertEqual(check_answer(rule, '{"count":3,"enabled":true}')["status"], "pass")

    def test_arithmetic_explanation_does_not_pass_integer_contract(self):
        rule = dict(kind="integer", expected=641)
        self.assertEqual(check_answer(rule, "373\n最终为641")["status"], "fail")
        self.assertEqual(check_answer(rule, "641")["status"], "pass")

    def test_translation_requires_negation_and_identifier(self):
        rule = dict(
            kind="translation_constraints",
            expected=dict(identifier="Orin-101", groups=[["不要", "不得"], ["重启"]]),
        )
        self.assertEqual(check_answer(rule, "重启 Orin-101")["status"], "fail")
        self.assertEqual(check_answer(rule, "不要重启 Orin-101")["status"], "pass")
        self.assertEqual(check_answer(rule, "不要重启 Orin-102")["status"], "fail")

    def test_missing_duplicate_and_unexpected_results_rejected(self):
        suite = dict(
            seed=1,
            scope="test",
            cases=[
                dict(id="a", category="extraction", validation=dict(kind="exact", expected="OK")),
                dict(id="b", category="extraction", validation=dict(kind="exact", expected="OK")),
            ],
        )
        for ids in [["a"], ["a", "a"], ["a", "c"]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                score_results(suite, [response(i, "OK") for i in ids])

    def test_truncation_and_runtime_error_never_count_as_pass(self):
        suite = dict(
            seed=1,
            scope="test",
            cases=[
                dict(id="a", category="extraction", validation=dict(kind="exact", expected="OK"))
            ],
        )
        scored = score_results(suite, [response("a", "OK", "length")])
        self.assertEqual(scored["strict_tasks"]["passed"], 0)
        self.assertEqual(scored["strict_tasks"]["truncated"], 1)
        row = response("a", "OK")
        row["error"] = {"message": "GPU failure"}
        self.assertEqual(score_results(suite, [row])["aggregate"]["runtime_errors"], 1)

    def test_translation_constraints_are_separate_from_strict_accuracy(self):
        suite = dict(
            seed=1,
            scope="test",
            cases=[
                dict(id="a", category="extraction", validation=dict(kind="exact", expected="OK")),
                dict(
                    id="b",
                    category="translation",
                    validation=dict(
                        kind="translation_constraints",
                        expected=dict(identifier="X", groups=[["缓存"]]),
                    ),
                ),
            ],
        )
        result = score_results(suite, [response("a", "OK"), response("b", "缓存X")])
        self.assertEqual(result["strict_tasks"]["total"], 1)
        self.assertEqual(result["translation_constraints"]["total"], 1)

    def test_suite_is_frozen_and_balanced(self):
        a = build_suite()
        b = build_suite()
        self.assertEqual(a, b)
        self.assertEqual(len(a["cases"]), 200)
        changed = copy.deepcopy(a)
        changed["cases"][0]["id"] = "changed"
        self.assertNotEqual(a, changed)


if __name__ == "__main__":
    unittest.main()
