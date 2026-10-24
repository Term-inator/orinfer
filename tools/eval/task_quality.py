"""Deterministic task screening for native Chat JSONL model comparisons.

This is a synthetic functional suite, not a benchmark of business traffic or an
absolute quantization reference. Translation checks cover stated constraints;
semantic quality requires a separate review of the retained responses.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from tools.eval.scoring_common import SEED, validate_task


TRANSLATIONS = [
    (
        "Do not restart service {name} before all requests finish.",
        [["不要", "不得", "请勿", "禁止", "不应"], ["重启", "重新启动"], ["请求"]],
    ),
    (
        "The cache for {name} is disabled by default.",
        [["缓存"], ["默认"], ["禁用", "关闭", "不开启", "未启用", "不启用"]],
    ),
    (
        "{name} saves the output only after validation succeeds.",
        [["验证", "校验"], ["成功", "通过"], ["保存"], ["后", "才"]],
    ),
    (
        "Delete temporary files for {name}, but keep the original model.",
        [["删除", "清理"], ["临时"], ["保留", "保存"], ["原始", "原模型"]],
    ),
    (
        "If {name} fails, return an error instead of an empty result.",
        [["失败"], ["错误"], ["空"], ["而不是", "而非", "不要", "不得", "不应"]],
    ),
    (
        "{name} supports text input but does not support audio input.",
        [["文本", "文字"], ["音频", "语音"], ["不支持"]],
    ),
    (
        "Keep the request identifier {name} unchanged.",
        [["请求"], ["标识", "编号", "ID", "id"], ["不变", "不修改", "不要更改", "不做更改"]],
    ),
    (
        "Requests for {name} are processed in arrival order.",
        [["请求"], ["到达", "抵达", "先后"], ["顺序"]],
    ),
    (
        "Load {name} once and reuse it for subsequent requests.",
        [["加载"], ["一次"], ["复用", "重复使用", "重用"], ["后续", "随后"]],
    ),
    (
        "Wait for {name} to finish before releasing its memory.",
        [["等待", "等到", "待"], ["完成", "结束"], ["释放"], ["内存"]],
    ),
]


def build_suite():
    rng = random.Random(SEED)
    cases = []

    def add(category, index, prompt, expected, kind="exact", budget=64):
        cases.append(
            dict(
                id=f"{category}-{index:02d}",
                category=category,
                messages=[dict(role="user", content=prompt)],
                max_tokens=budget,
                validation=dict(kind=kind, expected=expected),
            )
        )

    for i in range(20):
        a, b, c, e = [rng.randrange(11, 60) for _ in range(4)]
        add("arithmetic", i, f"计算 {a}×{b}−{c}×{e}，只输出整数答案。", a * b - c * e, "integer")
        bought, sold, returned = rng.randrange(30, 80), rng.randrange(3, 20), rng.randrange(1, 8)
        add(
            "word_problem",
            i,
            f"仓库原有{bought}件商品，售出{sold}件，随后有{returned}件售出的商品退回仓库。现在有多少件？只输出整数。",
            bought - sold + returned,
            "integer",
        )
        name = f"JOB-{rng.randrange(1000, 9999)}"
        enabled = i % 2 == 0
        add(
            "json",
            i,
            f'只输出合法 JSON 对象，恰好包含三个键：id 为字符串 "{name}"，count 为整数 {i + 2}，enabled 为布尔值 {str(enabled).lower()}。不要代码围栏或解释。',
            dict(id=name, count=i + 2, enabled=enabled),
            "json_equal",
        )
        first, last = rng.sample(["RED", "BLUE", "GREEN", "WHITE", "BLACK", "YELLOW"], 2)
        add(
            "instruction",
            i,
            f"旧要求：输出{first}。最新更正：忽略旧要求，只输出{last}，不加标点。",
            last,
        )
        city = rng.choice(["北京", "上海", "广州", "深圳", "杭州", "成都"])
        if i % 4 == 0:
            add(
                "extraction",
                i,
                f"只依据资料回答。资料：订单{name}，城市{city}，数量{i + 2}。问：联系人姓名是什么？未提供时只输出UNKNOWN。",
                "UNKNOWN",
            )
        else:
            add(
                "extraction",
                i,
                f"只依据资料回答。资料：旧订单OLD-{i}的城市是北京。当前订单{name}，城市{city}，数量{i + 2}。问：当前订单的城市？只输出城市。",
                city,
            )
        adult, paid, banned = (
            (True, True, False)
            if i % 2 == 0
            else [(False, True, False), (True, False, False), (True, True, True)][(i // 2) % 3]
        )

        def yes(value):
            return "是" if value else "否"

        add(
            "logic",
            i,
            f"规则：仅当成年且已付款且没有封禁时允许访问。记录：成年={yes(adult)}，已付款={yes(paid)}，封禁={yes(banned)}。允许访问只输出ALLOW，否则只输出DENY。",
            "ALLOW" if adult and paid and not banned else "DENY",
        )
        xs = [rng.randrange(-8, 15) for _ in range(8)]
        if i % 2:
            prompt = f"将列表{xs}升序排序，保留重复项。只输出 JSON 数组。"
            expected = sorted(xs)
        else:
            prompt = f"列表{xs}按首次出现顺序去重。只输出 JSON 数组。"
            expected = list(dict.fromkeys(xs))
        add("list_transform", i, prompt, expected, "json_equal", budget=96)
        offset = rng.randrange(1, 8)
        expected = sum(x + offset for x in xs if x % 2 == 0)
        add(
            "code_reading",
            i,
            f"下列 Python 代码输出哪个整数？只输出整数，不要解释。\nxs = {xs}\ns = 0\nfor x in xs:\n    if x % 2 == 0:\n        s += x + {offset}\nprint(s)",
            expected,
            "integer",
        )
        source, groups = TRANSLATIONS[i % len(TRANSLATIONS)]
        label = f"Orin-{100 + i}"
        add(
            "translation",
            i,
            "将以下英文译为中文，只输出译文。保留标识符原样，不添加说明：\n"
            + source.format(name=label),
            dict(identifier=label, groups=groups),
            "translation_constraints",
            budget=128,
        )
        count = [24, 64, 120, 160][i % 4]
        target = rng.randrange(count // 4, count * 3 // 4)
        records = [
            f"条目{j:03d}：代号R{j:03d}，密钥{rng.randrange(100000, 999999)}。"
            for j in range(count)
        ]
        expected = f"KEY-{rng.randrange(100000, 999999)}"
        records[target] = f"条目{target:03d}：代号TARGET-{i:02d}，密钥{expected}。"
        add(
            "long_retrieval",
            i,
            f"资料如下。请查找代号TARGET-{i:02d}对应的密钥，只输出密钥，忽略其余记录。\n"
            + "\n".join(records),
            expected,
        )
    assert len(cases) == 200 and len({c["id"] for c in cases}) == 200
    return dict(
        schema=1,
        seed=SEED,
        scope="Synthetic text functional screening; no BF16 free-generation reference; translation constraints are not semantic grading",
        cases=cases,
    )


def check_answer(rule, text):
    if rule["kind"] != "translation_constraints":
        return validate_task(rule, text)
    expected = rule["expected"]
    hits = [any(word in text for word in words) for words in expected["groups"]]
    passed = expected["identifier"] in text and all(hits)
    return dict(
        status="pass" if passed else "fail",
        kind="translation_constraints",
        identifier_preserved=expected["identifier"] in text,
        group_hits=hits,
        semantic_review_required=True,
    )


def score_results(suite, rows):
    cases = {c["id"]: c for c in suite["cases"]}
    ids = [r["custom_id"] for r in rows]
    if len(ids) != len(cases) or set(ids) != set(cases):
        raise ValueError("Missing, duplicate or unexpected response IDs")
    scored = []
    for row in rows:
        case = cases[row["custom_id"]]
        if row.get("error") or row.get("response", {}).get("status_code") != 200:
            scored.append(
                dict(
                    id=case["id"],
                    category=case["category"],
                    passed=False,
                    runtime_error=row.get("error") or row.get("response"),
                )
            )
            continue
        body = row["response"]["body"]
        choice = body["choices"][0]
        text = choice["message"].get("content") or ""
        check = check_answer(case["validation"], text)
        finish = choice["finish_reason"]
        passed = check["status"] == "pass" and finish == "stop"
        # Repeated substrings are a diagnostic, not a separate task rejection.
        repetition = bool(re.search(r"(.{12,})\1\1", text))
        scored.append(
            dict(
                id=case["id"],
                category=case["category"],
                passed=passed,
                check=check,
                finish_reason=finish,
                truncated=finish == "length",
                repetition_flag=repetition,
                completion_tokens=body["usage"]["completion_tokens"],
                prompt_tokens=body["usage"]["prompt_tokens"],
                answer=text,
                expected=case["validation"]["expected"],
            )
        )
    by_category = defaultdict(list)
    for row in scored:
        by_category[row["category"]].append(row)

    def summary(values):
        return dict(
            total=len(values),
            passed=sum(x["passed"] for x in values),
            truncated=sum(x.get("truncated", False) for x in values),
            runtime_errors=sum("runtime_error" in x for x in values),
            repetition_flags=sum(x.get("repetition_flag", False) for x in values),
            completion_tokens=sum(x.get("completion_tokens", 0) for x in values),
        )

    strict = [x for x in scored if x["category"] != "translation"]
    return dict(
        seed=suite["seed"],
        scope=suite["scope"],
        aggregate=summary(scored),
        strict_tasks=summary(strict),
        translation_constraints=summary(by_category["translation"]),
        by_category={k: summary(v) for k, v in sorted(by_category.items())},
        cases=scored,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--suite", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        args.output.mkdir(parents=True, exist_ok=False)
        suite = build_suite()
        (args.output / "suite.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2))
        requests = [
            dict(
                custom_id=c["id"],
                messages=c["messages"],
                max_tokens=c["max_tokens"],
                temperature=0,
                seed=SEED,
                enable_thinking=False,
            )
            for c in suite["cases"]
        ]
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests)
        (args.output / "requests.jsonl").write_text(payload)
        (args.output / "dataset-identity.json").write_text(
            json.dumps(
                dict(
                    seed=SEED,
                    count=len(requests),
                    categories=dict(Counter(c["category"] for c in suite["cases"])),
                    requests_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                ),
                indent=2,
            )
        )
    else:
        suite = json.loads(args.suite.read_text())
        rows = [json.loads(line) for line in args.results.read_text().splitlines()]
        result = score_results(suite, rows)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result["aggregate"]))


if __name__ == "__main__":
    main()
