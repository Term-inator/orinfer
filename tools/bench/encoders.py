"""Measure complete embedding/rerank HTTP requests on a frozen shared workload."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request


def request(task, backend, model, body):
    body = dict(body, model=model)
    if task == "rerank":
        return ("/rerank" if backend == "vllm" else "/v1/rerank"), body
    if backend == "orinfer":
        return "/v1/embeddings", body
    inputs = []
    for item in body.pop("input"):
        item = {"text": item} if isinstance(item, str) else item
        content = [
            {"type": "image_url", "image_url": {"url": image}} for image in item.get("images", [])
        ]
        if "text" in item:
            content.append({"type": "text", "text": item["text"]})
        inputs.append({"content": content})
    # Keep complete inputs; neither backend may truncate this workload.
    return "/v2/embed", dict(body, inputs=inputs, embedding_types=["float"], truncate="NONE")


def outputs(task, backend, response, size):
    if task == "rerank":
        results = response["results"]
        if sorted(r["index"] for r in results) != list(range(size)):
            raise ValueError("Reranker must return every document exactly once")
        scores = [r["relevance_score"] for r in sorted(results, key=lambda r: r["index"])]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
            raise ValueError("Invalid relevance probability")
        return scores
    if backend == "vllm":
        vectors = response["embeddings"]["float"]
    else:
        data = response["data"]
        if sorted(d["index"] for d in data) != list(range(size)):
            raise ValueError("Embedding input/output indices changed")
        vectors = [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
    if len(vectors) != size or any(len(v) != 2048 for v in vectors):
        raise ValueError("Expected one full 2048-dimensional vector per input")
    for vector in vectors:
        norm = sum(x * x for x in vector)
        if not math.isfinite(norm) or abs(norm - 1) > 0.01:
            raise ValueError("Embedding must be finite and L2-normalized")
    return vectors


def prompt_tokens(response):
    count = response.get("usage", {}).get("prompt_tokens")
    if count is None:
        count = response.get("meta", {}).get("billed_units", {}).get("input_tokens")
    if type(count) is not int or count < 1:
        raise ValueError("Encoder benchmark requires actual input token usage")
    return count


def measure(args):
    raw = args.requests.read_bytes()
    cases = json.loads(raw)["cases"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Workload requires nonempty, unique case IDs")
    if args.warmups < 1 or args.repetitions < 3:
        raise ValueError("Use at least one warmup and three measured repetitions")
    report = dict(
        task=args.task,
        backend=args.backend,
        requests_sha256=hashlib.sha256(raw).hexdigest(),
        identity=json.loads(args.identity.read_text()),
        warmups=args.warmups,
        repetitions=args.repetitions,
        cases=[],
    )
    for case in cases:
        path, body = request(args.task, args.backend, args.model, case["request"])
        size = len(case["request"]["documents" if args.task == "rerank" else "input"])
        if not 1 <= size <= 128 or "top_n" in body:
            raise ValueError("Use full batches of 1..128 items without top_n")
        wire = json.dumps(body, ensure_ascii=False).encode()
        durations = []
        tokens = None
        for iteration in range(args.warmups + args.repetitions):
            started = time.perf_counter()
            req = urllib.request.Request(
                args.url.rstrip("/") + path,
                data=wire,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                payload = response.read()
            # Include JSON decoding and complete output validation in latency.
            response = json.loads(payload)
            values = outputs(args.task, args.backend, response, size)
            actual_tokens = prompt_tokens(response)
            if tokens is not None and actual_tokens != tokens:
                raise ValueError("Repeated encoder requests changed input token count")
            tokens = actual_tokens
            elapsed = time.perf_counter() - started
            if iteration >= args.warmups:
                durations.append(elapsed)
        median = statistics.median(durations)
        report["cases"].append(
            dict(
                id=case["id"],
                items=size,
                prompt_tokens=tokens,
                durations_s=durations,
                median_s=median,
                items_per_s=size / median,
                outputs=values,
                response=response,
            )
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(case["id"], f"{size / median:.3f} items/s, {median * 1000:.3f} ms", flush=True)


def compare(baseline, candidate):
    if (
        baseline["task"] != candidate["task"]
        or baseline["requests_sha256"] != candidate["requests_sha256"]
    ):
        raise ValueError("Compare the same task and exact frozen workload")
    base = {c["id"]: c for c in baseline["cases"]}
    other = {c["id"]: c for c in candidate["cases"]}
    if base.keys() != other.keys():
        raise ValueError("Benchmark case coverage differs")
    comparisons = []
    for name, before in base.items():
        after = other[name]
        if before["items"] != after["items"]:
            raise ValueError("Benchmark batch size changed")
        before_tokens = prompt_tokens(before.get("response", {}))
        after_tokens = prompt_tokens(after.get("response", {}))
        if before_tokens != after_tokens:
            raise ValueError(
                f"Input token count differs for {name}: {before_tokens} vs {after_tokens}; "
                "align templates and truncation before comparing"
            )
        speedup = before["median_s"] / after["median_s"]
        quality = {}
        if baseline["task"] == "embedding":
            quality["min_cosine"] = min(
                sum(x * y for x, y in zip(a, b, strict=True))
                / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
                for a, b in zip(before["outputs"], after["outputs"], strict=True)
            )
        else:
            a, b = before["outputs"], after["outputs"]
            quality["top1_equal"] = max(range(len(a)), key=a.__getitem__) == max(
                range(len(b)), key=b.__getitem__
            )
            quality["mean_probability_error"] = statistics.mean(
                abs(x - y) for x, y in zip(a, b, strict=True)
            )
        comparisons.append(
            dict(
                id=name,
                prompt_tokens=before_tokens,
                speedup=speedup,
                reaches_2x=speedup >= 2,
                quality=quality,
            )
        )
    return dict(
        task=baseline["task"],
        baseline_identity=baseline["identity"],
        candidate_identity=candidate["identity"],
        requests_sha256=baseline["requests_sha256"],
        cases=comparisons,
        all_cases_reach_2x=all(c["reaches_2x"] for c in comparisons),
        geometric_mean_speedup=math.exp(
            statistics.mean(math.log(c["speedup"]) for c in comparisons)
        ),
        suite_speedup=sum(c["median_s"] for c in base.values())
        / sum(c["median_s"] for c in other.values()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare", nargs=2, type=Path)
    p.add_argument("--task", choices=["embedding", "rerank"])
    p.add_argument("--backend", choices=["vllm", "orinfer"])
    p.add_argument("--url")
    p.add_argument("--model")
    p.add_argument("--requests", type=Path)
    p.add_argument("--identity", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--timeout", type=float, default=300)
    args = p.parse_args()
    if args.compare:
        result = compare(*(json.loads(path.read_text()) for path in args.compare))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        if any(
            getattr(args, name) is None
            for name in ["task", "backend", "url", "model", "requests", "identity"]
        ):
            p.error("Measurement requires task, backend, URL, model, requests and identity")
        measure(args)


if __name__ == "__main__":
    main()
