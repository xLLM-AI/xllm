"""Paired timing: warm up 50, alternate 6 rounds, 200 iterations per round.

Protocol matches the task book: both sides are measured on the same inputs and
device, each round ends with a device synchronize, and the 95% interval is a
percentile bootstrap (10000 resamples, seed 20261031).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

WARMUP = 50
ROUNDS = 6
ITERATIONS = 200
CASE_IDS = ["packed_perf_small", "packed_perf_large", "packed_route_32736", "packed_route_32768"]
TARGETS = {"packed_perf_small": 40.0, "packed_perf_large": 75.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=pathlib.Path, default=pathlib.Path("/mnt/workspace/pkg"))
    parser.add_argument(
        "--implementation",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "evidence" / "result.json",
    )
    return parser.parse_args()


def bootstrap_ci(values: list[float], seed: int = 20261031, resamples: int = 10000):
    generator = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [values[generator.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.package))
    sys.path.insert(0, str(args.implementation))

    import generate_data
    import golden
    import torch
    import torch_npu

    from solution import prepare

    contract = json.loads((args.package / "case.json").read_text())["input_cases"]
    cases = {case["id"]: case for case in contract}

    def build(case):
        values, attrs = generate_data.generate(case)
        inputs = [torch.tensor(values[index], dtype=torch.int32).npu() for index in range(3)]
        return inputs, attrs

    def measure(fn) -> float:
        started = time.perf_counter()
        for _ in range(ITERATIONS):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - started) / ITERATIONS * 1e6

    report = {
        "benchmarks": [],
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "triton": __import__("triton").__version__,
        },
    }
    for case_id in CASE_IDS:
        inputs, attrs = build(cases[case_id])
        prepared = prepare(inputs, attrs)
        baseline = golden.reference("baseline").prepare(inputs, attrs, prepared.info)
        candidate = prepared.run

        for _ in range(WARMUP):
            baseline()
            candidate()
        torch.npu.synchronize()

        baseline_rounds = []
        candidate_rounds = []
        for index in range(ROUNDS):
            if index % 2 == 0:
                baseline_rounds.append(measure(baseline))
                candidate_rounds.append(measure(candidate))
            else:
                candidate_rounds.append(measure(candidate))
                baseline_rounds.append(measure(baseline))

        diffs = [b - c for b, c in zip(baseline_rounds, candidate_rounds)]
        low, high = bootstrap_ci(diffs)
        entry = {
            "id": case_id,
            "baseline_rounds_us": baseline_rounds,
            "candidate_rounds_us": candidate_rounds,
            "baseline_mean_us": statistics.fmean(baseline_rounds),
            "candidate_mean_us": statistics.fmean(candidate_rounds),
            "baseline_median_us": statistics.median(baseline_rounds),
            "candidate_median_us": statistics.median(candidate_rounds),
            "saved_mean_us": statistics.fmean(diffs),
            "saved_median_us": statistics.median(diffs),
            "saved_mean_ci95_us": [low, high],
        }
        report["benchmarks"].append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    checks = {}
    for entry in report["benchmarks"]:
        target = TARGETS.get(entry["id"])
        if target is None:
            continue
        checks[entry["id"]] = {
            "target_us": target,
            "mean_ok": entry["candidate_mean_us"] <= target,
            "mean_saved_positive": entry["saved_mean_us"] > 0,
            "median_saved_positive": entry["saved_median_us"] > 0,
            "ci_lower_positive": entry["saved_mean_ci95_us"][0] > 0,
        }
    report["acceptance"] = checks
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(checks, ensure_ascii=False))
    print("完成", len(report["benchmarks"]), "个用例")


if __name__ == "__main__":
    main()
