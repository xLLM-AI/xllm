#!/usr/bin/env python3
"""Validate absolute performance limits from saved reports, using only the CPU.

This checks performance evidence only. It neither executes an NPU operator nor
establishes that accuracy, memory, or other acceptance requirements have passed.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys


_CONFIG = {
    "schema_version": 2,
    "unit": "us",
    "metric": "api_to_completion_wall_time",
    "primary_entry": "run",
    "warmup": 50,
    "iterations": 200,
    "rounds": 5,
    "statistic": "median_of_round_means",
}


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    return value


def _index(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected a list")
    result = {}
    for entry in value:
        entry = _object(entry, label)
        name = entry.get("case_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}: missing or invalid case_name")
        if name in result:
            raise ValueError(f"{label}: duplicate case {name}")
        result[name] = entry
    return result


def _same_cases(actual, expected, label):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"{label}: missing cases {missing}; extra cases {extra}")


def _shapes(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}: expected a nonempty list of input shapes")
    for shape in value:
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"{label}: each shape must contain nonnegative integer dimensions")
    return value


def _positive_number(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label}: expected a positive finite number, not a boolean")
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label}: expected a positive finite number")
    return value


def evaluate(report, targets, public_cases):
    """Return per-case performance checks; raise ValueError for invalid evidence.

    The target and report case sets must each exactly match public performance
    cases. A limit is inclusive, applied to the recomputed, unrounded median of
    the five per-round average latencies. No per-round limit is imposed.
    """
    targets = _object(targets, "targets")
    for key, expected in _CONFIG.items():
        value = targets.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"targets.{key}: expected {expected!r}")

    public = _index(public_cases, "public_cases")
    required = {}
    for name, case in public.items():
        if type(case.get("performance")) is not bool:
            raise ValueError(f"public_cases {name}: performance must be a boolean")
        if case["performance"]:
            required[name] = case
    if not required:
        raise ValueError("public_cases: no performance cases are defined")

    configured = _index(targets.get("cases"), "targets.cases")
    _same_cases(configured, required, "targets.cases")
    for name, target in configured.items():
        threshold = target.get("max_wall_us")
        if type(threshold) is not int or threshold <= 0:
            raise ValueError(f"targets {name}: max_wall_us must be a positive integer")
        dtype = target.get("dtype")
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"targets {name}: unsupported dtype")
        shapes = _shapes(target.get("input_shapes"), f"targets {name}.input_shapes")
        inputs = required[name].get("input_desc")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"public_cases {name}: missing input_desc")
        descriptions = [_object(item, f"public_cases {name}.input_desc") for item in inputs]
        expected_shapes = _shapes([item.get("shape") for item in descriptions],
                                  f"public_cases {name}.input_desc shapes")
        if shapes != expected_shapes:
            raise ValueError(f"targets {name}: input_shapes do not match public case")
        if any(item.get("data_type") != dtype for item in descriptions):
            raise ValueError(f"targets {name}: dtype does not match every public input")

    report = _object(report, "report")
    if report.get("status") != "passed":
        raise ValueError("report.status must be 'passed' before checking performance evidence")
    if report.get("npu_executed") is not True:
        raise ValueError("report.npu_executed must be true")
    protocol = _object(report.get("protocol"), "report.protocol")
    for key in ("warmup", "iterations", "rounds"):
        if type(protocol.get(key)) is not int or protocol[key] != _CONFIG[key]:
            raise ValueError(f"report.protocol.{key}: expected integer {_CONFIG[key]}")
    measured = _index(report.get("cases"), "report.cases")
    _same_cases(measured, required, "report.cases")

    checks = []
    for name, target in configured.items():
        results = _object(measured[name].get("results"), f"report {name}.results")
        custom = _object(results.get("custom"), f"report {name}.results.custom")
        rounds = custom.get("wall_us_rounds")
        if not isinstance(rounds, list) or len(rounds) != _CONFIG["rounds"]:
            raise ValueError(f"report {name}: wall_us_rounds must contain exactly 5 values")
        values = [_positive_number(value, f"report {name}.wall_us_rounds[{index}]")
                  for index, value in enumerate(rounds)]
        cached = _positive_number(custom.get("wall_us_median"), f"report {name}.wall_us_median")
        median = statistics.median(values)
        if cached != median:
            raise ValueError(f"report {name}: cached median {cached!r} differs from recomputed median {median!r}")
        limit = target["max_wall_us"]
        checks.append({
            "case_name": name,
            "status": "passed" if median <= limit else "failed",
            "wall_us_rounds": values,
            "wall_us_median": median,
            "max_wall_us": limit,
        })
    return {
        "scope": "performance_only",
        "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
        "unit": _CONFIG["unit"],
        "metric": _CONFIG["metric"],
        "statistic": _CONFIG["statistic"],
        "checks": checks,
    }


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="CPU-only verification of saved absolute performance evidence; performance only.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=root / "performance_targets.json")
    parser.add_argument("--cases", type=Path, default=root / "case.json")
    parser.add_argument("--output", type=Path, default=Path("performance_acceptance.json"))
    args = parser.parse_args(argv)
    try:
        report, targets, public = [json.loads(path.read_text(encoding="utf-8"))
                                   for path in (args.report, args.targets, args.cases)]
        result = evaluate(report, targets, public)
    except (ValueError, OSError) as exc:
        result = {"scope": "performance_only", "status": "failed", "checks": [], "error": str(exc)}
    try:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except OSError as exc:
        print(f"Cannot save performance result: {exc}", file=sys.stderr)
        return 1
    print(f"Performance evidence: {result['status']}; result saved to {args.output}")
    if "error" in result:
        print(result["error"], file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
