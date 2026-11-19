# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail closed on missing, duplicate, failed, or non-device coverage evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _evaluate(directory: Any) -> Any:
    requirements = json.loads((ROOT / "reference/coverage_requirements.json").read_text(encoding="utf-8"))
    problems = []

    def _load(name: Any) -> Any:
        path = directory / name
        if not path.is_file():
            problems.append(f"missing {name}")
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "passed" or data.get("npu_executed") is not True:
            problems.append(f"{name}: not passed on device")
        return data

    def _keys(data: Any, field: Any = "records", default_entry: Any = None) -> Any:
        found = set()
        for item in data.get(field, []):
            key = (item.get("case_name"), item.get("entry", default_entry))
            if key in found:
                problems.append(f"duplicate record: {key}")
            if item.get("status", "passed") != "passed":
                problems.append(f"failed record: {key}")
            found.add(key)
        return found

    def _require(found: Any, expected: Any, label: Any) -> Any:
        for key in sorted(set(expected) - found):
            problems.append(f"{label}: missing {key}")

    public = _load("public_accuracy.json")
    _require(_keys(public, "cases", "run"), [(x, "run") for x in requirements["public_cases"]], "public")
    extra = _load("extended_accuracy.json")
    found = _keys(extra)
    expected = [(x, "run_direct") for x in requirements["public_cases"]]
    expected += [(x, e) for x in requirements["extra_input_case_ids"] for e in requirements["extra_entries"]]
    expected += [(x, e) for x in requirements["negative_scenarios"] for e in requirements["negative_entries"]]
    expected += [(x, "verifier") for x in requirements["verifier_checks"]]
    _require(found, expected, "extended")
    eng = _load("engineering.json")
    found = _keys(eng)
    expected = [(x, "prepared_stream") for x in requirements["stream_case_ids"]]
    expected += [(f"prepared_metadata_{i}", "prepare") for i in range(5)]
    expected += [
        (x, e)
        for x in ("sequential_cache_append", "output_lifetime", "input_release_cross_stream")
        for e in ("run", "run_direct")
    ]
    _require(found, expected, "engineering")
    for item in eng.get("records", []):
        if item.get("entry") == "prepared_stream" and item.get("calls", 0) < 10:
            problems.append("stream prepared calls < 10")
    mem = _load("memory.json")
    found = _keys(mem)
    _require(found, [(x, e) for x in requirements["performance_case_ids"] for e in ("run", "run_direct")], "memory")
    for item in mem.get("records", []):
        if item.get("workspace_bytes") != 0:
            problems.append("nonzero workspace")
        if item.get("peak_increment", float("inf")) > item.get("output_bytes", 0) + 16384:
            problems.append("memory peak exceeded")
        history = item.get("after_each_release", [])
        if len(history) != 20 or any(n != item.get("before") for n in history):
            problems.append("memory not reclaimed")
    performance = _load("performance.json")
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("frozen_performance_checker", ROOT / "reference/check_performance.py")
    checker = module_from_spec(spec)
    spec.loader.exec_module(checker)
    targets = json.loads((ROOT / "reference/performance_targets.json").read_text(encoding="utf-8"))
    cases = json.loads((ROOT / "reference/case.json").read_text(encoding="utf-8"))
    try:
        perf_result = checker.evaluate(performance, targets, cases)
        if perf_result["status"] != "passed":
            problems.append("fixed performance limits/protocol failed")
    except Exception as exc:
        problems.append(f"performance evidence invalid: {exc}")
    baseline = _load("baseline_comparison.json")
    _require(
        _keys(baseline, "cases", "baseline"),
        [(x, "baseline") for x in requirements["performance_case_ids"]],
        "baseline",
    )
    metadata = directory / "aclnn_invalid_metadata.log"
    if not metadata.exists():
        problems.append("missing raw ACLNN metadata checks")
    else:
        checks = [
            json.loads(line)
            for line in metadata.read_text(encoding="utf-8").splitlines()
            if line.startswith('{"metadata_scenario"')
        ]
        if {x["metadata_scenario"] for x in checks if x["status"] == "passed"} != set(range(5)):
            problems.append("incomplete raw ACLNN metadata checks")
    return {
        "status": "failed" if problems else "passed",
        "problems": problems,
        "scope": "Automated coverage only. Environment evidence, build artifacts and source fingerprints also require review.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=Path, required=True)
    args = p.parse_args()
    result = _evaluate(args.reports)
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "coverage_acceptance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
