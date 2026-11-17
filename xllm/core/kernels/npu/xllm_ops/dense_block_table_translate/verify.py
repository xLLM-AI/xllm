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

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_hashes() -> dict[str, str]:
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.suffix in (".py", ".cpp", ".json", ".md", ".txt")
        and ("__pycache__" not in p.parts)
        and ("task_submission" not in p.parts)
    }


def make_inputs(case: dict[str, Any], values: list[Any], torch: ModuleType) -> list[Any]:
    tensors = []
    for i, value in enumerate(values):
        tensor = torch.tensor(value, dtype=torch.int32)
        if i == 0 and "requests" in case:
            tensor = tensor.reshape(case["requests"], case["columns"])
        if case.get("offset"):
            storage = torch.full((tensor.numel() + 17,), -123456, dtype=torch.int32, device="npu:0")
            view = storage[1 : tensor.numel() + 1].reshape(tensor.shape)
            view.copy_(tensor)
            tensor = view
        else:
            tensor = tensor.to("npu:0")
        tensors.append(tensor)
    return tensors


def check_coverage(functional: dict[str, Any], contract: dict[str, Any]) -> None:
    expected = {case["id"] for case in contract["input_cases"]}
    observed = [case["id"] for case in functional["cases"]]
    if functional["status"] != "PASS" or len(observed) != len(expected) or set(observed) != expected:
        raise AssertionError("official functional coverage incomplete")
    if any(c["status"] != "PASS" for c in functional["cases"]):
        raise AssertionError("failed functional case")
    required = contract["expected_checks"]["result.json"]
    negative = functional["negative"]
    if len(negative) != len(required["negative"]) or {r["id"] for r in negative} != {
        r["id"] for r in required["negative"]
    }:
        raise AssertionError("illegal input coverage incomplete")
    if any(r["status"] != "PASS" or r["exception"] not in ("TypeError", "ValueError") for r in negative):
        raise AssertionError("illegal input check failed")
    for case_id, checks in required["dynamic_updates"].items():
        actual = functional["dynamic_updates"][case_id]
        if {r["id"] for r in actual} != {r["id"] for r in checks}:
            raise AssertionError("dynamic input coverage incomplete: " + case_id)
        if any(r["status"] != "PASS" or r["candidate_mismatches"] != 0 for r in actual):
            raise AssertionError("dynamic input failure: " + case_id)
    for case_id in required["engineering_by_case"]:
        record = functional["engineering_by_case"][case_id]
        for key in ("two_streams", "retained_outputs_independent", "repeat20_released"):
            if record[key] != "PASS":
                raise AssertionError("engineering failure: " + case_id + "/" + key)


def _stream_oracle_checks(
    prepared: Any, baseline: Callable[[], Any], expected: Any, torch: ModuleType
) -> dict[str, Any]:
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    for stream in streams:
        stream.wait_stream(torch.npu.current_stream())
    retained = []
    for stream in streams:
        with torch.npu.stream(stream):
            retained.extend(prepared.run() for _ in range(3))
            retained.extend(baseline() for _ in range(3))
    torch.npu.synchronize()
    ptrs = [x.untyped_storage().data_ptr() for x in retained]
    assert len(set(ptrs)) == len(ptrs)
    assert all(torch.equal(x.cpu(), expected) for x in retained)
    for x in retained:
        assert x.device == prepared.inputs[0].device and x.is_contiguous()
        assert x.dtype == torch.int32 and tuple(x.shape) == prepared.shape
    del retained, x
    gc.collect()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    for _ in range(20):
        output = baseline()
        del output
    torch.npu.synchronize()
    gc.collect()
    after = torch.npu.memory_allocated()
    assert before == after, ("baseline allocation leak", before, after)
    return {
        "status": "PASS",
        "candidate_retained": 6,
        "baseline_retained": 6,
        "compared_with": "independent CPU oracle",
        "distinct_storage": 12,
        "baseline_repeat20_before_bytes": before,
        "baseline_repeat20_after_bytes": after,
    }


def time_round(function: Callable[[], Any], iterations: int, torch: ModuleType) -> dict[str, int | float]:
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        output = function()
    torch.npu.synchronize()
    elapsed = time.perf_counter_ns() - start
    del output
    return {"elapsed_ns": elapsed, "iterations": iterations, "latency_us": elapsed / iterations / 1000}


def _benchmark(
    case_id: str,
    candidate: Callable[[], Any],
    baseline: Callable[[], Any],
    contract: dict[str, Any],
    output_dir: Path,
    torch: ModuleType,
) -> dict[str, Any]:
    from paired_statistics import summarize

    protocol = contract["acceptance"]["measurement_protocol"]
    for function in (baseline, candidate):
        for _ in range(protocol["warmup"]):
            output = function()
        torch.npu.synchronize()
        del output
    rows = []
    for round_id in range(protocol["rounds"]):
        order = ["baseline", "candidate"] if round_id % 2 == 0 else ["candidate", "baseline"]
        row = {"round": round_id, "order": order}
        for name in order:
            function = baseline if name == "baseline" else candidate
            row[name] = time_round(function, protocol["iterations_per_round"], torch)
        rows.append(row)
        _write_json(output_dir / (case_id + "_raw.json"), {"id": case_id, "rounds": rows, "protocol": protocol})
        print(json.dumps({"id": case_id, **row}), flush=True)
    result = summarize(
        [r["baseline"]["latency_us"] for r in rows],
        [r["candidate"]["latency_us"] for r in rows],
        contract["acceptance"],
        case_id,
    )
    result["measurement_protocol"] = protocol
    result["round_order"] = [r["order"] for r in rows]
    result["raw_file"] = case_id + "_raw.json"
    return result


def _run(args: argparse.Namespace, out: Path, report: dict[str, Any]) -> None:
    import torch
    import torch_npu
    import triton

    import solution

    sys.path.insert(0, str(ROOT / "official"))
    import generate_data
    import golden

    contract = json.loads((ROOT / "official/case.json").read_text(encoding="utf-8"))
    torch.npu.set_device(0)
    assert triton.runtime.driver.active.get_current_target().backend == "npu"
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "triton-ascend", "numpy")}
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "device": torch.npu.get_device_name(0),
        "versions": versions,
        "triton_target": str(triton.runtime.driver.active.get_current_target()),
        "source_sha256": _source_hashes(),
    }
    _write_json(out / "environment.json", environment)
    report["environment"] = environment
    command = [
        sys.executable,
        str(ROOT / "official/golden.py"),
        "--verify-functional",
        "--implementation",
        str(ROOT),
        "--output",
        str(out / "functional.json"),
    ]
    with (out / "functional.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    report["functional_returncode"] = completed.returncode
    if completed.returncode:
        raise RuntimeError("official functional verifier failed; see functional.log")
    functional = json.loads((out / "functional.json").read_text(encoding="utf-8"))
    check_coverage(functional, contract)
    report.update({key: functional[key] for key in ("cases", "negative", "dynamic_updates", "engineering_by_case")})
    report["supplemental"] = []
    report["benchmarks"] = []
    generalization = {"status": "RUNNING", "cases": [], "benchmarks": []}
    _write_json(out / "result.json", report)
    for case in contract["input_cases"]:
        values, attrs = generate_data.generate(case)
        inputs = make_inputs(case, values, torch)
        snapshots = [x.cpu().clone() for x in inputs]
        prepared = solution.prepare(inputs, attrs)
        baseline = golden.reference("baseline").prepare(inputs, attrs, prepared.info)
        expected_list = golden.calculate(values, attrs, prepared.info["single"])
        expected = torch.tensor(expected_list, dtype=torch.int32).reshape(prepared.shape)
        direct = solution.run(inputs, attrs)
        first, second = (prepared.run(), prepared.run())
        torch.npu.synchronize()
        assert all(torch.equal(x.cpu(), expected) for x in (direct, first, second))
        if first.numel():
            ptrs = [x.untyped_storage().data_ptr() for x in (direct, first, second)]
            assert len(set(ptrs)) == 3
        assert all((torch.equal(x.cpu(), old) for x, old in zip(inputs, snapshots)))
        record = {
            "id": case["id"],
            "status": "PASS",
            "backend": prepared.backend,
            "direct_run": "PASS",
            "repeated_outputs": "PASS",
        }
        if case.get("generalization"):
            digest = hashlib.sha256(json.dumps(first.cpu().tolist(), separators=(",", ":")).encode()).hexdigest()
            assert digest == case["generalization"]["expected_sha256"]
            generalization["cases"].append({"id": case["id"], "status": "PASS", "actual_sha256": digest})
        del direct, first, second
        if case.get("performance"):
            record["streams_cpu_oracle"] = _stream_oracle_checks(prepared, baseline, expected, torch)
        report["supplemental"].append(record)
        print(json.dumps(record), flush=True)
        if case.get("performance") or case.get("generalization", {}).get("measure"):
            result = _benchmark(case["id"], prepared.run, baseline, contract, out, torch)
            result["backend"] = prepared.backend
            if case.get("performance"):
                report["benchmarks"].append(result)
            else:
                generalization["benchmarks"].append(result)
        _write_json(out / "result.json", report)
        _write_json(out / "generalization.json", generalization)
    helper_root = out / "official_helpers"
    helper_root.mkdir()
    golden._write_sources(helper_root, golden.FUNCTIONAL_SOURCES)
    sys.path.insert(0, str(helper_root))
    spec = importlib.util.spec_from_file_location("_official_helpers", helper_root / "functional_checks.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    performance_case = next(c for c in contract["input_cases"] if c.get("performance"))
    report["negative_performance_fixture"] = helpers.negative_tests(performance_case)
    native_case = {
        "id": "additional_native_fallback",
        "requests": 3,
        "columns": 257,
        "layers": 3,
        "blocks": 4097,
        "offset": True,
        "seed": 417,
    }
    native_record, native_prepared, native_baseline = helpers.verify_case(native_case)
    native_record["dynamic_updates"] = helpers.dynamic_values(native_prepared)
    native_values = [x.cpu().tolist() for x in native_prepared.inputs]
    native_expected = torch.tensor(golden.calculate(native_values, {}, False), dtype=torch.int32).reshape(
        native_prepared.shape
    )
    native_record["streams_cpu_oracle"] = _stream_oracle_checks(
        native_prepared, native_baseline, native_expected, torch
    )
    report["additional_native_fallback"] = native_record
    generalization["status"] = "PASS"
    _write_json(out / "generalization.json", generalization)
    if len(report["benchmarks"]) != 2 or len(generalization["benchmarks"]) != 2:
        raise AssertionError("missing _benchmark")
    if any(r["status"] != "PASS" for r in report["benchmarks"]):
        raise AssertionError("one or more required performance gates failed")
    report["status"] = "PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("test results must be outside the submission source directory")
    out.mkdir(parents=True, exist_ok=False)
    report = {"task": "DenseBlockTableTranslate", "status": "RUNNING"}
    _write_json(out / "result.json", report)
    try:
        _run(args, out, report)
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        (out / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        _write_json(out / "result.json", report)
    print(json.dumps({"status": "PASS", "scope": "functional_engineering_performance"}), flush=True)


if __name__ == "__main__":
    main()
