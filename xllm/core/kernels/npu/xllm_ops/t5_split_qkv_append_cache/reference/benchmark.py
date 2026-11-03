#!/usr/bin/env python3
"""Measure public performance cases on NPU with the frozen 50/200/5 protocol.

The timed entry is run(inputs, attributes), including output allocation and API
submission through completion. CPU golden generation, input transfer and accuracy
comparison are outside timing. This is not the full acceptance suite.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parent
PROTOCOL = {"warmup": 50, "iterations": 200, "rounds": 5}


def measure(fn, npu, *, clock_ns=time.perf_counter_ns):
    """Return five round means and their medians; npu/clock are injectable."""
    for _ in range(PROTOCOL["warmup"]):
        fn()
    npu.synchronize()
    wall, events = [], []
    for _ in range(PROTOCOL["rounds"]):
        start, end = (npu.Event(enable_timing=True) for _ in range(2))
        npu.synchronize()
        before = clock_ns()
        start.record()
        for _ in range(PROTOCOL["iterations"]):
            fn()
        end.record()
        end.synchronize()
        wall.append((clock_ns() - before) / PROTOCOL["iterations"] / 1000)
        events.append(start.elapsed_time(end) * 1000 / PROTOCOL["iterations"])
        if not math.isfinite(wall[-1]) or wall[-1] <= 0 or not math.isfinite(events[-1]) or events[-1] < 0:
            raise ValueError("invalid wall or device-event timing")
    return {
        "wall_us_rounds": wall,
        "device_event_us_rounds": events,
        "wall_us_median": statistics.median(wall),
        "device_event_us_median": statistics.median(events),
    }


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cann_installation_info():
    """Record an existing installation file, without asserting a runtime version."""
    roots = [Path(os.environ[key]) for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if os.environ.get(key)]
    roots.append(Path("/usr/local/Ascend/ascend-toolkit/latest"))
    for root in roots:
        candidates = [root / "version.info", root / "ascend_toolkit_install.info"]
        candidates += [root / arch / "ascend_toolkit_install.info" for arch in ("aarch64-linux", "x86_64-linux")]
        for path in candidates:
            try:
                if path.is_file():
                    return {"source": str(path.resolve()), "content": path.read_text(encoding="utf-8")[:4096]}
            except (OSError, UnicodeError):
                continue
    return "unknown"


def environment_info(torch, torch_npu, device):
    try:
        device_name = torch.npu.get_device_name(device)
    except Exception:
        device_name = "unknown"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(getattr(torch, "__version__", "unknown")),
        "torch_npu": str(getattr(torch_npu, "__version__", "unknown")),
        "device_index": device,
        "device_name": device_name,
        "cann_installation_info": cann_installation_info(),
        "cann_runtime_version": "unknown",
        "driver_version": "unknown",
    }


def execute(args, report):
    # Support files are loaded from this public package, not from the submission.
    public_cases = json.loads((ROOT / "case.json").read_text(encoding="utf-8"))
    targets = json.loads((ROOT / "performance_targets.json").read_text(encoding="utf-8"))
    if any(type(targets.get(key)) is not int or targets[key] != value for key, value in PROTOCOL.items()):
        raise ValueError("performance_targets.json must specify warmup=50, iterations=200, rounds=5")
    if not isinstance(public_cases, list):
        raise ValueError("case.json must contain a list")
    selected = [case for case in public_cases if case.get("performance") is True]
    if not selected:
        raise ValueError("case.json contains no performance cases")
    sys.path.insert(0, str(ROOT))
    # Deliberately lazy: --help and CPU protocol tests need no torch installation.
    import numpy as np
    import torch
    import torch_npu
    import check_performance
    generate_data = load_module("generate_data", ROOT / "generate_data.py")
    golden = load_module("golden", ROOT / "golden.py")
    verifier = load_module("public_verify", ROOT / "verify.py")
    if not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable; no performance measurement was performed")
    torch.npu.set_device(args.device)
    report["environment"] = environment_info(torch, torch_npu, args.device)
    solution = load_module("submitted_solution", args.solution_file.resolve())
    if not callable(getattr(solution, "run", None)):
        raise ValueError("submitted solution must define run(inputs, attributes)")
    for case in selected:
        report["active_case"] = case["case_name"]
        arrays = generate_data.generate(case)
        attrs = {item["name"]: item["value"] for item in case["attr_desc"]}
        expected = golden.calculate(arrays, attrs)
        inputs = verifier.device_inputs(case, arrays, f"npu:{args.device}")
        outputs = tuple(solution.run(inputs, attrs))
        torch.npu.synchronize()
        report["npu_executed"] = True
        verifier.validate_ownership(inputs, outputs)
        accuracy = verifier.compare(case, arrays, outputs, expected)
        for tensor, original in zip(inputs, arrays):
            if not np.array_equal(verifier.cpu_array(tensor).view(np.uint16), original.view(np.uint16)):
                raise AssertionError(f"{case['case_name']}: input changed")
        del outputs
        # Call the public entry each time; the harness does not cache outputs.
        result = measure(lambda: solution.run(inputs, attrs), torch.npu)
        record = {"case_name": case["case_name"], "accuracy": accuracy, "results": {"custom": result}}
        report["cases"].append(record)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
    report.pop("active_case", None)
    report["status"] = "passed"
    acceptance = check_performance.evaluate(report, targets, public_cases)
    report["performance_checks"] = acceptance
    report["status"] = acceptance["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="NPU performance measurement: fixed 50 warmups, 200 calls/round, 5 rounds.")
    parser.add_argument("--solution-file", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("performance.json"))
    args = parser.parse_args(argv)
    report = {
        "status": "running",
        "npu_executed": False,
        "protocol": dict(PROTOCOL),
        "cases": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "solution_file": str(args.solution_file.resolve()),
        "solution_file_sha256": "unknown",
        "implementation_manifest_note": "This hash covers only the entry file; attach a manifest covering all implementation sources and built artifacts.",
        "environment": {"python": platform.python_version(), "torch": "unknown", "torch_npu": "unknown",
                        "device_name": "unknown", "cann_installation_info": "unknown", "cann_runtime_version": "unknown",
                        "driver_version": "unknown"},
    }
    exit_code = 1
    try:
        report["solution_file_sha256"] = hashlib.sha256(args.solution_file.read_bytes()).hexdigest()
        # Leave a non-success marker if the process is terminated during work.
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        execute(args, report)
        exit_code = 0 if report["status"] == "passed" else 1
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(report["error"], file=sys.stderr)
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"Cannot save performance report: {exc}", file=sys.stderr)
        return 1
    print(f"Performance measurement: {report['status']}; report saved to {args.report}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
