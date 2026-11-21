# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Paired complete-call latency measurements for the task acceptance protocol."""

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch_npu

from solution import prepare


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(case: dict[str, Any], values: list[Any]) -> list[torch.Tensor]:
    tensors = []
    for index, value in enumerate(values):
        cpu = torch.tensor(value, dtype=torch.int32)
        if index == 0 and "requests" in case:
            cpu = cpu.reshape(case["requests"], case["columns"])
        tensor = cpu.to("npu:0")
        if case.get("offset"):
            storage = torch.empty((tensor.numel() + 17,), dtype=torch.int32, device=tensor.device)
            tensor = storage[1 : tensor.numel() + 1].reshape(cpu.shape)
            tensor.copy_(cpu)
        tensors.append(tensor)
    return tensors


def _round(fn: Callable[[], torch.Tensor], iterations: int) -> tuple[float, int]:
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        output = fn()
    torch.npu.synchronize()
    elapsed_ns = time.perf_counter_ns() - start
    del output
    return elapsed_ns / iterations / 1000, elapsed_ns


def _measure(
    case: dict[str, Any], fixture: Any, golden: Any, contract: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    values, attributes = fixture.generate(case)
    inputs = _inputs(case, values)
    prepared = prepare(inputs, attributes)
    if args.backend:
        prepared._mode = args.backend
    if args.column_tile:
        prepared._column_tile = args.column_tile
    if args.layer_tile:
        prepared._layer_tile = args.layer_tile
    if args.backend or args.column_tile or args.layer_tile:
        prepared._configure()
    mapping = inputs[1]
    info = {
        "layers": 1 if mapping.ndim == 1 else mapping.shape[0],
        "blocks": mapping.shape[-1],
        "single": mapping.ndim == 1,
        "values": values,
    }
    baseline = golden.reference("baseline").prepare(inputs, attributes, info)
    expected = torch.tensor(golden.calculate(values, attributes, info["single"]), dtype=torch.int32).reshape(
        prepared.shape
    )
    snapshot = [tensor.cpu().clone() for tensor in inputs]
    for fn in (baseline, prepared.run):
        actual = fn()
        torch.npu.synchronize()
        assert actual.dtype == torch.int32 and actual.is_contiguous()
        assert actual.device == inputs[0].device and tuple(actual.shape) == tuple(expected.shape)
        assert torch.equal(actual.cpu(), expected)
        assert all(torch.equal(tensor.cpu(), old) for tensor, old in zip(inputs, snapshot))
        del actual
    protocol = contract["acceptance"]["measurement_protocol"]
    for fn in (baseline, prepared.run):
        for _ in range(protocol["warmup"]):
            output = fn()
        torch.npu.synchronize()
        del output
    gc.collect()
    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    raw = []
    for index in range(protocol["rounds"]):
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        record: dict[str, Any] = {"round": index + 1, "order": list(order)}
        for side in order:
            value, elapsed_ns = _round(
                baseline if side == "baseline" else prepared.run, protocol["iterations_per_round"]
            )
            assert np.isfinite(value) and value > 0
            samples[side].append(value)
            record[side + "_us"] = value
            record[side + "_elapsed_ns"] = elapsed_ns
        raw.append(record)
    base, candidate = samples["baseline"], samples["candidate"]
    deltas = np.asarray(base) - np.asarray(candidate)
    rng = np.random.default_rng(contract["acceptance"]["seed"])
    indices = rng.integers(0, len(deltas), (contract["acceptance"]["resamples"], len(deltas)))
    ci = np.percentile(deltas[indices].mean(axis=1), [2.5, 97.5], method="linear").tolist()
    maximum = contract["acceptance"]["max_mean_us"].get(case["id"])
    candidate_mean = statistics.mean(candidate)
    saved_mean = statistics.mean(base) - candidate_mean
    saved_median = statistics.median(base) - statistics.median(candidate)
    gates = {
        "target_mean": maximum is None or candidate_mean <= maximum,
        "mean_saving": saved_mean > 0,
        "median_saving": saved_median > 0,
        "paired_bootstrap_lower_positive": ci[0] > 0,
    }
    required = maximum is not None
    return {
        "id": case["id"],
        "status": "PASS" if not required or all(gates.values()) else "FAIL",
        "required": required,
        "correctness": "PASS",
        "baseline_rounds_us": base,
        "candidate_rounds_us": candidate,
        "raw_rounds": raw,
        "baseline_mean_us": statistics.mean(base),
        "candidate_mean_us": candidate_mean,
        "baseline_median_us": statistics.median(base),
        "candidate_median_us": statistics.median(candidate),
        "saved_mean_us": saved_mean,
        "saved_median_us": saved_median,
        "paired_savings_us": deltas.tolist(),
        "saved_mean_ci95_us": ci,
        "max_mean_us": maximum,
        "gates": gates,
        "warmup": protocol["warmup"],
        "iterations": protocol["iterations_per_round"],
        "rounds": protocol["rounds"],
        "unit": "us",
        "measurement_protocol": protocol,
        "bootstrap": {
            "resamples": contract["acceptance"]["resamples"],
            "seed": contract["acceptance"]["seed"],
            "percentile": "linear",
            "rng": "numpy.random.default_rng",
        },
        "configuration": {
            "mode": prepared._mode,
            "column_tile": prepared._column_tile,
            "layer_tile": prepared._layer_tile,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).with_name("fixtures"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--backend", choices=["fused", "gather"])
    parser.add_argument("--column-tile", type=int)
    parser.add_argument("--layer-tile", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new result file to preserve earlier measurements")
    for tile in (args.column_tile, args.layer_tile):
        if tile is not None and (tile < 1 or tile & (tile - 1)):
            parser.error("tile sizes must be positive powers of two")
    fixtures = args.fixtures.resolve()
    contract = json.loads((fixtures / "case.json").read_text())
    chosen = args.case or ["packed_perf_small", "packed_perf_large", "packed_route_32736", "packed_route_32768"]
    available = {case["id"]: case for case in contract["input_cases"]}
    if set(chosen) - available.keys():
        parser.error("unknown case ID")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    report: dict[str, Any] = {
        "status": "RUNNING",
        "benchmarks": [],
        "environment": {
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "triton-ascend")},
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.iterdir()
            if path.suffix in (".py", ".cpp")
        },
        "fixture_case_sha256": hashlib.sha256((fixtures / "case.json").read_bytes()).hexdigest(),
    }
    try:
        torch.npu.set_device(0)
        report["environment"]["device"] = torch.npu.get_device_name(0)
        fixture = _module(fixtures / "generate_data.py", "packed_fixture_generator")
        golden = _module(fixtures / "golden.py", "packed_fixture_golden")
        for name in chosen:
            record = _measure(available[name], fixture, golden, contract, args)
            report["benchmarks"].append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        report["status"] = "PASS" if all(r["status"] == "PASS" for r in report["benchmarks"]) else "FAIL"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
