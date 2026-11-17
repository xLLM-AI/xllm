# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the unmodified public fixtures on a real NPU; never substitutes CPU output."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from fixtures import ensure_fixtures
from statistics_utils import paired_statistics

ROOT = Path(__file__).resolve().parent


def _load_module(name: Any, path: Any) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump(path: Any, value: Any) -> Any:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stamp() -> Any:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _source_hashes() -> Any:
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.suffix in (".py", ".json", ".md", ".zip")
        and not any(s in p.parts for s in ("results", "task_submission", "verification", "__pycache__"))
    }


def _environment(torch: Any) -> Any:
    versions = {}
    for name in ("torch", "torch-npu", "triton-ascend", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_FOUND"
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "device_name": torch.npu.get_device_name(0),
        "device": "npu:0",
    }
    roots = {Path("/usr/local/Ascend/ascend-toolkit/latest"), Path.home() / "Ascend/cann-9.1.0"}
    roots.update(Path(os.environ[k]) for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if os.environ.get(k))
    files = {}
    for root in roots:
        for pattern in ("version.cfg", "version.info", "*/ascend_toolkit_install.info", "*/version.info"):
            for p in root.glob(pattern):
                if p.is_file():
                    files[str(p)] = p.read_text(errors="replace")[:12000]
    info["cann_version_files"] = files
    try:
        proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
        info["npu_smi"] = proc.stdout + proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["npu_smi"] = str(exc)
    # Unknown or mismatched evidence must never certify the official environment.
    import re

    configured_root = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    active_files = {p: v for p, v in files.items() if configured_root and Path(p).is_relative_to(Path(configured_root))}
    # Ascend A3 SoC names, consistent with vllm-ascend/device/hardware.py.
    info["a3_verified"] = info["device_name"].lower() in {
        "ascend910_9362",
        "ascend910_9363",
        "ascend910_9372",
        "ascend910_9381",
        "ascend910_9382",
        "ascend910_9391",
        "ascend910_9392",
    }
    info["cann_910_verified"] = any(
        re.search(r"(?im)^\s*(?:version|cann_version)\s*[:=]\s*9\.1\.0\s*$", v) for v in active_files.values()
    )
    info["official_environment_verified"] = info["a3_verified"] and info["cann_910_verified"]
    return info


class Harness:
    def __init__(self, torch: Any, solution: Any, golden: Any, generator: Any, log: Any) -> None:
        self._torch, self._solution, self._golden, self._generator = torch, solution, golden, generator
        self._log = log

    def _emit(self, kind: Any, record: Any) -> Any:
        entry = {"time": _stamp(), "kind": kind, **record}
        self._log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._log.flush()
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    def _inputs(self, case: Any) -> Any:
        values, attrs = self._generator.generate(case)
        tensors = []
        for v in values:
            if case.get("offset"):
                base = self._torch.tensor([123] + list(v), dtype=self._torch.int32, device="npu:0")
                tensors.append(base[1:])
            else:
                tensors.append(self._torch.tensor(v, dtype=self._torch.int32, device="npu:0"))
        expected = self._golden.calculate(values, attrs)
        if "expected" in case and expected != case["expected"]:
            raise AssertionError("fixture and independent golden disagree")
        return tensors, attrs, expected

    def _check_output(self, out: Any, expected: Any) -> Any:
        assert list(out.shape) == [len(expected)], "wrong output shape"
        assert out.dtype == self._torch.uint8, "wrong output dtype"
        assert out.device == self._torch.device("npu:0"), "wrong output device"
        assert out.is_contiguous(), "noncontiguous output"
        actual = out.detach().cpu().tolist()
        mismatches = sum(a != b for a, b in zip(actual, expected))
        assert mismatches == 0, f"{mismatches} output mismatches"
        return mismatches

    @staticmethod
    def _independent(outputs: Any, inputs: Any) -> Any:
        # Empty tensors need no allocated storage; null pointers are not aliasing.
        outs = [x.untyped_storage().data_ptr() for x in outputs if x.numel()]
        ins = {x.untyped_storage().data_ptr() for x in inputs if x.numel()}
        return len(set(outs)) == len(outs) and not (set(outs) & ins)

    def correctness(self, case: Any) -> Any:
        inputs, attrs, expected = self._inputs(case)
        before = [x.cpu().tolist() for x in inputs]
        prepared = self._solution.prepare(inputs, attrs)
        baseline = self._golden.reference("baseline").prepare(inputs, attrs, {"rows": len(inputs[2])})
        outs = [prepared.run(), prepared.run(), baseline(), baseline(), self._solution.run(inputs, attrs)]
        for out in outs:
            self._check_output(out, expected)
        assert self._independent(outs, inputs), "shared output storage"
        assert before == [x.cpu().tolist() for x in inputs], "input mutated"
        if case.get("offset"):
            assert all(x.storage_offset() == 1 for x in inputs)
        result = {
            "id": case["id"],
            "status": "PASS",
            "shape": [len(expected)],
            "input_shapes": [list(x.shape) for x in inputs],
            "attributes": attrs,
            "checks": {
                "candidate_mismatches": 0,
                "baseline_mismatches": 0,
                "shape": True,
                "dtype": True,
                "npu": True,
                "contiguous": True,
                "inputs_unchanged": True,
                "independent_storage": True,
            },
            "storage_offsets": [x.storage_offset() for x in inputs],
        }
        self._emit("accuracy", result)
        return result

    def negatives(self, case: Any) -> Any:
        result = []
        names = [
            "input_count",
            "unknown_attribute",
            "cpu_input",
            "dtype",
            "noncontiguous",
            "zero_tile",
            "oversize_tile",
            "non_dense_output_offsets",
            "length_exceeds_segment",
        ]
        for name in names:
            inputs, attrs, _ = self._inputs(case)
            if name == "input_count":
                inputs.pop()
            elif name == "unknown_attribute":
                attrs = dict(attrs, unknown=1)
            elif name == "cpu_input":
                inputs[0] = inputs[0].cpu()
            elif name == "dtype":
                inputs[0] = inputs[0].to(self._torch.int64)
            elif name == "noncontiguous":
                base = self._torch.empty(inputs[0].numel() * 2, dtype=self._torch.int32, device="npu:0")
                base[::2].copy_(inputs[0])
                inputs[0] = base[::2]
            elif name == "zero_tile":
                attrs = dict(attrs, tile_size=0)
            elif name == "oversize_tile":
                attrs = dict(attrs, tile_size=4097)
            elif name == "non_dense_output_offsets":
                inputs[3][1] = 0
            elif name == "length_exceeds_segment":
                inputs[2][0] = 2147483647
            try:
                self._solution.prepare(inputs, attrs)
            except (TypeError, ValueError) as exc:
                record = {"id": name, "status": "PASS", "exception": type(exc).__name__}
            else:
                raise AssertionError("illegal input accepted: " + name)
            result.append(record)
            self._emit("negative", record)
        return result

    def dynamic(self) -> Any:
        # Change offsets, boundaries and lengths without re-preparing. Enqueue a
        # producer and consumer on each caller-selected stream (no cross-stream race).
        case = {
            "values": [[0, 1, 2, 3, 8, 9, 10, 11], [0, 4, 8], [4, 4], [0, 2, 4]],
            "attributes": {"tile_size": 4, "max_tiles_per_seq": 2},
            "offset": True,
        }
        inputs, attrs, expected = self._inputs(case)
        prepared = self._solution.prepare(inputs, attrs)
        original = prepared.run()
        self._torch.npu.synchronize()
        retained = []
        sequences = [
            ([[4, 5, 6, 7, 9, 10, 11, 12], [0, 4, 8], [4, 4], [0, 2, 4]], [1, 0, 0, 0]),
            ([[4, 5, 6, 7, 8, 9, 10, 11], [0, 8, 8], [8, 0], [0, 2, 4]], [1, 1, 0, 0]),
        ]
        for values, expected_new in sequences:
            stream = self._torch.npu.Stream()
            with self._torch.npu.stream(stream):
                for x, value in zip(inputs, values):
                    x.copy_(self._torch.tensor(value, dtype=self._torch.int32, device="npu:0"))
                out = prepared.run()
                consumed = out.clone()
            stream.synchronize()
            self._check_output(out, expected_new)
            self._check_output(consumed, expected_new)
            retained.append(out)
        self._check_output(original, expected)
        assert self._independent([original, *retained], inputs)
        # Rebind the same tensor objects to different legal storage offsets.
        # A prepared call must dereference the current pointers, not old buffers.
        rebound_values = [[8, 9, 10, 11, 12, 13, 14, 15], [0, 4, 8], [4, 4], [0, 2, 4]]
        for tensor, values in zip(inputs, rebound_values):
            storage = self._torch.tensor([777, 888] + values, dtype=self._torch.int32, device="npu:0")
            tensor.set_(storage[2:])
        rebound = prepared.run()
        self._check_output(rebound, [1, 0, 1, 0])
        self._check_output(original, expected)
        assert self._independent([original, *retained, rebound], inputs)
        record = {
            "id": "current_values_current_stream",
            "status": "PASS",
            "updates": 2,
            "nonzero_storage_offset": True,
            "prior_outputs_unchanged": True,
            "storage_rebinding": True,
        }
        self._emit("dynamic", record)
        return record

    def benchmark(self, case: Any, limit: Any) -> Any:
        inputs, attrs, _ = self._inputs(case)
        candidate = self._solution.prepare(inputs, attrs)
        baseline = self._golden.reference("baseline").prepare(inputs, attrs, {"rows": len(inputs[2])})
        paths = {"baseline": baseline, "candidate": candidate.run}
        for fn in paths.values():
            for _ in range(50):
                fn()
        self._torch.npu.synchronize()
        rounds = {name: [] for name in paths}
        order = []
        for r in range(6):
            names = ["baseline", "candidate"] if r % 2 == 0 else ["candidate", "baseline"]
            order.append(names)
            for name in names:
                fn = paths[name]
                self._torch.npu.synchronize()
                start = time.perf_counter_ns()
                for _ in range(200):
                    fn()
                self._torch.npu.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 200 / 1000
                rounds[name].append(elapsed)
                self._emit("performance_round", {"id": case["id"], "round": r, "path": name, "latency_us": elapsed})
        record = {
            "id": case["id"],
            **paired_statistics(rounds["baseline"], rounds["candidate"], limit),
            "warmup": 50,
            "iterations": 200,
            "execution_order": order,
            "timed_entry": "complete prepared.run plus final device synchronization",
        }
        self._emit("performance", record)
        return record

    def _memory_path(self, fn: Any, expected: Any) -> Any:
        npu = self._torch.npu
        gc.collect()
        npu.synchronize()
        before = npu.memory_allocated()
        npu.reset_peak_memory_stats()
        out = fn()
        npu.synchronize()
        peak = npu.max_memory_allocated() - before
        mismatch = self._check_output(out, expected)
        output_bytes = out.numel() * out.element_size()
        del out
        gc.collect()
        npu.synchronize()
        after_one = npu.memory_allocated()
        repeats, active_after_each = [], []
        for _ in range(20):
            out = fn()
            repeats.append(self._check_output(out, expected))
            del out
            npu.synchronize()
            active_after_each.append(npu.memory_allocated())
        after = npu.memory_allocated()
        assert before == after_one == after and all(x == before for x in active_after_each), "allocation leak"
        return {
            "status": "PASS",
            "warmup_calls": 3,
            "active_before_bytes": before,
            "one_call_peak_increment_bytes": peak,
            "output_bytes": output_bytes,
            "single_call_golden_mismatches": mismatch,
            "active_after_one_release_bytes": after_one,
            "repeat_calls": 20,
            "repeat_golden_mismatches": repeats,
            "active_after_each_release_bytes": active_after_each,
            "active_after_repeat_release_bytes": after,
            "residual_active_bytes": after - before,
            "measurement_scope": "PyTorch active allocation; output and visible temporaries; "
            "both prepared objects and inputs retained; excludes total HBM and untracked runtime scratch",
        }

    def engineering(self, case: Any) -> Any:
        inputs, attrs, expected = self._inputs(case)
        before = [x.cpu().tolist() for x in inputs]
        candidate = self._solution.prepare(inputs, attrs)
        baseline = self._golden.reference("baseline").prepare(inputs, attrs, {"rows": len(inputs[2])})
        for fn in (candidate.run, baseline):
            for _ in range(3):
                fn()
        self._torch.npu.synchronize()
        outputs = []
        streams = [self._torch.npu.Stream() for _ in range(2)]
        for stream in streams:
            with self._torch.npu.stream(stream):
                outputs.extend(candidate.run() for _ in range(3))
        for stream in streams:
            stream.synchronize()
        mismatches = [self._check_output(x, expected) for x in outputs]
        assert self._independent(outputs, inputs), "two-stream storage alias"
        del outputs
        self._torch.npu.synchronize()
        record = {
            "id": case["id"],
            "status": "PASS",
            "common_warmup_calls_per_path": 3,
            "two_streams": {
                "status": "PASS",
                "streams": 2,
                "outputs_per_stream": 3,
                "golden_mismatches_per_output": mismatches,
                "retained_outputs_independent": True,
                "outputs_independent_of_inputs": True,
            },
            "candidate": self._memory_path(candidate.run, expected),
            "baseline": self._memory_path(baseline, expected),
        }
        assert before == [x.cpu().tolist() for x in inputs]
        record["inputs_unchanged"] = True
        record["baseline_minus_candidate_peak_bytes"] = (
            record["baseline"]["one_call_peak_increment_bytes"] - record["candidate"]["one_call_peak_increment_bytes"]
        )
        record["memory_difference_is_observation_only"] = True
        self._emit("memory", record)
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "official")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "npu")
    args = parser.parse_args()
    if args.fixtures == ROOT / "official":
        ensure_fixtures(ROOT)
    # New run directory prevents a failed attempt from leaving old PASS files.
    outdir = args.output / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    outdir.mkdir(parents=True)
    result = {
        "status": "RUNNING",
        "started_utc": _stamp(),
        "cases": [],
        "negative": [],
        "engineering": {},
        "benchmarks": [],
        "source_sha256": _source_hashes(),
    }
    generalization = {"status": "NOT_RUN", "cases": [], "benchmarks": []}
    engineering = {"status": "NOT_RUN", "cases": []}
    _dump(outdir / "result.json", result)
    code = 1
    with (outdir / "raw.log").open("w", encoding="utf-8") as log:
        try:
            import torch
            import torch_npu  # noqa: F401

            if not torch.npu.is_available():
                raise RuntimeError("No NPU available")
            torch.npu.set_device(0)
            result["environment"] = _environment(torch)
            import solution

            fixtures = json.loads((args.fixtures / "case.json").read_text(encoding="utf-8"))
            result["fixture_sha256"] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.fixtures.iterdir() if p.is_file()
            }
            golden = _load_module("task10_golden", args.fixtures / "golden.py")
            generator = _load_module("task10_generator", args.fixtures / "generate_data.py")
            h = Harness(torch, solution, golden, generator, log)
            cases = fixtures["input_cases"]
            for case in cases:
                record = h.correctness(case)
                target = generalization["cases"] if "generalization" in case else result["cases"]
                target.append(record)
                _dump(outdir / "result.json", result)
                _dump(outdir / "generalization.json", generalization)
            primary = [c for c in cases if c.get("performance")]
            result["negative"] = h.negatives(primary[0])
            result["dynamic"] = h.dynamic()
            for case in primary:
                engineering["cases"].append(h.engineering(case))
            engineering["status"] = "PASS"
            result["engineering"] = engineering
            for case in primary:
                result["benchmarks"].append(h.benchmark(case, fixtures["acceptance"]["max_mean_us"][case["id"]]))
            for case in cases:
                if case.get("generalization", {}).get("measure"):
                    generalization["benchmarks"].append(h.benchmark(case, None))
            generalization["status"] = "PASS"
            assert len(result["cases"]) == 10 and len(generalization["cases"]) == 4
            assert len(result["negative"]) == 9 and len(engineering["cases"]) == 2
            assert len(result["benchmarks"]) == len(generalization["benchmarks"]) == 2
            if not all(x["status"] == "PASS" for x in result["benchmarks"]):
                raise AssertionError("official performance gate failed")
            if not result["environment"]["official_environment_verified"]:
                result["status"] = "ENVIRONMENT_UNVERIFIED"
                result["error"] = (
                    "Actual test results retained; A3 and active CANN 9.1.0 must be verified before acceptance"
                )
                code = 2
            else:
                result["status"] = "PASS"
                code = 0
        except BaseException as exc:
            result["status"] = "BLOCKED" if isinstance(exc, (ImportError, ModuleNotFoundError)) else "FAIL"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
            log.write(result["traceback"])
            print(json.dumps({"error": result["error"]}), file=sys.stderr)
        finally:
            result["finished_utc"] = _stamp()
            _dump(outdir / "result.json", result)
            _dump(outdir / "generalization.json", generalization)
            _dump(outdir / "engineering_supplement.json", engineering)
            print(json.dumps({"results": str(outdir), "status": result["status"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
