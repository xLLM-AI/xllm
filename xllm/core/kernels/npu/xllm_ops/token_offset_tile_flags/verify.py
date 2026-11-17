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
import re
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

from fixtures import ensure_fixtures
from statistics_utils import paired_statistics

ROOT = Path(__file__).resolve().parent

FROZEN_FIXTURE_NAMES = ("任务书.md", "case.json", "op.json", "generate_data.py", "golden.py")
SUPPLEMENT_SCOPE = (
    "Two unchanged public performance fixtures, independent CPU golden, two-stream retained outputs, "
    "one-call allocator peaks and twenty-call release. No performance rerun, cold prepare timing, "
    "dynamic-input or storage-rebinding claim."
)
MEMORY_SCOPE = (
    "Warmed one-call PyTorch NPU active allocation increment; both prepared objects, input tensors and "
    "baseline static layout remain alive. Includes output and visible temporaries; excludes persistent "
    "prepared storage, total HBM and untracked CANN/UB scratch."
)
PEAK_DEFINITION = "PyTorch active allocation; not total HBM or all runtime scratch"


def _require(condition: Any, message: str) -> None:
    # Acceptance gates must also raise under `python -O` / PYTHONOPTIMIZE.
    if not condition:
        raise AssertionError(message)


def _load_module(name: Any, path: Any) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump(path: Any, value: Any) -> Any:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stamp() -> Any:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _golden_sha256(expected: Any) -> Any:
    return hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode("utf-8")).hexdigest()


def _source_hashes() -> Any:
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.suffix in (".py", ".json", ".md", ".zip")
        and not any(s in p.parts for s in ("results", "task_submission", "verification", "__pycache__"))
    }


def _fixtures_unmodified(fixtures: Any) -> Any:
    with zipfile.ZipFile(ROOT / "official.zip") as archive:
        return all((fixtures / name).read_bytes() == archive.read(name) for name in FROZEN_FIXTURE_NAMES)


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
        _require(list(out.shape) == [len(expected)], "wrong output shape")
        _require(out.dtype == self._torch.uint8, "wrong output dtype")
        _require(out.device == self._torch.device("npu:0"), "wrong output device")
        _require(out.is_contiguous(), "noncontiguous output")
        actual = out.detach().cpu().tolist()
        mismatches = sum(a != b for a, b in zip(actual, expected))
        _require(mismatches == 0, f"{mismatches} output mismatches")
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
        counts = [self._check_output(out, expected) for out in outs]
        independent = self._independent(outs, inputs)
        _require(independent, "shared output storage")
        unchanged = before == [x.cpu().tolist() for x in inputs]
        _require(unchanged, "input mutated")
        if case.get("offset"):
            _require(all(x.storage_offset() == 1 for x in inputs), "lost nonzero storage offset")
        result = {
            "id": case["id"],
            "status": "PASS",
            "shape": [len(expected)],
            "input_shapes": [list(x.shape) for x in inputs],
            "attributes": attrs,
            "checks": {
                "candidate_mismatches": counts[0] + counts[1] + counts[4],
                "baseline_mismatches": counts[2] + counts[3],
                "shape": all(list(x.shape) == [len(expected)] for x in outs),
                "dtype": all(x.dtype == self._torch.uint8 for x in outs),
                "npu": all(x.device == self._torch.device("npu:0") for x in outs),
                "contiguous": all(x.is_contiguous() for x in outs),
                "inputs_unchanged": unchanged,
                "independent_storage": independent,
            },
            "golden_sha256": _golden_sha256(expected),
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
        _require(self._independent([original, *retained], inputs), "dynamic outputs share storage")
        # Rebind the same tensor objects to different legal storage offsets.
        # A prepared call must dereference the current pointers, not old buffers.
        rebound_values = [[8, 9, 10, 11, 12, 13, 14, 15], [0, 4, 8], [4, 4], [0, 2, 4]]
        for tensor, values in zip(inputs, rebound_values):
            storage = self._torch.tensor([777, 888] + values, dtype=self._torch.int32, device="npu:0")
            tensor.set_(storage[2:])
        rebound = prepared.run()
        self._check_output(rebound, [1, 0, 1, 0])
        self._check_output(original, expected)
        _require(self._independent([original, *retained, rebound], inputs), "rebound outputs share storage")
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
        _require(before == after_one == after and all(x == before for x in active_after_each), "allocation leak")
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
            "measurement_scope": MEMORY_SCOPE,
        }

    def engineering(self, case: Any, frozen_verify_case: Any) -> Any:
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
        independent = self._independent(outputs, inputs)
        _require(independent, "two-stream storage alias")
        del outputs
        self._torch.npu.synchronize()
        record = {
            "id": case["id"],
            "status": "PASS",
            "frozen_verify_case": frozen_verify_case,
            "common_warmup_calls_per_path": 3,
            "two_streams": {
                "status": "PASS",
                "streams": 2,
                "outputs_per_stream": 3,
                "golden_mismatches_per_output": mismatches,
                "retained_outputs_independent": independent,
                "outputs_independent_of_inputs": independent,
            },
            "candidate": self._memory_path(candidate.run, expected),
            "baseline": self._memory_path(baseline, expected),
        }
        unchanged = before == [x.cpu().tolist() for x in inputs]
        _require(unchanged, "engineering inputs mutated")
        record["inputs_unchanged"] = unchanged
        record["baseline_minus_candidate_peak_bytes"] = (
            record["baseline"]["one_call_peak_increment_bytes"] - record["candidate"]["one_call_peak_increment_bytes"]
        )
        record["memory_difference_is_observation_only"] = True
        self._emit("memory", record)
        return record


def _engineering_summary(cases: Any) -> Any:
    small = next(c for c in cases if c["id"] == "flags_perf_small")
    candidate = small["candidate"]
    two_streams = all(c["two_streams"]["status"] == "PASS" for c in cases)
    independent = all(
        c["two_streams"]["retained_outputs_independent"] and c["two_streams"]["outputs_independent_of_inputs"]
        for c in cases
    )
    released = all(
        rec["residual_active_bytes"] == 0
        and rec["active_before_bytes"] == rec["active_after_repeat_release_bytes"]
        and rec["repeat_golden_mismatches"] == [0] * rec["repeat_calls"]
        for c in cases
        for rec in (c["candidate"], c["baseline"])
    )
    return {
        "two_streams": "PASS" if two_streams else "FAIL",
        "retained_outputs_independent": "PASS" if independent else "FAIL",
        "repeat20_released": "PASS" if released else "FAIL",
        "active_before_bytes": candidate["active_before_bytes"],
        "active_after_bytes": candidate["active_after_repeat_release_bytes"],
        "peak_increment_bytes": candidate["one_call_peak_increment_bytes"],
        "output_bytes": candidate["output_bytes"],
        "peak_definition": PEAK_DEFINITION,
    }


def _validate_supplement(supplement: Any, want: Any, cases: Any) -> None:
    for key in ("schema_version", "task", "kind", "device_requested"):
        _require(supplement.get(key) == want[key], f"engineering_supplement {key} != {want[key]!r}")
    for key in ("started_utc", "finished_utc", "scope"):
        _require(isinstance(supplement.get(key), str) and supplement[key], f"engineering_supplement missing {key}")
    _require(supplement["device_setup_attempted"] is True, "engineering_supplement: device setup was not attempted")
    _require(supplement["performance_measured"] is False, "engineering_supplement: performance is out of scope")
    _require(
        supplement["memory_gain_required"] is False, "engineering_supplement: memory gain must stay observation-only"
    )
    _require(supplement["frozen_results_modified"] is False, "engineering_supplement: frozen results were modified")
    environment = supplement["environment"]
    for key, value in want["environment"].items():
        if key == "versions":
            for name in value:
                _require(
                    environment.get("versions", {}).get(name) not in (None, "", "NOT_FOUND"),
                    f"engineering_supplement environment missing version: {name}",
                )
        else:
            _require(
                isinstance(environment.get(key), str) and environment[key],
                f"engineering_supplement environment missing {key}",
            )
    records = {c["id"]: c for c in supplement["cases"]}
    _require(
        len(records) == len(supplement["cases"]) == len(want["cases"]),
        "engineering_supplement case records do not match the mandatory list",
    )
    for want_case in want["cases"]:
        got = records.get(want_case["id"])
        _require(got is not None, f"missing engineering_supplement case: {want_case['id']}")
        _require(got["status"] == "PASS", f"engineering_supplement case not PASS: {want_case['id']}")
        frozen = got["frozen_verify_case"]
        source = cases[want_case["id"]]
        for key in ("id", "status", "checks", "shape", "input_shapes", "attributes"):
            _require(
                frozen.get(key) == source.get(key), f"frozen_verify_case of {want_case['id']} diverges from result.json"
            )
        _require(
            got["common_warmup_calls_per_path"] == want_case["common_warmup_calls_per_path"],
            f"{want_case['id']}: wrong common warmup count",
        )
        streams = got["two_streams"]
        _require(
            streams["status"] == "PASS" and streams["streams"] == 2 and streams["outputs_per_stream"] == 3,
            f"{want_case['id']}: two-stream protocol not fulfilled",
        )
        _require(streams["golden_mismatches_per_output"] == [0] * 6, f"{want_case['id']}: retained output mismatches")
        _require(
            streams["retained_outputs_independent"] is True and streams["outputs_independent_of_inputs"] is True,
            f"{want_case['id']}: retained outputs alias each other or the inputs",
        )
        for path in ("candidate", "baseline"):
            rec = got[path]
            _require(rec["status"] == "PASS", f"{want_case['id']} {path}: memory path not PASS")
            _require(
                rec["warmup_calls"] == 3 and rec["repeat_calls"] == 20, f"{want_case['id']} {path}: wrong call counts"
            )
            _require(rec["single_call_golden_mismatches"] == 0, f"{want_case['id']} {path}: single-call mismatch")
            _require(rec["repeat_golden_mismatches"] == [0] * 20, f"{want_case['id']} {path}: repeat-call mismatch")
            _require(
                rec["active_before_bytes"]
                == rec["active_after_one_release_bytes"]
                == rec["active_after_repeat_release_bytes"],
                f"{want_case['id']} {path}: active allocation did not return to its starting level",
            )
            _require(rec["residual_active_bytes"] == 0, f"{want_case['id']} {path}: residual allocation")
            _require(rec["output_bytes"] == source["shape"][0], f"{want_case['id']} {path}: wrong output byte count")
            _require(
                isinstance(rec["measurement_scope"], str) and rec["measurement_scope"],
                f"{want_case['id']} {path}: measurement_scope missing",
            )
        _require(got["inputs_unchanged"] is True, f"{want_case['id']}: inputs mutated")
        _require(
            got["baseline_minus_candidate_peak_bytes"]
            == got["baseline"]["one_call_peak_increment_bytes"] - got["candidate"]["one_call_peak_increment_bytes"],
            f"{want_case['id']}: peak difference does not match the measured peaks",
        )
        _require(
            got["memory_difference_is_observation_only"] is True, f"{want_case['id']}: memory gain claimed as a gate"
        )


def _validate_generalization(generalization: Any, fixtures: Any) -> None:
    config = fixtures["generalization"]
    records = {c["id"]: c for c in generalization["cases"]}
    _require(
        set(records) == set(config["case_ids"]) and len(records) == len(generalization["cases"]),
        "generalization case records do not match the mandatory list",
    )
    checks_template = fixtures["expected_checks"]["result.json"]["cases"][0]["checks"]
    for case in fixtures["input_cases"]:
        spec = case.get("generalization")
        if not spec:
            continue
        record = records[case["id"]]
        _require(record["status"] == "PASS", f"generalization case not PASS: {case['id']}")
        for key, value in checks_template.items():
            _require(record["checks"].get(key) == value, f"generalization {case['id']} check {key} is not {value}")
        _require(record["shape"] == spec["output_shape"], f"generalization {case['id']}: wrong output shape")
        _require(record["input_shapes"] == spec["input_shapes"], f"generalization {case['id']}: wrong input shapes")
        if "expected_sha256" in spec:
            _require(
                record["golden_sha256"] == spec["expected_sha256"],
                f"generalization {case['id']}: golden output differs from the frozen expected_sha256",
            )
    measured = {b["id"]: b for b in generalization["benchmarks"]}
    _require(
        set(measured) == set(config["diagnostic_ids"]) and len(measured) == len(generalization["benchmarks"]),
        "generalization paired timings do not match the mandatory list",
    )
    for case_id in config["diagnostic_ids"]:
        record = measured[case_id]
        _require(record["status"] == "MEASURED", f"generalization timing not MEASURED: {case_id}")
        _require(
            len(record["baseline_round_us"]) == len(record["candidate_round_us"]) == 6,
            f"generalization timing {case_id}: six paired rounds are required",
        )


def _validate_records(result: Any, generalization: Any, supplement: Any, fixtures: Any) -> None:
    """Consume every mandatory checklist record; record counts alone never certify a run."""
    expected = fixtures["expected_checks"]["result.json"]
    cases = {c["id"]: c for c in result["cases"]}
    _require(
        len(cases) == len(result["cases"]) == len(expected["cases"]),
        "result.json case records do not match the mandatory list",
    )
    for want in expected["cases"]:
        got = cases.get(want["id"])
        _require(got is not None, f"missing mandatory case record: {want['id']}")
        _require(got["status"] == "PASS", f"case not PASS: {want['id']}")
        for key, value in want["checks"].items():
            _require(got["checks"].get(key) == value, f"case {want['id']} check {key} is not {value}")
        for key in ("shape", "input_shapes", "attributes"):
            _require(got.get(key) == want[key], f"case {want['id']} {key} differs from the fixture")
    negative = {n["id"]: n for n in result["negative"]}
    _require(
        len(negative) == len(result["negative"]) == len(expected["negative"]),
        "negative records do not match the mandatory list",
    )
    for want in expected["negative"]:
        got = negative.get(want["id"])
        _require(got is not None and got["status"] == "PASS", f"negative check not PASS: {want['id']}")
        _require(got["exception"] in ("TypeError", "ValueError"), f"negative {want['id']} raised {got['exception']}")
    _require(result["dynamic"]["status"] == "PASS", "dynamic input/stream record not PASS")
    _validate_supplement(supplement, fixtures["supplement_checks"]["engineering_supplement"], cases)
    summary = result["engineering"]
    for key, value in expected["engineering"].items():
        _require(key in summary, f"engineering summary missing mandatory field: {key}")
        if value == "PASS":
            _require(summary[key] == "PASS", f"engineering summary {key} not PASS")
    _require(
        summary["active_before_bytes"] == summary["active_after_bytes"],
        "engineering summary: active allocation did not return to its starting level",
    )
    _require(
        summary["peak_increment_bytes"] >= summary["output_bytes"] > 0,
        "engineering summary: implausible peak/output byte counts",
    )
    _require(
        isinstance(summary["peak_definition"], str) and summary["peak_definition"],
        "engineering summary: peak_definition missing",
    )
    small = next(c for c in supplement["cases"] if c["id"] == "flags_perf_small")["candidate"]
    _require(
        summary["active_before_bytes"] == small["active_before_bytes"],
        "engineering summary disagrees with the measured small-case record",
    )
    _require(
        summary["active_after_bytes"] == small["active_after_repeat_release_bytes"],
        "engineering summary disagrees with the measured small-case record",
    )
    _require(
        summary["peak_increment_bytes"] == small["one_call_peak_increment_bytes"],
        "engineering summary disagrees with the measured small-case record",
    )
    _require(
        summary["output_bytes"] == small["output_bytes"],
        "engineering summary disagrees with the measured small-case record",
    )
    primary = [c for c in fixtures["input_cases"] if c.get("performance")]
    _require(len(result["benchmarks"]) == len(primary), "primary benchmark count mismatch")
    for index, (filename, path) in enumerate(fixtures["required_comparisons"]):
        _require(filename == "result.json", f"unsupported required comparison file: {filename}")
        match = re.fullmatch(r"benchmarks\[(\d+)\]", path)
        _require(match is not None, f"unsupported required comparison path: {path}")
        record = result["benchmarks"][int(match.group(1))]
        _require(record["id"] == primary[index]["id"], f"required comparison {path} is not {primary[index]['id']}")
        _require(record["status"] == "PASS", f"primary performance gate failed: {record['id']}")
    _validate_generalization(generalization, fixtures)


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
    supplement = {
        "schema_version": 1,
        "task": "flags",
        "kind": "engineering_supplement",
        "status": "NOT_RUN",
        "device_requested": "npu:0",
        "started_utc": _stamp(),
        "cases": [],
        "device_setup_attempted": False,
        "performance_measured": False,
        "scope": SUPPLEMENT_SCOPE,
        "memory_gain_required": False,
        "frozen_results_modified": False,
        "environment": {},
        "finished_utc": None,
    }
    _dump(outdir / "result.json", result)
    code = 1
    with (outdir / "raw.log").open("w", encoding="utf-8") as log:
        try:
            import torch
            import torch_npu  # noqa: F401

            if not torch.npu.is_available():
                raise RuntimeError("No NPU available")
            torch.npu.set_device(0)
            supplement["device_setup_attempted"] = True
            result["environment"] = _environment(torch)
            import triton

            supplement["environment"] = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "device": "npu:0",
                "device_name": result["environment"]["device_name"],
                "versions": {k: result["environment"]["versions"][k] for k in ("torch", "torch-npu", "triton-ascend")},
                "triton_target": str(triton.runtime.driver.active.get_current_target()),
            }
            import solution

            fixtures = json.loads((args.fixtures / "case.json").read_text(encoding="utf-8"))
            result["fixture_sha256"] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.fixtures.iterdir() if p.is_file()
            }
            if args.fixtures == ROOT / "official":
                _require(_fixtures_unmodified(args.fixtures), "frozen task attachments were modified")
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
            verified = {c["id"]: c for c in result["cases"]}
            for case in primary:
                supplement["cases"].append(h.engineering(case, verified[case["id"]]))
            result["engineering"] = _engineering_summary(supplement["cases"])
            for case in primary:
                result["benchmarks"].append(h.benchmark(case, fixtures["acceptance"]["max_mean_us"][case["id"]]))
            for case in cases:
                if case.get("generalization", {}).get("measure"):
                    generalization["benchmarks"].append(h.benchmark(case, None))
            _validate_records(result, generalization, supplement, fixtures)
            generalization["status"] = "PASS"
            supplement["status"] = "PASS"
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
            supplement["finished_utc"] = _stamp()
            _dump(outdir / "result.json", result)
            _dump(outdir / "generalization.json", generalization)
            _dump(outdir / "engineering_supplement.json", supplement)
            print(json.dumps({"results": str(outdir), "status": result["status"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
