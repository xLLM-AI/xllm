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

"""Required NPU coverage beyond the immutable public verify.py.

Every result is keyed by scenario and entry. A missing NPU is not a pass.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[5]), str(ROOT), str(ROOT / "reference")]
from scripts.logger import logger


def _execute(report: Any, args: Any) -> Any:
    import generate_data
    import golden
    import numpy as np
    import torch
    import torch_npu
    import verify

    import solution

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; required device checks were not executed")
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    cases = json.loads((ROOT / "reference/case.json").read_text(encoding="utf-8"))
    req = json.loads((ROOT / "reference/coverage_requirements.json").read_text(encoding="utf-8"))
    case_map = {c["case_name"]: c for c in cases}
    perf = [c for c in cases if c.get("performance")]
    entries = {"run": solution.run, "run_direct": solution.run_direct}

    def _record(name: Any, entry: Any, **data: Any) -> Any:
        item = {"case_name": name, "entry": entry, "status": "passed", **data}
        report["records"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def _attrs(case: Any) -> Any:
        return {a["name"]: a["value"] for a in case["attr_desc"]}

    def _make(case: Any, offset: Any = False) -> Any:
        arrays = generate_data.generate(case)
        inputs = verify.device_inputs(case, arrays, device)
        if offset:
            adjusted = []
            for tensor in inputs:
                storage = torch.empty(tensor.numel() + 16, dtype=tensor.dtype, device=tensor.device)
                view = storage[1 : 1 + tensor.numel()].view(tensor.shape)
                view.copy_(tensor)
                adjusted.append(view)
            inputs = adjusted
        return arrays, inputs

    def _check(case: Any, arrays: Any, inputs: Any, outputs: Any) -> Any:
        torch.npu.synchronize()
        verify.validate_ownership(inputs, outputs)
        checks = verify.compare(case, arrays, outputs, golden.calculate(arrays, _attrs(case)))
        for tensor, original in zip(inputs, arrays):
            if not np.array_equal(verify.cpu_array(tensor).view(np.uint16), original.view(np.uint16)):
                raise AssertionError("input changed")
        return checks

    def _expect_reject(fn: Any) -> Any:
        # Only an input-validation exception counts; unavailable extensions and
        # loader/runtime failures must never masquerade as rejected metadata.
        try:
            fn()
        except (ValueError, TypeError) as exc:
            return str(exc)
        except RuntimeError as exc:
            message = str(exc)
            markers = (
                "expected qkv",
                "inputs must",
                "FP16 or BF16",
                "input dtype",
                "same device",
                "contiguous ND",
                "input rank",
                "cache shape",
                "shape range",
                "packed QKV",
                "attribute/shape",
            )
            if not any(x in message for x in markers):
                raise
            return message
        raise AssertionError("invalid input was accepted")

    # Load the extension and prove one valid launch before any negative test.
    arrays, inputs = _make(cases[0])
    _check(cases[0], arrays, inputs, solution.run(inputs, _attrs(cases[0])))
    report["npu_executed"] = True
    del arrays, inputs

    if args.group in ("all", "accuracy"):
        for case in cases:
            arrays, inputs = _make(case)
            outputs = solution.run_direct(inputs, _attrs(case))
            _record(case["case_name"], "run_direct", outputs=_check(case, arrays, inputs, outputs))
            del outputs, arrays, inputs
        for name in req["extra_input_case_ids"]:
            case = copy.deepcopy(case_map[name.split("__extra_seed_")[0]])
            case.update(case_name=name, kind="random", performance=False)
            arrays, inputs = _make(case, offset=True)
            for entry, fn in entries.items():
                outputs = fn(inputs, _attrs(case))
                _record(name, entry, storage_offset=1, outputs=_check(case, arrays, inputs, outputs))
                del outputs
            del arrays, inputs
        # An odd large Q size makes the V-input cache-policy boundary unaligned.
        # Keep this outside the frozen public/performance case definitions.
        for template_id in ("T5SplitQkvAppendCache_007_float16_random", "T5SplitQkvAppendCache_017_bfloat16_random"):
            case = copy.deepcopy(case_map[template_id])
            b, h, cache_length, d = 15, 17, 2048, 63
            name = "cache_policy_odd_boundary_" + case["input_desc"][0]["data_type"]
            case.update(case_name=name, kind="random", performance=False)
            case["input_desc"][0]["shape"] = [b, 1, 3 * h * d]
            for desc in case["input_desc"][1:]:
                desc["shape"] = [b, h, cache_length, d]
            case["output_desc"][0]["shape"] = [b, h, 1, d]
            for desc in case["output_desc"][1:]:
                desc["shape"] = [b, h, cache_length + 1, d]
            for attr in case["attr_desc"]:
                attr["value"] = {"headNum": h, "headDim": d}[attr["name"]]
            arrays, inputs = _make(case, offset=True)
            for entry, fn in entries.items():
                outputs = fn(inputs, _attrs(case))
                _record(name, entry, storage_offset=1, outputs=_check(case, arrays, inputs, outputs))
                del outputs
            del arrays, inputs
        base = perf[0]
        arrays, valid = _make(base)
        # All three entrances validate the exact prescribed 28 scenarios.
        for name in req["negative_scenarios"]:
            inputs = list(valid)
            attributes = _attrs(base)
            if name == "attributes_type":
                attributes = []
            elif name == "cache_dtype":
                inputs[1] = inputs[1].float()
            elif name == "cache_mismatch":
                inputs[2] = torch.empty((4, 8, 1, 64), dtype=valid[0].dtype, device=device)
            elif name == "cache_rank":
                inputs[1] = torch.empty((2, 3), dtype=valid[0].dtype, device=device)
            elif name == "cpu_input":
                inputs[0] = inputs[0].cpu()
            elif name == "float32":
                inputs[0] = inputs[0].float()
            elif name.startswith(("headDim_", "headNum_")):
                key, value = name.split("_", 1)
                attributes[key] = ast.literal_eval(value)
            elif name == "input_count":
                inputs.pop()
            elif name == "noncontiguous":
                inputs[0] = torch.empty((4, 1, 3072), dtype=valid[0].dtype, device=device)[..., ::2]
            elif name == "packed_mismatch":
                inputs[0] = torch.empty((4, 1, 1537), dtype=valid[0].dtype, device=device)
            elif name == "packed_tokens":
                inputs[0] = torch.empty((4, 2, 1536), dtype=valid[0].dtype, device=device)
            elif name.startswith("shape_limit_"):
                b, h, cache_length, d = ast.literal_eval(name[len("shape_limit_") :])
                attributes = {}
                inputs = [
                    torch.empty(s, dtype=valid[0].dtype, device=device)
                    for s in ((b, 1, 3 * h * d), (b, h, cache_length, d), (b, h, cache_length, d))
                ]
            elif name == "unknown_attribute":
                attributes["unknown"] = 1
            else:
                raise AssertionError(f"Unimplemented prescribed scenario: {name}")
            for entry in req["negative_entries"]:
                message = _expect_reject(
                    lambda entry=entry, inputs=inputs, attributes=attributes: getattr(solution, entry)(
                        inputs, attributes
                    )
                )
                _record(name, entry, rejection=message)
        del arrays, valid, inputs

        # Verifier counterexamples use the original verification functions.
        case = case_map[req["stream_case_ids"][0]]
        arrays, inputs = _make(case)
        good = solution.run(inputs, _attrs(case))
        torch.npu.synchronize()
        cpu = tuple(t.cpu() for t in good)
        try:
            verify.compare(case, arrays, cpu, golden.calculate(arrays, _attrs(case)))
        except AssertionError:
            _record("CPU output rejected", "verifier")
        else:
            raise AssertionError("verifier accepted CPU output")
        try:
            verify.validate_ownership(inputs, (inputs[0], good[1], good[2]))
        except AssertionError:
            _record("input storage reuse rejected", "verifier")
        else:
            raise AssertionError("verifier accepted input storage")
        backing = torch.empty(good[0].numel() * 3 + 16, dtype=good[0].dtype, device=device)
        overlap = tuple(backing[i : i + good[0].numel()].view(good[0].shape) for i in range(3))
        assert len({t.data_ptr() for t in overlap}) == 3
        try:
            verify.validate_ownership(inputs, overlap)
        except AssertionError:
            _record("offset overlapping storage rejected", "verifier")
        else:
            raise AssertionError("verifier accepted overlapping storage")
        del cpu, good, backing, overlap, arrays, inputs

    if args.group in ("all", "engineering"):
        # An ACLNN metadata plan must refresh input and output addresses even
        # when shape/dtype/stream stay unchanged. Keep all old results alive.
        for case in (perf[1], perf[6]):
            seed_arrays, seed_inputs = _make(case)
            del seed_inputs
            retained = []
            for iteration in range(8):
                arrays = tuple(a.copy() for a in seed_arrays)
                for array in arrays:
                    bits = array.view(np.uint16)
                    np.bitwise_xor(bits, np.uint16(iteration + 1), out=bits)
                inputs = verify.device_inputs(case, arrays, device)
                outputs = solution.run(inputs, _attrs(case))
                retained.append((arrays, inputs, outputs))
            torch.npu.synchronize()
            for arrays, inputs, outputs in retained:
                _check(case, arrays, inputs, outputs)
            _record(
                "executor_address_refresh_" + case["case_name"],
                "run",
                calls=8,
                changed_inputs=True,
                retained_output_groups=len(retained),
            )
            del seed_arrays, retained, arrays, inputs, outputs, array, bits

        streams = [torch.npu.Stream(), torch.npu.Stream()]
        pending = []
        for stream, name in zip(streams, req["stream_case_ids"]):
            case = case_map[name]
            with torch.npu.stream(stream):
                arrays, inputs = _make(case)
                prepared = solution.prepare(inputs, _attrs(case))
                outputs = [prepared(inputs) for _ in range(req["min_prepared_calls_per_stream"])]
                pending.append((case, arrays, inputs, outputs))
        for stream, (case, arrays, inputs, groups) in zip(streams, pending):
            stream.synchronize()
            for outputs in groups:
                _check(case, arrays, inputs, outputs)
            _record(case["case_name"], "prepared_stream", calls=len(groups), stream_id=int(stream.npu_stream))
        del pending, arrays, inputs, outputs, groups, prepared

        # Prepare metadata rejects changed rank, dtype, shape, layout and attributes.
        case = perf[0]
        arrays, inputs = _make(case)
        prepared = solution.prepare(inputs, _attrs(case))
        variants = [
            (inputs[0].float(), inputs[1], inputs[2]),
            (inputs[0].reshape(-1), inputs[1], inputs[2]),
            (inputs[0][..., :-1], inputs[1], inputs[2]),
            (inputs[0], inputs[1], torch.empty((4, 8, 1, 64), dtype=inputs[0].dtype, device=device)),
            (torch.empty((4, 1, 3072), dtype=inputs[0].dtype, device=device)[..., ::2], inputs[1], inputs[2]),
        ]
        for index, variant in enumerate(variants):
            _record(
                f"prepared_metadata_{index}",
                "prepare",
                rejection=_expect_reject(lambda variant=variant: prepared(variant)),
            )
        del variants, variant, prepared, arrays, inputs

        case = next(c for c in cases if c["input_desc"][1]["shape"][2] == 31)
        for entry, fn in entries.items():
            arrays, inputs = _make(case)
            first = fn(inputs, _attrs(case))
            torch.npu.synchronize()
            second_inputs = (inputs[0], first[1], first[2])
            second = fn(second_inputs, _attrs(case))
            second_arrays = (arrays[0], *golden.calculate(arrays, _attrs(case))[1:])
            second_case = copy.deepcopy(case)
            for desc in second_case["output_desc"][1:]:
                desc["shape"][2] += 1
            _record("sequential_cache_append", entry, outputs=_check(second_case, second_arrays, second_inputs, second))
            del arrays, inputs, first, second_inputs, second, second_arrays

        for entry, fn in entries.items():
            case = perf[0]
            arrays, inputs = _make(case)
            held = [fn(inputs, _attrs(case)) for _ in range(8)]
            for _ in range(32):
                tmp = fn(inputs, _attrs(case))
                del tmp
                pressure = [torch.empty_like(t).fill_(7) for t in held[0]]
                del pressure
            torch.npu.synchronize()
            storages = [t.untyped_storage().data_ptr() for group in held for t in group]
            assert len(storages) == len(set(storages))
            for outputs in held:
                _check(case, arrays, inputs, outputs)
            _record("output_lifetime", entry, held_groups=8, early_release_calls=32)
            del held, outputs, arrays, inputs

            # Inputs produced on a different stream, released immediately after
            # enqueue; producer stream then reuses allocator storage aggressively.
            case = case_map[req["stream_case_ids"][1]]
            producer, consumer = streams
            with torch.npu.stream(producer):
                arrays, inputs = _make(case)
            consumer.wait_stream(producer)
            with torch.npu.stream(consumer):
                outputs = fn(inputs, _attrs(case))
            del inputs
            with torch.npu.stream(producer):
                for _ in range(32):
                    pressure = torch.empty((2**20,), dtype=torch.int16, device=device).fill_(0x1234)
                    del pressure
            consumer.synchronize()
            producer.synchronize()
            checks = verify.compare(case, arrays, outputs, golden.calculate(arrays, _attrs(case)))
            _record("input_release_cross_stream", entry, pressure_allocations=32, outputs=checks)
            del outputs, arrays

    if args.group in ("all", "memory"):
        for case in perf:
            arrays, inputs = _make(case)
            for entry, fn in entries.items():
                for _ in range(50):
                    fn(inputs, _attrs(case))
                torch.npu.synchronize()
                gc.collect()
                before = torch.npu.memory_stats()["active_bytes.all.current"]
                torch.npu.reset_peak_memory_stats()
                outputs = fn(inputs, _attrs(case))
                torch.npu.synchronize()
                peak = torch.npu.memory_stats()["active_bytes.all.peak"]
                output_bytes = sum(t.numel() * t.element_size() for t in outputs)
                checks = _check(case, arrays, inputs, outputs)
                assert peak - before <= output_bytes + 16384, (peak, before, output_bytes)
                del outputs
                gc.collect()
                torch.npu.synchronize()
                reclaimed = []
                for _ in range(20):
                    outputs = fn(inputs, _attrs(case))
                    del outputs
                    torch.npu.synchronize()
                    gc.collect()
                    reclaimed.append(torch.npu.memory_stats()["active_bytes.all.current"])
                assert all(x == before for x in reclaimed), (before, reclaimed)
                _record(
                    case["case_name"],
                    entry,
                    metric="active_tensor_bytes",
                    before=before,
                    peak_increment=peak - before,
                    output_bytes=output_bytes,
                    workspace_bytes=0,
                    allocation_allowance_bytes=16384,
                    after_each_release=reclaimed,
                    outputs=checks,
                )
            del arrays, inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--group", choices=["all", "accuracy", "engineering", "memory"], default="all")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "npu_executed": False, "group": args.group, "records": []}
    args.report.write_text(json.dumps(report), encoding="utf-8")
    code = 1
    try:
        _execute(report, args)
        report["status"] = "passed"
        code = 0
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logger.exception("Validation failed")
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
