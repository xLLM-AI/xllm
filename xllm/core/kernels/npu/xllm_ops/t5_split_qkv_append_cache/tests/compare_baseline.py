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

"""Supplemental same-device baseline ratios; does not replace fixed limits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[5]), str(ROOT), str(ROOT / "reference")]
from scripts.logger import logger


def _execute(report: Any) -> Any:
    import benchmark
    import generate_data
    import golden
    import numpy as np
    import torch
    import torch_npu
    import verify

    import solution

    torch.npu.set_device(0)
    cases = json.loads((ROOT / "reference/case.json").read_text(encoding="utf-8"))
    for case in (c for c in cases if c.get("performance")):
        arrays = generate_data.generate(case)
        attributes = {a["name"]: a["value"] for a in case["attr_desc"]}
        inputs = verify.device_inputs(case, arrays, "npu:0")
        b, h, cache_length, d = inputs[1].shape

        def _baseline(inputs: Any = inputs, b: int = b, h: int = h, d: int = d) -> Any:
            parts = inputs[0].reshape(b, 3, h, d)
            q = parts[:, 0].reshape(b, h, 1, d).clone(memory_format=torch.contiguous_format)
            k = parts[:, 1].reshape(b, h, 1, d)
            v = parts[:, 2].reshape(b, h, 1, d)
            return q, torch.cat((inputs[1], k), dim=2), torch.cat((inputs[2], v), dim=2)

        functions = {
            "views_concat_q_copy": _baseline,
            "custom": lambda inputs=inputs, attributes=attributes: solution.run(inputs, attributes),
        }
        results = {}
        for name, fn in functions.items():
            outputs = fn()
            torch.npu.synchronize()
            verify.validate_ownership(inputs, outputs)
            verify.compare(case, arrays, outputs, golden.calculate(arrays, attributes))
            for tensor, original in zip(inputs, arrays):
                assert np.array_equal(verify.cpu_array(tensor).view(np.uint16), original.view(np.uint16))
            del outputs
            results[name] = benchmark.measure(fn, torch.npu)
        record = {
            "case_name": case["case_name"],
            "results": results,
            "selected_baseline": "views_concat_q_copy",
            "excluded_candidates": [],
            "custom_over_baseline": results["custom"]["wall_us_median"]
            / results["views_concat_q_copy"]["wall_us_median"],
        }
        report["cases"].append(record)
        report["npu_executed"] = True
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    report = {
        "status": "running",
        "npu_executed": False,
        "protocol": {"warmup": 50, "iterations": 200, "rounds": 5},
        "ratio_acceptance": "undefined in inconsistent task prose; report measurements only",
        "cases": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report), encoding="utf-8")
    code = 1
    try:
        _execute(report)
        report["status"] = "passed"
        code = 0
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        logger.exception("Validation failed")
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(code)
