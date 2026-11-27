# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Task-specified negative tests based on the first nonempty performance input."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch_npu

from benchmark import _inputs, _module
from solution import prepare, run


def _check_invalid(name: str, inputs: list[torch.Tensor], attrs: dict[str, int]) -> dict[str, str]:
    try:
        prepare(inputs, attrs)
    except (TypeError, ValueError) as error:
        return {"id": name, "status": "PASS", "exception": type(error).__name__}
    raise AssertionError("invalid input accepted: " + name)


def _check() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    contract = json.loads((root / "fixtures/case.json").read_text())
    fixture = _module(root / "fixtures/generate_data.py", "contract_fixture")
    golden = _module(root / "fixtures/golden.py", "contract_golden")
    case = next(case for case in contract["input_cases"] if case.get("performance"))
    values, attrs = fixture.generate(case)
    inputs = _inputs(case, values)
    records = [
        _check_invalid("input_count", inputs[:-1], attrs),
        _check_invalid("unknown_attribute", inputs, {"unknown": 1}),
        _check_invalid("cpu_input", [inputs[0].cpu(), *inputs[1:]], attrs),
        _check_invalid("dtype", [inputs[0].to(torch.int64), *inputs[1:]], attrs),
    ]
    noncontiguous = torch.empty((inputs[0].numel() * 2,), dtype=torch.int32, device=inputs[0].device)[::2].reshape(
        inputs[0].shape
    )
    noncontiguous.copy_(inputs[0])
    records.append(_check_invalid("noncontiguous", [noncontiguous, *inputs[1:]], attrs))
    mapping = inputs[1].clone()
    mapping.reshape(-1)[0] = -1
    records.append(_check_invalid("negative_mapping", [inputs[0], mapping, inputs[2]], attrs))
    logical = inputs[0].clone()
    logical.reshape(-1)[0] = 2147483647
    records.append(_check_invalid("logical_address_out_of_range", [logical, *inputs[1:]], attrs))
    cu = inputs[2].clone()
    cu[1] = -1
    records.append(_check_invalid("nonmonotonic_cu", [*inputs[:2], cu], attrs))
    expected = torch.tensor(golden.calculate(values, attrs, False), dtype=torch.int32)
    first, second = run(inputs, attrs), run(inputs, attrs)
    torch.npu.synchronize()
    assert torch.equal(first.cpu(), expected) and torch.equal(second.cpu(), expected)
    assert first.data_ptr() != second.data_ptr()
    return {
        "status": "PASS",
        "base_case": case["id"],
        "negative": records,
        "direct_run": "PASS",
        "direct_run_independent": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "RUNNING"}
    try:
        torch.npu.set_device(0)
        report = _check()
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
