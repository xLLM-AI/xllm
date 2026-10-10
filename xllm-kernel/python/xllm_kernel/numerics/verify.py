# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Shared case validation and report lifecycle for offline kernel tools."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xllm_kernel.registry import PreparedKernel

if TYPE_CHECKING:
    import torch

    from xllm_kernel.numerics.rms_norm import NumericsCase


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# RMSNorm validation",
        "",
        f"Status: {report['status']}",
        "",
        "| Case | Implementation | Status |",
        "| --- | --- | --- |",
    ]
    for row in report["cases"]:
        lines.append(f"| {row['case_id']} | {row['implementation']} | {row['status']} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


@contextmanager
def validation_session(
    *,
    device: str,
    mode: str,
    implementation: str,
    revision: str,
    output: Path,
    report_writer: Callable[[Path, dict[str, Any]], None],
) -> Iterator[tuple[PreparedKernel, dict[str, Any]]]:
    """Prepare one implementation and preserve failures in the final report."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "not_run",
        "cases": [],
        "revision": revision,
        "mode": mode,
    }
    try:
        import torch
        import torch_npu

        from xllm_kernel.ops.normalization import RMS_NORM_CONTRACT
        from xllm_kernel.registry import KernelRegistry

        report["environment"] = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "platform": platform.platform(),
        }
        prepare_start = time.perf_counter()
        if implementation == "npu.xlite.rms_norm":
            from xllm_kernel.ops.normalization.xlite import register
        else:
            from xllm import xllm_export  # noqa: F401
            from xllm_kernel.ops.normalization.npu import register
        registry = KernelRegistry()
        register(registry)
        registry.freeze()
        plan = registry.prepare(RMS_NORM_CONTRACT, device=device, execution_mode=mode, implementation=implementation)
        report["preparation_s"] = time.perf_counter() - prepare_start
        report["selection"] = asdict(plan.spec)
        report["environment"]["device_name"] = torch.npu.get_device_name()
        cann_version = (
            Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")) / "version.cfg"
        )
        report["environment"]["cann_version"] = cann_version.read_text() if cann_version.is_file() else "unavailable"
        report["environment"]["npu_smi"] = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
        yield plan, report
        report["coverage"] = {
            state: sum(row["status"] == state for row in report["cases"])
            for state in ("passed", "failed", "not_applicable", "not_run")
        }
        report["status"] = (
            "passed"
            if report["coverage"]["passed"]
            and all(row["status"] in ("passed", "not_applicable") for row in report["cases"])
            else "failed"
        )
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        if report["cases"]:
            report["cases"][-1].update(status="error", error=report["error"])
        raise
    finally:
        report_writer(output, report)
    if report["status"] != "passed":
        raise SystemExit(1)


def generate_cases(rows: list[int], widths: list[int], dtypes: list[str]) -> Iterator[NumericsCase]:
    from xllm_kernel.numerics.rms_norm import NumericsCase

    for dtype, width, row_count in product(dtypes, widths, rows):
        yield NumericsCase(row_count, width, dtype)


def verify_case(
    plan: PreparedKernel,
    case: NumericsCase,
    report: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    """Record eligibility, then validate fresh inputs before any measurement."""
    import torch_npu

    from xllm_kernel.numerics.rms_norm import inputs, verify

    row = {"case_id": case.case_id, "implementation": plan.spec.name, "case": asdict(case), "status": "not_run"}
    report["cases"].append(row)
    if plan.spec.solution == "xlite" and (
        case.dtype == "float32" or case.width % 64 or case.width > 8192 or case.rows * case.width > 2**32 - 1
    ):
        row.update(status="not_applicable", reason="outside declared xlite RMSNorm tensor domain")
        return row, None, None
    value, weight = inputs(case)
    row["input_stride"] = list(value.stride())
    row["weight_stride"] = list(weight.stride())
    row["npu_format"] = torch_npu.get_npu_format(value)
    row.update(verify(plan.function, case, value, weight))
    return row, value, weight
