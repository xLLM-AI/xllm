# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Shared validation lifecycle; operation suites own inputs and correctness."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xllm_kernel.numerics.suite import KernelCase, OperationSuite, PreparedCase
from xllm_kernel.registry import PreparedKernel


@contextmanager
def validation_session(
    *,
    suite: OperationSuite,
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
        "operation": suite.operation,
        "status": "not_run",
        "cases": [],
        "revision": revision,
        "mode": mode,
    }
    try:
        import torch
        import torch_npu

        report["environment"] = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "platform": platform.platform(),
        }
        prepare_start = time.perf_counter()
        plan = suite.prepare(device, mode, implementation)
        if plan.spec.contract.operation != suite.operation:
            raise ValueError("prepared implementation does not match the operation suite")
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


def verify_case(
    plan: PreparedKernel,
    case: KernelCase,
    report: dict[str, Any],
) -> tuple[dict[str, Any], PreparedCase | None]:
    """Record eligibility, then validate fresh inputs before any measurement."""
    row = {
        "case_id": case.case_id,
        "operation": plan.spec.contract.operation,
        "implementation": plan.spec.name,
        "case": case.parameters,
        "status": "not_run",
    }
    report["cases"].append(row)
    reason = case.unsupported_reason(plan)
    if reason is not None:
        row.update(status="not_applicable", reason=reason)
        return row, None
    prepared = case.prepare(plan)
    row.update(prepared.metadata)
    row.update(prepared.validate(prepared.function))
    return row, prepared
