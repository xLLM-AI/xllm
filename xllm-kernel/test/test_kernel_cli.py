# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Offline CLI contract tests; CPU doubles do not validate NPU execution."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from xllm_kernel.numerics import rms_norm
from xllm_kernel.numerics import verify as verification
from xllm_kernel.ops.normalization import RMS_NORM_CONTRACT, xlite
from xllm_kernel.registry import KernelRegistry, KernelSpec


@pytest.fixture
def cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime = SimpleNamespace(function=rms_norm.reference, measurements=[])

    def register(registry: KernelRegistry) -> None:
        registry.register(
            KernelSpec(
                xlite.IMPLEMENTATION,
                RMS_NORM_CONTRACT,
                "npu",
                ("eager", "aclgraph"),
                "CPU test double",
                solution="xlite",
            ),
            runtime.function,
        )

    def inputs(case: rms_norm.NumericsCase) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = getattr(torch, case.dtype)
        return torch.ones(case.rows, case.width, dtype=dtype), torch.ones(case.width, dtype=dtype)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="test", get_npu_format=lambda _: 2))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(get_device_name=lambda: "CPU test double", synchronize=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(xlite, "register", register)
    monkeypatch.setattr(rms_norm, "inputs", inputs)
    monkeypatch.setattr(
        verification, "subprocess", SimpleNamespace(check_output=lambda *args, **kwargs: "test inventory")
    )

    from xllm_kernel.benchmark import npu

    def measure(function: Any, **kwargs: Any) -> dict[str, Any]:
        function()
        runtime.measurements.append(kwargs)
        return {"device_interval_us": {"median": 1.0}}

    monkeypatch.setattr(npu, "measure", measure)
    return runtime


def _invoke(module: str, output: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module,
            "--device",
            "npu",
            "--mode",
            "eager",
            "--implementation",
            xlite.IMPLEMENTATION,
            "--rows",
            "1",
            "--widths",
            "64",
            "--dtypes",
            "bfloat16",
            "--warmup",
            "0",
            "--revision",
            "cpu-test",
            "--output",
            str(output),
            *extra,
        ],
    )
    importlib.import_module(f"xllm_kernel.{module}.cli").main()


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
def test_help_does_not_import_torch_or_host(module: str, tmp_path: Path) -> None:
    script = f"""
import importlib.abc
import runpy
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'torch_npu', 'xllm'}}:
            raise AssertionError(f'unexpected runtime import: {{fullname}}')
sys.meta_path.insert(0, BlockRuntime())
sys.argv = ['xllm_kernel.{module}', '--help']
runpy.run_module('xllm_kernel.{module}', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, text=True, capture_output=True, check=True)
    assert "--implementation" in result.stdout
    assert ("--iterations" in result.stdout) == (module == "benchmark")


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
def test_successful_validation_and_optional_timing(
    module: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_runtime: SimpleNamespace,
) -> None:
    output = tmp_path / "report.json"
    _invoke(module, output, monkeypatch)
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["coverage"]["passed"] == 1
    assert ("timing" in report["cases"][0]) == (module == "benchmark")
    assert len(cpu_runtime.measurements) == int(module == "benchmark")
    assert "passed" in output.with_suffix(".md").read_text()


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
def test_failed_accuracy_or_empty_coverage_cannot_pass(
    module: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_runtime: SimpleNamespace,
) -> None:
    cpu_runtime.function = lambda value, *_: torch.zeros_like(value)
    for name, extra, status in [("wrong", (), "failed"), ("unsupported", ("--dtypes", "float32"), "not_applicable")]:
        output = tmp_path / f"{name}.json"
        with pytest.raises(SystemExit, match="1"):
            _invoke(module, output, monkeypatch, *extra)
        report = json.loads(output.read_text())
        assert report["status"] == "failed"
        assert report["cases"][0]["status"] == status
        assert "timing" not in report["cases"][0]
    assert not cpu_runtime.measurements


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
def test_graph_failure_is_reported_without_eager_fallback(
    module: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_runtime: SimpleNamespace,
) -> None:
    def unsupported_graph() -> None:
        raise RuntimeError("capture unavailable")

    monkeypatch.setattr(torch.npu, "NPUGraph", unsupported_graph, raising=False)
    output = tmp_path / "graph.json"
    with pytest.raises(RuntimeError, match="capture unavailable"):
        _invoke(module, output, monkeypatch, "--mode", "aclgraph")
    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["cases"][0]["status"] == "error"
    assert "capture unavailable" in report["error"]
    assert not cpu_runtime.measurements


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
@pytest.mark.parametrize("corrupt_output", [False, True])
def test_graph_output_is_validated_before_timing(
    module: str,
    corrupt_output: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_runtime: SimpleNamespace,
) -> None:
    outputs = []

    def candidate(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        output = rms_norm.reference(value, weight, eps)
        outputs.append(output)
        return output

    def replay() -> None:
        if corrupt_output:
            outputs[-1].zero_()

    cpu_runtime.function = candidate
    monkeypatch.setattr(torch.npu, "NPUGraph", lambda: SimpleNamespace(replay=replay), raising=False)
    monkeypatch.setattr(torch.npu, "graph", lambda graph: nullcontext(), raising=False)
    output = tmp_path / "graph.json"
    if corrupt_output:
        with pytest.raises(SystemExit, match="1"):
            _invoke(module, output, monkeypatch, "--mode", "aclgraph")
    else:
        _invoke(module, output, monkeypatch, "--mode", "aclgraph")
    report = json.loads(output.read_text())
    expected = "failed" if corrupt_output else "passed"
    assert report["status"] == report["cases"][0]["graph_accuracy"]["status"] == expected
    if module == "benchmark" and not corrupt_output:
        assert cpu_runtime.measurements[0]["divisor"] == 64
    else:
        assert not cpu_runtime.measurements
