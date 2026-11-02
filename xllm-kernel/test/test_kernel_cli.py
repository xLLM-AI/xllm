# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Offline CLI contract tests; CPU doubles do not validate NPU execution."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from xllm_kernel.numerics import verify as verification
from xllm_kernel.numerics.suite import OperationSuite, PreparedCase
from xllm_kernel.registry import KernelRegistry, KernelSpec, OpContract, PreparedKernel

_IMPLEMENTATION = "test.scale"


@dataclass(frozen=True)
class _CliCase:
    """Synthetic tensor case for CLI lifecycle tests."""

    rows: int
    width: int
    dtype: str

    @property
    def case_id(self) -> str:
        return f"test_scale/{self.dtype}/{self.rows}x{self.width}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"rows": self.rows, "width": self.width, "dtype": self.dtype}

    def unsupported_reason(self, plan: PreparedKernel) -> str | None:
        return None

    def prepare(self, plan: PreparedKernel) -> PreparedCase:
        value = torch.ones(self.rows, self.width, dtype=getattr(torch, self.dtype))
        expected = torch.full_like(value, 2)

        def validate(function: Any) -> dict[str, Any]:
            return {"status": "passed" if torch.equal(function(), expected) else "failed"}

        return PreparedCase(lambda: plan.function(value, 2.0), validate, {})


@pytest.fixture
def cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from xllm_kernel.numerics import catalog

    runtime = SimpleNamespace(function=lambda value, scale: value * scale, measurements=[])
    contract = OpContract("test_scale", 1, (), False, "test")
    registry = KernelRegistry()
    registry.register(
        KernelSpec(_IMPLEMENTATION, contract, "npu", ("eager", "aclgraph"), "CPU test double"),
        lambda *args: runtime.function(*args),
    )
    registry.freeze()

    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--rows", type=int, nargs="+", default=[1])
        parser.add_argument("--widths", type=int, nargs="+", default=[4])
        parser.add_argument("--dtypes", nargs="+", choices=["bfloat16", "float16", "float32"], default=["float32"])

    def cases(args: argparse.Namespace) -> tuple[_CliCase, ...]:
        if min(args.widths) <= 0 or min(args.rows) < 0:
            raise ValueError("widths must be positive; rows must be nonnegative")
        return tuple(
            _CliCase(rows, width, dtype) for dtype in args.dtypes for width in args.widths for rows in args.rows
        )

    def prepare(device: str, mode: str, implementation: str) -> PreparedKernel:
        return registry.prepare(contract, device=device, execution_mode=mode, implementation=implementation)

    provider = ModuleType("test_scale_suite")
    provider.SUITE = OperationSuite("test_scale", (_IMPLEMENTATION,), add_arguments, cases, prepare)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    monkeypatch.setattr(catalog, "_BUILTINS", (provider.__name__,))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="test"))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(get_device_name=lambda: "CPU test double", synchronize=lambda: None),
        raising=False,
    )
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
            _IMPLEMENTATION,
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
@pytest.mark.parametrize("status", ["failed", "not_applicable"])
def test_failed_accuracy_or_empty_coverage_cannot_pass(
    module: str,
    status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_runtime: SimpleNamespace,
) -> None:
    cpu_runtime.function = lambda value, *_: torch.zeros_like(value)
    if status == "not_applicable":
        monkeypatch.setattr(_CliCase, "unsupported_reason", lambda self, plan: "outside test domain")
    output = tmp_path / f"{status}.json"
    with pytest.raises(SystemExit, match="1"):
        _invoke(module, output, monkeypatch)
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

    def candidate(value: torch.Tensor, scale: float) -> torch.Tensor:
        output = value * scale
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


def test_mutating_graph_resets_inputs_and_captures_one_call(
    monkeypatch: pytest.MonkeyPatch, cpu_runtime: SimpleNamespace
) -> None:
    from xllm_kernel.benchmark.npu import measure_graph

    value = torch.zeros(1)
    output = torch.zeros(1)
    calls = []
    resets = []

    def run() -> torch.Tensor:
        calls.append(1)
        value.add_(1)
        output.copy_(value)
        return output

    def reset() -> None:
        resets.append(1)
        value.zero_()

    prepared = PreparedCase(
        run,
        lambda function: {"status": "passed" if function().item() == 1 else "failed"},
        {},
        reset,
    )
    monkeypatch.setattr(torch.npu, "NPUGraph", lambda: SimpleNamespace(replay=run), raising=False)
    monkeypatch.setattr(torch.npu, "graph", lambda graph: nullcontext(), raising=False)
    row = {}
    measure_graph(prepared, row, iterations=2, repeats=1)
    assert row["status"] == "passed"
    assert len(calls) == 3  # Capture, validation replay, and the timing double.
    assert len(resets) == 2
    assert cpu_runtime.measurements == [{"iterations": 2, "repeats": 1, "divisor": 1, "reset": reset}]


def test_mutating_wall_time_excludes_event_readback(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm_kernel.benchmark import npu

    clock = SimpleNamespace(now=0)

    def reset() -> None:
        clock.now += 1000

    def run() -> None:
        clock.now += 2000

    def elapsed_time(other: object) -> float:
        clock.now += 1_000_000
        return 0.002

    monkeypatch.setattr(npu, "time", SimpleNamespace(perf_counter_ns=lambda: clock.now))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Event=lambda **kwargs: SimpleNamespace(
                record=lambda: None, synchronize=lambda: None, elapsed_time=elapsed_time
            ),
            synchronize=lambda: None,
        ),
        raising=False,
    )
    timing = npu.measure(run, iterations=2, repeats=1, reset=reset)
    assert timing["samples"][0] == pytest.approx({"device_interval_us": 2.0, "host_enqueue_us": 2.0, "wall_us": 3.0})


@dataclass(frozen=True)
class _AffineCase:
    """A test operation with different argument names and structured outputs."""

    size: int
    scale: float

    @property
    def case_id(self) -> str:
        return f"test_affine/{self.size}/{self.scale}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"size": self.size, "scale": self.scale}

    def unsupported_reason(self, plan: PreparedKernel) -> str | None:
        return None

    def prepare(self, plan: PreparedKernel) -> PreparedCase:
        values = list(range(self.size))
        expected = {"values": tuple(value * self.scale for value in values), "count": self.size}

        def validate(function: Any) -> dict[str, Any]:
            return {"status": "passed" if function() == expected else "failed", "reference": "cpu_affine"}

        return PreparedCase(
            function=lambda: plan.function(values, self.scale),
            validate=validate,
            metadata={"argument_count": 2},
        )


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
@pytest.mark.parametrize("mode", ["eager", "aclgraph"])
def test_another_operation_uses_the_same_cli_validation_and_reports(
    module: str,
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_runtime: SimpleNamespace,
) -> None:
    from xllm_kernel.numerics import catalog

    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--size", type=int, required=True)
        parser.add_argument("--scale", type=float, required=True)

    def cases(args: argparse.Namespace) -> tuple[_AffineCase, ...]:
        return (_AffineCase(args.size, args.scale),)

    def prepare(device: str, mode: str, implementation: str) -> PreparedKernel:
        contract = OpContract("test_affine", 1, (), False, "reference")
        registry = KernelRegistry()
        registry.register(
            KernelSpec("test.affine", contract, device, ("eager", "aclgraph"), "CPU test double"),
            lambda values, scale: {"values": tuple(value * scale for value in values), "count": len(values)},
        )
        registry.freeze()
        return registry.prepare(contract, device=device, execution_mode=mode, implementation=implementation)

    provider = ModuleType("test_affine_suite")
    provider.SUITE = OperationSuite("test_affine", ("test.affine",), add_arguments, cases, prepare)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    monkeypatch.setattr(catalog, "_BUILTINS", (*catalog._BUILTINS, provider.__name__))
    monkeypatch.setattr(torch.npu, "NPUGraph", lambda: SimpleNamespace(replay=lambda: None), raising=False)
    monkeypatch.setattr(torch.npu, "graph", lambda graph: nullcontext(), raising=False)
    output = tmp_path / "affine.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module,
            "--operation",
            "test_affine",
            "--implementation",
            "test.affine",
            "--device",
            "npu",
            "--mode",
            mode,
            "--size",
            "3",
            "--scale",
            "2.5",
            "--warmup",
            "0",
            "--revision",
            "cpu-test",
            "--output",
            str(output),
        ],
    )
    importlib.import_module(f"xllm_kernel.{module}.cli").main()
    report = json.loads(output.read_text())
    row = report["cases"][0]
    assert report["operation"] == row["operation"] == "test_affine"
    assert report["status"] == row["status"] == "passed"
    assert row["case"] == {"size": 3, "scale": 2.5}
    assert row["argument_count"] == 2
    assert ("timing" in row) == (module == "benchmark")
    if module == "benchmark":
        assert cpu_runtime.measurements[0]["divisor"] == (64 if mode == "aclgraph" else 1)
    markdown = output.with_suffix(".md").read_text()
    assert "test_affine" in markdown and "RMSNorm" not in markdown


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
@pytest.mark.parametrize(
    "extra",
    [
        ("--operation", "unknown"),
        ("--implementation", "unknown"),
        ("--widths", "0"),
        ("--rows", "-1"),
        ("--warmup", "-1"),
    ],
)
def test_invalid_arguments_fail_before_runtime_preparation(
    module: str,
    extra: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_runtime: SimpleNamespace,
) -> None:
    output = tmp_path / "invalid.json"
    with pytest.raises(SystemExit, match="2"):
        _invoke(module, output, monkeypatch, *extra)
    assert not output.exists()


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
def test_reports_render_multiple_operations_without_loading_them(module: str, tmp_path: Path) -> None:
    report = {
        "status": "passed",
        "cases": [
            {
                "case_id": "add/1",
                "operation": "add",
                "implementation": "test.add",
                "solution": "torch",
                "variant": "",
                "binding": "torch",
                "status": "passed",
            },
            {
                "case_id": "matmul/2",
                "operation": "matmul",
                "implementation": "test.matmul",
                "solution": "torch",
                "variant": "",
                "binding": "torch",
                "status": "passed",
            },
        ],
    }
    output = tmp_path / "mixed.json"
    importlib.import_module(f"xllm_kernel.{module}.report").write_report(output, report)
    assert json.loads(output.read_text()) == report
    markdown = output.with_suffix(".md").read_text()
    assert "add/1 | add |" in markdown and "matmul/2 | matmul |" in markdown
    assert "RMSNorm" not in markdown


@pytest.mark.parametrize("module", ["numerics", "benchmark"])
@pytest.mark.parametrize("filename", ["report", "report.json"])
def test_distributed_reports_preserve_rank_in_both_formats(
    module: str, filename: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_runtime: SimpleNamespace
) -> None:
    from xllm_kernel.numerics.arguments import parse_arguments

    writer = importlib.import_module(f"xllm_kernel.{module}.report").write_report
    outputs = []
    for rank in range(2):
        monkeypatch.setenv("RANK", str(rank))
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
                _IMPLEMENTATION,
                "--revision",
                "cpu-test",
                "--output",
                str(tmp_path / filename),
                "--distributed",
            ],
        )
        args, _, _ = parse_arguments(argparse.ArgumentParser())
        report = {
            "status": "passed",
            "rank": rank,
            "cases": [
                {
                    "case_id": f"rank{rank}",
                    "operation": "test_scale",
                    "implementation": _IMPLEMENTATION,
                    "solution": "torch",
                    "variant": "",
                    "binding": "torch_npu",
                    "status": "passed",
                }
            ],
        }
        writer(args.output, report)
        outputs.append(args.output)
    assert len(set(outputs)) == 2
    for rank, path in enumerate(outputs):
        assert json.loads(path.read_text())["rank"] == rank
    markdown = list(tmp_path.glob("*.md"))
    assert len(markdown) == 2
    for rank in range(2):
        matching = [path for path in markdown if f"rank{rank}" in path.name]
        assert len(matching) == 1
        assert f"rank{rank}" in matching[0].read_text()
