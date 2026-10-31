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

"""Device-independent package, registry, and wheel checks."""

from __future__ import annotations

import importlib
import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import setuptools
from setuptools import Distribution
from setuptools.command.build_py import build_py
from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT
from xllm_kernel.registry import KernelRegistry, KernelSpec, register_kernel

_ROOT = Path(__file__).resolve().parents[2]


def _identity(value: object) -> object:
    return value


def _spec(name: str = "npu.native") -> KernelSpec:
    return KernelSpec(name, RMS_NORM_CONTRACT, "npu", ("eager", "aclgraph"), "native tensor domain")


@pytest.fixture
def _registry(monkeypatch: pytest.MonkeyPatch) -> KernelRegistry:
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    return registry


@pytest.fixture
def _backend_registry(monkeypatch: pytest.MonkeyPatch, _registry: KernelRegistry) -> Iterator[KernelRegistry]:
    rms_norm = importlib.import_module("xllm_kernel.ops.rms_norm")

    runtime = importlib.import_module("xllm_kernel.runtime")
    monkeypatch.setattr(runtime, "_prepared", None)
    monkeypatch.setattr(runtime, "_configuration", None)
    monkeypatch.setattr(rms_norm.OPERATION, "prepared", None)
    monkeypatch.setattr(runtime, "find_spec", lambda name: object() if name == "torch_npu" else None)
    yield _registry
    for backend in ("aclnn_xllm", "aclnn_torch_npu", "xlite"):
        sys.modules.pop(f"xllm_kernel.ops.rms_norm.{backend}", None)
        vars(rms_norm).pop(backend, None)


def test_decorator_preserves_callable_and_declares_selection_metadata(_registry: KernelRegistry) -> None:
    registry = KernelRegistry.get()
    assert registry is _registry
    decorated = register_kernel(
        name="npu.decorated",
        contract=RMS_NORM_CONTRACT,
        device="npu",
        execution_modes=("aclgraph",),
        input_domain="test tensor domain",
        solution="test",
        priority=7,
    )(_identity)
    assert decorated is _identity
    registry.register(_spec("npu.other"), lambda value: None)
    registry.freeze()
    plan = registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="aclgraph")
    assert plan.function is _identity
    assert plan.spec == KernelSpec(
        "npu.decorated", RMS_NORM_CONTRACT, "npu", ("aclgraph",), "test tensor domain", 7, "test"
    )
    assert plan.function(42) == 42
    with pytest.raises(ValueError, match="does not support execution mode eager"):
        registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="eager", implementation="npu.decorated")


@pytest.mark.parametrize("frozen", [False, True])
def test_decorator_obeys_registry_checks_when_applied(frozen: bool, _registry: KernelRegistry) -> None:
    registry = KernelRegistry.get()
    decorate = register_kernel(
        name="npu.native",
        contract=RMS_NORM_CONTRACT,
        device="npu",
        execution_modes=("eager", "aclgraph"),
        input_domain="test tensor domain",
        solution="native",
    )
    if frozen:
        registry.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            decorate(_identity)
    else:
        registry.register(_spec(), _identity)
        with pytest.raises(ValueError, match="already registered"):
            decorate(_identity)


def test_decorator_preserves_callable_instances(_registry: KernelRegistry) -> None:
    class _Offset:
        def __call__(self, value: int) -> int:
            return value + 1

    kernel = _Offset()
    decorated = register_kernel(
        name="callable_instance",
        contract=RMS_NORM_CONTRACT,
        device="npu",
        execution_modes=("eager",),
        input_domain="test tensor domain",
        solution="test",
    )(kernel)
    assert decorated is kernel
    _registry.freeze()
    plan = _registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="eager")
    assert plan.function(10) == 11
    independent = KernelRegistry()
    independent.freeze()
    with pytest.raises(ValueError, match="registry is empty"):
        independent.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="eager")


def test_registry_requires_freeze_and_rejects_late_or_duplicate_registration() -> None:
    registry = KernelRegistry()
    registry.register(_spec(), _identity)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_spec(), _identity)
    with pytest.raises(RuntimeError, match="freeze"):
        registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="eager")
    registry.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(_spec("another"), _identity)


def test_priority_and_stable_name_selection() -> None:
    registry = KernelRegistry()
    for name, priority in (("z", 5), ("a", 5), ("low", 0)):
        registry.register(replace(_spec(name), priority=priority), _identity)
    registry.freeze()
    plan = registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="aclgraph")
    assert plan.spec.name == "a"
    assert plan.generation == 3
    assert plan.function(42) == 42
    explicit = registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="eager", implementation="low")
    assert explicit.spec.name == "low"
    assert len(explicit.excluded) == 2


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"device": "cuda"}, "requires device cuda"),
        ({"execution_modes": ("eager",)}, "does not support execution mode aclgraph"),
        ({"contract": replace(RMS_NORM_CONTRACT, version=2)}, "operation contract mismatch"),
        ({"contract": replace(RMS_NORM_CONTRACT, aliases_inputs=True)}, "operation contract mismatch"),
    ],
)
def test_explicit_override_cannot_bypass_contract(change: dict, reason: str) -> None:
    registry = KernelRegistry()
    registry.register(replace(_spec(), **change), _identity)
    registry.freeze()
    with pytest.raises(ValueError, match=reason):
        registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="aclgraph", implementation="npu.native")


def test_unknown_override_does_not_fall_back() -> None:
    registry = KernelRegistry()
    registry.register(_spec(), _identity)
    registry.freeze()
    with pytest.raises(ValueError, match="explicit implementation is missing"):
        registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="eager", implementation="missing")


def test_import_without_torch_or_device_sdk(tmp_path: Path) -> None:
    script = """
import importlib.abc
import sys
class BlockDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'xllm'}:
            raise AssertionError(f'unexpected import: {fullname}')
sys.meta_path.insert(0, BlockDeviceImports())
import xllm_kernel
import xllm_kernel.ops
import xllm_kernel.custom_ops
from xllm_kernel import register_kernel
from xllm_kernel.registry import KernelRegistry, OpContract
registry = KernelRegistry.get()
@register_kernel(
    name='import_only', contract=OpContract('test', 1, (), False, 'test'),
    device='cpu', execution_modes=('eager',), input_domain='test', solution='test',
)
def identity(value: int) -> int:
    return value
registry.freeze()
assert registry.prepare(
    OpContract('test', 1, (), False, 'test'), device='cpu', execution_mode='eager',
).function(42) == 42
try:
    xllm_kernel.prepare_rms_norm()
except RuntimeError as error:
    assert 'initialize' in str(error)
else:
    raise AssertionError('uninitialized runtime accepted')
try:
    xllm_kernel.initialize(device='cuda')
except ValueError as error:
    assert 'adapter' in str(error)
else:
    raise AssertionError('unsupported platform accepted')
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


def test_native_module_import_registers_once_and_returns_normalized_output(
    monkeypatch: pytest.MonkeyPatch, _backend_registry: KernelRegistry
) -> None:
    calls = []
    value, weight, normalized, rstd = (object() for _ in range(4))

    def native_rms_norm(input: object, weight: object, *, epsilon: float) -> tuple[object, object]:
        calls.append((input, weight, epsilon))
        return normalized, rstd

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=native_rms_norm))
    module = importlib.import_module("xllm_kernel.ops.rms_norm.aclnn_torch_npu")
    assert importlib.import_module(module.__name__) is module
    _backend_registry.freeze()
    plan = _backend_registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="aclgraph")
    assert plan.function is module.rms_norm
    assert plan.generation == 1
    assert plan.function(value, weight, 0.125) is normalized
    assert calls == [(value, weight, 0.125)]


def test_native_missing_operator_fails_before_registration(
    monkeypatch: pytest.MonkeyPatch, _backend_registry: KernelRegistry
) -> None:
    from xllm_kernel import initialize

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    with pytest.raises(AttributeError, match="npu_rms_norm"):
        initialize(device="npu", operations=("rms_norm",))
    _backend_registry.freeze()
    with pytest.raises(ValueError, match="registry is empty"):
        _backend_registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="eager")


def test_native_execution_error_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch, _backend_registry: KernelRegistry
) -> None:
    from xllm_kernel import initialize

    calls = []

    def fail(input: object, weight: object, *, epsilon: float) -> tuple[object, object]:
        calls.append(epsilon)
        raise RuntimeError("native execution failed")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=fail))
    plan = initialize(device="npu", operations=("rms_norm",))["rms_norm"]
    with pytest.raises(RuntimeError, match="native execution failed"):
        plan.function(object(), object(), 1e-5)
    assert calls == [1e-5]


def test_initialize_selects_highest_priority_and_reuses_frozen_binding(
    monkeypatch: pytest.MonkeyPatch, _backend_registry: KernelRegistry
) -> None:
    from xllm_kernel import initialize

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=lambda *args, **kwargs: (42, 0)))
    _backend_registry.register(replace(_spec("npu.other"), priority=99), _identity)
    plan = initialize(device="npu", operations=("rms_norm",))["rms_norm"]
    assert plan.spec.name == "npu.other"
    assert initialize(device="npu", operations=("rms_norm",))["rms_norm"] is plan
    assert "xllm_kernel.ops.rms_norm.xlite" not in sys.modules
    with pytest.raises(RuntimeError, match="frozen"):
        _backend_registry.register(_spec("late"), _identity)
    with pytest.raises(RuntimeError, match="restart the worker"):
        initialize(device="npu", operations=("rms_norm",), overrides={"rms_norm": "xlite"})


@pytest.mark.parametrize("operation", ["missing", ""])
def test_unknown_initialization_operation_fails_without_loading_backend(
    operation: str, _backend_registry: KernelRegistry
) -> None:
    from xllm_kernel import initialize

    with pytest.raises(ValueError, match="unknown kernel override operation"):
        initialize(device="npu", operations=("rms_norm",), overrides={operation: "native"})
    assert "xllm_kernel.ops.rms_norm.aclnn_torch_npu" not in sys.modules
    assert "xllm_kernel.ops.rms_norm.xlite" not in sys.modules


@pytest.mark.parametrize("layout", ["source", "source_with_installed_package", "installed"])
def test_host_kernel_import_without_site_initialization(tmp_path: Path, layout: str) -> None:
    installed_root = tmp_path / "site-packages"
    installed_root.mkdir()
    if layout == "installed":
        (installed_root / "xllm/python").mkdir(parents=True)
        for relative in ("xllm/__init__.py", "xllm/python/__init__.py"):
            shutil.copy2(_ROOT / relative, installed_root / relative)
        shutil.copytree(
            _ROOT / "xllm-kernel/python/xllm_kernel",
            installed_root / "xllm_kernel",
            ignore=shutil.ignore_patterns("__pycache__", "*.so", "_xlite_build.json"),
        )
        python_path = str(installed_root)
        expected = installed_root / "xllm_kernel/__init__.py"
    else:
        python_path = str(_ROOT)
        expected = _ROOT / "xllm-kernel/python/xllm_kernel/__init__.py"
        if layout == "source_with_installed_package":
            (installed_root / "xllm_kernel").mkdir()
            (installed_root / "xllm_kernel/__init__.py").write_text(
                "raise AssertionError('loaded an unrelated installed kernel package')\n"
            )
            python_path += os.pathsep + str(installed_root)

    # A checkout-shaped working directory must not affect installed packages.
    (tmp_path / "setup.py").touch()
    (tmp_path / "CMakeLists.txt").touch()
    script = """
import importlib.util
import sys
from pathlib import Path

original_path = sys.path.copy()
import xllm.python as runtime
assert sys.path == original_path
assert 'xllm_kernel' not in sys.modules
if sys.argv[1] == 'source':
    assert importlib.util.find_spec('xllm_kernel') is None
package = runtime._load_kernel_package()
assert Path(package.__file__).resolve() == Path(sys.argv[2]).resolve()
prepared_path = sys.path.copy()
assert runtime._load_kernel_package() is package
assert sys.path == prepared_path
if sys.argv[1] == 'installed':
    assert sys.path == original_path
else:
    assert sys.path[1:] == original_path
assert not {'torch', 'torch_npu', 'xllm_export'} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script, layout, str(expected)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=python_path),
        check=True,
    )


def test_main_wheel_includes_independent_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_ROOT)
    from scripts.build_support import utils

    captured: dict[str, Any] = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(utils, "pre_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(utils, "check_and_install_pre_commit", lambda: None)
    monkeypatch.setattr(sys, "argv", ["setup.py", "build", "--device", "npu"])
    runpy.run_path(str(_ROOT / "setup.py"), run_name="__main__")
    distribution = Distribution({key: captured[key] for key in ("packages", "package_dir")})
    distribution.script_name = "setup.py"
    command = build_py(distribution)
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "lib")
    command.run()
    env = dict(os.environ, PYTHONPATH=command.build_lib)
    subprocess.run(
        [sys.executable, "-S", "-c", "import xllm_kernel, xllm_kernel.ops; assert xllm_kernel.__version__ == '0.1.0'"],
        cwd=tmp_path,
        env=env,
        check=True,
    )
