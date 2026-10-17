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

import os
import runpy
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import setuptools
from setuptools import Distribution
from setuptools.command.build_py import build_py
from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT
from xllm_kernel.registry import KernelRegistry, KernelSpec

_ROOT = Path(__file__).resolve().parents[2]


def _identity(value: object) -> object:
    return value


def _spec(name: str = "npu.native") -> KernelSpec:
    return KernelSpec(name, RMS_NORM_CONTRACT, "npu", ("eager", "aclgraph"), "native tensor domain")


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


@pytest.mark.parametrize("packaged", [False, True])
@pytest.mark.parametrize("host_available", [False, True])
def test_native_binding_resolves_package_or_host(
    monkeypatch: pytest.MonkeyPatch, packaged: bool, host_available: bool
) -> None:
    from xllm_kernel.ops.rms_norm import npu

    calls = []
    host_op = lambda *args: None
    package_op = lambda *args: None
    torch = SimpleNamespace(
        ops=SimpleNamespace(
            xllm_ops=SimpleNamespace(**({"rms_norm": host_op} if host_available else {})),
            xllm_kernel=SimpleNamespace(rms_norm=package_op),
            load_library=lambda path: calls.append(path),
        ),
        _C=SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=lambda *args: True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "xllm_kernel.fake.rms_norm", SimpleNamespace(register_native_fake=lambda: calls.append("fake"))
    )
    monkeypatch.setattr(npu, "_LOADED", False)
    monkeypatch.setattr(
        npu.importlib.util, "find_spec", lambda name: SimpleNamespace(origin="_npu.so") if packaged else None
    )
    if not packaged and not host_available:
        with pytest.raises(RuntimeError, match="XLLM_KERNEL_BUILD_NPU=1"):
            npu.register(KernelRegistry())
        return
    for _ in range(2):
        registry = KernelRegistry()
        npu.register(registry)
        registry.freeze()
        plan = registry.prepare(RMS_NORM_CONTRACT, device="npu", execution_mode="aclgraph")
        assert plan.function is (package_op if packaged else host_op)
    assert calls == (["_npu.so", "fake"] if packaged else [])


def test_native_extension_load_failure_does_not_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm_kernel.ops.rms_norm import npu

    def fail_load(path: str) -> None:
        raise OSError("incompatible NPU extension")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(load_library=fail_load, xllm_ops=SimpleNamespace(rms_norm=_identity))),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(npu, "_LOADED", False)
    monkeypatch.setattr(npu.importlib.util, "find_spec", lambda name: SimpleNamespace(origin="_npu.so"))
    with pytest.raises(OSError, match="incompatible NPU extension"):
        npu.register(KernelRegistry())
    assert not npu._LOADED


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
