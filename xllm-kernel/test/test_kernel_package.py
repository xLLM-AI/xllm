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
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import setuptools
from setuptools import Distribution
from setuptools.command.build_py import build_py
from xllm_kernel.ops.normalization import RMS_NORM_CONTRACT
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
