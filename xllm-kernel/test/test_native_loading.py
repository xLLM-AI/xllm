# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Check SDK-free imports and loading from a relocated binary distribution."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_import_and_missing_binary_are_sdk_free(tmp_path: Path) -> None:
    import xllm_kernel

    package = tmp_path / "xllm_kernel"
    shutil.copytree(Path(xllm_kernel.__file__).parent, package, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
    script = """
import importlib.abc
import sys
from pathlib import Path
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'xllm', 'scripts'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockSDK())
from xllm_kernel.native import load_native, resource_root
assert resource_root() == Path.cwd() / 'xllm_kernel/opp'
try:
    load_native()
except RuntimeError as error:
    assert 'XLLM_KERNEL_BUILD_NATIVE=1' in str(error)
else:
    raise AssertionError('accepted missing native binary')
binary = Path.cwd() / 'compiled/libxllm_kernel.so'
binary.parent.mkdir()
binary.touch()
(Path.cwd() / 'xllm_kernel/libxllm_kernel.so').symlink_to(binary)
assert resource_root() == binary.parent / 'opp'
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script], cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(tmp_path)), check=True
    )


def test_load_relocated_native_distribution_once(tmp_path: Path) -> None:
    artifact = os.environ.get("XLLM_KERNEL_NATIVE_TEST_PACKAGE")
    if artifact is None:
        pytest.skip("set XLLM_KERNEL_NATIVE_TEST_PACKAGE to an unpacked native wheel package")
    package = tmp_path / "xllm_kernel"
    shutil.copytree(artifact, package)
    script = """
import ctypes
from pathlib import Path
import torch
from xllm_kernel.native import load_native, resource_root
original = torch.ops.load_library
calls = []
def load(path):
    calls.append(path)
    original(path)
torch.ops.load_library = load
library = load_native()
assert library.parent == Path.cwd() / 'xllm_kernel'
assert load_native() is library
assert calls.count(str(library)) == 1
assert resource_root() == library.parent / 'opp'
assert ctypes.CDLL(str(library)).xllm_kernel_native_abi_version() == 1
assert not any(name in __import__('sys').modules for name in ('xllm', 'scripts'))
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path), TORCH_DEVICE_BACKEND_AUTOLOAD="0"),
        check=True,
    )
