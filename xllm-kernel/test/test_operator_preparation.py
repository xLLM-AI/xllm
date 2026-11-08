# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend imports register into the same startup registry used by models."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from xllm_kernel.catalog import load_operations
from xllm_kernel.platform import Platform, PlatformEnum
from xllm_kernel.registry import KernelRegistry


class _Vendor:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: (args[0], None)


def test_all_backend_modules_register_unique_variants_and_execute_public_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Platform, "enum", classmethod(lambda cls: PlatformEnum.NPU))
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setitem(sys.modules, "torch_npu", _Vendor())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda: "Ascend910B2"), raising=False)
    operations = load_operations()
    for operation in operations:
        for backend in operation.backends:
            monkeypatch.delitem(sys.modules, backend.module, raising=False)
    try:
        for operation in operations:
            for backend in operation.backends:
                module = importlib.import_module(backend.module)
                assert importlib.import_module(backend.module) is module
        specs = registry.specs()
        assert len(specs) == len({spec.name for spec in specs})
        assert {spec.solution for spec in specs} == {"torch"}
        registry.freeze()
        for operation in operations:
            assert {spec.name for spec in specs if spec.contract == operation.contract} == set(
                operation.implementations
            )
        rms_norm = importlib.import_module("xllm_kernel.ops.normalization")
        plan = registry.prepare(rms_norm.RMS_NORM_CONTRACT, device="npu", execution_mode="eager")
        monkeypatch.setattr(rms_norm.RMS_NORM_OPERATION, "prepared", plan)
        value = torch.tensor([2.0])
        assert rms_norm.rms_norm(value, torch.ones(1), 1e-5) is value
        assert plan.spec.solution == "torch"
        with pytest.raises(RuntimeError, match="frozen"):
            importlib.reload(sys.modules["xllm_kernel.ops.normalization.torch"])
    finally:
        for operation in operations:
            for backend in operation.backends:
                sys.modules.pop(backend.module, None)


@pytest.mark.parametrize("platform", [PlatformEnum.CPU, PlatformEnum.CUDA])
def test_torch_modules_do_not_import_or_register_npu_on_other_platforms(
    monkeypatch: pytest.MonkeyPatch, platform: PlatformEnum
) -> None:
    import importlib.abc

    class _BlockNpuImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
            if fullname == "torch_npu" or fullname.startswith("torch_npu."):
                raise AssertionError(f"unexpected NPU import on {platform.value}: {fullname}")

    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setattr(Platform, "enum", classmethod(lambda cls: platform))
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockNpuImports(), *sys.meta_path])
    modules = ("xllm_kernel.ops.activation.torch", "xllm_kernel.ops.normalization.torch")
    for name in modules:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        for name in modules:
            importlib.import_module(name)
        assert not registry.specs()
        assert "torch_npu" not in sys.modules
    finally:
        for name in modules:
            sys.modules.pop(name, None)
            vars(importlib.import_module(name.rsplit(".", 1)[0])).pop("torch", None)
