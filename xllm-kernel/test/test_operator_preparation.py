# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend imports register into the same startup registry used by models."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from xllm_kernel import custom_ops
from xllm_kernel.backends import collective, xlite, xllm
from xllm_kernel.catalog import load_operations
from xllm_kernel.registry import KernelRegistry


class _Vendor:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: (args[0], None)


def test_all_backend_modules_register_unique_variants_and_execute_public_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setitem(sys.modules, "torch_npu", _Vendor())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda: "Ascend910B2"), raising=False)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", lambda *args: True)
    monkeypatch.setattr(xllm, "load_operator", lambda name: lambda *args, **kwargs: args[0])
    monkeypatch.setattr(xlite, "load_extension", lambda: None)
    monkeypatch.setattr(collective, "prepare_hccl", lambda: (object(), "test_comm"))
    monkeypatch.setattr(custom_ops, "register_custom_op", lambda *args, **kwargs: lambda *args: args[0])
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
        assert {spec.solution for spec in specs} <= {"aclnn", "atb", "hccl", "torch", "tilelang", "xlite"}
        registry.freeze()
        for operation in operations:
            assert {spec.name for spec in specs if spec.contract == operation.contract} == set(
                operation.implementations
            )
        rms_norm = importlib.import_module("xllm_kernel.ops.rms_norm")
        plan = registry.prepare(rms_norm.CONTRACT, device="npu", execution_mode="eager")
        monkeypatch.setattr(rms_norm.OPERATION, "prepared", plan)
        value = torch.tensor([2.0])
        assert rms_norm.rms_norm(value, torch.ones(1), 1e-5) is value
        assert plan.spec.solution == "aclnn"
        with pytest.raises(RuntimeError, match="frozen"):
            importlib.reload(sys.modules["xllm_kernel.ops.rms_norm.aclnn_torch_npu"])
    finally:
        for operation in operations:
            for backend in operation.backends:
                sys.modules.pop(backend.module, None)
