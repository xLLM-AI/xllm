# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend imports register into the same startup registry used by models."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import xllm_kernel
from xllm_kernel.catalog import load_operations
from xllm_kernel.platform import Platform, PlatformEnum
from xllm_kernel.registry import KernelRegistry


class _Vendor:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: (args[0], None)


def test_attention_public_catalog_is_sdk_free(tmp_path: Path) -> None:
    script = """
import importlib.abc
import sys
from typing import Any

class BlockDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'triton', 'xllm'}:
            raise AssertionError(f'unexpected dependency: {fullname}')

sys.meta_path.insert(0, BlockDeviceImports())
from xllm_kernel.catalog import load_operations
from xllm_kernel.ops import fused_recurrent_gated_delta_rule, fused_sigmoid_gating_delta_rule_update, fused_qkvzba_split_reshape_cat
from xllm_kernel.ops import partial_rotary_embedding, reshape_paged_cache
from xllm_kernel.ops.attention.gdn import OPERATIONS as GDN_OPERATIONS
from xllm_kernel.ops.attention.kda import OPERATIONS as KDA_OPERATIONS, causal_conv1d_update_v2
from xllm_kernel.ops.gemm import OPERATIONS as GEMM_OPERATIONS, matmul_ein_sum, matmul_ein_sum_out
catalog = {operation.contract.operation: operation for operation in load_operations()}
for operation in (*GDN_OPERATIONS, *KDA_OPERATIONS):
    name = operation.contract.operation
    assert catalog[name] is operation
    assert operation.implementations == ('npu.triton.' + name,)
    assert operation.prepared is None
    assert operation.backends[0].requires == ('torch_npu', 'triton')
for operation in GEMM_OPERATIONS:
    name = operation.contract.operation
    assert catalog[name] is operation
    assert operation.implementations == ('npu.atb.' + name,)
    assert operation.prepared is None
assert callable(matmul_ein_sum) and callable(matmul_ein_sum_out)
assert callable(causal_conv1d_update_v2)
assert callable(reshape_paged_cache) and callable(partial_rotary_embedding)
cache = catalog["reshape_paged_cache"].contract
assert cache.mutates == ("key_cache", "value_cache") and cache.aliases_inputs
rope = catalog["partial_rotary_embedding"].contract
assert rope.mutates == () and not rope.aliases_inputs
assert all(callable(fn) for fn in (fused_recurrent_gated_delta_rule, fused_sigmoid_gating_delta_rule_update, fused_qkvzba_split_reshape_cat))
"""
    env = {**os.environ, "PYTHONPATH": str(Path(xllm_kernel.__file__).resolve().parent.parent)}
    subprocess.run([sys.executable, "-S", "-c", script], cwd=tmp_path, env=env, check=True)


@pytest.mark.parametrize(("generation", "solution"), [("a2", "atb"), ("a3", "atb"), ("a5", "aclnn")])
def test_cache_backend_follows_ascend_generation(
    monkeypatch: pytest.MonkeyPatch, generation: str, solution: str
) -> None:
    module_name = "xllm_kernel.ops.kv_cache.npu"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(Platform, "enum", classmethod(lambda cls: PlatformEnum.NPU))
    monkeypatch.setattr(Platform, "get_ascend_soc_generation", classmethod(lambda cls: generation))
    monkeypatch.setattr(KernelRegistry, "_instance", KernelRegistry())
    monkeypatch.setattr(xllm_kernel, "_npu_binding", object(), raising=False)
    atb, a5 = object(), object()
    monkeypatch.setattr(
        torch.ops,
        "xllm_kernel",
        SimpleNamespace(
            reshape_paged_cache_atb=SimpleNamespace(default=atb),
            reshape_paged_cache_a5=SimpleNamespace(default=a5),
        ),
    )
    from xllm_kernel.ops.kv_cache import RESHAPE_PAGED_CACHE_CONTRACT

    try:
        importlib.import_module(module_name)
        registry = KernelRegistry.get()
        registry.freeze()
        plan = registry.prepare(RESHAPE_PAGED_CACHE_CONTRACT, device="npu", execution_mode="aclgraph")
        assert plan.spec.solution == solution
        assert plan.function is (a5 if generation == "a5" else atb)
    finally:
        sys.modules.pop(module_name, None)


def test_torch_backend_modules_register_unique_variants_and_execute_public_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Platform, "enum", classmethod(lambda cls: PlatformEnum.NPU))
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setitem(sys.modules, "torch_npu", _Vendor())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda: "Ascend910B2"), raising=False)
    operations = tuple(
        operation
        for operation in load_operations()
        if any(backend.module.endswith(".torch") for backend in operation.backends)
    )
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
    modules = (
        "xllm_kernel.ops.activation.torch",
        "xllm_kernel.ops.normalization.torch",
        "xllm_kernel.ops.normalization.triton",
        "xllm_kernel.ops.attention.gdn.triton",
        "xllm_kernel.ops.attention.kda.triton",
        "xllm_kernel.ops.gemm.atb",
        "xllm_kernel.ops.kv_cache.npu",
    )
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
