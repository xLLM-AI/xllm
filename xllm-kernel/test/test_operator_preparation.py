# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend imports register into the same startup registry used by models."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from xllm_kernel.catalog import load_operations
from xllm_kernel.ops import normalization
from xllm_kernel.platform import Platform, PlatformEnum
from xllm_kernel.registry import KernelRegistry, KernelSpec


class _Vendor:
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: (args[0], None)


@pytest.fixture
def _triton_launchers(monkeypatch: pytest.MonkeyPatch) -> None:
    for leaf, function in (
        ("eh_norm", "fused_eh_norm"),
        ("rms_norm", "rms_norm_sigmoid_gated"),
        ("split_qkv_rmsnorm_rope", "split_qkv_rmsnorm_rope"),
    ):
        name = f"xllm_kernel.ops.normalization._triton.{leaf}"
        module = ModuleType(name)
        setattr(module, function, lambda *args, **kwargs: args[0])
        monkeypatch.setitem(sys.modules, name, module)


def test_all_backend_modules_register_unique_variants_and_execute_public_binding(
    monkeypatch: pytest.MonkeyPatch,
    _triton_launchers: None,
) -> None:
    monkeypatch.setattr(Platform, "enum", classmethod(lambda cls: PlatformEnum.NPU))
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setitem(sys.modules, "torch_npu", _Vendor())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda: "Ascend910B2"), raising=False)
    # Native schemas and DSL launchers are unavailable in this CPU test. The
    # implementation adapters, import-time registration and registry are real.
    monkeypatch.setattr(torch.ops, "xllm_ops", _Vendor(), raising=False)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name in normalization.TRITON_JIT_REQUIRES else real_find_spec(name),
    )
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
        assert {spec.solution for spec in specs} == {"torch", "aclnn", "triton", "tilelang"}
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
def test_backend_modules_do_not_load_device_dependencies_on_other_platforms(
    monkeypatch: pytest.MonkeyPatch, platform: PlatformEnum
) -> None:
    import importlib.abc

    class _BlockNpuImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
            if fullname.split(".")[0] in {"torch_npu", "triton", "tilelang", "xllm"}:
                raise AssertionError(f"unexpected NPU import on {platform.value}: {fullname}")

    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setattr(Platform, "enum", classmethod(lambda cls: platform))
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockNpuImports(), *sys.meta_path])
    modules = tuple(dict.fromkeys(backend.module for operation in load_operations() for backend in operation.backends))
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


# Values are opaque at this boundary: backend contracts own tensor validation.
_VALUE = object()
_WEIGHT = object()
_CASES = {
    "rms_norm": ((_VALUE, _WEIGHT, 1e-5), {}),
    "layer_norm": ((_VALUE, _WEIGHT, _VALUE, 1e-5), {}),
    "fused_add_rms_norm": ((_VALUE, _VALUE, _WEIGHT, 1e-5), {}),
    "gemma_rms_norm": ((_VALUE, _WEIGHT, 1e-5), {}),
    "fused_add_gemma_rms_norm": ((_VALUE, _VALUE, _WEIGHT, 1e-5), {}),
    "fused_add_rms_norm_dynamic_quant": ((_VALUE, _VALUE, _WEIGHT, 1e-5), {}),
    "rms_norm_dynamic_quant": ((_VALUE, _WEIGHT, 1e-5), {}),
    "l2_norm": ((_VALUE, 1e-5), {}),
    "rms_norm_gated": ((_VALUE, _VALUE, _WEIGHT, 1e-5), {}),
    "rms_norm_sigmoid_gated": ((_VALUE, _VALUE, _WEIGHT, 1e-5), {}),
    "fused_eh_norm": ((_VALUE, _VALUE, _WEIGHT, _WEIGHT, 1e-5), {}),
    "fused_qk_norm_rope": (
        (_VALUE,),
        dict(
            num_heads_q=4,
            num_heads_k=2,
            num_heads_v=2,
            head_dim=128,
            eps=1e-5,
            q_weight=_WEIGHT,
            k_weight=_WEIGHT,
            cos_sin_cache=_VALUE,
            position_ids=_VALUE,
            cos=None,
            sin=None,
            interleaved=False,
        ),
    ),
    "split_qkv_rmsnorm_mrope": ((_VALUE, _WEIGHT, _WEIGHT, _VALUE, _VALUE, 1e-5, 4, 2, 256), {}),
    "has_split_qkv_rmsnorm_mrope_specialization": ((4, 2, 256), {}),
    "build_split_qkv_rmsnorm_mrope_gather_pattern": ((64, [11, 11, 10], False, _VALUE), {}),
}


@pytest.mark.parametrize("name", _CASES)
def test_public_normalization_requires_preparation_and_uses_fixed_override(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xllm_kernel import runtime

    assert set(_CASES) == {op.contract.operation for op in normalization.OPERATIONS}
    for operation in load_operations():
        monkeypatch.setattr(operation, "prepared", None)
    monkeypatch.setattr(runtime, "_prepared", None)
    monkeypatch.setattr(runtime, "_configuration", None)
    # Missing default SDKs must not prevent an explicitly registered compatible
    # implementation from being selected, and forward must never import them.
    monkeypatch.setattr(runtime, "find_spec", lambda dependency: None)
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    operation = next(op for op in normalization.OPERATIONS if op.contract.operation == name)
    public = getattr(normalization, name)
    args, kwargs = _CASES[name]
    with pytest.raises(RuntimeError, match="initialize"):
        public(*args, **kwargs)

    calls = []
    expected = object()

    def selected(*values: Any, **keywords: Any) -> object:
        calls.append((values, keywords))
        if len(calls) == 3:
            raise RuntimeError("selected implementation failed")
        return expected

    implementation = f"npu.test.{name}"
    registry.register(
        KernelSpec(
            implementation,
            operation.contract,
            PlatformEnum.NPU,
            ("eager", "aclgraph"),
            "opaque test boundary",
            solution="test",
        ),
        selected,
    )
    bindings = runtime.initialize(device="npu", operations=(name,), overrides={name: implementation})
    assert bindings[name].function is selected

    def unexpected_selection(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("forward must not select or retry an implementation")

    monkeypatch.setattr(registry, "prepare", unexpected_selection)
    monkeypatch.setattr(runtime, "select_kernel", unexpected_selection)
    assert public(*args, **kwargs) is expected
    assert public(*args, **kwargs) is expected
    with pytest.raises(RuntimeError, match="selected implementation failed"):
        public(*args, **kwargs)
    assert calls == [(args, kwargs)] * 3
    assert runtime.initialize(device="npu", operations=(name,), overrides={name: implementation}) is bindings
    with pytest.raises(RuntimeError, match="restart"):
        runtime.initialize(device="npu", operations=(name,), overrides={name: "another"})


def test_missing_triton_excludes_backend_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm_kernel import runtime

    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setattr(runtime, "_prepared", None)
    monkeypatch.setattr(runtime, "_configuration", None)
    monkeypatch.setattr(runtime, "find_spec", lambda name: None if name == "triton" else object())

    def unexpected_import(name: str) -> None:
        raise AssertionError(f"missing dependency must exclude backend before import: {name}")

    monkeypatch.setattr(runtime, "import_module", unexpected_import)
    with pytest.raises(ValueError, match="missing dependencies: triton"):
        runtime.initialize(device="npu", operations=("rms_norm_sigmoid_gated",))
    assert runtime._prepared is None


@pytest.mark.parametrize("jit_available", [False, True])
def test_merged_triton_module_keeps_native_and_jit_dependencies_independent(
    jit_available: bool,
    monkeypatch: pytest.MonkeyPatch,
    _triton_launchers: None,
) -> None:
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setattr(Platform, "enum", classmethod(lambda cls: PlatformEnum.NPU))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (object() if jit_available else None)
        if name in normalization.TRITON_JIT_REQUIRES
        else real_find_spec(name),
    )
    native = SimpleNamespace() if jit_available else SimpleNamespace(l2_norm=lambda value, eps: value)
    monkeypatch.setattr(torch.ops, "xllm_ops", native, raising=False)
    name = "xllm_kernel.ops.normalization.triton"
    monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        importlib.import_module(name)
        expected = {"rms_norm_sigmoid_gated", "fused_eh_norm", "fused_qk_norm_rope"} if jit_available else {"l2_norm"}
        assert {spec.contract.operation for spec in registry.specs()} == expected
    finally:
        sys.modules.pop(name, None)
