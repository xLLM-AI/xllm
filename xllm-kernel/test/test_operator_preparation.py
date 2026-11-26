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
from unittest.mock import Mock

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
from xllm_kernel.ops.attention.fia import get_fused_attention
from xllm_kernel.ops.attention.gdn import OPERATIONS as GDN_OPERATIONS
from xllm_kernel.ops.attention.gdn import (
    fused_qkvzba_split_reshape_cat,
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from xllm_kernel.ops.attention.kda import OPERATIONS as KDA_OPERATIONS, causal_conv1d_update_v2
from xllm_kernel.ops.attention.mla import mla_preprocess_v2
from xllm_kernel.ops.gemm import (
    MATMUL_EIN_SUM_OPERATION,
    MATMUL_EIN_SUM_OUT_OPERATION,
    matmul_ein_sum,
    matmul_ein_sum_out,
    quant_matmul_out,
)
from xllm_kernel.ops.hyper_connection import hc_post, hc_pre
from xllm_kernel.ops.kv_cache import convert_xqa_pages, reshape_paged_cache, write_fp8_kv
from xllm_kernel.ops.quantization import quantize_query_fp8
from xllm_kernel.ops.positional_encoding import partial_rotary_embedding
catalog = {operation.contract.operation: operation for operation in load_operations()}
for api, mutates in ((write_fp8_kv, ("key_cache", "value_cache")), (convert_xqa_pages, ("table", "lengths")), (quantize_query_fp8, ())):
    operation = catalog[api.__name__]
    assert operation.implementations == ("cuda.triton." + api.__name__,)
    assert operation.contract.mutates == mutates and not operation.contract.aliases_inputs
    assert operation.prepared is None
for operation in (*GDN_OPERATIONS, *KDA_OPERATIONS):
    name = operation.contract.operation
    if name in ("causal_conv1d", "causal_conv1d_qkv_prefill", "chunk_gated_delta_rule"):
        continue
    assert catalog[name] is operation
    assert operation.implementations == ('npu.triton.' + name,)
    assert operation.prepared is None
    assert operation.backends[0].requires == ('torch_npu', 'triton')
for operation in (MATMUL_EIN_SUM_OPERATION, MATMUL_EIN_SUM_OUT_OPERATION):
    name = operation.contract.operation
    assert catalog[name] is operation
    assert operation.implementations == ('npu.atb.' + name,)
    assert operation.prepared is None
assert callable(matmul_ein_sum) and callable(matmul_ein_sum_out)
for name, mutates, aliases in (
    ("quant_matmul_out", ("out",), True),
    ("mla_preprocess_v2", ("kv_cache", "rope_cache"), False),
    ("hc_pre", (), False),
    ("hc_post", (), False),
    ("moe_gating_top_k_hash", (), False),
    ("moe_init_routing_v3", (), False),
    ("moe_grouped_matmul_swiglu_quant", (), False),
    ("dequant_swiglu_quant", (), False),
    ("causal_conv1d", ("conv_state",), False),
    ("causal_conv1d_qkv_prefill", ("conv_state",), False),
    ("chunk_gated_delta_rule", (), False),
    ("mega_gdn_prefill", ("conv_state", "ssm_cache"), False),
    ("mega_gdn_decode", ("conv_state", "ssm_state"), False),
    ("mega_gdn_mtp_decode", ("conv_state", "ssm_state"), False),
    ("mega_gdn_mtp_decode_varlen", ("conv_state", "ssm_state", "out"), False),
    ("compressor", ("kv_state", "score_state"), False),
    ("sparse_attn_sharedkv", (), False),
    ("sparse_attn_sharedkv_metadata", (), False),
    ("sparse_flash_attention_lse", (), False),
    ("sparse_flash_attention_lse_out", ("attention_output",), True),
    ("quant_lightning_indexer", (), False),
    ("quant_lightning_indexer_metadata", (), False),
    ("scatter_nd_update", ("value",), False),
):
    operation = catalog[name]
    assert operation.implementations == ("npu.ascendc." + name,)
    assert operation.contract.mutates == mutates and operation.contract.aliases_inputs == aliases
    assert operation.prepared is None
assert callable(quant_matmul_out) and callable(mla_preprocess_v2)
shared_indices = catalog["shared_linear_state_indices_out"]
assert shared_indices.implementations == ("npu.triton.shared_linear_state_indices_out",)
assert shared_indices.contract.mutates == ("out",) and not shared_indices.contract.aliases_inputs
assert shared_indices.prepared is None
assert callable(hc_pre) and callable(hc_post)
assert callable(causal_conv1d_update_v2)
assert callable(reshape_paged_cache) and callable(partial_rotary_embedding)
fia = catalog["fused_attention"]
assert callable(get_fused_attention)
assert fia.contract.mutates == ("workspace", "out") and fia.contract.aliases_inputs
assert fia.prepared is None
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
    monkeypatch.setattr(Platform, "is_npu", Mock(side_effect=AssertionError("unexpected platform check")))
    monkeypatch.setattr(Platform, "get_ascend_soc_generation", classmethod(lambda cls: generation))
    monkeypatch.setattr(KernelRegistry, "_instance", KernelRegistry())
    monkeypatch.setattr(xllm_kernel, "_npu_binding", object(), raising=False)
    monkeypatch.setattr(xllm_kernel, "_ascendc_opp", object(), raising=False)
    atb, a5 = object(), object()
    prepare_a5 = Mock()
    monkeypatch.setattr(
        torch.ops,
        "xllm_kernel",
        SimpleNamespace(
            prepare_ascendc_reshape_paged_cache_a5=prepare_a5,
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
        if generation == "a5":
            prepare_a5.assert_called_once_with()
        else:
            prepare_a5.assert_not_called()
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("has_split_rope", [False, True])
def test_fused_attention_prepares_optional_split_rope_and_preserves_graph_buffers(
    monkeypatch: pytest.MonkeyPatch, has_split_rope: bool
) -> None:
    from xllm_kernel.ops.attention.fia import FUSED_ATTENTION_OPERATION, get_fused_attention

    module_name = "xllm_kernel.ops.attention.fia.npu"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(Platform, "enum", classmethod(lambda cls: PlatformEnum.NPU))
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    v1, v2 = SimpleNamespace(default=Mock(), out=Mock()), SimpleNamespace(default=Mock(), out=Mock())
    workspace_v1, workspace_v2 = Mock(), Mock()
    npu = SimpleNamespace(npu_fused_infer_attention_score=v1)
    vendor = SimpleNamespace(_npu_fused_infer_attention_score_get_max_workspace=workspace_v1)
    if has_split_rope:
        npu.npu_fused_infer_attention_score_v2 = v2
        vendor._npu_fused_infer_attention_score_v2_get_max_workspace = workspace_v2
    monkeypatch.setattr(torch.ops, "npu", npu)
    monkeypatch.setitem(sys.modules, "torch_npu", vendor)
    try:
        importlib.import_module(module_name)
        registry.freeze()
        plan = registry.prepare(FUSED_ATTENTION_OPERATION.contract, device="npu", execution_mode="aclgraph")
        monkeypatch.setattr(FUSED_ATTENTION_OPERATION, "prepared", plan)
        fia = get_fused_attention()
        assert fia.run is v1.default and fia.out is v1.out and fia.workspace is workspace_v1
        if not has_split_rope:
            assert fia.with_separate_rope is None
            return

        query = torch.empty(2, 1, 8)
        lengths = [1, 3]
        out = [torch.empty_like(query), torch.empty(0)]
        workspace = torch.empty(16)
        split_rope = fia.with_separate_rope
        assert split_rope is not None
        split_rope.out(
            query,
            query,
            query,
            actual_seq_lengths_kv=lengths,
            num_heads=1,
            scale=0.25,
            softmax_lse_flag=False,
            workspace=workspace,
            out=out,
        )
        kwargs = v2.out.call_args.kwargs
        assert kwargs["actual_seq_kvlen"] is lengths
        assert kwargs["workspace"] is workspace and kwargs["out"] is out
        assert kwargs["num_query_heads"] == 1 and kwargs["softmax_scale"] == 0.25
        assert kwargs["return_softmax_lse"] is False
        v1.out.assert_not_called()
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


def test_npu_quantization_registers_only_its_two_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "xllm_kernel.ops.quantization.npu"
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setattr(xllm_kernel, "_npu_binding", object(), raising=False)
    monkeypatch.setattr(
        torch.ops,
        "xllm_kernel",
        SimpleNamespace(
            quantize_per_tensor=SimpleNamespace(default=Mock()), dynamic_quant=SimpleNamespace(default=Mock())
        ),
    )
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    try:
        importlib.import_module(module_name)
        assert {spec.name for spec in registry.specs()} == {"npu.aclnn.quantize_per_tensor", "npu.aclnn.dynamic_quant"}
    finally:
        sys.modules.pop(module_name, None)
