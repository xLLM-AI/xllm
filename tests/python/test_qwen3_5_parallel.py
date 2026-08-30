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

"""Parallel-layout tests for the Qwen3.5 Python CUDA model."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# conftest.py binds xllm.python.kernels and .distributed to empty stubs. Bind the
# CUDA MoE module without executing its package __init__, which would reach for
# operators from the C++ binary, so the architecture gate under test is the real
# one; the remaining kernels and collectives are mocked.
_python_root = Path(__file__).parents[2] / "xllm" / "python"
_kernels_cuda = types.ModuleType("xllm.python.kernels_cuda")
_kernels_cuda.__path__ = [str(_python_root / "kernels_cuda")]
sys.modules.setdefault("xllm.python.kernels_cuda", _kernels_cuda)

from xllm.python import distributed, kernels  # noqa: E402
from xllm.python.kernels_cuda.moe import supports_cutlass_moe  # noqa: E402

kernels.supports_cutlass_moe = supports_cutlass_moe
kernels.moe_fused_topk = MagicMock()
kernels.cutlass_fused_moe = MagicMock()
kernels.fused_moe = MagicMock()
distributed.all_gather_variable = MagicMock()
distributed.all_reduce_ = MagicMock()

from xllm.python.attention.backend import LayerCache  # noqa: E402
from xllm.python.kernels_npu.causal_conv1d import (  # noqa: E402
    causal_conv1d_decode as npu_causal_conv1d_decode,
)
from xllm.python.layers.cuda.qwen3_5.decoder_layer import (  # noqa: E402
    CudaQwen3_5DecoderLayer,
)
from xllm.python.layers.cuda.qwen3_5.gated_delta_net import (  # noqa: E402
    CudaQwen3_5GatedDeltaNet,
)
from xllm.python.layers.fused_moe import FusedMoE  # noqa: E402
from xllm.python.layers.gated_mlp import GatedMLP  # noqa: E402
from xllm.python.layers.layernorm import GemmaRMSNorm  # noqa: E402
from xllm.python.layers.npu.qwen3_5.decoder_layer import (  # noqa: E402
    NpuQwen3_5DecoderLayer,
)
from xllm.python.layers.npu.qwen3_5.gated_delta_net import (  # noqa: E402
    NpuQwen3_5GatedDeltaNet,
)
from xllm.python.layers.qwen3_5_decoder_layer import (  # noqa: E402
    Qwen3_5LoadContext,
    Qwen3_5SparseMoEBlock,
    get_qwen3_5_decoder_layer_class,
)
from xllm.python.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
)
from xllm.python.model_loader import (  # noqa: E402
    ScopedWeightLoader,
)
from xllm.python.models import qwen3_5 as qwen3_5_model  # noqa: E402
from xllm.python.models.qwen3_5 import (  # noqa: E402
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5Model,
)


@pytest.fixture(autouse=True)
def _use_cuda_decoder_for_cpu_model_tests(monkeypatch):
    monkeypatch.setattr(
        qwen3_5_model,
        "get_qwen3_5_decoder_layer_class",
        lambda _device: CudaQwen3_5DecoderLayer,
    )


def _config(**overrides) -> Qwen3_5Config:
    values = {
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "layer_types": ["full_attention"],
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "vocab_size": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 64,
        "tp_size": 2,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "world_size": 2,
        "moe_tp_size": 2,
        "moe_tp_rank": 0,
        "ep_size": 1,
        "ep_rank": 0,
    }
    values.update(overrides)
    return Qwen3_5Config.from_dict(values)


def _make_moe_layer(cfg: Qwen3_5Config, device: torch.device | None = None) -> FusedMoE:
    return FusedMoE(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.moe_intermediate_size,
        num_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        renormalize=True,
        moe_tp_size=cfg.moe_tp_size,
        moe_tp_rank=cfg.moe_tp_rank,
        ep_size=cfg.ep_size,
        ep_rank=cfg.ep_rank,
        dp_size=cfg.dp_size,
        dp_rank=cfg.dp_rank,
        dtype=torch.float32,
        device=device or torch.device("cpu"),
    )


def test_moe_layer_schedule_matches_config() -> None:
    cfg = _config(
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
        decoder_sparse_step=2,
        mlp_only_layers=[3],
    )

    assert not cfg.is_moe_layer(0)
    assert cfg.is_moe_layer(1)
    assert not cfg.is_moe_layer(2)
    assert not cfg.is_moe_layer(3)

    model = Qwen3_5Model(cfg, torch.float32, torch.device("cpu"))
    assert isinstance(model.layers[0].mlp, GatedMLP)
    assert isinstance(model.layers[1].mlp, Qwen3_5SparseMoEBlock)
    assert isinstance(model.layers[2].mlp, GatedMLP)
    assert isinstance(model.layers[3].mlp, GatedMLP)


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def test_backend_decoder_factory_selects_once_by_device() -> None:
    assert get_qwen3_5_decoder_layer_class("cuda") is CudaQwen3_5DecoderLayer
    assert get_qwen3_5_decoder_layer_class("privateuseone") is NpuQwen3_5DecoderLayer
    with pytest.raises(ValueError, match="no decoder implementation"):
        get_qwen3_5_decoder_layer_class("cpu")


def test_scoped_loader_resolves_supported_model_roots() -> None:
    value = torch.arange(8).view(2, 4)
    for prefix in ("model.language_model.", "model.", ""):
        state = _StateDict({prefix + "embed_tokens.weight": value})
        root = ScopedWeightLoader([state]).find_root(
            ("model.language_model.", "model.", ""),
            "embed_tokens.weight",
        )
        assert root.prefix == prefix
        assert root.tensor("embed_tokens.weight") is value


def test_backend_gdn_loads_native_conv_weight_layouts(monkeypatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    cfg = _config(
        hidden_size=8,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
    )
    cuda_layer = CudaQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.float32,
        torch.device("cpu"),
    )
    npu_layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.float32,
        torch.device("cpu"),
    )
    conv_dim = 24
    tensors = {
        "linear_attn.in_proj_qkv.weight": torch.zeros(conv_dim, 8),
        "linear_attn.in_proj_z.weight": torch.zeros(8, 8),
        "linear_attn.in_proj_b.weight": torch.zeros(2, 8),
        "linear_attn.in_proj_a.weight": torch.zeros(2, 8),
        "linear_attn.conv1d.weight": torch.arange(
            conv_dim * 4,
            dtype=torch.float32,
        ).view(conv_dim, 1, 4),
        "linear_attn.A_log": torch.zeros(2),
        "linear_attn.dt_bias": torch.zeros(2),
        "linear_attn.norm.weight": torch.zeros(4),
        "linear_attn.out_proj.weight": torch.zeros(8, 8),
    }
    state = ScopedWeightLoader([_StateDict(tensors)], "linear_attn.")
    context = Qwen3_5LoadContext(tp_rank=0, tp_size=1)

    cuda_layer.load_weights(state, context)
    npu_layer.load_weights(state, context)

    assert cuda_layer.conv1d_weight.shape == (conv_dim, 4)
    assert npu_layer.conv1d_weight.shape == (4, conv_dim)
    torch.testing.assert_close(
        npu_layer.conv1d_weight,
        cuda_layer.conv1d_weight.transpose(0, 1),
    )


class _ConstantModule(torch.nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, _hidden: torch.Tensor) -> torch.Tensor:
        return self.value


def _linear_config() -> Qwen3_5Config:
    return _config(
        hidden_size=8,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
    )


def _install_constant_gdn_projections(layer) -> None:
    layer.in_proj_qkv = _ConstantModule(torch.zeros(1, 24))
    layer.in_proj_z = _ConstantModule(torch.zeros(1, 8))
    layer.in_proj_b = _ConstantModule(torch.zeros(1, 2))
    layer.in_proj_a = _ConstantModule(torch.zeros(1, 2))
    layer.out_proj = torch.nn.Identity()


def _gdn_forward_context(*, is_prefill: bool) -> ForwardContext:
    metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([1], dtype=torch.int32),
        has_initial_state=(torch.tensor([False], dtype=torch.bool) if is_prefill else None),
        q_cu_seq_lens=(torch.tensor([0, 1], dtype=torch.int32) if is_prefill else None),
        is_prefill=is_prefill,
        is_chunked_prefill=False,
    )
    return ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[
            LayerCache(
                key=None,
                value=None,
                conv=torch.zeros(2, 3, 24),
                ssm=torch.zeros(2, 2, 4, 4),
            )
        ],
    )


def test_cuda_gdn_keeps_historical_prefill_kernel_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    calls = MagicMock()
    conv = MagicMock(return_value=torch.zeros(1, 24))
    post_conv = MagicMock(
        return_value=(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
        )
    )
    chunk = MagicMock(
        return_value=(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4, 4),
        )
    )
    rms = MagicMock(side_effect=lambda output, *_args: output)
    for name, mock in (
        ("conv", conv),
        ("post_conv", post_conv),
        ("chunk", chunk),
        ("rms", rms),
    ):
        calls.attach_mock(mock, name)
    monkeypatch.setattr(kernels, "causal_conv1d_prefill", conv, raising=False)
    monkeypatch.setattr(
        kernels,
        "fused_gdn_prefill_post_conv",
        post_conv,
        raising=False,
    )
    monkeypatch.setattr(kernels, "chunk_gated_delta_rule", chunk, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    layer = CudaQwen3_5GatedDeltaNet(
        _linear_config(),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    _install_constant_gdn_projections(layer)
    with forward_context(_gdn_forward_context(is_prefill=True)):
        output = layer(torch.zeros(1, 8))

    assert output.shape == (1, 8)
    assert [call[0] for call in calls.mock_calls] == [
        "conv",
        "post_conv",
        "chunk",
        "rms",
    ]


def test_npu_gdn_uses_npu_prefill_fusion_boundary(monkeypatch) -> None:
    calls = MagicMock()
    conv_qkv = MagicMock(
        return_value=(
            torch.zeros(1, 1, 2, 4),
            torch.zeros(1, 1, 2, 4),
            torch.zeros(1, 1, 2, 4),
        )
    )
    gating = MagicMock(
        return_value=(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
        )
    )
    chunk = MagicMock(
        return_value=(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4, 4),
        )
    )
    rms = MagicMock(side_effect=lambda output, *_args: output)
    for name, mock in (
        ("conv_qkv", conv_qkv),
        ("gating", gating),
        ("chunk", chunk),
        ("rms", rms),
    ):
        calls.attach_mock(mock, name)
    monkeypatch.setattr(
        kernels,
        "causal_conv1d_qkv_prefill",
        conv_qkv,
        raising=False,
    )
    monkeypatch.setattr(kernels, "fused_gdn_gating", gating, raising=False)
    monkeypatch.setattr(kernels, "chunk_gated_delta_rule", chunk, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    layer = NpuQwen3_5GatedDeltaNet(
        _linear_config(),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    _install_constant_gdn_projections(layer)
    with forward_context(_gdn_forward_context(is_prefill=True)):
        output = layer(torch.zeros(1, 8))

    assert output.shape == (1, 8)
    assert [call[0] for call in calls.mock_calls] == [
        "conv_qkv",
        "gating",
        "chunk",
        "rms",
    ]


def test_cuda_and_npu_decode_use_distinct_recurrent_kernels(monkeypatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    conv = MagicMock(return_value=torch.zeros(1, 24))
    cuda_recurrent = MagicMock(return_value=torch.zeros(1, 1, 2, 4))
    npu_recurrent = MagicMock(return_value=torch.zeros(1, 1, 2, 4))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    monkeypatch.setattr(kernels, "causal_conv1d_decode", conv, raising=False)
    monkeypatch.setattr(
        kernels,
        "fused_recurrent_gated_delta_rule_packed_decode",
        cuda_recurrent,
        raising=False,
    )
    monkeypatch.setattr(
        kernels,
        "fused_sigmoid_gating_delta_rule_decode",
        npu_recurrent,
        raising=False,
    )
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    cuda_layer = CudaQwen3_5GatedDeltaNet(
        _linear_config(),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    npu_layer = NpuQwen3_5GatedDeltaNet(
        _linear_config(),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    _install_constant_gdn_projections(cuda_layer)
    _install_constant_gdn_projections(npu_layer)

    with forward_context(_gdn_forward_context(is_prefill=False)):
        assert cuda_layer(torch.zeros(1, 8)).shape == (1, 8)
    with forward_context(_gdn_forward_context(is_prefill=False)):
        assert npu_layer(torch.zeros(1, 8)).shape == (1, 8)

    cuda_recurrent.assert_called_once()
    npu_recurrent.assert_called_once()


def test_npu_decode_passes_native_weight_and_cache_to_tilelang(
    monkeypatch,
) -> None:
    output = torch.zeros(1, 24)
    kernel = MagicMock(return_value=output)
    tilelang_wrapper = types.ModuleType("xllm.python.kernels_npu.tilelang.causal_conv1d_decode")
    tilelang_wrapper.DIM_PER_CORE = 2048
    tilelang_wrapper._build_decode_kernel_jit = MagicMock(return_value=kernel)
    monkeypatch.setitem(
        sys.modules,
        "xllm.python.kernels_npu.tilelang.causal_conv1d_decode",
        tilelang_wrapper,
    )
    value = torch.zeros(1, 24)
    weight = torch.zeros(4, 24)
    conv_state = torch.zeros(2, 3, 24)
    state_indices = torch.tensor([1], dtype=torch.int32)

    actual = npu_causal_conv1d_decode(
        value,
        weight,
        conv_state,
        state_indices,
    )

    assert actual is output
    args = kernel.call_args.args
    assert args[1] is weight
    assert args[2] is conv_state


def test_backend_split_preserves_public_parameter_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 16,
        "intermediate_size": 16,
        "layer_types": ["full_attention", "linear_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "vocab_size": 8,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 0,
        "shared_expert_intermediate_size": 0,
        "tp_size": 1,
        "world_size": 1,
        "moe_tp_size": 1,
        "dtype": "float32",
        "device": "cpu",
    }

    parameter_names = set(dict(Qwen3_5ForCausalLM(config).named_parameters()))

    assert "model.layers.0.self_attn.qkv_proj.weight" in parameter_names
    assert "model.layers.0.mlp.gate_up_proj.weight" in parameter_names
    assert "model.layers.1.linear_attn.conv1d_weight" in parameter_names
    assert "model.layers.1.mlp.gate_up_proj.weight" in parameter_names


def test_attention_biases_are_loaded() -> None:
    cfg_values = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 16,
        "intermediate_size": 16,
        "layer_types": ["full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "vocab_size": 8,
        "attention_bias": True,
        "attn_output_gate": False,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 0,
        "shared_expert_intermediate_size": 0,
        "tp_size": 1,
        "world_size": 1,
        "moe_tp_size": 1,
        "dtype": "float32",
        "device": "cpu",
    }
    model = Qwen3_5ForCausalLM(cfg_values)
    tensors = {
        "model.embed_tokens.weight": torch.zeros(8, 8),
        "model.layers.0.input_layernorm.weight": torch.zeros(8),
        "model.layers.0.post_attention_layernorm.weight": torch.zeros(8),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(4, 8),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(4, 8),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(8, 8),
        "model.layers.0.self_attn.q_proj.bias": torch.arange(8.0),
        "model.layers.0.self_attn.k_proj.bias": torch.arange(4.0) + 10,
        "model.layers.0.self_attn.v_proj.bias": torch.arange(4.0) + 20,
        "model.layers.0.self_attn.o_proj.bias": torch.arange(8.0) + 30,
        "model.layers.0.self_attn.q_norm.weight": torch.zeros(4),
        "model.layers.0.self_attn.k_norm.weight": torch.zeros(4),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(8, 16),
        "model.norm.weight": torch.zeros(8),
        "lm_head.weight": torch.zeros(8, 8),
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    expected_qkv_bias = torch.cat(
        (
            tensors["model.layers.0.self_attn.q_proj.bias"],
            tensors["model.layers.0.self_attn.k_proj.bias"],
            tensors["model.layers.0.self_attn.v_proj.bias"],
        )
    )
    torch.testing.assert_close(model.model.layers[0].self_attn.qkv_proj.bias, expected_qkv_bias)
    torch.testing.assert_close(
        model.model.layers[0].self_attn.o_proj.bias,
        tensors["model.layers.0.self_attn.o_proj.bias"],
    )


def test_full_attention_layers_share_rotary_table() -> None:
    cfg = _config(
        num_hidden_layers=2,
        layer_types=["full_attention", "full_attention"],
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        max_position_embeddings=16,
    )

    model = Qwen3_5Model(cfg, torch.float32, torch.device("cpu"))

    assert model.layers[0].self_attn.rotary is model.rotary
    assert model.layers[1].self_attn.rotary is model.rotary


def test_pre_sm90_moe_uses_triton_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (8, 0))
    assert not kernels.supports_cutlass_moe(torch.device("cuda:0"))
    cfg = _config(
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
    )
    layer = _make_moe_layer(cfg)
    assert not layer._use_cutlass
    layer.gate = torch.nn.Identity()
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(3, 1, dtype=torch.float32),
                torch.zeros(3, 1, dtype=torch.int32),
            )
        ),
    )
    triton_output = torch.ones(3, cfg.hidden_size)
    triton_moe = MagicMock(return_value=triton_output)
    monkeypatch.setattr(kernels, "fused_moe", triton_moe)

    output = layer(torch.zeros(3, cfg.hidden_size))

    assert output is triton_output
    triton_moe.assert_called_once()


def test_attention_tp_and_moe_tp_use_distinct_valid_topology():
    cfg = _config()
    cfg.validate()

    layer = _make_moe_layer(cfg)

    assert cfg.tp_size == 2
    assert cfg.dp_size == 1
    assert cfg.moe_tp_size == 2
    assert cfg.ep_size == 1
    assert layer.w13.shape == (8, 32, 64)
    assert layer.w2.shape == (8, 64, 16)


def test_attention_dp_and_moe_tp_use_distinct_valid_topology():
    cfg = _config(tp_size=1, dp_size=2)
    cfg.validate()

    layer = _make_moe_layer(cfg)

    assert cfg.tp_size == 1
    assert cfg.dp_size == 2
    assert cfg.moe_tp_size == 2
    assert cfg.ep_size == 1
    assert layer.w13.shape == (8, 32, 64)
    assert layer.w2.shape == (8, 64, 16)


def test_full_world_ep_partitions_experts_without_inner_tp_sharding():
    cfg = _config(
        world_size=4,
        ep_size=4,
        ep_rank=3,
        dp_size=2,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    cfg.validate()

    layer = _make_moe_layer(cfg)

    assert layer.w13.shape == (2, 64, 64)
    assert layer.w2.shape == (2, 64, 32)


def test_partial_world_ep_is_rejected():
    cfg = _config(world_size=4, dp_size=2, ep_size=2, moe_tp_size=2)

    with pytest.raises(ValueError, match="ep_size=1 or world_size"):
        cfg.validate()


def test_dense_model_does_not_require_expert_parallelism():
    cfg = _config(
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
    )

    cfg.validate()


def test_moe_parallel_product_must_equal_world_size():
    cfg = _config(world_size=4, dp_size=2, ep_size=4, moe_tp_size=2)

    with pytest.raises(ValueError, match=r"moe_tp_size \* ep_size"):
        cfg.validate()


def test_attention_parallel_product_must_equal_world_size():
    cfg = _config(world_size=4)

    with pytest.raises(ValueError, match=r"tp_size \* dp_size"):
        cfg.validate()


def test_gemma_rms_norm_matches_fp32_weight_reference():
    layer = GemmaRMSNorm(4, dtype=torch.bfloat16, device=torch.device("cpu"))
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.3242, -0.8164, 2.4844, -0.2383]))

    hidden = torch.tensor([[0.4258, -0.2656, 1.4922, -0.7344]], dtype=torch.bfloat16)
    residual = torch.tensor([[-0.1064, 0.9844, -0.3789, 0.5469]], dtype=torch.bfloat16)
    actual, actual_residual = layer(hidden, residual)

    summed = hidden.float() + residual.float()
    expected_residual = summed.to(torch.bfloat16)
    variance = summed.pow(2).mean(dim=-1, keepdim=True)
    expected = summed * torch.rsqrt(variance + layer.eps)
    expected = (expected * (layer.weight + 1.0)).to(torch.bfloat16)

    assert layer.weight.dtype == torch.float32
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
