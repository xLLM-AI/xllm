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

from xllm.python.layers.fused_moe import FusedMoE  # noqa: E402
from xllm.python.layers.gated_mlp import GatedMLP  # noqa: E402
from xllm.python.layers.layernorm import GemmaRMSNorm  # noqa: E402
from xllm.python.models.qwen3_5 import (  # noqa: E402
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5Model,
    Qwen3_5SparseMoEBlock,
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
