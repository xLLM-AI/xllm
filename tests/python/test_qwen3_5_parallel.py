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

"""Backend-independent parallel-contract tests for Python Qwen3.5."""

from __future__ import annotations

import pytest
import torch

from tests.python.qwen3_5_test_utils import (
    StateDict as _StateDict,
)
from tests.python.qwen3_5_test_utils import (
    gemma_rms_norm as _gemma_rms_norm,
)
from tests.python.qwen3_5_test_utils import (
    make_config as _config,
)
from xllm.python import kernels
from xllm.python.layers.layernorm import GemmaRMSNorm
from xllm.python.layers.qwen3_5_decoder_layer import (
    get_qwen3_5_decoder_layer_class,
)
from xllm.python.model_loader import ScopedWeightLoader

kernels.gemma_rms_norm = _gemma_rms_norm


def test_backend_decoder_factory_rejects_unknown_device() -> None:
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


def test_partial_world_ep_and_moe_tp_form_a_valid_topology():
    cfg = _config(world_size=4, dp_size=2, ep_size=2, moe_tp_size=2)

    cfg.validate()

    assert cfg.tp_size == 2
    assert cfg.dp_size == 2
    assert cfg.moe_tp_size == 2
    assert cfg.ep_size == 2


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


def test_full_attention_kv_heads_support_replication() -> None:
    cfg = _config(
        tp_size=4,
        world_size=4,
        moe_tp_size=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )

    cfg.validate()

    assert cfg.head_split() == (2, 1)


def test_full_attention_rejects_invalid_kv_replication() -> None:
    cfg = _config(
        tp_size=4,
        world_size=4,
        moe_tp_size=4,
        num_attention_heads=8,
        num_key_value_heads=3,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )

    with pytest.raises(ValueError, match="tp_size must be divisible by KV heads"):
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


def test_gemma_rms_norm_delegates_to_backend_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def backend_gemma_rms_norm(
        value: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        calls.append((value, weight, eps))
        variance = value.pow(2).mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(variance + eps) * (weight + 1.0)

    monkeypatch.setattr(
        kernels,
        "gemma_rms_norm",
        backend_gemma_rms_norm,
        raising=False,
    )
    layer = GemmaRMSNorm(4, dtype=torch.bfloat16, device=torch.device("cpu"))
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.3242, -0.8164, 2.4844, -0.2383]))

    hidden = torch.tensor(
        [[0.4258, -0.2656, 1.4922, -0.7344]],
        dtype=torch.bfloat16,
    )
    residual = torch.tensor(
        [[-0.1064, 0.9844, -0.3789, 0.5469]],
        dtype=torch.bfloat16,
    )
    actual, actual_residual = layer(hidden, residual)

    assert len(calls) == 1
    native_input, native_weight, native_eps = calls[0]
    torch.testing.assert_close(
        native_input,
        hidden.float() + residual.float(),
        rtol=0,
        atol=0,
    )
    assert native_input.dtype == torch.float32
    assert native_weight is layer.weight
    assert native_eps == layer.eps
    torch.testing.assert_close(
        actual_residual,
        native_input.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    variance = native_input.pow(2).mean(dim=-1, keepdim=True)
    expected = native_input * torch.rsqrt(variance + native_eps)
    expected = (expected * (native_weight + 1.0)).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
