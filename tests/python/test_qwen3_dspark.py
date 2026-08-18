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

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from xllm.python import kernels
from xllm.python.models.qwen3_dspark import (
    Qwen3DSparkConfig,
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)


def _config(**overrides) -> Qwen3DSparkConfig:
    values = _config_dict(**overrides)
    config = Qwen3DSparkConfig.from_dict(values)
    config.validate()
    return config


def _config_dict(**overrides) -> dict:
    values = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 16,
        "vocab_size": 8,
        "draft_vocab_size": 8,
        "markov_rank": 2,
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def test_config_rejects_reduced_draft_vocabulary() -> None:
    with pytest.raises(ValueError, match="reduced-vocabulary"):
        _config(draft_vocab_size=4)


def test_draft_attention_is_non_causal() -> None:
    model = Qwen3DSparkModel(_config(), torch.float32, torch.device("cpu"))

    assert not model.layers[0].self_attn.attn.causal


def test_context_projection_writes_each_layer_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    model = Qwen3DSparkModel(config, torch.float32, torch.device("cpu"))
    attention = model.layers[0].self_attn
    with torch.no_grad():
        model.fc.load_weight(torch.eye(config.hidden_size))
        model.hidden_norm.weight.fill_(1.0)
        attention.qkv_proj.weight.zero_()
        attention.qkv_proj.weight[attention.q_size : attention.q_size + attention.kv_size].copy_(
            torch.eye(config.hidden_size)
        )
        attention.qkv_proj.weight[attention.q_size + attention.kv_size :].copy_(2.0 * torch.eye(config.hidden_size))
        attention.k_norm.weight.fill_(1.0)
    monkeypatch.setattr(
        kernels,
        "rms_norm",
        lambda hidden, weight, eps: hidden
        * torch.rsqrt(hidden.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        * weight,
        raising=False,
    )
    reshape_paged_cache = Mock()
    monkeypatch.setattr(kernels, "reshape_paged_cache", reshape_paged_cache, raising=False)
    synchronizer = Mock()
    key_cache = torch.empty(1, 1, 1, config.head_dim)
    value_cache = torch.empty_like(key_cache)
    target_hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    projected = model.write_context_kv(
        target_hidden,
        torch.tensor([0]),
        torch.tensor([0], dtype=torch.int32),
        [(key_cache, value_cache, None, None, None)],
        synchronizer,
    )

    reshape_paged_cache.assert_called_once()
    call_args = reshape_paged_cache.call_args.args
    torch.testing.assert_close(call_args[2], 2.0 * projected.view(1, 1, config.head_dim))
    assert call_args[3] is key_cache
    assert call_args[4] is value_cache
    synchronizer.record_event.assert_called_once_with(0)


def test_checkpoint_weight_names_load_into_fused_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        lambda weight: (weight, False),
        raising=False,
    )
    model = Qwen3DSparkForCausalLM(_config_dict())
    tensors = {
        "fc.weight": torch.eye(4),
        "hidden_norm.weight": torch.ones(4),
        "layers.0.input_layernorm.weight": torch.ones(4),
        "layers.0.post_attention_layernorm.weight": torch.ones(4),
        "layers.0.self_attn.q_norm.weight": torch.ones(4),
        "layers.0.self_attn.k_norm.weight": torch.ones(4),
        "layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
        "layers.0.self_attn.k_proj.weight": torch.full((4, 4), 2.0),
        "layers.0.self_attn.v_proj.weight": torch.full((4, 4), 3.0),
        "layers.0.self_attn.o_proj.weight": torch.full((4, 4), 4.0),
        "layers.0.mlp.gate_proj.weight": torch.full((8, 4), 5.0),
        "layers.0.mlp.up_proj.weight": torch.full((8, 4), 6.0),
        "layers.0.mlp.down_proj.weight": torch.full((4, 8), 7.0),
        "norm.weight": torch.ones(4),
        "markov_head.markov_w1.weight": torch.full((8, 2), 8.0),
        "markov_head.markov_w2.weight": torch.full((8, 2), 9.0),
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    qkv_weight = model.model.layers[0].self_attn.qkv_proj.weight
    torch.testing.assert_close(qkv_weight[:4], tensors["layers.0.self_attn.q_proj.weight"])
    torch.testing.assert_close(qkv_weight[4:8], tensors["layers.0.self_attn.k_proj.weight"])
    torch.testing.assert_close(qkv_weight[8:12], tensors["layers.0.self_attn.v_proj.weight"])
    assert model.model._fused_kv_weight.shape == (8, 4)
    assert model.model.embed_tokens is None
    assert model.lm_head is None
