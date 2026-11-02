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

import pytest
import torch

from xllm.python import kernels
from xllm.python.models.qwen3_dspark import (
    Qwen3DSparkConfig,
    Qwen3DSparkForCausalLM,
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


def test_config_requires_positive_markov_rank() -> None:
    with pytest.raises(ValueError, match="markov_rank > 0"):
        _config(markov_rank=0)


@pytest.mark.parametrize("non_causal", [None, False, True])
def test_windowed_dspark_keeps_checkpoint_block_visibility(non_causal: bool | None) -> None:
    config = _config_dict(sliding_window=2048)
    if non_causal is not None:
        config["sliding_window_non_causal"] = non_causal
    model = Qwen3DSparkForCausalLM(config)
    attention = model.model.layers[0].self_attn.attn
    assert attention.causal == (not non_causal)
    assert attention.sliding_window == 2048
    assert attention.fia_sparse_mode == 0


@pytest.mark.parametrize("window", [None, 0, -1])
def test_dspark_without_window_keeps_non_causal_block_visibility(window: int | None) -> None:
    model = Qwen3DSparkForCausalLM(_config_dict(sliding_window=window, sliding_window_non_causal=False))
    assert not model.model.layers[0].self_attn.attn.causal


@pytest.mark.parametrize("window", [None, 0, -1, 2048])
def test_disabled_window_keeps_dspark_full_block_visibility(window: int | None) -> None:
    model = Qwen3DSparkForCausalLM(_config_dict(sliding_window=window, use_sliding_window=False))
    attention = model.model.layers[0].self_attn.attn
    assert attention.sliding_window == 0
    assert not attention.causal


def _checkpoint_tensors(draft_vocab_size: int = 8) -> dict[str, torch.Tensor]:
    return {
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
        "markov_head.markov_w2.weight": torch.full((draft_vocab_size, 2), 9.0),
        "confidence_head.proj.weight": torch.full((1, 6), 10.0),
        "confidence_head.proj.bias": torch.full((1,), 11.0),
    }


def test_checkpoint_loads_dspark_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        lambda weight: (weight, False),
        raising=False,
    )
    model = Qwen3DSparkForCausalLM(
        _config_dict(
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
    )
    tensors = _checkpoint_tensors()

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    torch.testing.assert_close(
        model.markov_head.markov_w1.weight,
        tensors["markov_head.markov_w1.weight"],
    )
    torch.testing.assert_close(
        model.markov_head.markov_w2.weight,
        tensors["markov_head.markov_w2.weight"],
    )
    assert model.confidence_head is not None
    torch.testing.assert_close(
        model.confidence_head.proj.weight,
        tensors["confidence_head.proj.weight"],
    )
    torch.testing.assert_close(
        model.confidence_head.proj.bias,
        tensors["confidence_head.proj.bias"],
    )


def _load_reduced_model(monkeypatch: pytest.MonkeyPatch, tensors: dict) -> Qwen3DSparkForCausalLM:
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        lambda weight: (weight, False),
        raising=False,
    )
    model = Qwen3DSparkForCausalLM(_config_dict(draft_vocab_size=4))
    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)
    return model


def test_reduced_dspark_loads_d2t_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    tensors = _checkpoint_tensors(draft_vocab_size=4)
    tensors["d2t"] = torch.tensor([0, 1, 3, 4], dtype=torch.int64)
    model = _load_reduced_model(monkeypatch, tensors)

    assert model.hot_token_id.tolist() == [0, 2, 5, 7]


def test_reduced_dspark_requires_d2t_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="must provide a d2t table"):
        _load_reduced_model(monkeypatch, _checkpoint_tensors(draft_vocab_size=4))
