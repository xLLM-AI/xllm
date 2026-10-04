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

from xllm.python import distributed, kernels
from xllm.python.models.qwen3 import Qwen3Model
from xllm.python.models.qwen3_dflash import (
    DFlashContextProjection,
    DFlashQwen3Config,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


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
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


def _config(**overrides) -> DFlashQwen3Config:
    config = DFlashQwen3Config.from_dict(_config_dict(**overrides))
    config.validate()
    return config


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def test_top_level_and_nested_rope_config_formats_are_supported() -> None:
    top_level_rope = _config(model_type="qwen3", rope_theta=10000.0)
    nested_rope = _config(
        model_type="qwen3",
        rope_theta=None,
        rope_parameters={"rope_theta": 1e7},
    )

    assert top_level_rope.rope_theta == 10000.0
    assert nested_rope.rope_theta == 1e7


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dp_size": 2, "world_size": 1}, "world_size must equal"),
        ({"dp_size": 2, "dp_rank": 2, "world_size": 2}, "dp_rank must be"),
        ({"tp_rank": 1}, "tp_rank must be"),
    ],
)
def test_config_rejects_invalid_parallel_settings(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_config_rejects_reduced_draft_vocabulary() -> None:
    with pytest.raises(ValueError, match="reduced-vocabulary"):
        _config(draft_vocab_size=4)


def test_config_rejects_aux_hidden_capture() -> None:
    with pytest.raises(ValueError, match="does not support layers_to_capture"):
        _config(layers_to_capture=[0])


def test_draft_attention_is_non_causal() -> None:
    model = DFlashQwen3Model(_config(), torch.float32, torch.device("cpu"))

    assert isinstance(model, Qwen3Model)
    assert model.embed_tokens is None
    assert not model.layers[0].self_attn.attn.causal


def test_context_projection_uses_tensor_parallel_output_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = DFlashContextProjection(
        out_features=4,
        tp_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    hidden = torch.tensor([[1.0, 2.0, 3.0]])
    rank_zero_output = torch.nn.functional.linear(hidden, weight[:2])
    all_gather = Mock(
        side_effect=lambda local_output, **_: torch.cat(
            (rank_zero_output, local_output),
            dim=-1,
        )
    )
    monkeypatch.setattr(distributed, "all_gather", all_gather, raising=False)

    projection.load_weight(weight, tp_rank=1)
    output = projection(hidden)

    torch.testing.assert_close(projection.weight, weight[2:])
    torch.testing.assert_close(output, torch.nn.functional.linear(hidden, weight))
    all_gather.assert_called_once()
    torch.testing.assert_close(
        all_gather.call_args.args[0],
        torch.nn.functional.linear(hidden, weight[2:]),
    )
    assert all_gather.call_args.kwargs == {"dim": -1, "world_size": 2}


def test_context_projection_writes_each_layer_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(hidden_size=2, head_dim=2, num_hidden_layers=2)
    model = DFlashQwen3Model(config, torch.float32, torch.device("cpu"))
    key_weights = torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[0.0, 1.0], [3.0, 0.0]]])
    with torch.no_grad():
        model.fc.load_weight(torch.eye(2), tp_rank=0)
        for layer_id, layer in enumerate(model.layers):
            value_weight = (layer_id + 2) * torch.eye(2)
            layer.self_attn.qkv_proj.weight.copy_(torch.cat((torch.zeros(2, 2), key_weights[layer_id], value_weight)))
        model._build_context_kv_buffers()

    def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps) * weight

    trace: list[tuple[str, int]] = []
    caches = [(torch.full((1, 4, 1, 2), -10.0), torch.full((1, 4, 1, 2), -20.0)) for _ in range(2)]

    def _write_cache(
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> None:
        key_cache.view(4, 1, 2).index_copy_(0, slots.long(), keys)
        value_cache.view(4, 1, 2).index_copy_(0, slots.long(), values)
        trace.append(("cache", next(i for i, cache in enumerate(caches) if cache[0] is key_cache)))

    def _record_event(layer_id: int) -> None:
        trace.append(("event", layer_id))

    monkeypatch.setattr(kernels, "rms_norm", _rms_norm, raising=False)
    monkeypatch.setattr(kernels, "reshape_paged_cache", _write_cache, raising=False)
    synchronizer = Mock(record_event=Mock(side_effect=_record_event))
    hidden = torch.tensor([[2.0, 2.0], [2.0, -2.0]])
    positions, slots = torch.tensor([1, 2]), torch.tensor([3, 1], dtype=torch.int32)
    projected = model.write_context_kv(hidden, positions, slots, caches, synchronizer)

    expected_hidden = hidden / (4.0 + config.rms_norm_eps) ** 0.5
    torch.testing.assert_close(projected, expected_hidden)
    for layer_id, (key_cache, value_cache) in enumerate(caches):
        keys = expected_hidden @ key_weights[layer_id].T
        keys /= torch.sqrt(keys.square().mean(-1, keepdim=True) + config.rms_norm_eps)
        cos, sin = positions.float().cos(), positions.float().sin()
        rotated = torch.stack((keys[:, 0] * cos - keys[:, 1] * sin, keys[:, 1] * cos + keys[:, 0] * sin), dim=-1)
        expected_keys, expected_values = torch.full((1, 4, 1, 2), -10.0), torch.full((1, 4, 1, 2), -20.0)
        expected_keys[0, [3, 1]] = rotated.unsqueeze(1)
        expected_values[0, [3, 1]] = ((layer_id + 2) * expected_hidden).unsqueeze(1)
        torch.testing.assert_close(key_cache, expected_keys)
        torch.testing.assert_close(value_cache, expected_values)
    assert trace == [("cache", 0), ("event", 0), ("cache", 1), ("event", 1)]


def test_checkpoint_weight_names_load_into_fused_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        lambda weight: (weight, False),
        raising=False,
    )
    model = DFlashQwen3ForCausalLM(_config_dict())
    tensors = {
        "fc.weight": torch.ones(4, 8),
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
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    qkv_weight = model.model.layers[0].self_attn.qkv_proj.weight
    torch.testing.assert_close(
        qkv_weight[:4],
        tensors["layers.0.self_attn.q_proj.weight"],
    )
    torch.testing.assert_close(
        qkv_weight[4:8],
        tensors["layers.0.self_attn.k_proj.weight"],
    )
    torch.testing.assert_close(
        qkv_weight[8:12],
        tensors["layers.0.self_attn.v_proj.weight"],
    )
    assert model.model._fused_kv_weight.shape == (8, 4)
    assert model.model.fc.weight.shape == (4, 8)
    assert model.model.embed_tokens is None
    assert model.lm_head is None
