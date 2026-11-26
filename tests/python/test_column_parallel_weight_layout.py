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

"""Behavior tests for backend-selected ColumnParallel weight preparation."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from xllm.python.layers import linear
from xllm.python.models.qwen3 import Qwen3Config, Qwen3DecoderLayer, Qwen3ForCausalLM
from xllm.python.models.qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionTransformer


def test_column_parallel_weight_preparation_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = linear.ColumnParallelLinear(3, 5, 1, bias=True, prepare_weight_layout=True)
    layer.weight.data.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3))
    layer.bias.data.copy_(torch.arange(5, dtype=torch.float32))
    x = torch.tensor([[1.0, -2.0, 3.0]])
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
    prepare = Mock(side_effect=lambda weight: (weight.transpose(0, 1).contiguous(), True))
    monkeypatch.setattr(
        linear.kernels,
        "prepare_column_parallel_weight",
        prepare,
    )

    layer.process_weights_after_loading()
    actual = layer(x)
    layer.process_weights_after_loading()

    torch.testing.assert_close(actual, expected)
    assert layer.weight.shape == (3, 5)
    prepare.assert_called_once()
    prepared_data_ptr = layer.weight.data_ptr()

    layer.prepare_weight_for_loading()
    assert layer.weight.shape == (5, 3)
    reloaded = torch.full((5, 3), 2.0)
    layer.weight.data.copy_(reloaded)
    expected_after_reload = torch.nn.functional.linear(x, reloaded, layer.bias)
    layer.process_weights_after_loading()

    assert layer.weight.shape == (3, 5)
    assert layer.weight.data_ptr() == prepared_data_ptr
    torch.testing.assert_close(layer(x), expected_after_reload)
    assert prepare.call_count == 2


def test_column_parallel_weight_layout_requires_layer_eligibility(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = linear.ColumnParallelLinear(3, 5, 1)
    prepare = Mock()
    monkeypatch.setattr(
        linear.kernels,
        "prepare_column_parallel_weight",
        prepare,
    )

    layer.process_weights_after_loading()

    assert layer.weight.shape == (5, 3)
    prepare.assert_not_called()


@pytest.mark.parametrize("prepare_weight_layout", [False, True])
def test_greedy_tokens_uses_vocab_shard_size_after_weight_layout(
    monkeypatch: pytest.MonkeyPatch,
    prepare_weight_layout: bool,
) -> None:
    layer = linear.ColumnParallelLinear(
        3,
        5,
        2,
        prepare_weight_layout=prepare_weight_layout,
    )
    layer.weight.data.copy_(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        )
    )
    if prepare_weight_layout:
        monkeypatch.setattr(
            linear.kernels,
            "prepare_column_parallel_weight",
            lambda weight: (weight.transpose(0, 1).contiguous(), True),
        )
        layer.process_weights_after_loading()

    def gather_rank_candidates(value: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
        assert dim == -1
        assert world_size == 2
        rank_one = torch.tensor([[8.0, 3.0]], dtype=value.dtype)
        return torch.cat((value, rank_one), dim=dim)

    monkeypatch.setattr(linear.distributed, "tp_all_gather", gather_rank_candidates)

    actual = layer.greedy_tokens(torch.tensor([[1.0, 0.0, 0.0]]))

    assert torch.equal(actual, torch.tensor([8]))


def test_qwen3_enables_backend_layout_for_all_target_projections() -> None:
    cfg = Qwen3Config(
        hidden_size=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=4,
        intermediate_size=16,
        vocab_size=16,
    )
    layer = Qwen3DecoderLayer(cfg, 0, torch.float32, torch.device("cpu"))
    model = Qwen3ForCausalLM(
        {
            "hidden_size": 8,
            "n_layers": 0,
            "n_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "vocab_size": 16,
            "dtype": "float32",
            "device": "cpu",
        }
    )

    assert layer.self_attn.qkv_proj._prepare_weight_layout
    assert layer.mlp.gate_up_proj._prepare_weight_layout
    assert model.lm_head is not None
    assert model.lm_head._prepare_weight_layout


def test_qwen3_vl_enables_backend_layout_for_vision_projections() -> None:
    vision = Qwen3VLVisionTransformer(
        Qwen3VLVisionConfig(
            deepstack_visual_indexes=[],
            depth=1,
            hidden_size=8,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            intermediate_size=16,
            out_hidden_size=8,
            num_position_embeddings=16,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert vision.blocks[0].attn.qkv._prepare_weight_layout
    assert vision.blocks[0].mlp.linear_fc1._prepare_weight_layout
