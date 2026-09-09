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

"""Shared backend-neutral helpers for Qwen3.5 Python tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.forward_context import ForwardContext
from xllm.python.models.qwen3_5 import Qwen3_5Config


def gemma_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(variance + eps)
    return normalized * (weight + 1.0)


def make_config(**overrides) -> Qwen3_5Config:
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


class StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


class ConstantModule(torch.nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, _hidden: torch.Tensor) -> torch.Tensor:
        return self.value


def make_linear_config() -> Qwen3_5Config:
    return make_config(
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


def install_constant_gdn_projections(layer) -> None:
    layer.in_proj_qkv = ConstantModule(torch.zeros(1, 24))
    layer.in_proj_z = ConstantModule(torch.zeros(1, 8))
    layer.in_proj_b = ConstantModule(torch.zeros(1, 2))
    layer.in_proj_a = ConstantModule(torch.zeros(1, 2))
    layer.out_proj = torch.nn.Identity()


def make_gdn_forward_context(*, is_prefill: bool) -> ForwardContext:
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
