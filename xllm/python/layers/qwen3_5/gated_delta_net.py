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

"""Backend-neutral checkpoint helpers for Qwen3.5 gated delta nets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from xllm.python.layers.qwen3_5.common import Qwen3_5GatedDeltaNetConfig
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader

if TYPE_CHECKING:
    from xllm.python.layers.linear import RowParallelLinear


class _GdnWeightViews(Protocol):
    cfg: Qwen3_5GatedDeltaNetConfig
    conv1d_weight: torch.nn.Parameter
    norm_weight: torch.nn.Parameter
    out_proj: RowParallelLinear


def shard_qkv_rows(
    state: ScopedWeightLoader,
    tensor: torch.Tensor,
    local_name: str,
    cfg: Qwen3_5GatedDeltaNetConfig,
    context: ParallelLoadContext,
) -> torch.Tensor:
    """Split checkpoint Q/K/V rows and pack this rank's local TP shard."""
    global_key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    global_value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    q, k, v = tensor.split(
        (global_key_dim, global_key_dim, global_value_dim),
        dim=0,
    )
    return torch.cat(
        [
            state.shard_value(
                part,
                f"{local_name}[{tag}]",
                0,
                context.tp_rank,
                context.tp_size,
            )
            for part, tag in ((q, "q"), (k, "k"), (v, "v"))
        ]
    )


def load_gdn_weights(
    layer: _GdnWeightViews,
    state: ScopedWeightLoader,
    context: ParallelLoadContext,
    input_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    transpose_conv: bool = False,
) -> None:
    """Load the shared Qwen3.5 GDN checkpoint layout into backend weight views."""
    qkv_weight, z_weight, b_weight, a_weight = input_weights
    state.copy(
        qkv_weight,
        shard_qkv_rows(
            state,
            state.get_tensor("in_proj_qkv.weight"),
            "in_proj_qkv.weight",
            layer.cfg,
            context,
        ),
        "in_proj_qkv.weight",
    )
    for target, name in (
        (z_weight, "in_proj_z"),
        (b_weight, "in_proj_b"),
        (a_weight, "in_proj_a"),
    ):
        state.load_tensor(
            target,
            f"{name}.weight",
            dim=0,
            rank=context.tp_rank,
            world_size=context.tp_size,
        )
    local_conv = shard_qkv_rows(
        state,
        state.get_tensor("conv1d.weight").squeeze(1),
        "conv1d.weight",
        layer.cfg,
        context,
    )
    if transpose_conv:
        local_conv = local_conv.transpose(0, 1).contiguous()
    state.copy(layer.conv1d_weight, local_conv, "conv1d.weight")
    for name in ("A_log", "dt_bias"):
        state.load_tensor(
            getattr(layer, name),
            name,
            dim=0,
            rank=context.tp_rank,
            world_size=context.tp_size,
        )
    state.load_tensor(layer.norm_weight, "norm.weight")
    state.load_tensor(
        layer.out_proj.weight,
        "out_proj.weight",
        dim=1,
        rank=context.tp_rank,
        world_size=context.tp_size,
    )
    layer.out_proj.process_weights_after_loading()
