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

"""NPU-owned Qwen3.5 full-attention composition."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers.attention import Attention
from xllm.python.layers.layernorm import GemmaRMSNorm
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.qwen3_5_decoder_layer import (
    PartialRotaryEmbedding,
    Qwen3_5LayerConfig,
)
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    copy_parameter,
)


class NpuQwen3_5Attention(nn.Module):
    def __init__(
        self,
        cfg: Qwen3_5LayerConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.num_heads, self.num_kv_heads = cfg.head_split()
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        q_multiplier = 2 if cfg.attn_output_gate else 1
        self.attn_output_gate = cfg.attn_output_gate
        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            q_multiplier * self.q_size + 2 * self.kv_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            cfg.tp_size,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.q_norm = GemmaRMSNorm(
            self.head_dim,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.k_norm = GemmaRMSNorm(
            self.head_dim,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.rotary = rotary
        self.attn = Attention(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.head_dim**-0.5,
            0,
            layer_id,
        )

    def _shard_kv(
        self,
        state: ScopedWeightLoader,
        name: str,
        context: ParallelLoadContext,
    ) -> torch.Tensor:
        if self.cfg.n_kv_heads >= context.tp_size:
            return state.shard(name, 0, context.tp_rank, context.tp_size)
        replicas = context.tp_size // self.cfg.n_kv_heads
        return state.shard(
            name,
            0,
            context.tp_rank // replicas,
            self.cfg.n_kv_heads,
        )

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        q = state.shard("q_proj.weight", 0, context.tp_rank, context.tp_size)
        k = self._shard_kv(state, "k_proj.weight", context)
        v = self._shard_kv(state, "v_proj.weight", context)
        copy_parameter(
            self.qkv_proj.weight,
            torch.cat((q, k, v)),
            state.prefix + "{q,k,v}_proj.weight",
        )
        copy_parameter(
            self.o_proj.weight,
            state.shard("o_proj.weight", 1, context.tp_rank, context.tp_size),
            state.prefix + "o_proj.weight",
        )
        if self.cfg.attention_bias:
            q_bias = state.shard(
                "q_proj.bias",
                0,
                context.tp_rank,
                context.tp_size,
            )
            k_bias = self._shard_kv(state, "k_proj.bias", context)
            v_bias = self._shard_kv(state, "v_proj.bias", context)
            assert self.qkv_proj.bias is not None
            assert self.o_proj.bias is not None
            copy_parameter(
                self.qkv_proj.bias,
                torch.cat((q_bias, k_bias, v_bias)),
                state.prefix + "{q,k,v}_proj.bias",
            )
            copy_parameter(
                self.o_proj.bias,
                state.tensor("o_proj.bias"),
                state.prefix + "o_proj.bias",
            )
        copy_parameter(
            self.q_norm.weight,
            state.tensor("q_norm.weight"),
            state.prefix + "q_norm.weight",
        )
        copy_parameter(
            self.k_norm.weight,
            state.tensor("k_norm.weight"),
            state.prefix + "k_norm.weight",
        )
        # TODO: Prepare the NPU row-parallel weight after TileLang and
        # CANN/TBE TVM runtimes can coexist in the same process.

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [2 * self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
            q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim)
            q, gate = q_gate.chunk(2, dim=-1)
        else:
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
            q = q.view(-1, self.num_heads, self.head_dim)
            gate = None
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rotary(positions, q).reshape(-1, self.q_size)
        k = self.rotary(positions, k).reshape(-1, self.kv_size)
        output = self.attn(q, k, v)
        if gate is not None:
            output = output * torch.sigmoid(gate.reshape(-1, self.q_size))
        return self.o_proj(output)
