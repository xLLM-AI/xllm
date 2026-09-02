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

"""CUDA-native Qwen3.5 gated delta network composition."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.qwen3_5_decoder_layer import (
    Qwen3_5LayerConfig,
)
from xllm.python.model_executor.forward_context import get_forward_context
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    copy_parameter,
)


class CudaQwen3_5GatedDeltaNet(nn.Module):
    """CUDA graph matching the pre-NPU Qwen3.5 implementation."""

    def __init__(
        self,
        cfg: Qwen3_5LayerConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.num_k_heads = cfg.linear_num_key_heads // cfg.tp_size
        self.num_v_heads = cfg.linear_num_value_heads // cfg.tp_size
        self.key_head_dim = cfg.linear_key_head_dim
        self.value_head_dim = cfg.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.norm_eps = cfg.rms_norm_eps
        self.gdn_prefill_backend = kernels.resolve_gdn_prefill_backend()

        self.in_proj_qkv = ColumnParallelLinear(
            cfg.hidden_size,
            self.conv_dim,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.in_proj_z = ColumnParallelLinear(
            cfg.hidden_size,
            self.value_dim,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.in_proj_b = ColumnParallelLinear(
            cfg.hidden_size,
            self.num_v_heads,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.in_proj_a = ColumnParallelLinear(
            cfg.hidden_size,
            self.num_v_heads,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.conv1d_weight = nn.Parameter(
            torch.empty(
                self.conv_dim,
                self.conv_kernel_size,
                dtype=dtype,
                device=device,
            )
        )
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32, device=device))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, dtype=dtype, device=device))
        self.norm_weight = nn.Parameter(torch.ones(self.value_head_dim, dtype=dtype, device=device))
        self.out_proj = RowParallelLinear(
            self.value_dim,
            cfg.hidden_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )

    def _cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache = get_forward_context().layer_caches[self.layer_id]
        if cache.conv is None or cache.ssm is None:
            raise RuntimeError(f"linear-attention cache is missing for layer {self.layer_id}")
        if cache.conv.dim() != 3 or self.conv_dim not in cache.conv.shape[1:]:
            raise ValueError("CUDA Qwen3.5 conv cache has an unexpected shape")
        return cache.conv, cache.ssm

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        global_key = self.cfg.linear_num_key_heads * self.key_head_dim
        global_value = self.cfg.linear_num_value_heads * self.value_head_dim
        q, k, v = state.tensor("in_proj_qkv.weight").split(
            (global_key, global_key, global_value),
            dim=0,
        )
        local_qkv = torch.cat(
            (
                q.chunk(context.tp_size, dim=0)[context.tp_rank],
                k.chunk(context.tp_size, dim=0)[context.tp_rank],
                v.chunk(context.tp_size, dim=0)[context.tp_rank],
            )
        )
        copy_parameter(
            self.in_proj_qkv.weight,
            local_qkv,
            state.prefix + "in_proj_qkv.weight",
        )
        for name in ("in_proj_z", "in_proj_b", "in_proj_a"):
            projection = getattr(self, name)
            copy_parameter(
                projection.weight,
                state.shard(
                    f"{name}.weight",
                    0,
                    context.tp_rank,
                    context.tp_size,
                ),
                state.prefix + f"{name}.weight",
            )

        conv = state.tensor("conv1d.weight").squeeze(1)
        cq, ck, cv = conv.split((global_key, global_key, global_value), dim=0)
        local_conv = torch.cat(
            (
                cq.chunk(context.tp_size, dim=0)[context.tp_rank],
                ck.chunk(context.tp_size, dim=0)[context.tp_rank],
                cv.chunk(context.tp_size, dim=0)[context.tp_rank],
            )
        )
        copy_parameter(
            self.conv1d_weight,
            local_conv,
            state.prefix + "conv1d.weight",
        )
        for name in ("A_log", "dt_bias"):
            copy_parameter(
                getattr(self, name),
                state.shard(name, 0, context.tp_rank, context.tp_size),
                state.prefix + name,
            )
        copy_parameter(
            self.norm_weight,
            state.tensor("norm.weight"),
            state.prefix + "norm.weight",
        )
        copy_parameter(
            self.out_proj.weight,
            state.shard(
                "out_proj.weight",
                1,
                context.tp_rank,
                context.tp_size,
            ),
            state.prefix + "out_proj.weight",
        )
        self.out_proj.process_weights_after_loading()

    def _gdn_prefill(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        non_null_state = state_indices > 0
        use_initial_state = non_null_state & has_initial_state
        cache_indices = state_indices.to(torch.long)
        null_state = ssm_state[0].clone()
        initial_state = ssm_state.index_select(0, cache_indices).float().contiguous()
        initial_state = torch.where(
            use_initial_state[:, None, None, None],
            initial_state,
            torch.zeros_like(initial_state),
        )
        q, k, v, g, beta = kernels.fused_gdn_prefill_post_conv(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            num_key_heads=self.num_k_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
        )
        output, final_state = kernels.chunk_gated_delta_rule(
            q,
            k,
            v.contiguous(),
            g,
            beta,
            initial_state,
            cu_seqlens,
            self.gdn_prefill_backend,
        )
        sequence_lengths = cu_seqlens.diff().to(dtype=torch.long)
        token_mask = torch.repeat_interleave(
            non_null_state,
            sequence_lengths,
            output_size=mixed_qkv.shape[0],
        )
        output = torch.where(token_mask[:, None, None], output, 0.0)
        ssm_state.index_copy_(
            0,
            cache_indices,
            final_state.to(dtype=ssm_state.dtype),
        )
        ssm_state[0].copy_(null_state)
        return output

    def _gdn_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        output = kernels.fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv.contiguous(),
            a.contiguous(),
            b.contiguous(),
            self.A_log,
            self.dt_bias,
            ssm_state,
            state_indices.contiguous(),
            self.key_head_dim**-0.5,
        )
        return output.view(-1, self.num_v_heads, self.value_head_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        metadata = get_forward_context().metadata
        state_indices = metadata.linear_state_indices
        if state_indices is None:
            raise RuntimeError("linear_state_indices are required by Qwen3.5")
        state_indices = state_indices.to(device=hidden.device, dtype=torch.int32)
        has_initial_state = metadata.has_initial_state
        cu_seqlens = metadata.q_cu_seq_lens
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                state_indices.numel() + 1,
                dtype=torch.int32,
                device=hidden.device,
            )

        conv_state, ssm_state = self._cache()
        mixed_qkv = self.in_proj_qkv(hidden)
        z = self.in_proj_z(hidden).view(
            -1,
            self.num_v_heads,
            self.value_head_dim,
        )
        b = self.in_proj_b(hidden)
        a = self.in_proj_a(hidden)
        is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
        if is_prefill:
            if has_initial_state is None:
                raise RuntimeError("has_initial_state is required by Qwen3.5 prefill")
            has_initial_state = has_initial_state.to(
                device=hidden.device,
                dtype=torch.bool,
            )
            mixed_qkv = kernels.causal_conv1d_prefill(
                mixed_qkv,
                self.conv1d_weight,
                conv_state,
                state_indices,
                has_initial_state,
                cu_seqlens,
            )
            output = self._gdn_prefill(
                mixed_qkv,
                a,
                b,
                ssm_state,
                state_indices,
                has_initial_state,
                cu_seqlens,
            )
        else:
            mixed_qkv = kernels.causal_conv1d_decode(
                mixed_qkv,
                self.conv1d_weight,
                conv_state,
                state_indices,
            )
            output = self._gdn_decode(
                mixed_qkv,
                a,
                b,
                ssm_state,
                state_indices,
            )
        output = kernels.rms_norm_gated(
            output,
            z,
            self.norm_weight,
            self.norm_eps,
        )
        return self.out_proj(output.reshape(-1, self.value_dim))
