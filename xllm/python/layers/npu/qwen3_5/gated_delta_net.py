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

"""NPU-native Qwen3.5 gated delta network composition."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.qwen3_5.common import Qwen3_5GatedDeltaNetConfig
from xllm.python.layers.qwen3_5.gated_delta_net import (
    Qwen3_5GatedDeltaNetBase,
)
from xllm.python.model_executor.forward_context import get_forward_context


class NpuQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNetBase):
    """NPU graph whose nodes match the native fused operator boundaries."""

    def __init__(
        self,
        cfg: Qwen3_5GatedDeltaNetConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, layer_id, dtype, device)
        self.conv1d_weight = nn.Parameter(
            torch.empty(
                self.conv_kernel_size,
                self.conv_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.register_buffer(
            "conv1d_bias",
            torch.zeros(self.conv_dim, dtype=dtype, device=device),
            persistent=False,
        )

    def _to_backend_conv_layout(
        self,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return weight.transpose(0, 1).contiguous()

    def _cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache = get_forward_context().layer_caches[self.layer_id]
        if cache.conv is None or cache.ssm is None:
            raise RuntimeError(f"linear-attention cache is missing for layer {self.layer_id}")
        if cache.conv.dim() != 3 or cache.conv.size(2) != self.conv_dim:
            raise ValueError("NPU Qwen3.5 conv cache must use [slot, state_len, dim]")
        return cache.conv, cache.ssm

    def _prefill(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = kernels.causal_conv1d_qkv_prefill(
            mixed_qkv,
            self.conv1d_weight,
            conv_state,
            state_indices,
            has_initial_state,
            cu_seqlens,
            self.num_k_heads,
            self.num_v_heads,
            self.key_head_dim,
            self.value_head_dim,
        )
        g, beta = kernels.fused_gdn_gating(
            self.A_log,
            a,
            b,
            self.dt_bias,
        )
        q, k, v, g, beta = (
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            g.squeeze(0),
            beta.squeeze(0),
        )
        initial_state, cache_indices, null_state, non_null_state = self._prepare_ssm_state(
            ssm_state,
            state_indices,
            has_initial_state,
        )
        output, final_state = kernels.chunk_gated_delta_rule(
            q,
            k,
            v.contiguous(),
            g,
            beta,
            initial_state,
            cu_seqlens,
        )
        return self._finalize_prefill(
            output,
            final_state,
            ssm_state,
            cache_indices,
            null_state,
            non_null_state,
            cu_seqlens,
            q.shape[0],
        )

    def _decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        mixed_qkv = kernels.causal_conv1d_decode(
            mixed_qkv,
            self.conv1d_weight,
            conv_state,
            state_indices,
            self.conv1d_bias,
        )
        output = kernels.fused_sigmoid_gating_delta_rule_decode(
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
