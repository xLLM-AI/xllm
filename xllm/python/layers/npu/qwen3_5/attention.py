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

"""NPU-specific Qwen3.5 full-attention fusion."""

from __future__ import annotations

import torch

from xllm.python import kernels
from xllm.python.layers.npu.layernorm import NpuGemmaRMSNorm
from xllm.python.layers.qwen3_5.attention import Qwen3_5Attention
from xllm.python.layers.qwen3_5.common import (
    PartialRotaryEmbedding,
    Qwen3_5AttentionConfig,
)
from xllm.python.model_executor.forward_context import get_forward_context_or_none
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader


class NpuQwen3_5Attention(Qwen3_5Attention):
    """Qwen3.5 attention using the same fused QKV kernel as native C++."""

    normalization_cls = NpuGemmaRMSNorm

    def __init__(
        self,
        cfg: Qwen3_5AttentionConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__(cfg, layer_id, dtype, device, rotary)
        rotary_dim = int(self.head_dim * cfg.partial_rotary_factor)
        is_npu = device.type in ("npu", "privateuseone")
        self.use_fused_qkv = (
            is_npu
            and dtype == torch.bfloat16
            and self.attn_output_gate
            and len(cfg.rope_scaling_mrope_section) == 3
            and rotary_dim == 64
            and kernels.has_split_qkv_rmsnorm_mrope_specialization(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        )
        self._weights_reordered = False
        if self.use_fused_qkv:
            gather_pattern = kernels.build_split_qkv_rmsnorm_mrope_gather_pattern(
                rotary_dim,
                cfg.rope_scaling_mrope_section,
                cfg.rope_scaling_mrope_interleaved,
                device,
            )
        else:
            gather_pattern = None
        self.register_buffer(
            "_mrope_gather_pattern",
            gather_pattern,
            persistent=False,
        )

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        self._weights_reordered = False
        super().load_weights(state, context)

    def _finish_loading(self) -> None:
        if not self.use_fused_qkv or self._weights_reordered:
            return

        with torch.no_grad():
            for parameter in (self.qkv_proj.weight, self.qkv_proj.bias):
                if parameter is not None:
                    qg_rows = parameter[: 2 * self.q_size]
                    qg_rows.copy_(qg_rows.view(self.num_heads, 2, -1).transpose(0, 1).reshape_as(qg_rows))
            self.q_norm.weight.add_(1.0)
            self.k_norm.weight.add_(1.0)
        self._weights_reordered = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_fused_qkv:
            return super().forward(positions, hidden)
        if self._mrope_gather_pattern is None:
            raise RuntimeError("fused QKV mRoPE gather pattern is unavailable")

        mrope_cos_sin: object
        context = get_forward_context_or_none()
        if context is None:
            mrope_cos_sin = self.rotary.build_mrope_cos_sin(positions)
        else:
            cache_key = (__name__, "mrope_cos_sin", id(self.rotary))
            mrope_cos_sin = context.layer_shared_cache.get(cache_key)
            if mrope_cos_sin is None:
                mrope_cos_sin = self.rotary.build_mrope_cos_sin(positions)
                context.layer_shared_cache[cache_key] = mrope_cos_sin
            if not isinstance(mrope_cos_sin, torch.Tensor):
                raise TypeError("invalid cached Qwen3.5 mRoPE cosine/sine tensor")
        qkvg = self.qkv_proj(hidden)
        query, key, value, gate = kernels.split_qkv_rmsnorm_mrope(
            qkvg,
            self.q_norm.weight,
            self.k_norm.weight,
            mrope_cos_sin,
            self._mrope_gather_pattern,
            self.cfg.rms_norm_eps,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        num_tokens = qkvg.shape[0]
        output = self.attn(
            query.view(num_tokens, self.q_size),
            key.view(num_tokens, self.kv_size),
            value.view(num_tokens, self.kv_size),
        )
        output = output * torch.sigmoid_(gate.view(num_tokens, self.q_size))
        return self.o_proj(output)
