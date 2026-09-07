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

"""Shared decoder orchestration and backend dispatch for Qwen3.5."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.layernorm import GemmaRMSNorm
from xllm.python.layers.qwen3_5_attention import Qwen3_5Attention
from xllm.python.layers.qwen3_5_common import (
    PartialRotaryEmbedding,
    Qwen3_5DecoderConfig,
)
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader


class Qwen3_5DecoderLayer(nn.Module):
    """Backend-neutral Qwen3.5 decoder orchestration.

    Concrete backend layers supply the attention, gated-delta-net, and sparse
    MoE implementations. Runtime sequencing and checkpoint traversal remain
    shared.
    """

    attention_cls: type[Qwen3_5Attention]
    gated_delta_net_cls: type[nn.Module]
    sparse_moe_cls: type[nn.Module]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        missing = [
            name for name in ("attention_cls", "gated_delta_net_cls", "sparse_moe_cls") if not hasattr(cls, name)
        ]
        if missing:
            raise TypeError(f"{cls.__name__} must set backend class vars: {', '.join(missing)}")

    def __init__(
        self,
        cfg: Qwen3_5DecoderConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        rotary: PartialRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.layer_type = cfg.layer_types[layer_id]
        self.input_layernorm = GemmaRMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if self.layer_type == "full_attention":
            self.self_attn = self.attention_cls(
                cfg,
                layer_id,
                dtype,
                device,
                rotary,
            )
        elif self.layer_type == "linear_attention":
            self.linear_attn = self.gated_delta_net_cls(
                cfg,
                layer_id,
                dtype,
                device,
            )
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {self.layer_type}")
        self.post_attention_layernorm = GemmaRMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        if cfg.is_moe_layer(layer_id):
            self.mlp = self.sparse_moe_cls(cfg, dtype, device)
        else:
            self.mlp = GatedMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.tp_size,
                dtype,
                device,
            )

    def _prepare_forward(self) -> None:
        """Run the backend-specific pre-forward hook, if any."""

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        state.load_tensor(
            self.input_layernorm.weight,
            "input_layernorm.weight",
        )
        state.load_tensor(
            self.post_attention_layernorm.weight,
            "post_attention_layernorm.weight",
        )
        if self.layer_type == "full_attention":
            self.self_attn.load_weights(
                state.with_prefix("self_attn."),
                context,
            )
        else:
            self.linear_attn.load_weights(
                state.with_prefix("linear_attn."),
                context,
            )
        self.mlp.load_weights(state.with_prefix("mlp."), context)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._prepare_forward()
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        if self.layer_type == "full_attention":
            hidden = self.self_attn(positions, hidden)
        else:
            hidden = self.linear_attn(hidden)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


def get_qwen3_5_decoder_layer_class(
    device: torch.device | str,
) -> type[Qwen3_5DecoderLayer]:
    device_type = torch.device(device).type
    if device_type == "cuda":
        from xllm.python.layers.cuda.qwen3_5.decoder_layer import (
            CudaQwen3_5DecoderLayer,
        )

        return CudaQwen3_5DecoderLayer
    if device_type in ("npu", "privateuseone"):
        from xllm.python.layers.npu.qwen3_5.decoder_layer import (
            NpuQwen3_5DecoderLayer,
        )

        return NpuQwen3_5DecoderLayer
    raise ValueError(f"Qwen3.5 Python has no decoder implementation for device {device_type!r}")
