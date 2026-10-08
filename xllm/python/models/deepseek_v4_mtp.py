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

"""DeepSeek-V4 multi-token prediction draft model."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.model_executor.forward_context import get_forward_context, record_layer_event
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4DecoderLayer,
    _build_rotary_tables,
    _DeepseekV4ForwardSetup,
    _hc_head_merge,
)


class DeepseekV4MtpLayer(DeepseekV4DecoderLayer):
    """One DeepSeek-V4 MTP fusion layer followed by a V4 decoder layer."""

    def __init__(
        self,
        cfg: DeepseekV4Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, layer_id, dtype, device)
        self.cfg = cfg
        self.enorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.hnorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.e_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, dtype=dtype, device=device)
        self.h_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, dtype=dtype, device=device)
        hc_dim = cfg.hc_mult * cfg.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(cfg.hc_mult, hc_dim, dtype=torch.float32, device=device))
        self.hc_head_base = nn.Parameter(torch.empty(cfg.hc_mult, dtype=torch.float32, device=device))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device))

    def _fuse_hidden_states(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_size = self.cfg.hidden_size
        hc_mult = self.cfg.hc_mult
        input_width = previous_hidden_states.size(-1)
        expected_aux_width = hc_mult * hidden_size
        enorm_out = self.enorm(inputs_embeds)
        if input_width == expected_aux_width:
            previous = self.hnorm(previous_hidden_states.reshape(-1, hidden_size))
            embedded = self.e_proj(enorm_out)
            projected = self.h_proj(previous).reshape(-1, hc_mult, hidden_size)
            return embedded.unsqueeze(1) + projected
        if input_width == hidden_size:
            previous = self.hnorm(previous_hidden_states)
            fused = self.e_proj(enorm_out) + self.h_proj(previous)
            return fused.unsqueeze(1).repeat(1, hc_mult, 1)
        raise ValueError(
            "DeepSeek-V4 MTP input hidden size must be hidden_size "
            f"({hidden_size}) or hc_mult * hidden_size ({expected_aux_width}), "
            f"but got {input_width}"
        )

    def _merge_hc_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return _hc_head_merge(
            hidden,
            self.hc_head_fn,
            self.hc_head_base,
            self.hc_head_scale,
            self.cfg.rms_norm_eps,
            self.cfg.hc_eps,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._fuse_hidden_states(inputs_embeds, previous_hidden_states)
        hidden, _ = super().forward(
            hidden,
            None,
            positions,
            cos_sin_cache,
            input_ids,
        )
        return self._merge_hc_hidden(hidden), hidden


class DeepseekV4MtpModel(_DeepseekV4ForwardSetup, nn.Module):
    """DeepSeek-V4 MTP body with checkpoint-owned or target-shared endpoints."""

    def __init__(self, cfg: DeepseekV4Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        tp = cfg.tp_size
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [DeepseekV4MtpLayer(cfg, layer_id, dtype, device) for layer_id in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary, self.compress_rotary_c4, self.compress_rotary_c128 = _build_rotary_tables(cfg, dtype, device)

    def make_dummy_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a zero target-hidden row for an empty DP MTP shard."""
        return torch.zeros(
            input_ids.shape[0],
            self.cfg.hc_mult * self.cfg.hidden_size,
            dtype=self.norm.weight.dtype,
            device=input_ids.device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        position_zero = positions == 0
        hidden = hidden.masked_fill(position_zero.unsqueeze(-1), 0)

        context = get_forward_context()
        backend = context.attention_backend
        metadata = context.metadata
        is_dummy = bool(getattr(metadata, "is_dummy", False))
        if is_dummy and input_embedding is None:
            input_embedding = self.make_dummy_input_embedding(input_ids)
        if input_embedding is None:
            raise ValueError("DeepSeek-V4 MTP requires input_embedding from the target or previous draft step")

        self._prepare_dsa_forward(backend, metadata, positions, "DeepSeek-V4 MTP")
        cp_ctx = self._build_cp_context(backend, metadata, positions, allow_spec_verify=True)

        if cp_ctx is not None and cp_ctx.enabled():
            hidden = cp_ctx.shard_rows(hidden)
            input_embedding = cp_ctx.shard_rows(input_embedding)
            positions = cp_ctx.local_positions

        previous_hidden_states = input_embedding
        aux_hidden = torch.empty(0, dtype=hidden.dtype, device=hidden.device)
        for layer_id, layer in enumerate(self.layers):
            compress_ratio = self.cfg.compress_ratios[layer_id]
            if compress_ratio == 4:
                layer_cos_sin_cache = self.compress_rotary_c4.cos_sin_cache
            elif compress_ratio == 128:
                layer_cos_sin_cache = self.compress_rotary_c128.cos_sin_cache
            else:
                layer_cos_sin_cache = self.rotary.cos_sin_cache
            select_layer_rope = getattr(backend, "select_dsa_layer_rope", None)
            if select_layer_rope is None:
                raise RuntimeError("DeepSeek-V4 MTP requires select_dsa_layer_rope")
            select_layer_rope(layer_id, layer_cos_sin_cache, metadata)
            if cp_ctx is not None and cp_ctx.enabled():
                kv_cos, kv_sin = cp_ctx.global_rope(compress_ratio)
                dsa = getattr(metadata, "dsa_metadata", None)
                if dsa is None:
                    raise RuntimeError("DeepSeek-V4 MTP CP requires prepared DSA metadata")
                dsa.kv_cos = kv_cos
                dsa.kv_sin = kv_sin
            hidden, aux_hidden = layer(
                hidden,
                previous_hidden_states,
                positions,
                layer_cos_sin_cache,
                input_ids,
            )
            record_layer_event(layer_id)

        if cp_ctx is not None and cp_ctx.enabled():
            hidden = cp_ctx.gather_restore(hidden)
            aux_hidden = cp_ctx.gather_restore(aux_hidden)
        hidden = self.norm(hidden, None)
        return hidden, aux_hidden.flatten(1)
