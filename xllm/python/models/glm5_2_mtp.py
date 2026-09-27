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

"""GLM-5.2/5.3 MTP graph for ``model_type=glm_moe_dsa_mtp``."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, RMSNorm
from xllm.python.model_executor.cp_utils import cp_merge_rows, cp_shard_positions, cp_shard_rows
from xllm.python.model_executor.forward_context import get_forward_context, record_layer_event
from xllm.python.models.glm5_2 import (
    Glm52Config,
    Glm52DecoderLayer,
    Glm52ForCausalLM,
    Glm52YarnRotaryEmbedding,
    _select_indexer_query_cos_sin,
)
from xllm.python.models.weight_utils import W8A8WeightLoader

_MTP_NORM_ALIASES: dict[str, tuple[str, ...]] = {
    "model.norm.weight": (
        "model.norm.weight",
        "model.final_norm.weight",
        "model.shared_head.norm.weight",
    ),
}


def _resolve_mtp_topk_reuse(cfg: Glm52Config) -> tuple[bool, ...]:
    """Resolve the native DSA cross-layer/cross-draft reuse plan."""
    if not cfg.index_share_for_mtp_iteration:
        return (False,) * cfg.n_layers

    pattern = cfg.index_topk_pattern
    if pattern:
        symbols = list(pattern) if isinstance(pattern, str) else list(pattern)
        if len(symbols) != cfg.n_layers:
            raise ValueError("MTP DSA top-k sharing pattern length must equal num_hidden_layers")
        reuse = []
        for symbol in symbols:
            normalized = str(symbol).upper()
            if normalized not in ("F", "S", "FULL", "SHARED"):
                raise ValueError(f"MTP DSA top-k sharing only supports F/S, got {symbol!r}")
            reuse.append(normalized in ("S", "SHARED"))
        return tuple(reuse)

    frequency = cfg.index_topk_freq
    if frequency <= 1:
        return (False,) * cfg.n_layers
    offset = cfg.index_skip_topk_offset
    if offset < 0:
        raise ValueError("MTP DSA top-k sharing offset must be non-negative")
    if offset > 0:
        return tuple(max(layer_id - offset + 1, 0) % frequency != 0 for layer_id in range(cfg.n_layers))
    return tuple(max(layer_id - 1, 0) % frequency != 0 for layer_id in range(cfg.n_layers))


class Glm52MtpModel(nn.Module):
    """MTP decoder body with optional DSA top-k reuse across draft steps."""

    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0

        self.cfg = cfg
        self.embed_tokens: nn.Module | None = None
        self.eh_proj = ColumnParallelLinear(
            2 * cfg.hidden_size,
            cfg.hidden_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        self.rot = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.hidden_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        self.enorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hnorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.layers = nn.ModuleList([Glm52DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.rotary = Glm52YarnRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )
        self.enable_rot = False
        self._reuse_topk_by_layer = _resolve_mtp_topk_reuse(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor | None]:
        assert self.embed_tokens is not None
        token_hidden = self.embed_tokens(input_ids)
        token_hidden = torch.where(positions.ne(0).unsqueeze(-1), token_hidden, torch.zeros_like(token_hidden))
        if input_embedding is None:
            input_embedding = token_hidden

        hnorm_input = self.rot(input_embedding) if self.enable_rot else input_embedding
        hidden = self.eh_proj(torch.cat((self.enorm(token_hidden), self.hnorm(hnorm_input)), dim=-1))

        positions = positions.to(torch.int64).contiguous()
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            hidden = cp_shard_rows(hidden, cp_context)
            positions = cp_shard_positions(positions, cp_context).contiguous()

        half_rope_cos, half_rope_sin, rope_cos, rope_sin = self.rotary(positions)
        query_cos_sin = _select_indexer_query_cos_sin(
            self.cfg.indexer_rope_interleave, half_rope_cos, half_rope_sin, rope_cos, rope_sin, cp_context
        )
        residual: torch.Tensor | None = None
        topk_indices = mtp_topk_indices
        for layer_id, layer in enumerate(self.layers):
            reuse_topk = self._reuse_topk_by_layer[layer_id] and topk_indices is not None
            hidden, residual, topk_indices = layer(
                hidden,
                residual,
                half_rope_cos,
                half_rope_sin,
                rope_cos,
                rope_sin,
                query_cos_sin,
                topk_indices,
                reuse_topk,
            )
            record_layer_event(layer_id)

        if residual is not None:
            hidden = hidden + residual
        if cp_context is not None:
            hidden = cp_merge_rows(hidden, cp_context)

        output_topk = topk_indices if self.cfg.index_share_for_mtp_iteration else None
        return hidden, None, output_topk


class Glm52MtpForCausalLM(Glm52ForCausalLM):
    """GLM-5.2/5.3 MTP calculator; scheduling remains in the C++ worker."""

    def __init__(self, config: dict) -> None:
        super().__init__(config, build_model=False)
        self.model = Glm52MtpModel(self.cfg, self.dtype, self.device)

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        normalized = self.model.norm(hidden)
        assert isinstance(normalized, torch.Tensor)
        assert self.lm_head is not None
        return self.lm_head(normalized)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        loader = W8A8WeightLoader(
            self,
            state_dicts,
            self.cfg.tp_size,
            self.cfg.tp_rank,
            src_prefixes=("", "model."),
            name_aliases=_MTP_NORM_ALIASES,
        )
        super().load_weights(
            state_dicts,
            tp_rank,
            tp_size,
            load_lm_head=False,
            load_embedding=False,
            loader=loader,
        )

        def copy_if_present(module_name: str, required: bool = False) -> bool:
            key = module_name + ".weight"
            if not loader.has(key):
                if required:
                    raise KeyError(f"missing required MTP weight: {key}")
                return False
            tensor = loader.load_tensor(key)
            parameter = self.get_parameter("model." + key)
            if tensor.shape != parameter.shape and tensor.dim() == 2:
                tensor = loader.shard(tensor, dim=0)
            loader.copy_in("model." + key, tensor)
            return True

        copy_if_present("eh_proj", required=True)
        copy_if_present("enorm", required=True)
        copy_if_present("hnorm", required=True)
        self.model.enable_rot = copy_if_present("rot")
