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

from xllm.python.model_executor.cp_utils import CpContext
from xllm.python.model_executor.forward_context import get_forward_context, record_layer_event
from xllm.python.models.deepseek_v32_mtp import DeepseekV32MtpModel, _load_mtp_weights
from xllm.python.models.glm5_2 import (
    Glm52Config,
    Glm52DecoderLayer,
    Glm52ForCausalLM,
    _attn_dp_gather_inputs,
    _attn_dp_owner_rows,
)


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


class Glm52MtpModel(DeepseekV32MtpModel):
    """GLM checkpoint, position-zero, top-k and recurrent-output adapters."""

    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, dtype, device)
        self._reuse_topk_by_layer = _resolve_mtp_topk_reuse(cfg)

    def _make_decoder(
        self, cfg: Glm52Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> Glm52DecoderLayer:
        return Glm52DecoderLayer(cfg, layer_id, dtype, device)

    def _record_layer_event(self, layer_id: int) -> None:
        record_layer_event(layer_id)

    def _cp_context(self) -> CpContext | None:
        return get_forward_context().cp_context

    def _indexer_interleaved(self) -> bool:
        return self.cfg.indexer_rope_interleave

    def _prepare_layer_inputs(
        self, hidden: torch.Tensor, positions: torch.Tensor, cp_context: CpContext | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]:
        if self.cfg.enable_attn_dp_weight_sharding:
            if cp_context is not None:
                raise ValueError("attention DP sharding does not support CP")
            hidden, positions = _attn_dp_gather_inputs(hidden, positions, self.cfg)
        return super()._prepare_layer_inputs(hidden, positions, cp_context)

    def _prepare_token_hidden(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # A scalar zero broadcasts in the where kernel; avoid materializing a
        # full hidden-shaped zeros_like tensor for the position-zero rows.
        return torch.where(positions.ne(0).unsqueeze(-1), hidden, 0)

    def _recurrent_hidden(self, hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        return hidden if residual is None else hidden + residual

    def _format_mtp_output(
        self, hidden: torch.Tensor, topk: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, torch.Tensor | None]:
        if self.cfg.enable_attn_dp_weight_sharding:
            hidden = _attn_dp_owner_rows(hidden, self.cfg)
        # Both recurrence and the vocabulary head consume post-norm hidden.
        if hidden.shape[0] > 0:
            hidden = self.norm(hidden)
        return hidden, None, topk if self.cfg.index_share_for_mtp_iteration else None


class Glm52MtpForCausalLM(Glm52ForCausalLM):
    """GLM-5.2/5.3 MTP calculator; scheduling remains in the C++ worker."""

    def __init__(self, config: dict) -> None:
        super().__init__(config, build_model=False)
        self.model = Glm52MtpModel(self.cfg, self.dtype, self.device)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        _load_mtp_weights(self, super().load_weights, state_dicts, tp_rank, tp_size)
