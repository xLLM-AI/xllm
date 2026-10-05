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

"""DeepSeek-V3.2 MTP model graph.

The speculative worker and MTP scheduling remain in C++.  This module only
describes the MTP model computation.  In particular, ``input_embedding`` is
the hidden state produced by the target model and supplied by the caller for
the next MTP step.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, RMSNorm
from xllm.python.model_executor.cp_utils import cp_merge_rows, cp_shard_rows
from xllm.python.model_loader import W8A8WeightLoader
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
    DeepseekYarnRotaryEmbedding,
)

# The loader strips "model.", so a bare "shared_head.norm.weight" needs no entry.
_MTP_NORM_ALIASES: dict[str, tuple[str, ...]] = {
    "model.norm.weight": (
        "model.norm.weight",
        "model.final_norm.weight",
        "model.shared_head.norm.weight",
    ),
}


class DeepseekV32MtpModel(DeepseekV3Model):
    """MTP body matching ``MtpModelImplBase`` and ``DeepseekV32MtpModel``."""

    def __init__(self, cfg: DeepseekV3Config, dtype: torch.dtype, device: torch.device) -> None:
        nn.Module.__init__(self)
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
        self.layers = nn.ModuleList([self._make_decoder(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.rotary = DeepseekYarnRotaryEmbedding(
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
        self._reuse_topk_by_layer = (False,) * cfg.n_layers

    def _prepare_token_hidden(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return hidden

    def _recurrent_hidden(self, hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        hidden, _ = self.norm(hidden, residual)
        return hidden

    def _prepare_logits_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    def _format_mtp_output(
        self, hidden: torch.Tensor, topk: torch.Tensor | None
    ) -> torch.Tensor | tuple[torch.Tensor, None, torch.Tensor | None]:
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, None, torch.Tensor | None]:
        assert self.embed_tokens is not None
        cp_context = self._cp_context()
        if cp_context is not None and mtp_topk_indices is not None:
            if mtp_topk_indices.ndim < 2 or mtp_topk_indices.shape[0] != input_ids.shape[0]:
                raise ValueError("CP MTP top-k indices must have one leading row per global input token")
        token_hidden = self._prepare_token_hidden(self.embed_tokens(input_ids), positions)
        if input_embedding is None:
            input_embedding = token_hidden
        hnorm_input = self.rot(input_embedding) if self.enable_rot else input_embedding
        hidden = self.eh_proj(torch.cat((self.enorm(token_hidden), self.hnorm(hnorm_input)), dim=-1))
        hidden, rope, query_cos_sin = self._prepare_layer_inputs(hidden, positions, cp_context)
        residual: torch.Tensor | None = None
        topk = mtp_topk_indices
        if cp_context is not None and topk is not None:
            topk = cp_shard_rows(topk, cp_context)
        for layer_id, layer in enumerate(self.layers):
            reuse = self._reuse_topk_by_layer[layer_id] and topk is not None
            hidden, residual, topk = layer(hidden, residual, *rope, query_cos_sin, topk, reuse)
            self._record_layer_event(layer_id)
        hidden = self._recurrent_hidden(hidden, residual)
        if cp_context is not None:
            hidden = cp_merge_rows(hidden, cp_context)
            if topk is not None:
                # The worker selects next-step rows in global token order,
                # matching hidden states even after a padded CP prefill.
                topk = cp_merge_rows(topk, cp_context)
        return self._format_mtp_output(hidden, topk)


def _compute_mtp_logits(
    model: DeepseekV32MtpModel,
    lm_head: nn.Module | None,
    hidden: torch.Tensor,
    selected_idxes: torch.Tensor | None,
) -> torch.Tensor:
    if selected_idxes is not None and selected_idxes.numel() > 0:
        hidden = hidden.index_select(0, selected_idxes)
    assert lm_head is not None
    return lm_head(model._prepare_logits_hidden(hidden))


def _load_mtp_weights(
    model: PyModelBase,
    load_decoder_weights: Callable[..., None],
    state_dicts: list,
    tp_rank: int,
    tp_size: int,
) -> None:
    loader = W8A8WeightLoader(
        model,
        state_dicts,
        model.cfg.tp_size,
        model.cfg.tp_rank,
        src_prefixes=("", "model."),
        name_aliases=_MTP_NORM_ALIASES,
    )
    load_decoder_weights(
        state_dicts,
        tp_rank,
        tp_size,
        load_lm_head=False,
        load_embedding=False,
        loader=loader,
    )

    def _copy_if_present(module_name: str, required: bool = False) -> bool:
        key = module_name + ".weight"
        if not loader.has(key):
            if required:
                raise KeyError(f"missing required MTP weight: {key}")
            return False
        tensor = loader.get_tensor(key)
        parameter = model.get_parameter("model." + key)
        if tensor.shape != parameter.shape and tensor.dim() == 2:
            tensor = loader.shard(tensor, dim=0)
        loader.copy_in("model." + key, tensor)
        return True

    _copy_if_present("eh_proj", required=True)
    _copy_if_present("enorm", required=True)
    _copy_if_present("hnorm", required=True)
    model.model.enable_rot = _copy_if_present("rot")


class DeepseekV32MtpForCausalLM(DeepseekV3ForCausalLM):
    """DeepSeek-V3.2 MTP calculator; scheduling stays in the C++ worker."""

    def __init__(self, config: dict) -> None:
        super().__init__(config, build_model=False)
        self.model = DeepseekV32MtpModel(self.cfg, self.dtype, self.device)

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        return _compute_mtp_logits(self.model, self.lm_head, hidden, selected_idxes)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        _load_mtp_weights(self, super().load_weights, state_dicts, tp_rank, tp_size)
