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

"""Qwen3.5/3.8 MTP draft, matching the native Qwen3_5MtpModelImplBase."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from xllm.python.layers.flash_comm1 import get_flash_comm1_context
from xllm.python.layers.qwen3_5.decoder_layer import get_qwen3_5_decoder_layer_class
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader, StateDictLike, load_causal_lm_weights
from xllm.python.models.base import TokenEmbeddingRunner
from xllm.python.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5Model


class Qwen3_5MtpModel(Qwen3_5Model):
    def __init__(self, cfg: Qwen3_5Config, dtype: torch.dtype, device: torch.device) -> None:
        if cfg.num_experts or any(kind != "full_attention" for kind in cfg.layer_types):
            raise ValueError("Qwen3.5 Python MTP requires dense full-attention draft layers")
        super().__init__(cfg, dtype, device)
        normalization_cls = get_qwen3_5_decoder_layer_class(device).normalization_cls
        self.pre_fc_norm_embedding = normalization_cls(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.pre_fc_norm_hidden = normalization_cls(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        # The native fusion projection is replicated, including under TP.
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False, dtype=dtype, device=device)

    def prepare_inputs_embeds(self, input_ids: torch.Tensor, input_embedding: torch.Tensor | None) -> torch.Tensor:
        embedding = self.embed_tokens(input_ids)
        if input_embedding is None:
            input_embedding = embedding
        elif input_embedding.shape != embedding.shape:
            raise ValueError("Qwen3.5 MTP requires one carried hidden state per token")
        fused = torch.cat((self.pre_fc_norm_embedding(embedding), self.pre_fc_norm_hidden(input_embedding)), dim=-1)
        return cast("torch.Tensor", self.fc(fused))

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        mrope_cos_sin = self.rotary.build_mrope_cos_sin(positions) if self._use_fused_mrope else None
        flash_comm1 = get_flash_comm1_context()
        if flash_comm1 is not None:
            hidden = flash_comm1.shard(hidden)
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, positions, mrope_cos_sin)
            record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        return hidden if flash_comm1 is None else flash_comm1.gather(hidden)


class _Qwen35MtpRunner(TokenEmbeddingRunner):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        backbone = cast("Qwen3_5MtpModel", self.backbone)
        return cast("torch.Tensor", backbone(backbone.prepare_inputs_embeds(input_ids, input_embedding), positions))


class Qwen3_5MtpForCausalLM(Qwen3_5ForCausalLM):
    model_cls = Qwen3_5MtpModel
    runner_model_cls: type[nn.Module] | None = _Qwen35MtpRunner
    model: Qwen3_5MtpModel

    def load_weights(self, state_dicts: list[StateDictLike], tp_rank: int, tp_size: int) -> None:
        context = ParallelLoadContext.from_config(self.cfg, tp_rank, tp_size)
        shared = ScopedWeightLoader(
            state_dicts,
            src_prefixes=("model.language_model.", "language_model.model.", "model.", ""),
        ).bind_source_root("embed_tokens.weight")
        shared.load_tensor(
            self.model.embed_tokens.weight, "embed_tokens.weight", dim=1, rank=tp_rank, world_size=tp_size
        )
        if self.cfg.tie_word_embeddings:
            shared.load_tensor(self.lm_head.weight, "embed_tokens.weight", dim=0, rank=tp_rank, world_size=tp_size)
        else:
            ScopedWeightLoader(
                state_dicts,
                src_prefixes=("", "model.", "language_model.", "model.language_model."),
            ).load_tensor(self.lm_head.weight, "lm_head.weight", dim=0, rank=tp_rank, world_size=tp_size)
        mtp = load_causal_lm_weights(
            self.model,
            None,
            ScopedWeightLoader(state_dicts, src_prefixes=("mtp.", "model.mtp.")),
            context,
            tie_word_embeddings=self.cfg.tie_word_embeddings,
            load_embedding=False,
        )
        mtp.load_tensor(self.model.pre_fc_norm_embedding.weight, "pre_fc_norm_embedding.weight")
        mtp.load_tensor(self.model.pre_fc_norm_hidden.weight, "pre_fc_norm_hidden.weight")
        mtp.load_tensor(self.model.fc.weight, "fc.weight")
