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

import torch
import torch.nn as nn

from xllm.python.layers.flash_comm1 import get_flash_comm1_context
from xllm.python.layers.qwen3_5.decoder_layer import get_qwen3_5_decoder_layer_class
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader, StateDictLike, load_causal_lm_weights
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mrope_cos_sin = self.rotary.build_mrope_cos_sin(positions) if self._use_fused_mrope else None
        embedding = self.embed_tokens(input_ids)
        carried = embedding if input_embedding is None else input_embedding
        if carried.shape != embedding.shape:
            raise ValueError("Qwen3.5 MTP requires one carried hidden state per token")
        hidden = self.fc(torch.cat((self.pre_fc_norm_embedding(embedding), self.pre_fc_norm_hidden(carried)), dim=-1))
        flash_comm1 = get_flash_comm1_context()
        if flash_comm1 is not None:
            hidden = flash_comm1.shard(hidden)
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, positions, mrope_cos_sin)
            record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        return hidden if flash_comm1 is None else flash_comm1.gather(hidden)


class Qwen3_5MtpForCausalLM(Qwen3_5ForCausalLM):
    model_cls = Qwen3_5MtpModel
    model: Qwen3_5MtpModel

    def load_weights(self, state_dicts: list[StateDictLike], tp_rank: int, tp_size: int) -> None:
        context = ParallelLoadContext.from_config(self.cfg, tp_rank, tp_size)
        shared = ScopedWeightLoader(
            state_dicts,
            src_prefixes=("model.language_model.", "language_model.model.", "model.", ""),
        )
        if shared.has("embed_tokens.weight"):
            shared.load_tensor(
                self.model.embed_tokens.weight, "embed_tokens.weight", dim=1, rank=tp_rank, world_size=tp_size
            )
        else:
            self.model.add_module("embed_tokens", None)
        if self.cfg.tie_word_embeddings:
            head_weights, head_key = shared, "embed_tokens.weight"
        else:
            head_weights = ScopedWeightLoader(
                state_dicts,
                src_prefixes=("", "model.", "language_model.", "model.language_model."),
            )
            head_key = "lm_head.weight"
        if head_weights.has(head_key):
            head_weights.load_tensor(self.lm_head.weight, head_key, dim=0, rank=tp_rank, world_size=tp_size)
        else:
            self.add_module("lm_head", None)
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
