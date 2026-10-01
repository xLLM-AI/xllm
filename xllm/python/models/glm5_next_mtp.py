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
"""GLM-5.3-Flash MTP (multi-token prediction) draft model.

The speculative worker and MTP scheduling remain in C++ (MTPWorkerImpl). This
module only describes the draft computation, mirroring the DeepSeek-V3.2
python MTP scheme (``deepseek_v32_mtp.py``):

- The draft layer is the checkpoint's appended layer
  ``model.language_model.layers.<n_layers>`` — a full DSA sparse-attention MoE
  layer WITHOUT mHC residual streams (its checkpoint keys carry
  ``input_layernorm`` / ``post_attention_layernorm`` and no ``hc_*``
  weights), so the decoder layer here is a plain pre-norm residual variant.
- ``eh_proj`` fuses ``[enorm(token_embed) ‖ hnorm(carried_hidden)]`` —
  embedding first, carried hidden second (matches the C++ NPU MTP base,
  ``mtp_model_base.h``).
- ``input_embedding`` is the hidden state produced by the target model (or
  the previous draft step) and supplied by the caller; when None the draft
  runs in prefill mode off its own token embedding.
- The draft owns its ``embed_tokens`` / ``lm_head`` copies (the w8a8
  checkpoint materializes them inside the appended layer; the exporter copies
  them for bf16), so no target->draft weight sharing is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.linear import ColumnParallelLinear
from xllm.python.layers.qlinear import QLinearWeightLoader
from xllm.python.models.glm5_next import (
    Glm5NextConfig,
    Glm5NextForCausalLM,
    Glm5NextMlaAttention,
    Glm5NextMoE,
    Glm5NextRMSNorm,
)

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class Glm5NextMtpDecoderLayer(nn.Module):
    """One pre-norm decoder layer without mHC (the MTP checkpoint layer shape)."""

    def __init__(self, cfg: Glm5NextConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = 0
        self.input_layernorm = Glm5NextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        # Draft schedule: layer 0 is deepseek_sparse_attention (the appended
        # MTP layer carries a full indexer), MoE MLP (first_k_dense_replace=0).
        self.self_attn = Glm5NextMlaAttention(cfg, 0, dtype, device)
        self.post_attention_layernorm = Glm5NextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.mlp = Glm5NextMoE(cfg, dtype, device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_ids, attention_mask)
        # MLA attention returns [B*S, D] (2D); reshape back to [B, S, D] to
        # match residual — parity with Glm5NextDecoderLayer._forward_ref.
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.view(residual.shape[0], residual.shape[1], -1)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Glm5NextMtpModel(nn.Module):
    """MTP draft body: eh_proj fusion + one decoder layer + shared-head norm."""

    def __init__(self, cfg: Glm5NextConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        tp = cfg.tp_size
        for name, val in (("hidden_size", cfg.hidden_size), ("vocab_size", cfg.vocab_size)):
            if val % tp != 0:
                raise ValueError(f"{name} {val} not divisible by tp_size {tp}")
        # The MTP draft is one appended decoder layer; layer_id / load_weights
        # both hardcode index 0. Fail fast if the config drifts from that so
        # the model doesn't silently forward extra unloaded init-value layers.
        if cfg.n_layers != 1:
            raise ValueError(f"glm5_next_mtp draft expects exactly 1 layer, got {cfg.n_layers}")
        # The decoder layer and load_weights hardcode layer 0 as MoE + full
        # (non-shared) indexer DSA; a config drift here would silently load the
        # wrong weight branch in _load_mlp or crash forward on the missing
        # indexer, so fail fast on both.
        if not cfg.is_dsa(0):
            raise ValueError("glm5_next_mtp draft layer 0 must be deepseek_sparse_attention")
        if not cfg.is_moe(0):
            raise ValueError(
                "glm5_next_mtp draft layer 0 must be MoE; set first_k_dense_replace=0 "
                f"(got first_k_dense_replace={cfg.first_k_dense_replace})"
            )
        if cfg.indexer_shared(0):
            raise ValueError("glm5_next_mtp draft layer 0 must carry a full indexer (indexer_types[0] != 'shared')")
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        self.enorm = Glm5NextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hnorm = Glm5NextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        # Fuses [norm(token_embed) ‖ norm(carried hidden)] -> hidden.
        self.eh_proj = ColumnParallelLinear(
            2 * cfg.hidden_size,
            cfg.hidden_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        # Single-layer MTP draft (n_layers guaranteed 1 by the fail-fast above).
        self.layers = nn.ModuleList([Glm5NextMtpDecoderLayer(cfg, dtype, device)])
        # Checkpoint name: model.layers.<n>.shared_head.norm — exported as
        # model.norm.weight by the MTP exporter.
        self.norm = Glm5NextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Normalize flat [N] runner inputs to [B, S] like Glm5NextModel.
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        token_hidden = self.embed_tokens(input_ids)

        carried = token_hidden if input_embedding is None else input_embedding.reshape(token_hidden.shape)

        e = self.enorm(token_hidden)
        h = self.hnorm(carried)
        hidden_states = self.eh_proj(torch.cat((e, h), dim=-1))

        batch_size, seq_len = hidden_states.shape[:2]
        attention_mask = torch.ones(
            batch_size,
            seq_len,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        hidden_states = self.layers[0](hidden_states, position_ids, attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states.view(-1, self.cfg.hidden_size)


class Glm5NextMtpForCausalLM(Glm5NextForCausalLM):
    """GLM-5.3-Flash MTP draft calculator; scheduling stays in the C++ worker."""

    def __init__(self, config: dict) -> None:
        super().__init__(config, build_model=False)
        self.model = Glm5NextMtpModel(self.cfg, self.dtype, self.device)
        self.to(device=self.device, dtype=self.dtype)

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        """Load the exported MTP draft checkpoint (loader-native key names).

        The exporter (tools/export_mtp.py) remaps the appended checkpoint layer
        ``model.language_model.layers.<n>`` to ``model.layers.0`` plus top-level
        ``model.{enorm,hnorm,eh_proj}``, ``model.norm`` (shared_head.norm),
        ``model.embed_tokens`` and ``lm_head``, so no real-checkpoint aliasing
        is needed here.
        """
        L = QLinearWeightLoader(self, state_dicts, tp_size, tp_rank)
        # embed_tokens: HiddenParallelEmbedding — shard the hidden dim.
        L.load_fp("model.embed_tokens.weight", dim=1)
        p = "model.layers.0."
        L.load_fp(p + "input_layernorm.weight")
        L.load_fp(p + "post_attention_layernorm.weight")
        self._load_dsa_attn(L, p + "self_attn.", 0)
        self._load_mlp(L, p + "mlp.", 0)
        L.load_fp("model.norm.weight")
        # lm_head: ColumnParallelLinear — shard the vocab dim.
        L.load_fp("lm_head.weight", dim=0)
        # MTP-specific: replicated norms + column-parallel fusion projection.
        L.load_fp("model.enorm.weight")
        L.load_fp("model.hnorm.weight")
        L.load_fp("model.eh_proj.weight", dim=0)


# Registration is centralised in xllm.python.registry; import there.
