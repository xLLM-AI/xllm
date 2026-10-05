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

"""Qwen3-style DFlash2 draft model for Python NPU execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.layers import DFlash2GroupedConv
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader
from xllm.python.models.qwen3 import Qwen3DecoderLayer
from xllm.python.models.qwen3_dflash import (
    DFlashQwen3Config,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


@dataclass
class DFlash2Qwen3Config(DFlashQwen3Config):
    block_size: int = 0
    conv_group_size: int = 0
    conv_kernel_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> DFlash2Qwen3Config:
        base = DFlashQwen3Config.from_dict(d)
        nested = d.get("dflash_config")
        dflash_config = nested if isinstance(nested, dict) else {}

        def pick(reflected_name: str, nested_name: str) -> int:
            value = d.get(reflected_name)
            if value is None:
                value = d.get(nested_name)
            if value is None:
                value = dflash_config.get(nested_name, 0)
            return int(value)

        return cls(
            **base.__dict__,
            block_size=pick("dflash2_block_size", "block_size"),
            conv_group_size=pick("dflash2_conv_group_size", "conv_group_size"),
            conv_kernel_size=pick("dflash2_conv_kernel_size", "conv_kernel_size"),
            selector_rank=pick("dflash2_selector_rank", "selector_rank"),
            selector_top_k=pick("dflash2_selector_top_k", "selector_top_k"),
        )

    def validate(self) -> None:
        super().validate()
        if (
            min(
                self.block_size,
                self.conv_group_size,
                self.conv_kernel_size,
                self.selector_rank,
                self.selector_top_k,
            )
            <= 0
        ):
            raise ValueError("DFlash2 geometry values must be positive")
        if self.hidden_size % self.conv_group_size:
            raise ValueError("DFlash2 conv_group_size must divide hidden_size")
        if self.sliding_window <= 0:
            raise ValueError("DFlash2 requires a positive sliding_window")
        if self.selector_top_k > self.vocab_size:
            raise ValueError("DFlash2 selector_top_k must not exceed vocab_size")


class DFlash2CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.hidden_projection = nn.Linear(
            hidden_size,
            rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=dtype, device=device))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank, dtype=dtype, device=device))

    def forward(
        self,
        hidden_states: torch.Tensor,
        unary_logits: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 3 or unary_logits.dim() != 3:
            raise ValueError("DFlash2 selector hidden states and logits must be three-dimensional")
        if hidden_states.shape[:2] != unary_logits.shape[:2]:
            raise ValueError("DFlash2 selector hidden/logit geometry mismatch")
        if unary_logits.size(2) != self.vocab_size:
            raise ValueError("DFlash2 selector vocabulary size mismatch")
        if anchor_token_ids.shape != (hidden_states.size(0),):
            raise ValueError("DFlash2 selector anchor-token shape mismatch")

        values, candidate_ids = torch.topk(unary_logits, self.top_k, dim=-1)
        values = values.to(torch.float32)
        candidate_ids = candidate_ids.to(torch.long)
        hidden = self.hidden_projection(hidden_states)
        successors = F.embedding(candidate_ids, self.successor_codebook)
        anchor = anchor_token_ids.view(-1, 1, 1).expand(-1, 1, self.top_k)
        predecessor_ids = torch.cat((anchor, candidate_ids[:, :-1]), dim=1)
        predecessors = F.embedding(predecessor_ids, self.predecessor_codebook)
        pair_scores = torch.einsum(
            "blpr,blcr->blpc",
            predecessors * hidden.unsqueeze(2),
            successors,
        )
        return candidate_ids, values.unsqueeze(2) + pair_scores


class DFlash2Qwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        cfg: DFlash2Qwen3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = False,
    ) -> None:
        super().__init__(cfg, layer_id, dtype, device, causal=causal)
        self.self_attn.attn.fia_sparse_mode = 4
        self.self_attn.attn.fia_pre_tokens = cfg.sliding_window - 1
        self.self_attn.attn.fia_next_tokens = cfg.block_size - 1
        self.self_attn.attn.fia_use_attention_mask = True
        self.attention_conv = DFlash2GroupedConv(
            cfg.hidden_size,
            cfg.conv_kernel_size,
            cfg.conv_group_size,
            cfg.block_size,
            dtype,
            device,
        )
        self.mlp_conv = DFlash2GroupedConv(
            cfg.hidden_size,
            cfg.conv_kernel_size,
            cfg.conv_group_size,
            cfg.block_size,
            dtype,
            device,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden, attention_coefficients = self.attention_conv.prepare(hidden)
        hidden = self.self_attn(positions, hidden, cos_sin_cache, cos, sin, mrope_section)
        hidden = self.attention_conv.finish(hidden, attention_coefficients)

        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden, mlp_coefficients = self.mlp_conv.prepare(hidden)
        hidden = self.mlp(hidden)
        hidden = self.mlp_conv.finish(hidden, mlp_coefficients)
        return hidden, residual

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        super().load_weights(weights, context)
        for name in ("attention_conv", "mlp_conv"):
            module = getattr(self, name)
            weights.load_tensor(module.base_kernel, f"{name}.base_kernel")
            weights.load_tensor(module.kernel_projection.weight, f"{name}.kernel_projection.weight")


class DFlash2Qwen3Model(DFlashQwen3Model):
    def __init__(self, cfg: DFlash2Qwen3Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(
            cfg,
            dtype,
            device,
            decoder_layer_type=DFlash2Qwen3DecoderLayer,
        )
        self.candidate_selector = DFlash2CandidateSelector(
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.selector_rank,
            cfg.selector_top_k,
            dtype,
            device,
        )

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> ScopedWeightLoader:
        weights = super().load_weights(state_dicts, tp_rank, tp_size)
        for name in (
            "candidate_selector.hidden_projection.weight",
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
        ):
            weights.load_tensor(self.get_parameter(name), name)
        return weights


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model: DFlash2Qwen3Model
    config_cls = DFlash2Qwen3Config
    model_cls = DFlash2Qwen3Model

    def dflash2_candidates(
        self,
        hidden_states: torch.Tensor,
        unary_logits: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.candidate_selector(hidden_states, unary_logits, anchor_token_ids)
