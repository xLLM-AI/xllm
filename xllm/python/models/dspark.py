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

"""Shared DSpark heads and runtime methods for Python model adapters."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.models.base import PyModelBase


class DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank, dtype=dtype, device=device)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False, dtype=dtype, device=device)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.embed(token_ids))


class DSparkConfidenceHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.with_markov = with_markov
        input_size = hidden_size + markov_rank if with_markov else hidden_size
        self.proj = nn.Linear(input_size, 1, bias=True, dtype=torch.float32, device=device)

    def forward(self, hidden: torch.Tensor, markov_embedding: torch.Tensor | None) -> torch.Tensor:
        if self.with_markov:
            if markov_embedding is None:
                raise ValueError("DSpark confidence head requires Markov embeddings")
            hidden = torch.cat((hidden, markov_embedding), dim=-1)
        return torch.sigmoid(self.proj(hidden.float())).squeeze(-1).to(torch.float32)


class DSparkForCausalLMBase(PyModelBase):
    """Owns model-independent DSpark heads and runtime bridge methods."""

    def __init__(
        self,
        *,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        enable_confidence_head: bool,
        confidence_head_with_markov: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.markov_head = DSparkMarkovHead(
            vocab_size,
            draft_vocab_size,
            markov_rank,
            dtype,
            device,
        )
        self.confidence_head = (
            DSparkConfidenceHead(
                hidden_size,
                markov_rank,
                confidence_head_with_markov,
                device,
            )
            if enable_confidence_head
            else None
        )

    def dspark_markov_bias(self, previous_token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(previous_token_ids)

    def dspark_confidence_probs(
        self,
        hidden_all: torch.Tensor,
        prev_matrix: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.confidence_head is None:
            raise RuntimeError("DSpark confidence head is not enabled")
        markov_embedding = None
        if prev_matrix is not None:
            markov_embedding = self.markov_head.embed(prev_matrix)
        return self.confidence_head(hidden_all, markov_embedding)

    def has_dspark_confidence_head(self) -> bool:
        return self.confidence_head is not None
