# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention

KVCache = tuple[torch.Tensor, torch.Tensor]


class AttentionMetadata(Protocol):
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None
    q_cu_seq_lens: torch.Tensor | None
    kv_cu_seq_lens: torch.Tensor | None
    kv_seq_lens_host: torch.Tensor | None
    paged_kv_indptr_host: torch.Tensor | None
    paged_kv_last_page_len_host: torch.Tensor | None
    block_table: torch.Tensor | None
    kv_seq_lens: torch.Tensor | None
    is_prefill: bool
    is_chunked_prefill: bool


class AttentionBackend(ABC):
    @abstractmethod
    def bind_kv_caches(self, kv_caches: list[KVCache]) -> None:
        pass

    @abstractmethod
    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_kv_blocks(self) -> int:
        pass

    @property
    @abstractmethod
    def page_size(self) -> int:
        pass
