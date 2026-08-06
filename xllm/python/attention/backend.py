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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention

KVCache = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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


@dataclass(frozen=True)
class MlaIndexContext:
    """Public contract handed to an optional LightningIndexer.

    Replaces direct model access to ``backend._metadata`` / ``backend._kv_caches``
    for MLA layers. The backend owns the paged index cache (the third slot of
    ``KVCache``) and prepares the paging / sequence-length metadata once per
    step; the indexer receives this view and produces ``topk``.
    """

    index_cache: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor | None
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor


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

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent: torch.Tensor,
        k_pe: torch.Tensor,
        layer: "Attention",
        topk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Absorbed-MLA attention over paged latent (nope) + rope caches.

        Returns ``[T, H, kv_lora]``; caller bmm's ``W_UV``. When ``topk`` is
        provided, dispatches to the sparse SFA path driven by an optional
        LightningIndexer; otherwise a dense MLA path is requested. Backends
        that do not implement MLA raise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support MLA"
        )

    def mla_index_context(self, layer: "Attention") -> MlaIndexContext:
        """Public hook for an optional LightningIndexer.

        Hands out the paged index cache view (third ``KVCache`` slot) plus the
        paging / sequence-length metadata the indexer needs, so the model never
        touches ``backend._metadata`` / ``backend._kv_caches`` directly.
        Backends that do not support the sparse MLA indexer raise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support MLA indexer"
        )
