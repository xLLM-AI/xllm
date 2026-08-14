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
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

import torch

from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadataLike,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


@dataclass(frozen=True, slots=True)
class LayerCache:
    """Every cache a layer may own, named rather than positional.

    A layer holds a subset: full attention uses ``key``/``value``, the MLA
    sparse indexer adds ``index``, and a linear-attention layer uses
    ``conv``/``ssm`` instead of K/V. An absent slot is ``None`` rather than an
    empty tensor, so a layer that reads the wrong slot fails loudly.
    """

    key: torch.Tensor | None
    value: torch.Tensor | None
    index: torch.Tensor | None = None
    conv: torch.Tensor | None = None
    ssm: torch.Tensor | None = None
    # DeepSeek-V4 DSA cache slots. Generic models leave these as None; the
    # tuple order is shared with PyExecutorImpl::bind_kv_caches.
    swa: torch.Tensor | None = None
    compress_kv_state: torch.Tensor | None = None
    compress_score_state: torch.Tensor | None = None
    compress_index_kv_state: torch.Tensor | None = None
    compress_index_score_state: torch.Tensor | None = None
    indexer_scale: torch.Tensor | None = None


#: Field order of the tuple form, which is what the C++ executor hands over.
_LAYER_CACHE_SLOTS = (
    "key",
    "value",
    "index",
    "conv",
    "ssm",
    "swa",
    "compress_kv_state",
    "compress_score_state",
    "compress_index_kv_state",
    "compress_index_score_state",
    "indexer_scale",
)

LayerCacheInput = LayerCache | tuple[torch.Tensor | None, ...]


def normalize_layer_caches(caches: Sequence[LayerCacheInput]) -> list[LayerCache]:
    """Accept the tuple form and return named caches with empty slots dropped."""
    normalized: list[LayerCache] = []
    for cache in caches:
        if isinstance(cache, LayerCache):
            normalized.append(cache)
            continue
        if not 2 <= len(cache) <= len(_LAYER_CACHE_SLOTS):
            raise ValueError(f"layer cache must hold between K/V and {'/'.join(_LAYER_CACHE_SLOTS)} tensors")
        slots = [None if tensor is None or not tensor.numel() else tensor for tensor in cache]
        slots.extend([None] * (len(_LAYER_CACHE_SLOTS) - len(slots)))
        normalized.append(LayerCache(*slots))
    return normalized


class AttentionMetadata(Protocol):
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None
    q_cu_seq_lens: torch.Tensor | None
    kv_cu_seq_lens: torch.Tensor | None
    kv_seq_lens_host: torch.Tensor | None
    kv_seq_lens_host_values: list[int] | None
    q_seq_lens_host: torch.Tensor | None
    paged_kv_indptr_host: torch.Tensor | None
    paged_kv_last_page_len_host: torch.Tensor | None
    block_table: torch.Tensor | None
    kv_seq_lens: torch.Tensor | None
    max_query_len: int
    max_seq_len: int
    multi_block_tables: Sequence[torch.Tensor | None]
    dsa_metadata: object | None
    dsa_positions: torch.Tensor | None
    dsa_cos_sin: torch.Tensor | None
    dsa_c4_cos_sin: torch.Tensor | None
    dsa_c128_cos_sin: torch.Tensor | None
    dsa_graph_block_table_cols: int
    dsa_graph_mode: bool
    linear_state_indices: torch.Tensor | None
    has_initial_state: torch.Tensor | None
    dp_token_counts: Sequence[int]
    q_seq_lens: torch.Tensor | None
    expanded_decode_metadata: ExpandedDecodeMetadataLike
    is_prefill: bool
    is_chunked_prefill: bool


@dataclass(frozen=True)
class MlaIndexContext:
    """Public contract handed to an optional LightningIndexer.

    Replaces direct model access to ``backend._metadata`` / ``backend._kv_caches``
    for MLA layers. The backend owns the paged index cache (``LayerCache.index``)
    and prepares the paging / sequence-length metadata once per step; the indexer
    receives this view and produces ``topk``.
    """

    index_cache: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor | None
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor
    update_index_cache: Callable[[torch.Tensor], None]


@dataclass(frozen=True)
class DsaIndexContext:
    """Per-forward cache and metadata view consumed by the DSV4 indexer."""

    index_cache: torch.Tensor
    indexer_scale: torch.Tensor | None
    slot_mapping: torch.Tensor
    block_table: torch.Tensor | None
    cmp_block_table: torch.Tensor | None
    kv_state: torch.Tensor | None
    score_state: torch.Tensor | None
    kv_block_table: torch.Tensor | None
    score_block_table: torch.Tensor | None
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor
    start_pos: torch.Tensor | None
    qli_metadata: torch.Tensor | None


class AttentionBackend(ABC):
    @abstractmethod
    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
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
        layer: Attention,
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
        layer: Attention,
        topk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Absorbed-MLA attention over paged latent (nope) + rope caches.

        Returns ``[T, H, kv_lora]``; caller bmm's ``W_UV``. When ``topk`` is
        provided, dispatches to the sparse SFA path driven by an optional
        LightningIndexer; otherwise a dense MLA path is requested. Backends
        that do not implement MLA raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support MLA")

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        """Public hook for an optional LightningIndexer.

        Hands out the paged index cache view (``LayerCache.index``) plus the
        paging / sequence-length metadata the indexer needs, so the model never
        touches ``backend._metadata`` / ``backend._kv_caches`` directly.
        Backends that do not support the sparse MLA indexer raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support MLA indexer")
