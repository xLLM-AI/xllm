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
#

"""Typed algorithm-specific contexts for the generic MLA index contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from xllm.python.attention.backend import MlaIndexContext


@dataclass(frozen=True, slots=True)
class KPoolIndexContext:
    """KPool state owned by an MLA indexer with compressed index tails."""

    tail: torch.Tensor
    tail_read_indices: torch.Tensor | None
    tail_write_indices: torch.Tensor | None
    query_lens: Sequence[int] = ()
    query_lens_device: torch.Tensor | None = None
    triton_compatible: bool = False


class MlaIndexContextBuilder(Protocol):
    """Build a typed MLA context from backend-owned common state."""

    @staticmethod
    def specialized(*, layer_cache: object, metadata: object, triton_compatible: bool) -> object | None: ...

    @staticmethod
    def build(common: MlaIndexContext, specialized: object | None) -> MlaIndexContext: ...


@dataclass(frozen=True)
class KPoolMlaIndexContext(MlaIndexContext):
    """MLA index context with the typed state required by KPool."""

    kpool: KPoolIndexContext | None = None


class KPoolMlaIndexContextBuilder:
    """Build the KPool specialization without coupling the backend to GLM."""

    @staticmethod
    def specialized(*, layer_cache: object, metadata: object, triton_compatible: bool) -> KPoolIndexContext | None:
        tail = getattr(layer_cache, "kpool_tail", None)
        if tail is None:
            return None
        from xllm.python.attention.backend import resolve_linear_state_io_indices

        read_indices, write_indices = resolve_linear_state_io_indices(metadata)
        return KPoolIndexContext(
            tail=tail,
            tail_read_indices=read_indices,
            tail_write_indices=write_indices,
            query_lens=tuple(getattr(metadata, "kpool_query_lens", ())),
            query_lens_device=getattr(metadata, "kpool_query_lens_device", None),
            triton_compatible=triton_compatible,
        )

    @staticmethod
    def build(
        common: MlaIndexContext,
        specialized: object | None,
    ) -> KPoolMlaIndexContext:
        if specialized is not None and not isinstance(specialized, KPoolIndexContext):
            raise TypeError(f"expected KPoolIndexContext, got {type(specialized).__name__}")
        return KPoolMlaIndexContext(
            index_cache=common.index_cache,
            slot_mapping=common.slot_mapping,
            block_table=common.block_table,
            actual_seq_q=common.actual_seq_q,
            actual_seq_kv=common.actual_seq_kv,
            index_cache_scale=common.index_cache_scale,
            get_quant_indexer_metadata=common.get_quant_indexer_metadata,
            update_index_cache=common.update_index_cache,
            materialize_index_cache=common.materialize_index_cache,
            cp_context=common.cp_context,
            actual_seq_kv_host=common.actual_seq_kv_host,
            kpool=specialized,
        )
