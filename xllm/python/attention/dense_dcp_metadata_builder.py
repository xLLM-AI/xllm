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

"""Normalize dense DCP attention inputs and update entry-owned metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from xllm.python.attention.dense_dcp_metadata import DenseDcpMetadata
from xllm.python.attention.kv_shard_layout import KVShardLayout

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata, LayerCache
    from xllm.python.model_executor.input_batch import InputBatch


class DenseDcpMetadataBuilder:
    metadata_type = DenseDcpMetadata

    def __init__(self, dcp_size: int, dcp_rank: int) -> None:
        if dcp_size <= 1 or not 0 <= dcp_rank < dcp_size:
            raise ValueError("Dense DCP requires a valid rank in a group with more than one member")
        self._dcp_size = dcp_size
        self._dcp_rank = dcp_rank
        self._kv_layout: KVShardLayout | None = None

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        page_sizes = {int(cache.key.shape[1]) for cache in layer_caches if cache.key is not None}
        if len(page_sizes) != 1:
            raise RuntimeError("Dense DCP requires full-attention caches with a common page size")
        self._kv_layout = KVShardLayout(page_sizes.pop(), self._dcp_size, self._dcp_rank)

    def build(self, input_batch: InputBatch, metadata: AttentionMetadata) -> DenseDcpMetadata:
        slots = self._local_slots(input_batch, metadata)
        query_ends, local_lengths, empty_rows, has_context = self._attention_inputs(input_batch, metadata)
        block_table = metadata.block_table
        if block_table is not None:
            block_table = block_table.to(torch.int32)
        return DenseDcpMetadata(
            local_slot_mapping=slots,
            query_seq_ends=query_ends,
            local_kv_lengths=local_lengths,
            empty_kv_shards=torch.tensor(empty_rows, dtype=torch.bool, device=slots.device).view(-1, 1),
            block_table=block_table,
            is_prefill=metadata.is_prefill,
            is_chunked_prefill=metadata.is_chunked_prefill,
            has_context=has_context,
            has_local_context=any(local_lengths),
        )

    def allocate_persistent(self, input_batch: InputBatch, metadata: AttentionMetadata) -> DenseDcpMetadata:
        source = self._local_slots(input_batch, metadata)
        if metadata.is_prefill or metadata.is_chunked_prefill:
            raise NotImplementedError("Dense DCP graph metadata supports only decode")
        self._attention_inputs(input_batch, metadata)
        capacity = input_batch.num_tokens_after_padding
        return DenseDcpMetadata(
            local_slot_mapping=torch.full((capacity,), -1, dtype=source.dtype, device=source.device),
            query_seq_ends=list(range(1, capacity + 1)),
            local_kv_lengths=[0] * capacity,
            empty_kv_shards=torch.ones((capacity, 1), dtype=torch.bool, device=source.device),
            # Reuse the runner-owned table; the execution adapter binds it in prepare().
            block_table=None,
            is_prefill=False,
            is_chunked_prefill=False,
            has_context=False,
            has_local_context=False,
        )

    def update_persistent(
        self,
        persistent_metadata: object,
        input_batch: InputBatch,
        metadata: AttentionMetadata,
    ) -> None:
        if not isinstance(persistent_metadata, DenseDcpMetadata):
            raise TypeError("Dense DCP received invalid persistent metadata")
        source = self._local_slots(input_batch, metadata)
        if metadata.is_prefill or metadata.is_chunked_prefill:
            raise NotImplementedError("Dense DCP graph metadata supports only decode")
        _, local_lengths, empty_rows, has_context = self._attention_inputs(input_batch, metadata)
        target = persistent_metadata.local_slot_mapping
        if target.shape != (input_batch.num_tokens_after_padding,):
            raise RuntimeError("Dense DCP metadata capacity does not match the execution batch")
        if target.dtype != source.dtype or target.device != source.device:
            raise RuntimeError("Dense DCP local slot dtype or device changed after graph allocation")
        target[: input_batch.num_tokens].copy_(source)
        target[input_batch.num_tokens :].fill_(-1)
        padding = target.numel() - input_batch.num_tokens
        persistent_metadata.local_kv_lengths[:] = local_lengths + [0] * padding
        persistent_metadata.empty_kv_shards.copy_(
            torch.tensor(empty_rows + [True] * padding, dtype=torch.bool, device="cpu").view(-1, 1)
        )
        persistent_metadata.has_context = has_context
        persistent_metadata.has_local_context = any(local_lengths)

    def _attention_inputs(
        self, input_batch: InputBatch, metadata: AttentionMetadata
    ) -> tuple[list[int], list[int], list[bool], bool]:
        if self._kv_layout is None:
            raise RuntimeError("Dense DCP metadata requires bound layer caches")
        is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
        is_pure_prefill = metadata.is_prefill and not metadata.is_chunked_prefill
        if input_batch.is_dummy and input_batch.num_reqs == 0:
            query_lengths = [input_batch.num_tokens] if is_prefill else [1] * input_batch.num_tokens
            query_ends = [input_batch.num_tokens] if is_prefill else list(range(1, input_batch.num_tokens + 1))
            global_lengths = [0] * len(query_lengths)
        else:
            query_lengths = list(input_batch.num_scheduled_tokens)
            query_ends = list(input_batch.query_start_loc[1:])
            if not is_prefill and any(length != 1 for length in query_lengths):
                raise NotImplementedError("Dense DCP decode requires one query token per request")
            if is_pure_prefill:
                if any(input_batch.num_computed_tokens):
                    raise RuntimeError("Dense DCP prefill with cached context requires chunked-prefill metadata")
                # Pure prefill attends directly to this step's K/V, not the sharded cache.
                return query_ends, [], [False] * input_batch.num_tokens, False
            values = metadata.kv_seq_lens_host_values
            if values is None:
                raise RuntimeError("Dense DCP requires upstream host KV lengths")
            global_lengths = list(values[: len(query_lengths)])
            if len(global_lengths) != len(query_lengths) or any(length < 0 for length in global_lengths):
                raise RuntimeError("Dense DCP requires one non-negative KV length per execution sequence")
            if metadata.is_chunked_prefill:
                global_lengths = [
                    kv_length - query_length
                    for kv_length, query_length in zip(global_lengths, query_lengths, strict=True)
                ]
                if any(length < 0 for length in global_lengths):
                    raise RuntimeError("Dense DCP chunked prefill query length exceeds its KV length")

        if not is_pure_prefill:
            table = metadata.block_table
            if table is None or table.ndim != 2 or table.shape[0] < len(query_lengths):
                raise RuntimeError("Dense DCP cached attention requires a block-table row per execution sequence")
            if table.device != input_batch.input_ids.device or table.dtype not in (torch.int32, torch.int64):
                raise RuntimeError("Dense DCP block table must use an integer dtype on the input device")
        local_lengths = [self._kv_layout.local_token_count(length) for length in global_lengths]
        empty_rows = [
            length == 0
            for length, query_length in zip(local_lengths, query_lengths, strict=True)
            for _ in range(query_length)
        ]
        return query_ends, local_lengths, empty_rows, any(global_lengths)

    def _local_slots(self, input_batch: InputBatch, metadata: AttentionMetadata) -> torch.Tensor:
        if getattr(metadata, "is_spec_verify", False):
            raise NotImplementedError("Dense DCP attention does not support speculative verification")
        if not metadata.has_kv_shard:
            raise RuntimeError("Dense DCP requires upstream KV-shard metadata")
        if int(metadata.kv_split_size) != self._dcp_size or int(metadata.kv_split_rank) != self._dcp_rank:
            raise RuntimeError("Dense DCP metadata topology does not match the active process group")
        slots = metadata.local_slot_mapping
        if slots is None:
            raise RuntimeError("Dense DCP requires upstream local slot metadata")
        if slots.shape != (input_batch.num_tokens,):
            raise RuntimeError("Dense DCP local slot metadata must contain one entry per execution token")
        if slots.dtype not in (torch.int32, torch.int64):
            raise TypeError("Dense DCP local slots must use an integer dtype")
        if slots.device != input_batch.input_ids.device:
            raise RuntimeError("Dense DCP local slots must be on the input device")
        return slots
