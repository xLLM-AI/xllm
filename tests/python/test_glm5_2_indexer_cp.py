# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM indexer/backend contracts; accelerator kernels and transport use CPU fakes."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.model_executor.cp_utils import CpContext, cp_shard_positions, cp_shard_rows
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models import glm5_2


def _three_token_plan(rank: int) -> CpContext:
    shard = [0, -1] if rank == 0 else [1, 2]
    q_ends = [1] if rank == 0 else [1, 2]
    kv_lengths = [1] if rank == 0 else [2, 3]
    return CpContext(
        cp_size=2,
        cp_rank=rank,
        total_local=2,
        shard_index=torch.tensor(shard),
        shard_gather_index=torch.tensor(shard).clamp_min(0),
        shard_valid_mask=torch.tensor(shard) >= 0,
        restore_index=torch.tensor([0, 2, 3]),
        query_index=torch.tensor([0] if rank == 0 else [0, 1]),
        q_cu_seqlens=q_ends,
        q_cu_seqlens_tensor=torch.tensor(q_ends, dtype=torch.int32),
        kv_gather_index=torch.tensor([0] if rank == 0 else [0, 1, 0, 1, 2]),
        kv_cu_seqlens=[1] if rank == 0 else [2, 5],
        segment_seq_indices=torch.tensor([0] if rank == 0 else [0, 0]),
        segment_kv_seq_lens=kv_lengths,
        segment_kv_seq_lens_tensor=torch.tensor(kv_lengths, dtype=torch.int32),
        has_prefix=False,
    )


def _indexer() -> glm5_2.Glm52Indexer:
    cfg = glm5_2.Glm52Config(
        hidden_size=2,
        q_lora_rank=2,
        index_n_heads=1,
        index_head_dim=2,
        qk_rope_head_dim=2,
        index_topk=1,
        indexer_rope_interleave=False,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        indexer.wq_b.weight.copy_(torch.eye(2, dtype=torch.int8))
        indexer.wq_b.deq_scale.fill_(1)
        indexer.wq_b.quant_bias.zero_()
        indexer.wq_b.input_scale.fill_(1)
        indexer.wq_b.input_offset.zero_()
        indexer.wk.weight.copy_(torch.eye(2))
        indexer.k_norm.weight.zero_()
        indexer.k_norm.bias.copy_(torch.tensor([5.0, 7.0]))
        indexer.weights_proj.weight.fill_(1)
    return indexer


def _backend_and_metadata(
    quantized: bool, num_blocks: int = 2
) -> tuple[NpuPagedAttentionBackend, SimpleNamespace, LayerCache]:
    backend = NpuPagedAttentionBackend(1, 1, 2, 1.0, -1, True, torch.device("cpu"), torch.float32)
    cache = LayerCache(
        key=torch.zeros(num_blocks, 2, 1, 2),
        value=torch.zeros(num_blocks, 2, 1, 2),
        index=torch.full((num_blocks, 2, 1, 2), -9, dtype=torch.int8 if quantized else torch.float32),
        indexer_scale=torch.full((num_blocks, 2, 1, 1), -9.0, dtype=torch.float16) if quantized else None,
    )
    backend.bind_kv_caches([cache])
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2]),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([3]),
        kv_seq_lens_host_values=[3],
        q_cu_seq_lens=torch.tensor([0, 3]),
        q_seq_lens=torch.tensor([3]),
        expanded_decode_metadata=None,
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        has_kv_shard=False,
        kv_split_size=1,
    )
    return backend, metadata, cache


def _quantize(x: torch.Tensor, *_args: object) -> torch.Tensor:
    if x.shape[0] == 0:
        raise AssertionError("empty-query ranks must not launch the Q projection")
    return x.round().to(torch.int8)


def _quant_matmul(x: torch.Tensor, weight: torch.Tensor, *_args: object) -> torch.Tensor:
    return x.float() @ weight.float().T


def _dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones_like(x, dtype=torch.int8), torch.ones(x.shape[:-1])


def _scatter(cache: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
    for slot, value in zip(indices.flatten().tolist(), values, strict=True):
        if slot >= 0:
            cache[slot].copy_(value)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("quantized", [False, True])
def test_short_prompt_indexer_keeps_queries_local_and_returns_padded_topk(rank: int, quantized: bool) -> None:
    plan = _three_token_plan(rank)
    indexer = _indexer()
    backend, metadata, cache = _backend_and_metadata(quantized)
    global_hidden = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    local_hidden = cp_shard_rows(global_hidden, plan)
    local_positions = cp_shard_positions(torch.arange(3), plan)
    rope = torch.tensor([[1.0, 0.0]]).expand(3, -1)
    expected = torch.tensor([[[0]], [[-1]]] if rank == 0 else [[[1]], [[2]]], dtype=torch.int32)

    def all_gather(local: torch.Tensor, dim: int, world_size: int, group: str) -> torch.Tensor:
        assert (dim, world_size, group) == (0, 2, "cp")
        assert local.shape == (2, 2), "only padded K rows may enter the CP collective"
        return local.new_tensor([[5, 7], [5, 7], [5, 7], [5, 7]])

    def indexer_metadata(
        heads_q: int,
        heads_k: int,
        head_dim: int,
        q_ends: torch.Tensor,
        kv_lengths: torch.Tensor,
        max_q: int,
        max_k: int,
        topk: int,
        ratio: int,
    ) -> torch.Tensor:
        assert (heads_q, heads_k, head_dim, max_q, max_k, topk, ratio) == (64, 1, 2, 1, 1 if rank == 0 else 3, 1, 1)
        assert q_ends.tolist() == ([1] if rank == 0 else [1, 2])
        assert kv_lengths.tolist() == ([1] if rank == 0 else [2, 3])
        return torch.empty(0, dtype=torch.int32)

    def select(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, *args: object) -> torch.Tensor:
        q_ends, kv_lengths, blocks = args[3:6] if quantized else args[:3]
        assert query.shape[0] == (1 if rank == 0 else 2)
        assert weights.shape[0] == query.shape[0]
        assert q_ends.tolist() == ([1] if rank == 0 else [1, 2])
        assert kv_lengths.tolist() == ([1] if rank == 0 else [2, 3])
        assert blocks.tolist() == ([[0, 1]] if rank == 0 else [[0, 1], [0, 1]])
        if not quantized:
            assert query[:, 0, 0].tolist() == ([1.0] if rank == 0 else [2.0, 3.0])
        return torch.tensor([[[0]]] if rank == 0 else [[[1]], [[2]]], dtype=torch.int32)

    context = ForwardContext(backend, torch.device("cpu"), metadata, [cache], cp_context=plan)
    with (
        forward_context(context),
        patch.object(glm5_2.kernels, "quantize_per_tensor", side_effect=_quantize, create=True),
        patch.object(glm5_2.kernels, "quant_matmul", side_effect=_quant_matmul, create=True),
        patch.object(glm5_2.kernels, "dynamic_quant", side_effect=_dynamic_quant, create=True),
        patch.object(glm5_2.kernels, "scatter_nd_update", side_effect=_scatter, create=True),
        patch.object(glm5_2.kernels, "quant_lightning_indexer_metadata", side_effect=indexer_metadata, create=True),
        patch.object(
            glm5_2.kernels,
            "quant_lightning_indexer" if quantized else "lightning_indexer",
            side_effect=select,
            create=True,
        ),
        patch("xllm.python.model_executor.cp_utils.distributed.all_gather", side_effect=all_gather, create=True),
    ):
        backend.prepare(metadata)
        output = indexer.select_qli(
            local_hidden.index_select(0, plan.query_index),
            local_hidden.index_select(0, plan.query_index),
            local_positions.index_select(0, plan.query_index),
            backend.mla_index_context(SimpleNamespace(layer_id=0)),
            rope,
            cache_hidden=local_hidden,
            cache_positions=local_positions,
        )

    torch.testing.assert_close(output, expected)
    expected_keys = torch.ones(3, 2, dtype=torch.int8) if quantized else torch.tensor([[5.0, 7.0]]).expand(3, -1)
    torch.testing.assert_close(cache.index.view(4, 2)[:3], expected_keys)
    assert cache.index.view(4, 2)[3].tolist() == [-9, -9]
    if quantized:
        assert cache.index_scale.view(-1).tolist() == [1.0, 1.0, 1.0, -9.0]


@pytest.mark.parametrize("quantized", [False, True])
def test_empty_query_rank_still_populates_index_cache(quantized: bool) -> None:
    empty = torch.empty(0, dtype=torch.int64)
    plan = replace(
        _three_token_plan(1),
        shard_index=torch.tensor([-1, -1]),
        shard_gather_index=torch.tensor([0, 0]),
        shard_valid_mask=torch.tensor([False, False]),
        restore_index=torch.tensor([0]),
        query_index=empty,
        q_cu_seqlens=[],
        q_cu_seqlens_tensor=empty.to(torch.int32),
        kv_gather_index=empty,
        kv_cu_seqlens=[],
        segment_seq_indices=empty,
        segment_kv_seq_lens=[],
        segment_kv_seq_lens_tensor=empty.to(torch.int32),
    )
    indexer = _indexer()
    backend, metadata, cache = _backend_and_metadata(quantized)
    metadata.slot_mapping = torch.tensor([0])
    metadata.kv_seq_lens = torch.tensor([1])
    metadata.kv_seq_lens_host_values = [1]
    metadata.q_cu_seq_lens = torch.tensor([0, 1])
    metadata.q_seq_lens = torch.tensor([1])

    def all_gather(local: torch.Tensor, dim: int, world_size: int, group: str) -> torch.Tensor:
        assert (dim, world_size, group) == (0, 2, "cp")
        assert local.shape == (2, 2)
        return local.new_tensor([[5, 7], [5, 7], [5, 7], [5, 7]])

    def select(*_args: object) -> torch.Tensor:
        raise AssertionError("empty-query ranks must not launch a lightning indexer")

    context = ForwardContext(backend, torch.device("cpu"), metadata, [cache], cp_context=plan)
    with (
        forward_context(context),
        patch.object(glm5_2.kernels, "quantize_per_tensor", side_effect=_quantize, create=True),
        patch.object(glm5_2.kernels, "quant_matmul", side_effect=_quant_matmul, create=True),
        patch.object(glm5_2.kernels, "dynamic_quant", side_effect=_dynamic_quant, create=True),
        patch.object(glm5_2.kernels, "scatter_nd_update", side_effect=_scatter, create=True),
        patch.object(glm5_2.kernels, "quant_lightning_indexer", side_effect=select, create=True),
        patch.object(glm5_2.kernels, "lightning_indexer", side_effect=select, create=True),
        patch("xllm.python.model_executor.cp_utils.distributed.all_gather", side_effect=all_gather, create=True),
    ):
        backend.prepare(metadata)
        output = indexer.select_qli(
            torch.empty(0, 2),
            torch.empty(0, 2),
            empty,
            backend.mla_index_context(SimpleNamespace(layer_id=0)),
            torch.tensor([[1.0, 0.0]]),
            cache_hidden=torch.zeros(2, 2),
            cache_positions=torch.zeros(2, dtype=torch.int64),
        )

    assert output.dtype == torch.int32
    assert output.tolist() == [[[-1]], [[-1]]]
    assert cache.index.view(4, 2)[0].tolist() == ([1, 1] if quantized else [5.0, 7.0])
    assert cache.index.view(4, 2)[1:].tolist() == [[-9, -9]] * 3
    if quantized:
        assert cache.index_scale.view(-1).tolist() == [1.0, -9.0, -9.0, -9.0]


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("quantized", [False, True])
def test_packed_chunked_pcp4_preserves_segment_owners_and_prefix(quantized: bool, rank: int) -> None:
    # Requests extend prefixes of lengths 3/2 by 5/1 tokens. PCP=4 pads each
    # request to eight chunks. Rank 0 owns tokens 0/5; rank 3 owns 3/4.
    shard = [0, -1, 5, -1] if rank == 0 else [3, 4, -1, -1]
    query_rows = [0, 2] if rank == 0 else [0, 1]
    visible_lengths = [4, 3] if rank == 0 else [7, 8]
    sequences = [0, 1] if rank == 0 else [0, 0]
    plan = replace(
        _three_token_plan(0),
        cp_size=4,
        cp_rank=rank,
        total_local=4,
        shard_index=torch.tensor(shard),
        shard_gather_index=torch.tensor(shard).clamp_min(0),
        shard_valid_mask=torch.tensor(shard) >= 0,
        restore_index=torch.tensor([0, 4, 8, 12, 13, 2]),
        query_index=torch.tensor(query_rows),
        q_cu_seqlens=[1, 2],
        q_cu_seqlens_tensor=torch.tensor([1, 2], dtype=torch.int32),
        kv_gather_index=torch.tensor([0, 5] if rank == 0 else [0, 1, 2, 3, 0, 1, 2, 3, 4]),
        kv_cu_seqlens=[1, 2] if rank == 0 else [4, 9],
        segment_seq_indices=torch.tensor(sequences),
        segment_kv_seq_lens=visible_lengths,
        segment_kv_seq_lens_tensor=torch.tensor(visible_lengths, dtype=torch.int32),
        has_prefix=True,
    )
    indexer = _indexer()
    backend, metadata, cache = _backend_and_metadata(quantized, num_blocks=8)
    metadata.slot_mapping = torch.tensor([3, 4, 5, 6, 7, 10])
    metadata.block_table = torch.tensor([[0, 1, 2, 3], [4, 5, -1, -1]], dtype=torch.int32)
    metadata.kv_seq_lens = torch.tensor([8, 3])
    metadata.kv_seq_lens_host_values = [8, 3]
    metadata.q_cu_seq_lens = torch.tensor([0, 5, 6])
    metadata.q_seq_lens = torch.tensor([5, 1])
    metadata.is_prefill = False
    metadata.is_chunked_prefill = True
    hidden = cp_shard_rows(torch.arange(1, 7).float().unsqueeze(1).expand(-1, 2), plan)
    positions = cp_shard_positions(torch.tensor([3, 4, 5, 6, 7, 2]), plan)

    def all_gather(local: torch.Tensor, dim: int, world_size: int, group: str) -> torch.Tensor:
        assert (dim, world_size, group) == (0, 4, "cp")
        assert local.shape == (4, 2)
        return local.new_tensor([[5, 7]] * 16)

    def indexer_metadata(
        heads_q: int,
        heads_k: int,
        head_dim: int,
        q_ends: torch.Tensor,
        kv_lengths: torch.Tensor,
        max_q: int,
        max_k: int,
        topk: int,
        ratio: int,
    ) -> torch.Tensor:
        assert (max_q, max_k) == (1, 4 if rank == 0 else 8)
        assert q_ends.tolist() == [1, 2]
        assert kv_lengths.tolist() == visible_lengths
        return torch.empty(0, dtype=torch.int32)

    def select(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, *args: object) -> torch.Tensor:
        q_ends, kv_lengths, blocks = args[3:6] if quantized else args[:3]
        assert query.shape[0] == weights.shape[0] == 2
        assert q_ends.tolist() == [1, 2]
        assert kv_lengths.tolist() == visible_lengths
        assert blocks.tolist() == ([[0, 1, 2, 3], [4, 5, -1, -1]] if rank == 0 else [[0, 1, 2, 3]] * 2)
        return torch.tensor([[[3]], [[2]]] if rank == 0 else [[[6]], [[7]]], dtype=torch.int32)

    with (
        forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [cache], cp_context=plan)),
        patch.object(glm5_2.kernels, "quantize_per_tensor", side_effect=_quantize, create=True),
        patch.object(glm5_2.kernels, "quant_matmul", side_effect=_quant_matmul, create=True),
        patch.object(glm5_2.kernels, "dynamic_quant", side_effect=_dynamic_quant, create=True),
        patch.object(glm5_2.kernels, "scatter_nd_update", side_effect=_scatter, create=True),
        patch.object(glm5_2.kernels, "quant_lightning_indexer_metadata", side_effect=indexer_metadata, create=True),
        patch.object(
            glm5_2.kernels,
            "quant_lightning_indexer" if quantized else "lightning_indexer",
            side_effect=select,
            create=True,
        ),
        patch("xllm.python.model_executor.cp_utils.distributed.all_gather", side_effect=all_gather, create=True),
    ):
        backend.prepare(metadata)
        output = indexer.select_qli(
            hidden.index_select(0, plan.query_index),
            hidden.index_select(0, plan.query_index),
            positions.index_select(0, plan.query_index),
            backend.mla_index_context(SimpleNamespace(layer_id=0)),
            torch.tensor([[1.0, 0.0]]).expand(8, -1),
            cache_hidden=hidden,
            cache_positions=positions,
        )

    assert output.view(-1).tolist() == ([3, -1, 2, -1] if rank == 0 else [6, 7, -1, -1])
    # Existing prefixes and unused blocks are never rewritten by the chunk.
    assert cache.index.view(16, 2)[[0, 1, 2, 8, 9, 11, 12, 13, 14, 15]].tolist() == [[-9, -9]] * 10
    if quantized:
        assert cache.index_scale.view(-1)[[0, 1, 2, 8, 9, 11, 12, 13, 14, 15]].tolist() == [-9.0] * 10
