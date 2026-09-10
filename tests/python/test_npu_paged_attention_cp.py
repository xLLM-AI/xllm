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

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))

from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
    _build_stable_sfa_page_layout,
)
from xllm.python.model_executor.cp_utils import build_cp_context  # noqa: E402
from xllm.python.model_executor.runners.decode_acl_graph import (  # noqa: E402
    _StaticAttentionMetadata,
)


def test_decode_prepare_does_not_require_cp_metadata_fields() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._is_mla = True
    backend._kv_owner_representatives = torch.tensor([0])
    backend._materialized_block_table = torch.tensor([[0]], dtype=torch.int32)
    backend._sfa_page_layout = _build_stable_sfa_page_layout(
        torch.tensor([[0]], dtype=torch.int32),
    )

    backend._prepare_kv_shard_materialization(SimpleNamespace(is_prefill=False, is_chunked_prefill=False))

    assert backend._kv_owner_representatives is None
    assert backend._materialized_block_table is None
    assert backend._sfa_page_layout is None


def test_mla_index_context_accepts_decode_graph_static_metadata() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    slot_mapping = torch.arange(2, dtype=torch.int64)
    backend._metadata = _StaticAttentionMetadata(
        slot_mapping=slot_mapping,
        paged_kv_indptr=torch.arange(3, dtype=torch.int32),
        paged_kv_indices=torch.arange(2, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(2, dtype=torch.int32),
    )
    backend._block_table_i32 = torch.arange(2, dtype=torch.int32).view(2, 1)
    backend._mla_actual_seq_q = torch.arange(1, 3, dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.arange(1, 3, dtype=torch.int32)
    backend._kv_caches = [
        SimpleNamespace(
            index=torch.zeros(2, 1, 1),
            index_scale=None,
        )
    ]

    with patch(
        "xllm.python.attention.npu_paged_attention.get_forward_context",
        return_value=SimpleNamespace(cp_context=None),
    ):
        context = backend.mla_index_context(SimpleNamespace(layer_id=0))

    assert context.slot_mapping.data_ptr() == slot_mapping.data_ptr()


def test_owner_local_index_write_ignores_non_owned_slots() -> None:
    cache = torch.full((2, 2, 1, 1), -1.0)
    slots = torch.tensor([0, -1, 1, -1, 2], dtype=torch.int64)
    values = torch.arange(5, dtype=torch.float32).view(-1, 1)

    def scatter(var: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> None:
        rows = indices.flatten()
        valid = rows >= 0
        var.index_copy_(0, rows[valid], updates[valid])

    with patch(
        "xllm.python.attention.npu_paged_attention.kernels.scatter_nd_update",
        side_effect=scatter,
        create=True,
    ):
        NpuPagedAttentionBackend._update_mla_index_cache(
            cache,
            None,
            slots,
            values,
            None,
        )

    torch.testing.assert_close(cache.view(-1), torch.tensor([0.0, 2.0, 4.0, -1.0]))


def test_proper_divisor_materialization_selects_one_replica_per_owner() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._is_mla = True
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[3, 7, -1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        has_kv_shard=True,
        kv_split_size=2,
        kv_split_rank=0,
    )
    cp_context = SimpleNamespace(cp_size=4)

    def all_gather(tensor: torch.Tensor, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0
        assert world_size == 4
        assert group_name == "cp"
        if tensor.shape == (1,):
            return torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        return torch.cat([tensor + rank * 100 for rank in range(4)], dim=0)

    cache = torch.arange(16, dtype=torch.float32).view(8, 2, 1)
    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=4,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            side_effect=all_gather,
            create=True,
        ),
    ):
        backend._prepare_kv_shard_materialization(metadata)
        materialized, block_table = backend._materialize_cp_cache(cache, metadata, cp_context)

    assert block_table.tolist() == [[0, 1, 2, 3, -1, -1]]
    torch.testing.assert_close(materialized[0], cache[3])
    torch.testing.assert_close(materialized[1], cache[3] + 200)
    torch.testing.assert_close(materialized[2], cache[7])
    torch.testing.assert_close(materialized[3], cache[7] + 200)


def test_kv1_materialization_keeps_persistent_cache_view() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._is_mla = True
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[3, 1, -1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        has_kv_shard=True,
        kv_split_size=1,
        kv_split_rank=0,
    )
    cp_context = SimpleNamespace(cp_size=4)
    cache = torch.arange(8, dtype=torch.float32).view(4, 2, 1)

    with patch(
        "xllm.python.attention.npu_paged_attention.distributed.all_gather",
        create=True,
    ) as all_gather:
        backend._prepare_kv_shard_materialization(metadata)
        materialized, block_table = backend._materialize_cp_cache(cache, metadata, cp_context)

    all_gather.assert_not_called()
    assert materialized.data_ptr() == cache.data_ptr()
    assert block_table.data_ptr() == backend._block_table_i32.data_ptr()


def test_stable_sfa_layout_handles_multiple_sequences_and_invalid_tail() -> None:
    materialized_block_table = torch.tensor(
        [
            [4, 7, -1],
            [9, -1, -1],
            [3, 2, 8],
        ],
        dtype=torch.int32,
    )

    layout = _build_stable_sfa_page_layout(materialized_block_table)

    assert layout.source_page_ids.tolist() == [4, 7, 9, 3, 2, 8]
    assert layout.target_page_ids.tolist() == [1, 0, 3, 7, 6, 8]
    assert layout.block_table.tolist() == [
        [1, 0, -1],
        [3, -1, -1],
        [7, 6, 8],
    ]
    assert layout.page_count == 9


def test_build_cp_context_materializes_mla_segment_lengths_on_device() -> None:
    q_cu_seqlens = [2, 5]
    segment_kv_seq_lens = [4, 9]
    op_result = (
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([True, True]),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        q_cu_seqlens,
        [4, 9],
        torch.tensor([0, 1], dtype=torch.int64),
        segment_kv_seq_lens,
        2,
    )
    device = torch.device("cpu")

    with patch.object(
        torch.ops.xllm_ops,
        "build_cp_context",
        return_value=op_result,
        create=True,
    ):
        context = build_cp_context([2, 3], [4, 5], 2, 0, device)

    assert context.q_cu_seqlens is q_cu_seqlens
    assert context.segment_kv_seq_lens is segment_kv_seq_lens
    assert context.q_cu_seqlens_tensor.dtype == torch.int32
    assert context.q_cu_seqlens_tensor.device == device
    assert context.q_cu_seqlens_tensor.tolist() == q_cu_seqlens
    assert context.segment_kv_seq_lens_tensor.dtype == torch.int32
    assert context.segment_kv_seq_lens_tensor.device == device
    assert context.segment_kv_seq_lens_tensor.tolist() == segment_kv_seq_lens


def test_materialization_rejects_incomplete_owner_distribution() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._is_mla = True
    backend.device = torch.device("cpu")
    backend._block_table_i32 = torch.tensor([[0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        has_kv_shard=True,
        kv_split_size=2,
        kv_split_rank=0,
    )

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.cp_world_size",
            return_value=4,
            create=True,
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.distributed.all_gather",
            return_value=torch.tensor([0, 0, 0, 1], dtype=torch.int64),
            create=True,
        ),
        pytest.raises(RuntimeError, match="KV owner distribution"),
    ):
        backend._prepare_kv_shard_materialization(metadata)


@pytest.mark.parametrize("has_kv_shard", [False, True])
def test_mla_cp_uses_one_paged_sequence_per_zigzag_segment_and_reuses_lengths(
    has_kv_shard: bool,
) -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._metadata = SimpleNamespace(
        has_kv_shard=has_kv_shard,
        kv_split_size=1,
        local_slot_mapping=torch.arange(6, dtype=torch.int64),
        slot_mapping=torch.arange(6, dtype=torch.int64),
    )
    backend._block_table_i32 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    nope_cache = torch.zeros(4, 8, 2)
    rope_cache = torch.zeros(4, 8, 1)
    backend._kv_caches = [
        SimpleNamespace(key=nope_cache, value=rope_cache),
        SimpleNamespace(key=nope_cache, value=rope_cache),
    ]
    backend._mla_actual_seq_q = torch.tensor([3, 6], dtype=torch.int32)
    backend._mla_actual_seq_kv = torch.tensor([12, 7], dtype=torch.int32)

    q_cu_seqlens_tensor = torch.tensor([2, 3, 5], dtype=torch.int32)
    segment_kv_seq_lens_tensor = torch.tensor([10, 12, 7], dtype=torch.int32)
    cp_context = SimpleNamespace(
        query_index=torch.tensor([0, 1, 3, 4, 5], dtype=torch.int64),
        segment_seq_indices=torch.tensor([0, 0, 1], dtype=torch.int64),
        q_cu_seqlens=[2, 3, 5],
        q_cu_seqlens_tensor=q_cu_seqlens_tensor,
        segment_kv_seq_lens=[10, 12, 7],
        segment_kv_seq_lens_tensor=segment_kv_seq_lens_tensor,
    )
    q_latent = torch.zeros(6, 1, 2)
    q_pe = torch.zeros(6, 1, 1)
    k_latent = torch.zeros(6, 1, 2)
    k_pe = torch.zeros(6, 1, 1)
    topk = torch.arange(6, dtype=torch.int32).view(6, 1)

    with (
        patch(
            "xllm.python.attention.npu_paged_attention.get_forward_context",
            return_value=SimpleNamespace(cp_context=cp_context),
        ),
        patch(
            "xllm.python.attention.npu_paged_attention.cp_gather_kv",
            side_effect=lambda tensor, _context: tensor,
        ),
        patch.object(torch.ops.xllm_ops, "reshape_paged_cache", create=True),
        patch.object(
            backend,
            "_mla_sparse",
            side_effect=lambda query, *_args: torch.ones_like(query),
        ) as sparse,
    ):
        outputs = [
            backend.execute_mla(
                q_latent,
                q_pe,
                k_latent,
                k_pe,
                SimpleNamespace(layer_id=layer_id),
                topk,
            )
            for layer_id in range(2)
        ]

    assert sparse.call_count == 2
    sparse_args = sparse.call_args_list[0].args
    torch.testing.assert_close(
        sparse_args[5],
        torch.tensor([[0, 1], [0, 1], [2, 3]], dtype=torch.int32),
    )
    for call in sparse.call_args_list:
        assert call.args[6] is q_cu_seqlens_tensor
        assert call.args[7] is segment_kv_seq_lens_tensor
        assert call.args[6].data_ptr() == q_cu_seqlens_tensor.data_ptr()
        assert call.args[7].data_ptr() == segment_kv_seq_lens_tensor.data_ptr()
    for output in outputs:
        torch.testing.assert_close(output[cp_context.query_index], torch.ones(5, 1, 2))
        torch.testing.assert_close(output[2], torch.zeros(1, 2))
