# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""CPU tests for SFA DCP graph-prepare metadata."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import MlaIndexContext
from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.attention.sfa_dcp_backend import SfaDcpAttentionBackend
from xllm.python.layers.sfa_dcp import (
    AscendSFADCPMetadataBuilder,
    _lse_as_token_head,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    copy_into_execution_buffer,
    forward_context,
)
from xllm.python.models import glm5_2


def _cpu_context(execution_state: AclGraphExecutionState | None) -> ForwardContext:
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=execution_state,
    )


def _builder() -> AscendSFADCPMetadataBuilder:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0)
    return AscendSFADCPMetadataBuilder(
        layout=layout,
        device=torch.device("cpu"),
        max_num_reqs=4,
    )


def test_builder_graph_decode_skips_prefill_count() -> None:
    builder = _builder()
    slots = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([8, 4], dtype=torch.int32)

    metadata = builder.build(
        slots,
        block_table,
        seq_lens,
        num_reqs=2,
        num_input_tokens=2,
        num_prefills=0,
    )

    assert metadata.num_prefills == 0
    assert metadata.dcp_context.kv_gather_block_ids is None
    assert metadata.dcp_context.kv_gather_block_table is None
    assert torch.equal(metadata.dcp_context.seq_lens, torch.tensor([4, 4], dtype=torch.int32))


def test_copy_into_execution_buffer_reuses_graph_storage() -> None:
    state = AclGraphExecutionState({})
    with forward_context(_cpu_context(state)):
        first = torch.tensor([1, 2, 3], dtype=torch.int32)
        buffer = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (3,)), first)
        pointer = buffer.data_ptr()
        second = torch.tensor([4, 5, 6], dtype=torch.int32)
        reused = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (3,)), second)

        assert reused.data_ptr() == pointer
        assert torch.equal(reused, second)


def test_copy_into_execution_buffer_eager_returns_source() -> None:
    source = torch.tensor([1, 2], dtype=torch.int32)
    with forward_context(_cpu_context(None)):
        out = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (2,)), source)
        assert out.data_ptr() == source.data_ptr()


def test_sfa_dcp_backend_constructs_as_mla_backend() -> None:
    dcp_group = MagicMock()
    dcp_group.size.return_value = 2
    dcp_group.rank.return_value = 0

    backend = SfaDcpAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=0.125,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        dcp_group=dcp_group,
        index_topk=512,
        max_num_reqs=4,
    )

    assert backend.is_mla


def test_mla_index_context_uses_expanded_table_and_preserves_cp_context() -> None:
    backend = object.__new__(SfaDcpAttentionBackend)
    expanded_block_table = torch.tensor([[3, 1]], dtype=torch.int32)
    backend._expanded_indexer_block_table = expanded_block_table
    index_cache = torch.empty(1)
    index_cache_scale = torch.empty(1)
    materialized_cache = torch.empty(2)
    materialized_scale = torch.empty(2)
    materialized_block_table = torch.tensor([[5, 7]], dtype=torch.int32)
    materialize_index_cache = MagicMock(return_value=(materialized_cache, materialized_scale, materialized_block_table))
    cp_context = MagicMock()
    base_context = MlaIndexContext(
        index_cache=index_cache,
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        block_table=torch.tensor([[1]], dtype=torch.int32),
        actual_seq_q=torch.tensor([1], dtype=torch.int32),
        actual_seq_kv=torch.tensor([1], dtype=torch.int32),
        index_cache_scale=index_cache_scale,
        get_quant_indexer_metadata=MagicMock(),
        update_index_cache=MagicMock(),
        materialize_index_cache=materialize_index_cache,
        cp_context=cp_context,
    )

    with patch.object(
        NpuPagedAttentionBackend,
        "mla_index_context",
        return_value=base_context,
    ):
        remapped_context = backend.mla_index_context(MagicMock())

    assert remapped_context.block_table is expanded_block_table
    actual_cache, actual_scale, materialized_table = remapped_context.materialize_index_cache()
    assert actual_cache is materialized_cache
    assert actual_scale is materialized_scale
    assert materialized_table is expanded_block_table
    materialize_index_cache.assert_called_once_with()
    for field in fields(MlaIndexContext):
        if field.name in {"block_table", "materialize_index_cache"}:
            continue
        assert getattr(remapped_context, field.name) is getattr(base_context, field.name)


def test_mla_index_materialization_keeps_cache_scale_and_table_together() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    index_cache = torch.empty(1)
    index_cache_scale = torch.empty(1)
    materialized_cache = torch.empty(2)
    materialized_scale = torch.empty(2)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = MagicMock()
    cp_context = SimpleNamespace(segment_seq_indices=torch.tensor([0, 0]))
    materialize_cp_cache = MagicMock(
        side_effect=[
            (materialized_cache, block_table),
            (materialized_scale, block_table),
        ]
    )
    backend._materialize_cp_cache = materialize_cp_cache

    actual_cache, actual_scale, actual_table = backend._materialize_mla_index_cache(
        index_cache,
        index_cache_scale,
        metadata,
        cp_context,
    )

    assert actual_cache is materialized_cache
    assert actual_scale is materialized_scale
    torch.testing.assert_close(actual_table, torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))
    assert [call.args[0] for call in materialize_cp_cache.call_args_list] == [
        index_cache,
        index_cache_scale,
    ]


def test_mla_index_materialization_without_scale_keeps_legacy_path() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    index_cache = torch.empty(1)
    materialized_cache = torch.empty(2)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = MagicMock()
    cp_context = SimpleNamespace(segment_seq_indices=torch.tensor([0, 0]))
    materialize_cp_cache = MagicMock(return_value=(materialized_cache, block_table))
    backend._materialize_cp_cache = materialize_cp_cache

    actual_cache, actual_scale, actual_table = backend._materialize_mla_index_cache(
        index_cache,
        None,
        metadata,
        cp_context,
    )

    assert actual_cache is materialized_cache
    assert actual_scale is None
    torch.testing.assert_close(actual_table, torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))
    materialize_cp_cache.assert_called_once_with(index_cache, metadata, cp_context)


def test_glm_quant_indexer_without_cp_uses_materialized_scale() -> None:
    indexer = glm5_2.Glm52Indexer.__new__(glm5_2.Glm52Indexer)
    nn.Module.__init__(indexer)
    indexer.n_head = 1
    indexer.head_dim = 2
    indexer.rope_dim = 0
    indexer.topk = 2
    indexer.indexer_rope_interleave = False
    indexer.wq_b = MagicMock(return_value=torch.ones(2, 2))
    indexer.wk = MagicMock(return_value=torch.ones(2, 2))
    indexer.k_norm = nn.Identity()
    indexer.weights_proj = MagicMock(return_value=torch.ones(2, 1))
    indexer.hadamard = torch.eye(2)

    persistent_cache = torch.empty(2, 2, 1, dtype=torch.int8)
    persistent_scale = torch.empty(2, 2, 1, dtype=torch.float16)
    materialized_cache = torch.empty(4, 2, 1, dtype=torch.int8)
    materialized_scale = torch.empty(4, 2, 1, dtype=torch.float16)
    materialized_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    topk = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    update_index_cache = MagicMock()
    materialize_index_cache = MagicMock(return_value=(materialized_cache, materialized_scale, materialized_table))
    context = MlaIndexContext(
        index_cache=persistent_cache,
        slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        actual_seq_q=torch.tensor([2], dtype=torch.int32),
        actual_seq_kv=torch.tensor([4], dtype=torch.int32),
        index_cache_scale=persistent_scale,
        get_quant_indexer_metadata=MagicMock(return_value=torch.empty(0)),
        update_index_cache=update_index_cache,
        materialize_index_cache=materialize_index_cache,
        cp_context=None,
    )

    def dynamic_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantized = torch.ones_like(value, dtype=torch.int8)
        scale = torch.ones(value.shape[:-1], dtype=torch.float32)
        return quantized, scale

    with (
        patch.object(glm5_2, "_apply_half_rope", side_effect=lambda _cache, value, _positions: value),
        patch.object(
            glm5_2.kernels,
            "dynamic_quant",
            side_effect=dynamic_quant,
            create=True,
        ),
        patch.object(
            indexer,
            "_pad_q_heads_to_kernel_gsize",
            side_effect=lambda q, q_scale, weights, _required_heads: (q, q_scale, weights),
        ),
        patch.object(
            glm5_2.kernels,
            "quant_lightning_indexer",
            return_value=topk,
            create=True,
        ) as quant_lightning_indexer,
    ):
        output = indexer.select_qli(
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.tensor([0, 1]),
            context,
            torch.empty(0),
        )

    assert output is topk
    materialize_index_cache.assert_called_once_with()
    assert quant_lightning_indexer.call_args.args[1] is materialized_cache
    assert quant_lightning_indexer.call_args.args[4] is materialized_scale
    assert quant_lightning_indexer.call_args.args[8] is materialized_table


def test_lse_as_token_head_squeezes_graph_leading_one() -> None:
    num_tokens = 8
    num_heads = 16
    lse = torch.randn(1, num_tokens, num_heads, dtype=torch.float32)
    out = _lse_as_token_head(lse, num_tokens, num_heads)
    assert tuple(out.shape) == (num_tokens, num_heads)
    torch.testing.assert_close(out, lse[0])
