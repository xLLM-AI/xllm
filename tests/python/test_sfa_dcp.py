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
    AscendSFADCPImpl,
    AscendSFADCPMetadataBuilder,
    _lse_as_token_head,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    copy_into_execution_buffer,
    forward_context,
)
from xllm.python.models import deepseek_v32, glm5_2


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


@pytest.mark.parametrize("with_scale", [False, True])
def test_mla_index_materialization_keeps_cache_scale_and_table_together(with_scale: bool) -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    index_cache = torch.empty(1)
    index_cache_scale = torch.empty(1) if with_scale else None
    materialized_cache = torch.empty(2)
    materialized_scale = torch.empty(2) if with_scale else None
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = MagicMock()
    cp_context = SimpleNamespace(segment_seq_indices=torch.tensor([0, 0]))
    materialize_cp_cache = MagicMock(side_effect=[(materialized_cache, block_table), (materialized_scale, block_table)])
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
    expected_calls = [((index_cache, metadata, cp_context), {})]
    if with_scale:
        expected_calls.append(((index_cache_scale, metadata, cp_context), {}))
    assert materialize_cp_cache.call_args_list == expected_calls


def test_cp_block_table_segmentation_reuses_forward_cache() -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    cp_context = SimpleNamespace(segment_seq_indices=torch.tensor([1, 0]))

    first = backend._segment_block_table(block_table, cp_context)
    second = backend._segment_block_table(block_table, cp_context)

    assert first is second
    torch.testing.assert_close(
        first,
        torch.tensor([[3, 4], [1, 2]], dtype=torch.int32),
    )


def test_glm_quant_indexer_without_cp_uses_materialized_scale() -> None:
    indexer = glm5_2.Glm52Indexer.__new__(glm5_2.Glm52Indexer)
    nn.Module.__init__(indexer)
    indexer.n_head = 1
    indexer.head_dim = 2
    indexer.rope_dim = 0
    indexer.topk = 2
    indexer.indexer_rope_interleave = False
    indexer._q_stream = None
    indexer._weights_stream = None
    indexer._wk_weights_proj_ready = False
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
        patch.object(deepseek_v32, "_apply_half_rope_with_cos_sin", side_effect=lambda value, _cos, _sin: value),
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
        forward_context(_cpu_context(None)),
    ):
        output = indexer.select_qli(
            torch.ones(2, 3),
            torch.ones(2, 3),
            context,
            (torch.empty(2, 0), torch.empty(2, 0)),
            (torch.empty(2, 0), torch.empty(2, 0)),
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


class _IdentityDcpGroup:
    world_size = 1
    rank_in_group = 0
    device_group = None


def _dcp_impl() -> AscendSFADCPImpl:
    return AscendSFADCPImpl(
        _IdentityDcpGroup(),
        scale=0.1,
        index_topk=8,
        layout=KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0),
    )


def test_query_gather_nope_returns_none_q_pe() -> None:
    impl = _dcp_impl()
    ql_nope = torch.randn(2, 4, 8)
    gather_context = impl._start_dcp_query_gather(ql_nope, None)
    assert gather_context.split_sizes == (8,)
    gathered_ql, gathered_q_pe = impl._finish_dcp_gather(gather_context)
    torch.testing.assert_close(gathered_ql, ql_nope)
    assert gathered_q_pe is None


def test_query_gather_with_rope_keeps_fused_split() -> None:
    impl = _dcp_impl()
    ql_nope = torch.randn(2, 4, 8)
    q_pe = torch.randn(2, 4, 4)
    gather_context = impl._start_dcp_query_gather(ql_nope, q_pe)
    assert gather_context.split_sizes == (8, 4)
    gathered_ql, gathered_q_pe = impl._finish_dcp_gather(gather_context)
    torch.testing.assert_close(gathered_ql, ql_nope)
    assert gathered_q_pe is not None
    torch.testing.assert_close(gathered_q_pe, q_pe)


def test_decode_nope_query_gather_unpacks_none_before_sfa() -> None:
    impl = _dcp_impl()
    builder = _builder()
    metadata = builder.build(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[1, 2]], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        num_reqs=1,
        num_input_tokens=2,
        num_prefills=0,
    )
    ql_nope = torch.randn(2, 4, 8)
    impl._record_query_gather_context(ql_nope, None, metadata)
    sfa_out = torch.randn(2, 4, 8)
    softmax_max = torch.zeros(2, 4)
    softmax_sum = torch.ones(2, 4)
    merged = torch.randn(2, 2, 8)
    topk = torch.zeros(2, 1, 8, dtype=torch.int32)
    kv = torch.randn(4, 4, 1, 8)
    with (
        patch.object(impl, "_remap_sparse_indices", side_effect=lambda indices: indices),
        patch.object(
            impl,
            "_npu_sparse_flash_attention",
            return_value=(sfa_out, softmax_max, softmax_sum),
        ) as sparse_attn,
        patch.object(impl, "_merge_dcp_outputs", return_value=merged) as merge,
    ):
        out = impl._execute_sparse_flash_attention_process(
            ql_nope,
            None,
            (kv,),
            topk,
            metadata,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
        )
    assert out is merged
    assert sparse_attn.call_args.args[1] is None
    merge.assert_called_once()


def test_prefill_kv_gather_with_rope_still_fuses() -> None:
    impl = _dcp_impl()
    builder = _builder()
    metadata = builder.build(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_input_tokens=1,
        num_prefills=1,
    )
    nope = torch.randn(4, 4, 1, 16)
    rope = torch.randn(4, 4, 1, 4)
    impl._record_dcp_kv_gather_context((nope, rope), metadata)
    gather_context = metadata.dcp_context.gather_context
    assert gather_context is not None
    assert gather_context.split_sizes == (16, 4)
    gathered_kv, gathered_rope = impl._finish_dcp_gather(gather_context)
    assert gathered_rope is not None
    assert gathered_kv.shape[-1] == 16
    assert gathered_rope.shape[-1] == 4


def test_prefill_kv_gather_without_rope_cache() -> None:
    impl = _dcp_impl()
    builder = _builder()
    metadata = builder.build(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[1, 2], [3, 0]], dtype=torch.int32),
        torch.tensor([8, 4], dtype=torch.int32),
        num_reqs=2,
        num_input_tokens=2,
        num_prefills=2,
    )
    nope = torch.randn(8, 4, 1, 16)
    impl._record_dcp_kv_gather_context((nope,), metadata)
    gather_context = metadata.dcp_context.gather_context
    assert gather_context is not None
    assert gather_context.split_sizes == (16,)
    gathered_kv, gathered_rope = impl._finish_dcp_gather(gather_context)
    assert gathered_rope is None
    assert gathered_kv.shape[-1] == 16

    sfa_out = torch.randn(2, 4, 16)
    ql_nope = torch.randn(2, 4, 16)
    topk = torch.zeros(2, 1, 8, dtype=torch.int32)
    metadata.dcp_context.gather_context = gather_context
    with patch.object(impl, "_npu_sparse_flash_attention", return_value=sfa_out) as sparse_attn:
        out = impl._execute_sparse_flash_attention_process(
            ql_nope,
            None,
            (nope, None),
            topk,
            metadata,
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([8, 4], dtype=torch.int32),
        )
    assert out is sfa_out
    kv_arg = sparse_attn.call_args.args[2]
    assert kv_arg[1] is None


def test_prefill_kv_gather_skips_zero_width_rope_cache() -> None:
    impl = _dcp_impl()
    builder = _builder()
    metadata = builder.build(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_input_tokens=1,
        num_prefills=1,
    )
    nope = torch.randn(4, 4, 1, 16)
    rope = torch.empty(4, 4, 1, 0)
    impl._record_dcp_kv_gather_context((nope, rope), metadata)
    gather_context = metadata.dcp_context.gather_context
    assert gather_context is not None
    assert gather_context.split_sizes == (16,)
    _, gathered_rope = impl._finish_dcp_gather(gather_context)
    assert gathered_rope is None


def test_npu_sparse_flash_attention_passes_none_rope() -> None:
    impl = _dcp_impl()
    query = torch.randn(2, 4, 8)
    key = torch.randn(4, 4, 1, 8)
    topk = torch.zeros(2, 1, 8, dtype=torch.int32)
    block_table = torch.zeros(1, 2, dtype=torch.int32)
    actual_q = torch.tensor([2], dtype=torch.int32)
    actual_kv = torch.tensor([8], dtype=torch.int32)
    attn_out = torch.randn(2, 4, 8)
    softmax_max = torch.zeros(1)
    softmax_sum = torch.ones(1)
    fake_ops = MagicMock()
    fake_ops.sparse_flash_attention_lse.return_value = (attn_out, softmax_max, softmax_sum)
    with patch.object(torch.ops, "xllm_ops", fake_ops, create=True):
        out = impl._npu_sparse_flash_attention(
            query,
            None,
            (key, None),
            topk,
            actual_q,
            actual_kv,
            block_table,
            sparse_mode=0,
            return_lse=False,
        )
    assert out is attn_out
    kwargs = fake_ops.sparse_flash_attention_lse.call_args.kwargs
    assert kwargs["query_rope"] is None
    assert kwargs["key_rope"] is None
