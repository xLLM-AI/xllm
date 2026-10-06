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

import os
from types import SimpleNamespace

import pytest
import torch

from xllm.python.models import glm5_next
from xllm.python.models.glm5_next_kpool import compress_completed_pools


def _indexer(device: torch.device) -> glm5_next.Glm5NextIndexer:
    config = glm5_next.Glm5NextConfig(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=128,
        index_kpool=4,
        index_topk=8,
        index_kpool_compress=True,
    )
    return glm5_next.Glm5NextIndexer(config, 1, torch.bfloat16, device).to(device=device, dtype=torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 4, 17])
@torch.inference_mode()
def test_indexer_merged_key_weights_projection_preserves_parameters(num_tokens: int) -> None:
    torch.manual_seed(42)
    indexer = _indexer(torch.device("cpu"))
    hidden = torch.randn(1, num_tokens, 16, dtype=torch.bfloat16)
    expected = indexer.wk(hidden), indexer.weights_proj(hidden)
    indexer.process_weights_after_loading()
    torch.testing.assert_close(indexer._project_key_weights(hidden), expected, rtol=0, atol=0)
    for weight in (indexer.wk.weight, indexer.weights_proj.weight):
        assert weight.untyped_storage().data_ptr() == indexer._wk_weights_weight.untyped_storage().data_ptr()
    indexer.wk.weight.add_(0.01)
    expected = indexer.wk(hidden), indexer.weights_proj(hidden)
    indexer.process_weights_after_loading()
    torch.testing.assert_close(indexer._project_key_weights(hidden), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_indexer_merged_key_weights_reuse_storage_across_reloads() -> None:
    # A captured decode ACL graph records _wk_weights_weight's storage address,
    # so a weight hot-reload must keep the packed buffer at a stable address
    # instead of reallocating via torch.cat; otherwise replay reads stale
    # indexer weights and produces wrong top-k. Reproduces the reload path:
    # copy_in writes fresh weights into the existing .data storage, then
    # process_weights_after_loading re-packs.
    torch.manual_seed(42)
    indexer = _indexer(torch.device("cpu"))
    indexer.process_weights_after_loading()
    captured = indexer._wk_weights_weight
    captured_ptr = captured.data_ptr()
    generator = torch.Generator().manual_seed(7)
    for _ in range(2):
        new_wk = torch.randn(indexer.wk.weight.shape, generator=generator).to(torch.bfloat16)
        new_weights = torch.randn(indexer.weights_proj.weight.shape, generator=generator).to(torch.bfloat16)
        indexer.wk.weight.data.copy_(new_wk)
        indexer.weights_proj.weight.data.copy_(new_weights)
        indexer.process_weights_after_loading()
        assert indexer._wk_weights_weight.data_ptr() == captured_ptr
        torch.testing.assert_close(captured[: indexer.head_dim], new_wk, rtol=0, atol=0)
        torch.testing.assert_close(captured[indexer.head_dim :], new_weights, rtol=0, atol=0)


@pytest.mark.parametrize("query_lengths", [[1, 1, 1], [2, 3, 1]])
@torch.inference_mode()
def test_indexer_paged_selection_reuses_merged_weights_for_varlen_queries(
    query_lengths: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(42)
    indexer = _indexer(torch.device("cpu"))
    indexer.index_kpool_compress_gate.normal_(std=0.1)
    hidden = torch.randn(1, sum(query_lengths), 16, dtype=torch.bfloat16)
    query = torch.randn(sum(query_lengths), 16, dtype=torch.bfloat16)
    history = torch.randn(3, 8, 257, dtype=torch.bfloat16)
    history[..., -1] = 1
    max_length = max(query_lengths)
    padded_hidden = torch.zeros(3, max_length, 16, dtype=torch.bfloat16)
    padded_query = torch.zeros_like(padded_hidden)
    valid = torch.zeros(3, max_length, dtype=torch.bool)
    padded_positions = torch.zeros(3, max_length, dtype=torch.long)
    offset = 0
    for sequence, length in enumerate(query_lengths):
        padded_hidden[sequence, :length] = hidden[0, offset : offset + length]
        padded_query[sequence, :length] = query[offset : offset + length]
        valid[sequence, :length] = True
        padded_positions[sequence, :length] = torch.arange(offset, offset + length)
        offset += length
    expected = indexer.select_topk(
        padded_query, padded_hidden, valid, 8, 8, packed_states=history, query_positions=padded_positions
    )[valid]
    context = SimpleNamespace(
        index_cache=None, slot_mapping=None, block_table=None, actual_seq_kv=torch.tensor([8, 8, 8])
    )
    backend = SimpleNamespace(gather_index_history=lambda *_args: history)
    monkeypatch.setattr(glm5_next, "_current_q_seq_lens", lambda *_args: query_lengths)
    indexer.process_weights_after_loading()

    def _unexpected_projection(*_args: object) -> None:
        pytest.fail("merged indexer must not dispatch independent key or weight projections")

    monkeypatch.setattr(indexer.wk, "forward", _unexpected_projection)
    monkeypatch.setattr(indexer.weights_proj, "forward", _unexpected_projection)
    actual = indexer.select_qli(
        hidden,
        query,
        torch.arange(sum(query_lengths)),
        torch.ones(hidden.shape[:2], dtype=torch.bool),
        context,
        None,
        backend,
    )
    torch.testing.assert_close(actual[:, 0], expected.to(torch.int32), rtol=0, atol=0)


@torch.inference_mode()
def test_indexer_mixed_projection_dtypes_keep_independent_calls() -> None:
    indexer = _indexer(torch.device("cpu"))
    hidden = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    indexer.process_weights_after_loading()
    indexer.weights_proj.float()
    expected = indexer.wk(hidden), indexer.weights_proj(hidden.float())
    indexer.process_weights_after_loading()
    assert indexer._wk_weights_weight is None
    torch.testing.assert_close(indexer._project_key_weights(hidden), expected, rtol=0, atol=0)


@pytest.mark.parametrize("positions", [[0], [3], [3, 6, 7, 15], [3, 0, 0, 0], [7, 7, 7, 7], [-1, 3, -1, 7]])
@pytest.mark.parametrize("aliased_tables", [False, True])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("table_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_batched_pool_compression_matches_ordered_sequence_writes(
    positions: list[int], aliased_tables: bool, position_dtype: torch.dtype, table_dtype: torch.dtype
) -> None:
    torch.manual_seed(42)
    sequence_count = len(positions)
    index_cache = torch.randn(2 * sequence_count, 8, 1, 257, dtype=torch.bfloat16)
    pool_cache = torch.randn(2 * sequence_count, 2, 1, 128, dtype=torch.bfloat16)
    tables = torch.arange(2 * sequence_count, dtype=table_dtype).reshape(sequence_count, 2)
    if aliased_tables:
        tables = tables[:1].expand(sequence_count, -1)
    position_tensor = torch.tensor(positions, dtype=position_dtype)
    ape = torch.randn(4, 128, dtype=torch.bfloat16)
    expected = pool_cache.clone()
    for sequence in range(sequence_count):
        compress_completed_pools(
            index_cache,
            expected,
            tables[sequence : sequence + 1],
            position_tensor[sequence : sequence + 1],
            ape,
            128,
            4,
        )
    compress_completed_pools(index_cache, pool_cache, tables, position_tensor, ape, 128, 4, batched=True)
    torch.testing.assert_close(pool_cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize("positions", [[0, 1, 2], list(range(8)), list(range(3, 12))])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("table_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_prefill_pool_compression_matches_ordered_token_writes(
    positions: list[int], position_dtype: torch.dtype, table_dtype: torch.dtype
) -> None:
    torch.manual_seed(42)
    index_cache = torch.randn(2, 8, 1, 257, dtype=torch.bfloat16)
    pool_cache = torch.randn(2, 2, 1, 128, dtype=torch.bfloat16)
    tables = torch.tensor([[1, 0]], dtype=table_dtype)
    position_tensor = torch.tensor(positions, dtype=position_dtype)
    ape = torch.randn(4, 128, dtype=torch.bfloat16)
    expected = pool_cache.clone()
    for position in position_tensor.split(1):
        compress_completed_pools(index_cache, expected, tables, position, ape, 128, 4)
    compress_completed_pools(index_cache, pool_cache, tables, position_tensor, ape, 128, 4)
    torch.testing.assert_close(pool_cache, expected, rtol=0, atol=0)


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("table_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_batched_pool_compression_graph_updates_positions_and_blocks(table_dtype: torch.dtype) -> None:
    runtime = pytest.importorskip("torch_npu")
    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    torch.manual_seed(42)
    index_cache = torch.randn(8, 8, 1, 257, device=device, dtype=torch.bfloat16)
    initial_pool = torch.randn(8, 2, 1, 128, device=device, dtype=torch.bfloat16)
    pool_cache = initial_pool.clone()
    tables = torch.arange(8, device=device, dtype=table_dtype).reshape(4, 2)
    positions = torch.tensor([3, 6, 7, 15], device=device, dtype=torch.int32)
    ape = torch.randn(4, 128, device=device, dtype=torch.bfloat16)

    def _compress() -> None:
        compress_completed_pools(index_cache, pool_cache, tables, positions, ape, 128, 4, batched=True)

    for _ in range(3):
        _compress()
    torch.npu.synchronize()
    graph = runtime.npu.NPUGraph()
    with runtime.npu.graph(graph):
        _compress()
    for step in range(4):
        positions.copy_(torch.tensor([3 + step, 7 + step, 11 + step, step], device=device))
        tables.copy_(torch.arange(8, device=device).roll(step).reshape(4, 2))
        index_cache.copy_(torch.randn_like(index_cache))
        pool_cache.copy_(initial_pool)
        graph.replay()
        torch.npu.synchronize()
        expected = initial_pool.clone()
        for sequence in range(4):
            compress_completed_pools(
                index_cache, expected, tables[sequence : sequence + 1], positions[sequence : sequence + 1], ape, 128, 4
            )
        torch.testing.assert_close(pool_cache, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_pool_selection_length_input_matches_dense_fallback() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")
    indexer = _indexer(device)
    lengths = torch.tensor([3, 8, 17], dtype=torch.int32)
    key_valid = torch.arange(32)[None] < lengths[:, None]
    cache = torch.randn(8, 4, 1, 128, dtype=torch.bfloat16)
    tables = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    hidden = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    mask = torch.ones(3, 1, dtype=torch.bool)
    expected = indexer.select_topk(
        hidden, hidden, mask, 32, 32, key_valid=key_valid, pool_cache=cache, pool_block_table=tables
    )
    actual = indexer.select_topk(
        hidden, hidden, mask, 32, 32, kv_seq_lens=lengths, pool_cache=cache, pool_block_table=tables
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("always_select_tail", [False, True])
@torch.inference_mode()
def test_fused_pool_selection_does_not_materialize_history_mask(
    monkeypatch: pytest.MonkeyPatch, always_select_tail: bool
) -> None:
    pytest.importorskip("torch_npu")
    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    indexer = _indexer(device)
    indexer.index_kpool_always_select_tail = always_select_tail
    lengths = torch.tensor([3, 8, 32768], device=device, dtype=torch.int32)
    hidden = torch.randn(3, 1, 16, device=device, dtype=torch.bfloat16)
    mask = torch.ones(3, 1, device=device, dtype=torch.bool)
    cache = torch.zeros(8, 4, 1, 128, device=device, dtype=torch.bfloat16)
    tables = torch.ones(3, 2048, device=device, dtype=torch.int32)
    output = torch.zeros(3, 1, 11, device=device, dtype=torch.int32)
    calls = []

    def _pool_indexer(
        query: torch.Tensor,
        keys: torch.Tensor,
        weights: torch.Tensor,
        tail: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        calls.append((tail, kwargs["actual_seq_k"]))
        return output, None

    def _unexpected_arange(*_args: object, **_kwargs: object) -> None:
        pytest.fail("fused paged selection must not construct a length-sized mask")

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _pool_indexer, raising=False)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    monkeypatch.setattr(torch, "arange", _unexpected_arange)
    actual = indexer.select_topk(
        hidden, hidden, mask, 32768, 32768, kv_seq_lens=lengths, pool_cache=cache, pool_block_table=tables
    )
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0].cpu(), torch.tensor([3, 0, 0], dtype=torch.int32))
    torch.testing.assert_close(calls[0][1].cpu(), torch.tensor([0, 2, 8192], dtype=torch.int32))
    output_width = 11 if always_select_tail else 8
    torch.testing.assert_close(actual, output[..., :output_width].long())
