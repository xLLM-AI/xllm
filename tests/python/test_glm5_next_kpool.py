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
from unittest.mock import patch

import pytest
import torch

from xllm.python.models import glm5_next
from xllm.python.models.glm5_next_kpool import update_compressed_kpool


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


@pytest.mark.parametrize(
    ("query_lens", "rate", "expected"),
    [
        ([], 4, None),
        ([1, 1], 4, 1),
        ([4, 4], 4, 4),
        ([8, 8], 4, 8),
        ([9, 9], 4, 9),
        ([4, 8], 4, None),
        ([0], 4, None),
        ([1], 0, None),
    ],
)
def test_compact_kpool_query_width_spans_multiple_pools(query_lens: list[int], rate: int, expected: int | None) -> None:
    assert glm5_next._compact_kpool_triton_query_len(query_lens, rate) == expected


def test_graph_kpool_prefix_drops_short_bucket_tail() -> None:
    hidden = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    flat = torch.arange(8, dtype=torch.float32)
    ctx = SimpleNamespace(kpool_query_lens=(6,))

    assert glm5_next._prefix_token_tensor(hidden, 8, 6).shape == (1, 6, 2)
    assert glm5_next._prefix_token_tensor(flat, 8, 6).tolist() == list(range(6))
    batched = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    with pytest.raises(RuntimeError, match="cannot keep the first"):
        glm5_next._prefix_token_tensor(batched, 8, 6)
    with pytest.raises(RuntimeError, match="cover the current token rows"):
        glm5_next._kpool_update_query_lens(ctx, 8)
    with patch("xllm.python.models.glm5_next.in_acl_graph", return_value=True):
        assert glm5_next._kpool_update_query_lens(ctx, 8) == [6]


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
        padded_query,
        padded_hidden,
        valid,
        8,
        8,
        packed_states=history,
        query_positions=padded_positions,
    )[valid]
    context = SimpleNamespace(
        index_cache=None,
        slot_mapping=None,
        block_table=torch.zeros(3, 1, dtype=torch.int32),
        actual_seq_kv=torch.tensor([8, 8, 8]),
        kpool_tail=None,
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

    prefill_indexer = SimpleNamespace(
        wq_b=torch.nn.Linear(2, 2, bias=False),
        weights_proj=torch.nn.Linear(2, 1, bias=False),
        n_heads=1,
        head_dim=2,
        index_kpool=4,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        topk=4,
        softmax_scale=1.0,
    )
    pool_data = (
        torch.zeros(2, 1, 2),
        torch.tensor([[[0, 1, 2, 3]], [[0, 1, 2, 3]]], dtype=torch.int64),
        torch.ones(2, 1, dtype=torch.bool),
    )
    selected = glm5_next.Glm5NextIndexer.select_topk(
        prefill_indexer,
        q_resid=torch.zeros(2, 4, 2),
        hidden_states=torch.zeros(2, 4, 2),
        attention_mask=torch.tensor([[True, True, True, True], [True, True, False, False]]),
        kv_len=4,
        current_length=4,
        pool_data=pool_data,
        key_valid=torch.ones(2, 4, dtype=torch.bool),
        query_positions=torch.tensor([[0, 1, 2, 3], [0, 1, -1, -1]], dtype=torch.int64),
        append_unscored_tail=True,
    )
    torch.testing.assert_close(
        selected,
        torch.tensor(
            [
                [
                    [0, -1, -1, -1, -1, -1, -1],
                    [0, 1, -1, -1, -1, -1, -1],
                    [0, 1, 2, -1, -1, -1, -1],
                    [0, 1, 2, 3, -1, -1, -1],
                ],
                [
                    [0, -1, -1, -1, -1, -1, -1],
                    [0, 1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1],
                ],
            ],
            dtype=torch.int64,
        ),
    )


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize(
    ("always_select_tail", "query_len"),
    [(False, 1), (True, 4)],
)
@torch.inference_mode()
def test_fused_pool_selection_does_not_materialize_history_mask(
    monkeypatch: pytest.MonkeyPatch,
    always_select_tail: bool,
    query_len: int,
) -> None:
    pytest.importorskip("torch_npu")
    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    indexer = _indexer(device)
    indexer.index_kpool_always_select_tail = always_select_tail
    lengths = torch.tensor([9, 12, 32768], device=device, dtype=torch.int32)
    hidden = torch.randn(3, query_len, 16, device=device, dtype=torch.bfloat16)
    mask = torch.ones(3, query_len, device=device, dtype=torch.bool)
    positions = lengths[:, None] - query_len + torch.arange(query_len, device=device)
    cache = torch.zeros(8, 4, 1, 128, device=device, dtype=torch.bfloat16)
    tables = torch.ones(3, 2048, device=device, dtype=torch.int32)
    output = torch.zeros(3 * query_len, 1, 11, device=device, dtype=torch.int32)
    calls = []

    def _pool_indexer(
        query: torch.Tensor,
        keys: torch.Tensor,
        weights: torch.Tensor,
        tail: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        calls.append(
            (
                query.shape,
                tail,
                kwargs["actual_seq_k"],
                kwargs["block_table"].shape,
                kwargs["mask_mode"],
            )
        )
        return output, None

    def _unexpected_arange(*_args: object, **_kwargs: object) -> None:
        pytest.fail("fused paged selection must not construct a length-sized mask")

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _pool_indexer, raising=False)
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    monkeypatch.setattr(torch, "arange", _unexpected_arange)
    actual = indexer.select_topk(
        hidden,
        hidden,
        mask,
        32768,
        32768,
        kv_seq_lens=lengths,
        pool_cache=cache,
        pool_block_table=tables,
        query_positions=positions,
    )
    assert len(calls) == 1
    row_lengths = lengths if query_len == 1 else positions.reshape(-1) + 1
    assert calls[0][0] == torch.Size([3 * query_len, 1, indexer.n_heads, indexer.head_dim])
    expected_tail = torch.remainder(row_lengths, 4).to(torch.int32).cpu()
    expected_pools = torch.div(row_lengths, 4, rounding_mode="floor").to(torch.int32).cpu()
    torch.testing.assert_close(calls[0][1].cpu(), expected_tail)
    torch.testing.assert_close(calls[0][2].cpu(), expected_pools)
    assert calls[0][3] == torch.Size([3 * query_len, 2048])
    assert calls[0][4] == 3
    output_width = 11 if always_select_tail else 8
    torch.testing.assert_close(actual, output.reshape(3, query_len, -1)[..., :output_width].long())


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@torch.inference_mode()
def test_fused_pool_selection_mtp_preserves_causal_tail_across_pool_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch_npu")
    from xllm.python.kernels_npu.sparse_attention import pool_key_indexer

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", pool_key_indexer, raising=False)
    indexer = _indexer(device)
    indexer.index_kpool_always_select_tail = True

    query = torch.ones(2, 4, indexer.n_heads, indexer.head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.ones(2, 4, indexer.n_heads, device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([14, 17], device=device, dtype=torch.int32)
    positions = lengths[:, None] - 4 + torch.arange(4, device=device)
    mask = torch.ones(2, 4, device=device, dtype=torch.bool)
    cache = torch.zeros(2, 16, 1, indexer.head_dim, device=device, dtype=torch.bfloat16)
    pool_values = torch.arange(1, 5, device=device, dtype=torch.bfloat16).view(1, 4, 1, 1)
    cache[:, :4].copy_(pool_values.expand(2, -1, 1, indexer.head_dim))
    tables = torch.tensor([[0], [1]], device=device, dtype=torch.int32)

    actual = indexer._select_topk_fused_pa(
        query,
        weights,
        lengths,
        mask,
        cache,
        tables,
        positions,
    )
    assert actual is not None

    expected_positions = (
        (range(11), range(4, 12), range(4, 13), range(4, 14)),
        (range(4, 14), range(4, 15), range(8, 16), range(8, 17)),
    )
    for batch_idx, batch_rows in enumerate(expected_positions):
        for query_idx, expected_row in enumerate(batch_rows):
            valid = actual[batch_idx, query_idx]
            valid = valid[valid >= 0].sort().values.cpu()
            torch.testing.assert_close(valid, torch.tensor(list(expected_row), dtype=torch.int64))


def test_compressed_kpool_completes_pool_across_forward_boundary() -> None:
    compressed_cache = torch.zeros(1, 2, 1, 2, dtype=torch.bfloat16)
    tail_cache = torch.zeros(3, 2, 4, 2, dtype=torch.bfloat16)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    tail_ids = torch.tensor([1], dtype=torch.int32)
    ape = torch.zeros(4, 2, dtype=torch.float32)

    update_compressed_kpool(
        raw_k=torch.tensor([[1, 10], [3, 30], [5, 50]], dtype=torch.bfloat16),
        gate_scores=torch.zeros(3, 2, dtype=torch.bfloat16),
        valid_rows=torch.ones(3, dtype=torch.bool),
        positions=torch.tensor([0, 1, 2]),
        compressed_cache=compressed_cache,
        tail_cache=tail_cache,
        tail_read_ids=tail_ids,
        tail_write_ids=tail_ids,
        block_table=block_table,
        query_lens=[3],
        ape=ape,
        rate=4,
    )
    update_compressed_kpool(
        raw_k=torch.tensor([[7, 70]], dtype=torch.bfloat16),
        gate_scores=torch.zeros(1, 2, dtype=torch.bfloat16),
        valid_rows=torch.ones(1, dtype=torch.bool),
        positions=torch.tensor([3]),
        compressed_cache=compressed_cache,
        tail_cache=tail_cache,
        tail_read_ids=tail_ids,
        tail_write_ids=torch.tensor([2], dtype=torch.int32),
        block_table=block_table,
        query_lens=[1],
        ape=ape,
        rate=4,
        graph_mode=True,
    )

    torch.testing.assert_close(
        compressed_cache[0, 0, 0],
        torch.tensor([4, 40], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        tail_cache[2, 0],
        torch.tensor([[1, 10], [3, 30], [5, 50], [7, 70]], dtype=torch.bfloat16),
    )

    # Position 4 wraps onto the restored position-0 slot. An invalid row must
    # invalidate that slot so positions 5-7 cannot complete a stale pool.
    restored_tail_ids = torch.tensor([2], dtype=torch.int32)
    compressed_cache[0, 1, 0].fill_(123)
    update_compressed_kpool(
        torch.tensor([[9, 90]], dtype=torch.bfloat16),
        torch.zeros(1, 2, dtype=torch.bfloat16),
        torch.zeros(1, dtype=torch.bool),
        torch.tensor([4]),
        compressed_cache,
        tail_cache,
        restored_tail_ids,
        restored_tail_ids,
        block_table,
        [1],
        ape,
        4,
    )
    update_compressed_kpool(
        torch.tensor([[11, 110], [13, 130], [15, 150]], dtype=torch.bfloat16),
        torch.zeros(3, 2, dtype=torch.bfloat16),
        torch.ones(3, dtype=torch.bool),
        torch.tensor([5, 6, 7]),
        compressed_cache,
        tail_cache,
        restored_tail_ids,
        restored_tail_ids,
        block_table,
        [3],
        ape,
        4,
    )
    torch.testing.assert_close(
        compressed_cache[0, 1, 0],
        torch.full((2,), 123, dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )

    placeholder_tail = torch.zeros(2, 2, 4, 2, dtype=torch.bfloat16)
    placeholder_tail_ids = torch.tensor([1], dtype=torch.int32)

    update_compressed_kpool(
        raw_k=torch.tensor([[1, 10], [3, 30]], dtype=torch.bfloat16),
        gate_scores=torch.tensor([[2, 20], [4, 40]], dtype=torch.bfloat16),
        valid_rows=torch.tensor([False, True]),
        positions=torch.tensor([0, 1]),
        compressed_cache=torch.zeros(1, 2, 1, 2, dtype=torch.bfloat16),
        tail_cache=placeholder_tail,
        tail_read_ids=placeholder_tail_ids,
        tail_write_ids=placeholder_tail_ids,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        query_lens=[2],
        ape=torch.zeros(4, 2, dtype=torch.bfloat16),
        rate=4,
    )

    torch.testing.assert_close(
        placeholder_tail[1, :, 0],
        torch.tensor(
            [[0, 0], [float("-inf"), float("-inf")]],
            dtype=torch.bfloat16,
        ),
    )
    torch.testing.assert_close(
        placeholder_tail[1, :, 1],
        torch.tensor([[3, 30], [4, 40]], dtype=torch.bfloat16),
    )


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("query_len", [1, 4, 7, 8, 9])
@pytest.mark.parametrize("position_offset", [0, 1, 2, 3])
@torch.inference_mode()
def test_compact_kpool_triton_matches_torch_reference(query_len: int, position_offset: int) -> None:
    runtime = pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    from xllm.python.kernels_npu.triton.kpool_compress import (
        update_compact_kpool,
    )

    torch.manual_seed(42)
    rate = 4
    head_dim = 128
    first_positions = (position_offset, 28 + position_offset, 5)
    num_requests = len(first_positions)
    positions = torch.cat(
        [torch.arange(start, start + query_len, dtype=torch.int64) for start in first_positions],
    )
    raw_k = torch.randn(num_requests * query_len, head_dim, dtype=torch.bfloat16)
    gate_scores = torch.randn_like(raw_k)
    valid_rows = torch.ones(num_requests * query_len, dtype=torch.bool)
    valid_rows[0] = False
    valid_rows[-query_len:] = False
    ape = torch.randn(rate, head_dim, dtype=torch.bfloat16)
    tail_ids = torch.tensor([1, 2, 0], dtype=torch.int32)
    block_table = torch.tensor([[2, 0], [3, 1], [0, 0]], dtype=torch.int64)
    compressed_cache = torch.zeros(4, 4, 1, head_dim, dtype=torch.bfloat16)
    tail_cache = torch.randn(num_requests, 2, rate + query_len - 1, head_dim, dtype=torch.bfloat16)
    tail_cache[0].zero_()

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    device_raw_k = raw_k.to(device)
    device_gate_scores = gate_scores.to(device)
    device_valid_rows = valid_rows.to(device)
    device_positions = positions.to(device)
    device_tail_ids = tail_ids.to(device)
    device_block_table = block_table.to(device)
    device_ape = ape.to(device)
    initial_cache = compressed_cache.to(device)
    initial_tail = tail_cache.to(device)
    expected_cache = initial_cache.clone()
    expected_tail = initial_tail.clone()
    update_compressed_kpool(
        device_raw_k,
        device_gate_scores,
        device_valid_rows,
        device_positions,
        expected_cache,
        expected_tail,
        device_tail_ids,
        device_tail_ids,
        device_block_table,
        [query_len] * num_requests,
        device_ape,
        rate,
        graph_mode=True,
    )
    actual_cache = initial_cache.clone()
    actual_tail = initial_tail.clone()

    def _update() -> None:
        update_compact_kpool(
            device_raw_k,
            device_gate_scores,
            device_valid_rows,
            device_positions,
            actual_cache,
            actual_tail,
            device_tail_ids,
            device_block_table,
            query_len,
            device_ape,
            rate,
        )

    for _ in range(3):
        actual_cache.copy_(initial_cache)
        actual_tail.copy_(initial_tail)
        _update()
    torch.npu.synchronize()
    graph = runtime.npu.NPUGraph()
    actual_cache.copy_(initial_cache)
    actual_tail.copy_(initial_tail)
    with runtime.npu.graph(graph):
        _update()
    actual_cache.copy_(initial_cache)
    actual_tail.copy_(initial_tail)
    graph.replay()
    torch.npu.synchronize()

    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(actual_tail, expected_tail, rtol=0, atol=0)

    # Replay with a partially accepted prefix, then a full-window advance.
    # Both steps must read the live circular tail and write every new pool.
    for accepted_prefix in (1, query_len):
        device_positions.add_(accepted_prefix)
        device_raw_k.mul_(0.75).add_(0.125)
        device_gate_scores.mul_(0.5).sub_(0.25)
        expected_cache = actual_cache.clone()
        expected_tail = actual_tail.clone()
        update_compressed_kpool(
            device_raw_k,
            device_gate_scores,
            device_valid_rows,
            device_positions,
            expected_cache,
            expected_tail,
            device_tail_ids,
            device_tail_ids,
            device_block_table,
            [query_len] * num_requests,
            device_ape,
            rate,
            graph_mode=True,
        )
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(actual_tail, expected_tail, rtol=0, atol=0)
