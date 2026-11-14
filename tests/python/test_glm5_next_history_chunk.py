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

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

pytest.importorskip("torch_npu", reason="history chunk tests import the NPU attention backend")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.models import glm5_next

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


def _packed_indexer(device: torch.device) -> glm5_next.Glm5NextIndexer:
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


def _raw_indexer(device: torch.device) -> glm5_next.Glm5NextIndexer:
    config = glm5_next.Glm5NextConfig(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=128,
        index_kpool=1,
        index_topk=8,
        index_kpool_compress=False,
    )
    return glm5_next.Glm5NextIndexer(config, 1, torch.bfloat16, device).to(device=device, dtype=torch.bfloat16)


def _make_backend(num_blocks: int, page_size: int, width: int) -> NpuPagedAttentionBackend:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
        scale=0.1,
        sliding_window=0,
        is_mla=True,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    index = torch.zeros(num_blocks, page_size, 1, width, dtype=torch.bfloat16)
    ids = torch.arange(num_blocks * page_size, dtype=torch.float32).view(num_blocks, page_size)
    index[:, :, 0, -1] = ids
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(num_blocks, page_size, 1, 128),
                value=torch.empty(num_blocks, page_size, 1, 0),
                index=index,
            )
        ]
    )
    return backend


def _prepare_metadata(block_table: torch.Tensor, kv_seq_lens: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        prepared_attention_state=None,
        slot_mapping=torch.zeros(0, dtype=torch.int32),
        block_table=block_table,
        kv_seq_lens=kv_seq_lens,
        kv_seq_lens_host_values=kv_seq_lens.tolist(),
        kv_seq_lens_host=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        has_kv_shard=False,
        kv_split_size=1,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
    )


@torch.inference_mode()
def test_gather_row_range_matches_full_gather() -> None:
    torch.manual_seed(0)
    num_seqs, page_size, width = 5, 8, 257
    block_table = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=torch.int32)
    kv = torch.tensor([16, 8, 12, 16, 4], dtype=torch.int32)
    backend = _make_backend(12, page_size, width)
    backend.prepare(_prepare_metadata(block_table, kv), graph_mode=False)

    layer = SimpleNamespace(layer_id=0)
    full = backend.gather_index_history(layer)
    assert full.shape == (num_seqs, 16, width)
    for begin, end in [(0, 2), (2, 4), (4, 5)]:
        chunk = backend.gather_index_history(layer, begin, end)
        chunk_max = int(kv[begin:end].max())
        torch.testing.assert_close(chunk, full[begin:end, :chunk_max], rtol=0, atol=0)
    assert torch.equal(full[1, 8:, -1], torch.zeros(8))
    assert full[0, 15, -1].item() == (2 * page_size + 7)


@torch.inference_mode()
def test_graph_gather_row_range_keeps_full_capacity() -> None:
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    num_seqs, page_size, width = 4, 8, 128
    block_table = torch.tensor([[1, 3], [2, 4], [5, 7], [6, 8]], dtype=torch.int32)
    kv = torch.tensor([10, 16, 5, 0], dtype=torch.int32)
    backend = _make_backend(10, page_size, width)
    metadata = _prepare_metadata(block_table, kv)
    with forward_context(
        ForwardContext(
            attention_backend=backend,
            device=torch.device("cpu"),
            metadata=metadata,
            layer_caches=[],
            execution_state=AclGraphExecutionState(persistent_buffers={}),
        )
    ):
        backend.prepare(metadata, graph_mode=True, owned_metadata=True)
        layer = SimpleNamespace(layer_id=0)
        full = backend.gather_index_history(layer)
        assert full.shape == (num_seqs, 2 * page_size, width)
        for begin, end in [(0, 1), (1, 3), (3, 4)]:
            chunk = backend.gather_index_history(layer, begin, end)
            torch.testing.assert_close(chunk, full[begin:end], rtol=0, atol=0)
        assert torch.equal(full[2, 5:, -1], torch.zeros(2 * page_size - 5))
        assert torch.equal(full[3], torch.zeros_like(full[3]))


class _ChunkRecordingBackend:
    def __init__(self, backend: NpuPagedAttentionBackend) -> None:
        self._backend = backend
        self.gathered_shapes: list[tuple[int, ...]] = []

    def indexer_block_table(self) -> torch.Tensor:
        return self._backend.indexer_block_table()

    def gather_index_history(
        self,
        layer: Attention,
        seq_start: int | None = None,
        seq_stop: int | None = None,
    ) -> torch.Tensor:
        out = self._backend.gather_index_history(layer, seq_start, seq_stop)
        self.gathered_shapes.append(tuple(out.shape))
        return out


def _raw_context(backend: NpuPagedAttentionBackend, block_table: torch.Tensor, kv: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        index_cache=backend._kv_caches[0].index,
        slot_mapping=None,
        block_table=block_table,
        actual_seq_q=None,
        actual_seq_kv=kv,
        index_cache_scale=None,
        cp_context=None,
        actual_seq_kv_host=kv.tolist(),
        update_index_cache=lambda *_args: None,
        materialize_index_cache=lambda *_args: None,
        get_quant_indexer_metadata=lambda *_args: None,
    )


@torch.inference_mode()
def test_select_qli_chunks_history_and_matches_single_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    torch.manual_seed(3)
    query_len = 2

    def _run(indexer: glm5_next.Glm5NextIndexer, width: int) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
        page_size = 4
        num_seqs = 6
        kv = torch.tensor([16, 16, 16, 16, 16, 16], dtype=torch.int32)
        columns = 4
        block_table = torch.arange(1, num_seqs * columns + 1, dtype=torch.int32).view(num_seqs, columns)
        backend = _make_backend(num_seqs * columns + 1, page_size, width)
        backend.prepare(_prepare_metadata(block_table, kv), graph_mode=False)
        layer = SimpleNamespace(layer_id=0)
        recorder = _ChunkRecordingBackend(backend)

        hidden = torch.randn(1, num_seqs * query_len, 16, dtype=torch.bfloat16)
        query = torch.randn(num_seqs * query_len, 16, dtype=torch.bfloat16)
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
        positions = (kv[:, None] - query_len + torch.arange(query_len)).reshape(-1).to(torch.int64)
        q_cu = torch.arange(0, num_seqs * query_len + 1, query_len, dtype=torch.int32)
        ctx = _raw_context(backend, block_table[:1], kv)

        metadata = SimpleNamespace(q_cu_seq_lens=q_cu)
        with forward_context(
            ForwardContext(
                attention_backend=backend,
                device=torch.device("cpu"),
                metadata=metadata,
                layer_caches=[],
            )
        ):
            history_budget = 3 * 16 * width * backend._kv_caches[0].index.element_size()
            monkeypatch.setattr(glm5_next, "_INDEX_HISTORY_CHUNK_BYTES", history_budget)
            actual = indexer.select_qli(hidden, query, positions, mask, ctx, layer, recorder)
            chunk_rows = {shape[0] for shape in recorder.gathered_shapes}
            assert len(recorder.gathered_shapes) > 1, "test must exercise multi-chunk selection"
            assert all(rows < num_seqs for rows in chunk_rows), recorder.gathered_shapes
            assert all(
                2 * shape[0] * shape[1] * shape[2] * backend._kv_caches[0].index.element_size() <= history_budget
                for shape in recorder.gathered_shapes
            ), recorder.gathered_shapes

            recorder.gathered_shapes.clear()
            monkeypatch.setattr(glm5_next, "_INDEX_HISTORY_CHUNK_BYTES", 1 << 40)
            reference = indexer.select_qli(hidden, query, positions, mask, ctx, layer, recorder)
            assert len(recorder.gathered_shapes) == 1
            assert recorder.gathered_shapes[0][0] == num_seqs
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        return actual, list(recorder.gathered_shapes)

    packed, packed_shapes = _run(_packed_indexer(torch.device("cpu")), 257)
    assert all(shape[2] == 257 for shape in packed_shapes)
    raw, raw_shapes = _run(_raw_indexer(torch.device("cpu")), 128)
    assert all(shape[2] == 128 for shape in raw_shapes)
    assert packed.shape == raw.shape
    assert packed.dtype == raw.dtype


@torch.inference_mode()
def test_select_qli_retains_token_beyond_old_history_limit() -> None:
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    page_size, width, num_seqs = 4, 128, 2
    kv = torch.tensor([33024, 33024], dtype=torch.int32)
    pages_per_seq = 33024 // page_size
    num_blocks = pages_per_seq * num_seqs + 1
    table = torch.zeros(num_seqs, pages_per_seq, dtype=torch.int32)
    block = 1
    for seq in range(num_seqs):
        for column in range(pages_per_seq):
            table[seq, column] = block
            block += 1
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=128,
        scale=0.1,
        sliding_window=0,
        is_mla=True,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    index = torch.zeros(num_blocks, page_size, 1, width, dtype=torch.bfloat16)
    index[:, :, 0, -1] = 1  # valid channel
    index[:, :, 0, 0] = torch.arange(num_blocks * page_size, dtype=torch.float32).view(num_blocks, page_size)
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(num_blocks, page_size, 1, 128),
                value=torch.empty(num_blocks, page_size, 1, 0),
                index=index,
            )
        ]
    )
    backend.prepare(_prepare_metadata(table, kv), graph_mode=False)
    layer = SimpleNamespace(layer_id=0)

    indexer = _raw_indexer(torch.device("cpu"))
    indexer.wq_b.weight.zero_()
    indexer.wq_b.weight[0, 0] = 1.0
    indexer.weights_proj.weight.zero_()
    indexer.weights_proj.weight[0, 0] = 1.0
    hidden = torch.ones(1, num_seqs, 16, dtype=torch.bfloat16)
    query = torch.ones(num_seqs, 16, dtype=torch.bfloat16)
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
    positions = torch.tensor([33023, 33023], dtype=torch.int64)
    q_cu = torch.arange(num_seqs + 1, dtype=torch.int32)
    ctx = _raw_context(backend, table, kv)
    recorder = _ChunkRecordingBackend(backend)

    with forward_context(
        ForwardContext(
            attention_backend=backend,
            device=torch.device("cpu"),
            metadata=SimpleNamespace(q_cu_seq_lens=q_cu),
            layer_caches=[],
        )
    ):
        actual = indexer.select_qli(hidden, query, positions, mask, ctx, layer, recorder)
    selected = actual.reshape(num_seqs, -1).to(torch.int64)
    first = selected[0]
    valid_tokens = first[first >= 0]
    assert valid_tokens.numel() > 0
    assert int(valid_tokens.max()) <= 33023
    assert int(valid_tokens.max()) > 32767


@torch.inference_mode()
def test_graph_selection_replay_uses_refreshed_lengths_and_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    page_size, width, num_seqs = 4, 128, 2
    capacity_blocks = 4
    kv = torch.tensor([8, 16], dtype=torch.int32)
    block_table = torch.tensor([[1, 3, 3, 3], [2, 4, 4, 4]], dtype=torch.int32)
    backend = _make_backend(5, page_size, width)
    layer = SimpleNamespace(layer_id=0)

    indexer = _raw_indexer(torch.device("cpu"))
    hidden = torch.ones(1, num_seqs, 16, dtype=torch.bfloat16)
    query = torch.ones(num_seqs, 16, dtype=torch.bfloat16)
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
    ctx = _raw_context(backend, block_table, kv)
    recorder = _ChunkRecordingBackend(backend)
    monkeypatch.setattr(glm5_next, "_INDEX_HISTORY_CHUNK_BYTES", page_size * width * 2 * 2)

    execution_state = AclGraphExecutionState(persistent_buffers={})
    with forward_context(
        ForwardContext(
            attention_backend=backend,
            device=torch.device("cpu"),
            metadata=_prepare_metadata(block_table, kv),
            layer_caches=[],
            execution_state=execution_state,
        )
    ):
        backend.prepare(_prepare_metadata(block_table, kv), graph_mode=True, owned_metadata=True)
        first = indexer.select_qli(hidden, query, torch.tensor([7, 15], dtype=torch.int64), mask, ctx, layer, recorder)
        assert all(shape[1] == capacity_blocks * page_size for shape in recorder.gathered_shapes)
        assert len(recorder.gathered_shapes) > 1

        kv2 = torch.tensor([16, 8], dtype=torch.int32)
        backend._mla_actual_seq_kv.copy_(kv2)
        index = backend._kv_caches[0].index
        index[3, :, 0, -1] = 2  # block 3 was stale for seq0; now live
        recorder.gathered_shapes.clear()
        second = indexer.select_qli(hidden, query, torch.tensor([15, 7], dtype=torch.int64), mask, ctx, layer, recorder)
        assert all(shape[1] == capacity_blocks * page_size for shape in recorder.gathered_shapes)
        assert all(shape[0] < num_seqs for shape in recorder.gathered_shapes)

    first_sel = first.reshape(num_seqs, -1)
    second_sel = second.reshape(num_seqs, -1)
    assert first_sel.shape == second_sel.shape
    for row, bound in enumerate((15, 7)):
        row_tokens = second_sel[row]
        live = row_tokens[row_tokens >= 0]
        if live.numel() > 0:
            assert int(live.max()) <= bound


@torch.inference_mode()
def test_chunk_budget_includes_batch_in_scores_slab(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(5)
    indexer = _raw_indexer(torch.device("cpu"))
    batch, seq_len, kv_len, width = 3, 4, 8, 128
    q_resid = torch.randn(batch, seq_len, 16, dtype=torch.bfloat16)
    hidden = torch.randn(batch, seq_len, 16, dtype=torch.bfloat16)
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    packed = torch.randn(batch, kv_len, width, dtype=torch.bfloat16)
    packed[..., -1] = 1
    positions = (torch.arange(seq_len) + kv_len - seq_len).expand(batch, -1).contiguous()

    n_pools = kv_len  # raw rate=1: one pool per token
    cap = batch * indexer.n_heads * n_pools * 4
    matmul_shapes: list[tuple[int, ...]] = []
    real_matmul = torch.matmul
    score_shapes: list[tuple[int, ...]] = []
    real_topk = torch.Tensor.topk

    def _spy_matmul(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        result = real_matmul(a, b, out=out)
        matmul_shapes.append(tuple(result.shape))
        return result

    def _spy_topk(
        scores: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score_shapes.append(tuple(scores.shape))
        return real_topk(scores, k, dim=dim, largest=largest, sorted=sorted)

    monkeypatch.setattr(glm5_next, "_SCORES_SLAB_CAP_BYTES", cap)
    monkeypatch.setattr(torch, "matmul", _spy_matmul)
    monkeypatch.setattr(torch.Tensor, "topk", _spy_topk)
    chunked = indexer.select_topk(
        q_resid,
        hidden,
        mask,
        kv_len=kv_len,
        current_length=kv_len,
        packed_states=packed,
        query_positions=positions,
    )
    monkeypatch.undo()

    for shape in matmul_shapes:
        assert shape[1] == 1, shape
        assert shape[0] * shape[1] * shape[2] * shape[3] * 4 <= cap, shape
    assert len(matmul_shapes) >= seq_len
    assert score_shapes == [(batch, 1, n_pools)] * seq_len

    baseline = indexer.select_topk(
        q_resid,
        hidden,
        mask,
        kv_len=kv_len,
        current_length=kv_len,
        packed_states=packed,
        query_positions=positions,
    )
    torch.testing.assert_close(chunked, baseline, rtol=0, atol=0)
