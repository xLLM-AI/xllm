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

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="history chunk tests import the NPU attention backend")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.layers.attention import Attention
from xllm.python.models import glm5_next


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


def _make_backend(num_blocks: int, page_size: int, index_size: int) -> NpuPagedAttentionBackend:
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
    index = torch.zeros(num_blocks, page_size, 1, index_size, dtype=torch.bfloat16)
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
    )


class _ChunkRecordingBackend:
    def __init__(self, backend: NpuPagedAttentionBackend) -> None:
        self._backend = backend
        self.gathered_shapes: list[tuple[int, ...]] = []

    def indexer_block_table(self) -> torch.Tensor:
        return self._backend.indexer_block_table()

    def indexer_history_block_table(self, page_size: int) -> torch.Tensor:
        return self._backend.indexer_history_block_table(page_size)

    def gather_index_history(
        self,
        layer: Attention,
        start_request_index: int,
        end_request_index: int,
    ) -> torch.Tensor:
        out = self._backend.gather_index_history(layer, start_request_index, end_request_index)
        self.gathered_shapes.append(tuple(out.shape))
        return out


@torch.inference_mode()
def test_select_qli_chunks_history_and_matches_single_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    torch.manual_seed(3)
    q_len = 2

    def _run(indexer: glm5_next.Glm5NextIndexer, index_size: int) -> None:
        page_size = 4
        num_reqs = 6
        kv = torch.tensor([16, 16, 16, 16, 16, 16], dtype=torch.int32)
        num_pages_per_req = 4
        block_table = torch.arange(1, num_reqs * num_pages_per_req + 1, dtype=torch.int32).view(
            num_reqs, num_pages_per_req
        )
        backend = _make_backend(num_reqs * num_pages_per_req + 1, page_size, index_size)
        backend.prepare(_prepare_metadata(block_table, kv), graph_mode=False)
        layer = Attention(num_heads=8, num_kv_heads=1, head_dim=128, scale=0.1, sliding_window=0, layer_id=0)
        recorder = _ChunkRecordingBackend(backend)

        hidden = torch.randn(1, num_reqs * q_len, 16, dtype=torch.bfloat16)
        query = torch.randn(num_reqs * q_len, 16, dtype=torch.bfloat16)
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
        positions = (kv[:, None] - q_len + torch.arange(q_len)).reshape(-1).to(torch.int64)
        cu_seqlens_q = torch.arange(0, num_reqs * q_len + 1, q_len, dtype=torch.int32)
        metadata = SimpleNamespace(q_cu_seq_lens=cu_seqlens_q)
        with forward_context(
            ForwardContext(
                attention_backend=backend,
                device=torch.device("cpu"),
                metadata=metadata,
                layer_caches=[],
            )
        ):
            ctx = replace(backend.mla_index_context(layer), block_table=block_table[:1])
            history_budget = 3 * 16 * index_size * backend._kv_caches[0].index.element_size()
            monkeypatch.setattr(glm5_next, "_INDEX_HISTORY_CHUNK_BYTES", history_budget)
            actual = indexer.select_qli(hidden, query, positions, mask, ctx, layer, recorder)
            assert len(recorder.gathered_shapes) > 1
            assert all(shape[0] < num_reqs for shape in recorder.gathered_shapes)
            assert all(
                2 * shape[0] * shape[1] * shape[2] * backend._kv_caches[0].index.element_size() <= history_budget
                for shape in recorder.gathered_shapes
            ), recorder.gathered_shapes

            recorder.gathered_shapes.clear()
            monkeypatch.setattr(glm5_next, "_INDEX_HISTORY_CHUNK_BYTES", 1 << 40)
            reference = indexer.select_qli(hidden, query, positions, mask, ctx, layer, recorder)
            assert len(recorder.gathered_shapes) == 1
            assert recorder.gathered_shapes[0][0] == num_reqs
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    _run(_packed_indexer(torch.device("cpu")), 257)
    _run(_raw_indexer(torch.device("cpu")), 128)


@torch.inference_mode()
def test_select_qli_retains_token_beyond_old_history_limit() -> None:
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    page_size, index_size, num_reqs = 4, 128, 2
    kv = torch.tensor([33024, 33024], dtype=torch.int32)
    num_pages_per_req = 33024 // page_size
    num_blocks = num_pages_per_req * num_reqs + 1
    table = torch.arange(1, num_blocks, dtype=torch.int32).view(num_reqs, num_pages_per_req)
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
    index = torch.zeros(num_blocks, page_size, 1, index_size, dtype=torch.bfloat16)
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
    layer = Attention(num_heads=8, num_kv_heads=1, head_dim=128, scale=0.1, sliding_window=0, layer_id=0)

    indexer = _raw_indexer(torch.device("cpu"))
    indexer.wq_b.weight.zero_()
    indexer.wq_b.weight[0, 0] = 1.0
    indexer.weights_proj.weight.zero_()
    indexer.weights_proj.weight[0, 0] = 1.0
    hidden = torch.ones(1, num_reqs, 16, dtype=torch.bfloat16)
    query = torch.ones(num_reqs, 16, dtype=torch.bfloat16)
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
    positions = torch.tensor([33023, 33023], dtype=torch.int64)
    cu_seqlens_q = torch.arange(num_reqs + 1, dtype=torch.int32)
    recorder = _ChunkRecordingBackend(backend)

    with forward_context(
        ForwardContext(
            attention_backend=backend,
            device=torch.device("cpu"),
            metadata=SimpleNamespace(q_cu_seq_lens=cu_seqlens_q),
            layer_caches=[],
        )
    ):
        ctx = backend.mla_index_context(layer)
        actual = indexer.select_qli(hidden, query, positions, mask, ctx, layer, recorder)
    topk_token_indices = actual.reshape(num_reqs, -1).to(torch.int64)
    first = topk_token_indices[0]
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

    page_size, index_size, num_reqs = 4, 128, 2
    kv = torch.tensor([8, 16], dtype=torch.int32)
    block_table = torch.tensor([[1, 1, 3, 3], [2, 4, 4, 4]], dtype=torch.int32)
    history_token_capacity = block_table.size(1) * page_size
    backend = _make_backend(5, page_size, index_size)
    layer = Attention(num_heads=8, num_kv_heads=1, head_dim=128, scale=0.1, sliding_window=0, layer_id=0)

    indexer = _raw_indexer(torch.device("cpu"))
    indexer.wq_b.weight.fill_(1)
    indexer.weights_proj.weight.fill_(1)
    hidden = torch.ones(1, num_reqs, 16, dtype=torch.bfloat16)
    query = torch.ones(num_reqs, 16, dtype=torch.bfloat16)
    mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
    recorder = _ChunkRecordingBackend(backend)
    monkeypatch.setattr(glm5_next, "_INDEX_HISTORY_CHUNK_BYTES", page_size * index_size * 2 * 2)

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
        backend.prepare(_prepare_metadata(block_table, kv), graph_mode=True)
        ctx = backend.mla_index_context(layer)
        first = indexer.select_qli(hidden, query, torch.tensor([7, 15], dtype=torch.int64), mask, ctx, layer, recorder)
        assert all(shape[1] == history_token_capacity for shape in recorder.gathered_shapes)
        assert len(recorder.gathered_shapes) > 1

        kv2 = torch.tensor([16, 8], dtype=torch.int32)
        ctx.actual_seq_kv.copy_(kv2)
        index = backend._kv_caches[0].index
        index[3, :, 0, -1] = 100
        recorder.gathered_shapes.clear()
        second = indexer.select_qli(hidden, query, torch.tensor([15, 7], dtype=torch.int64), mask, ctx, layer, recorder)
        assert all(shape[1] == history_token_capacity for shape in recorder.gathered_shapes)
        assert all(shape[0] < num_reqs for shape in recorder.gathered_shapes)

    assert first.shape == second.shape
    second_topk_token_indices = second.reshape(num_reqs, -1)
    assert ((second_topk_token_indices[0] >= kv[0]) & (second_topk_token_indices[0] < kv2[0])).any()
    assert (second_topk_token_indices[1] >= 0).any()
    assert (second_topk_token_indices[1] < kv2[1]).all()


@torch.inference_mode()
def test_chunk_budget_includes_batch_in_scores_slab(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(5)
    indexer = _raw_indexer(torch.device("cpu"))
    num_reqs, q_len, kv_len, index_size = 3, 4, 8, 128
    q_resid = torch.randn(num_reqs, q_len, 16, dtype=torch.bfloat16)
    hidden = torch.randn(num_reqs, q_len, 16, dtype=torch.bfloat16)
    mask = torch.ones(num_reqs, q_len, dtype=torch.bool)
    packed = torch.randn(num_reqs, kv_len, index_size, dtype=torch.bfloat16)
    packed[..., -1] = 1
    positions = (torch.arange(q_len) + kv_len - q_len).expand(num_reqs, -1).contiguous()

    num_pools = kv_len
    score_budget_bytes = num_reqs * indexer.n_heads * num_pools * 4
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

    monkeypatch.setattr(glm5_next, "_SCORES_SLAB_CAP_BYTES", score_budget_bytes)
    monkeypatch.setattr(torch, "matmul", _spy_matmul)
    monkeypatch.setattr(torch.Tensor, "topk", _spy_topk)
    chunked = indexer.select_topk(
        q_resid,
        indexer.weights_proj(hidden),
        mask,
        kv_len=kv_len,
        kv_seq_lens=torch.full((num_reqs,), kv_len),
        packed_states=packed,
        query_positions=positions,
    )
    monkeypatch.undo()

    for shape in matmul_shapes:
        assert shape[1] == 1, shape
        assert shape[0] * shape[1] * shape[2] * shape[3] * 4 <= score_budget_bytes, shape
    assert len(matmul_shapes) >= q_len
    assert score_shapes == [(num_reqs, 1, num_pools)] * q_len

    baseline = indexer.select_topk(
        q_resid,
        indexer.weights_proj(hidden),
        mask,
        kv_len=kv_len,
        kv_seq_lens=torch.full((num_reqs,), kv_len),
        packed_states=packed,
        query_positions=positions,
    )
    torch.testing.assert_close(chunked, baseline, rtol=0, atol=0)
