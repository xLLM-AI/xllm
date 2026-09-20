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

"""Tests for the NPU paged-attention backend."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import LayerCache  # noqa: E402
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
)


def test_uses_first_nonempty_key_cache() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    linear_cache = LayerCache(
        key=None,
        value=None,
        conv=torch.empty(8, 3, 64),
        ssm=torch.empty(8, 2, 4, 4),
    )
    key_cache = torch.empty(17, 128, 2, 64)
    value_cache = torch.empty_like(key_cache)

    backend.bind_kv_caches(
        [
            linear_cache,
            LayerCache(key=key_cache, value=value_cache),
        ]
    )

    assert backend.num_kv_blocks == 17
    assert backend.page_size == 128


@pytest.mark.parametrize("host_ends", [[3, 5], [0, 3, 5]])
def test_prepared_host_query_ends_avoid_device_readback(host_ends: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    cache = torch.empty(4, 128, 2, 64)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    metadata = SimpleNamespace(
        q_cu_seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        q_cu_seq_lens_host_values=host_ends,
        q_seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        block_table=None,
        kv_seq_lens=None,
        is_prefill=True,
    )

    def reject_readback(self: torch.Tensor) -> torch.Tensor:
        raise AssertionError("prepared metadata must not copy Device lengths to Host")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_readback)
    backend.prepare(metadata)
    assert backend._cumulative_seq_lens(metadata, 5) == [3, 5]


def _ordinary_metadata(*, paged: bool) -> SimpleNamespace:
    return SimpleNamespace(
        q_cu_seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        q_cu_seq_lens_host_values=[0, 1, 2],
        q_seq_lens=torch.ones(2, dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32) if paged else None,
        kv_seq_lens=torch.tensor([6, 4], dtype=torch.int32),
        kv_seq_lens_host_values=[6, 4],
        is_prefill=not paged,
        is_spec_verify=False,
        has_kv_shard=False,
    )


def _ordinary_backend() -> NpuPagedAttentionBackend:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    cache = torch.empty(4, 128, 2, 64)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    return backend


@pytest.mark.parametrize("paged", [False, True])
def test_private_metadata_preparation_preserves_active_slot(paged: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _ordinary_backend()
    active = _ordinary_metadata(paged=True)
    backend.prepare(active)
    old_query = backend._actual_seq_q
    old_kv = backend._actual_seq_kv
    old_table = backend._block_table_i32
    metadata = _ordinary_metadata(paged=paged)

    def reject_tensor_work(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("private metadata preparation and activation must use prepared Host values/views")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_tensor_work)
    monkeypatch.setattr(torch.Tensor, "to", reject_tensor_work)
    monkeypatch.setattr(torch, "empty", reject_tensor_work)
    monkeypatch.setattr(torch, "arange", reject_tensor_work)
    state = backend.prepare_metadata(metadata)
    assert backend._metadata is active
    assert backend._actual_seq_q is old_query
    assert backend._actual_seq_kv is old_kv
    assert backend._block_table_i32 is old_table
    metadata.q_cu_seq_lens_host_values[:] = [-99]
    metadata.kv_seq_lens_host_values[:] = [-99]
    metadata.prepared_attention_state = state
    backend.prepare(metadata)
    assert backend._metadata is metadata
    assert backend._actual_seq_lens == [1, 2]
    assert backend._actual_seq_q == ([1, 2] if paged else [])
    assert backend._actual_seq_kv == ([6, 4] if paged else [])
    assert backend._block_table_i32 is metadata.block_table


@pytest.mark.parametrize("invalid", ["dtype", "query", "kv", "verify", "shard", "expanded"])
def test_private_metadata_rejection_preserves_active_slot(invalid: str) -> None:
    backend = _ordinary_backend()
    active = _ordinary_metadata(paged=True)
    backend.prepare(active)
    old_kv = backend._actual_seq_kv
    candidate = _ordinary_metadata(paged=True)
    if invalid == "dtype":
        candidate.block_table = candidate.block_table.to(torch.int64)
    elif invalid == "query":
        candidate.q_cu_seq_lens_host_values = [1]
    elif invalid == "kv":
        candidate.kv_seq_lens_host_values = [6]
    elif invalid == "verify":
        candidate.is_spec_verify = True
    elif invalid == "shard":
        candidate.has_kv_shard = True
    else:
        candidate.expanded_decode_metadata = SimpleNamespace(enabled=True)
    with pytest.raises(ValueError):
        backend.prepare_metadata(candidate)
    assert backend._metadata is active
    assert backend._actual_seq_kv is old_kv


def _mla_backend() -> NpuPagedAttentionBackend:
    backend = NpuPagedAttentionBackend(
        num_heads=64,
        num_kv_heads=1,
        head_dim=256,
        scale=0.0625,
        sliding_window=0,
        is_mla=True,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    cache = torch.empty(4, 128, 1, 512)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    return backend


def _mla_metadata(*, prefill: bool = False, chunked: bool = False) -> SimpleNamespace:
    metadata = _ordinary_metadata(paged=True)
    metadata.is_prefill = prefill
    metadata.is_chunked_prefill = chunked
    metadata.q_cu_seq_lens = torch.tensor([3, 5] if prefill or chunked else [1, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens_host_values = [0, 3, 5] if prefill or chunked else [0, 1, 2]
    metadata.slot_mapping = torch.arange(5 if prefill or chunked else 2, dtype=torch.int32)
    return metadata


@pytest.mark.parametrize("prefill,chunked", [(True, False), (False, True), (False, False)])
def test_prepared_mla_borrows_final_views_without_device_work(
    prefill: bool,
    chunked: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _mla_backend()
    active = object()
    backend._metadata = active
    metadata = _mla_metadata(prefill=prefill, chunked=chunked)
    backend._mla_quant_indexer_metadata["previous_slot"] = object()

    def reject_tensor_work(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("prepared MLA must borrow final views without Device work")

    for name in ("cpu", "to", "copy_", "clone", "item"):
        monkeypatch.setattr(torch.Tensor, name, reject_tensor_work)
    for name in ("empty", "arange", "tensor"):
        monkeypatch.setattr(torch, name, reject_tensor_work)
    state = backend.prepare_metadata(metadata)
    assert backend._metadata is active
    assert "previous_slot" in backend._mla_quant_indexer_metadata
    metadata.q_cu_seq_lens_host_values[:] = [-99]
    metadata.kv_seq_lens_host_values[:] = [-99]
    metadata.prepared_attention_state = state
    backend.prepare(metadata)
    assert backend._metadata is state
    assert state.query_ends == ([3, 5] if prefill or chunked else [1, 2])
    assert state.kv_lengths == [6, 4]
    assert backend._mla_actual_seq_q is metadata.q_cu_seq_lens
    assert backend._mla_actual_seq_kv is metadata.kv_seq_lens
    assert backend._block_table_i32 is metadata.block_table
    assert state.slot_mapping is metadata.slot_mapping
    assert backend._mla_max_seqlen_q == (3 if prefill or chunked else 1)
    assert backend._mla_max_seqlen_k == 6
    assert not backend._mla_quant_indexer_metadata


@pytest.mark.parametrize("invalid", ["table", "query_dtype", "kv_shape", "slots"])
def test_prepared_mla_rejects_invalid_views_before_activation(invalid: str) -> None:
    backend = _mla_backend()
    metadata = _mla_metadata()
    active = object()
    backend._metadata = active
    if invalid == "table":
        metadata.block_table = None
    elif invalid == "query_dtype":
        metadata.q_cu_seq_lens = metadata.q_cu_seq_lens.to(torch.int64)
    elif invalid == "kv_shape":
        metadata.kv_seq_lens = metadata.kv_seq_lens[:1]
    elif invalid == "slots":
        metadata.slot_mapping = metadata.slot_mapping[:1]
    with pytest.raises(ValueError):
        backend.prepare_metadata(metadata)
    assert backend._metadata is active


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("prefill,chunked", [(True, False), (False, True), (False, False)])
def test_prepared_graph_rejection_preserves_active_slot(is_mla: bool, prefill: bool, chunked: bool) -> None:
    backend = _mla_backend() if is_mla else _ordinary_backend()
    active = _mla_metadata()
    active.prepared_attention_state = backend.prepare_metadata(active)
    backend.prepare(active)
    active_state = backend._metadata
    backend._mla_quant_indexer_metadata["active_slot"] = object()
    # Even a warmed graph must not accept prepared eager metadata.
    backend._graph_workspace = object()
    backend._graph_outputs[2] = object()
    backend._graph_lses[2] = object()
    metadata = _mla_metadata(prefill=prefill, chunked=chunked)
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)

    with pytest.raises(ValueError, match="requires eager execution"):
        backend.prepare(metadata, graph_mode=True)

    assert backend._metadata is active_state
    assert backend._block_table_i32 is active.block_table
    assert "active_slot" in backend._mla_quant_indexer_metadata
    assert backend._current_graph_output is None
    assert backend._current_graph_lse is None
