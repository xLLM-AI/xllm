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
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import LayerCache  # noqa: E402
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
    PagedAttentionGraphState,
)
from xllm.python.kernels_npu import _custom_op, sparse_attention  # noqa: E402
from xllm.python.model_executor.forward_context import ForwardContext, forward_context  # noqa: E402


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


@pytest.mark.parametrize("masked", [False, True])
def test_prefill_allocates_mask_only_for_consumer(masked: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _ordinary_backend()
    metadata = _ordinary_metadata(paged=False)
    backend.prepare(metadata)
    assert backend._causal_mask is None
    masks: list[torch.Tensor | None] = []

    def attention(q: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        masks.append(kwargs["atten_mask"])
        return q, None

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", attention, raising=False)
    q = torch.zeros(2, 8, 64)
    kv = torch.zeros(2, 2, 64)
    cache = backend._kv_caches[0].key
    layer = SimpleNamespace(
        causal=masked,
        fia_use_attention_mask=False,
        fia_sparse_mode=None,
        fia_pre_tokens=2147483647,
        fia_next_tokens=0,
    )
    for _ in range(2):
        backend._prefill(q, kv, kv, cache, cache, metadata, 2, layer)
    if masked:
        assert masks[0] is masks[1]
        assert masks[0].dtype == torch.int8
        assert masks[0].shape == (2048, 2048)
        assert masks[0].sum().item() == 2048 * 2047 // 2
        assert torch.equal(masks[0][:3, :3], torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=torch.int8))
    else:
        assert masks == [None, None]
        assert backend._causal_mask is None


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


def test_prepared_block_query_uses_cumulative_query_ends() -> None:
    backend = _ordinary_backend()
    metadata = SimpleNamespace(
        q_cu_seq_lens=torch.tensor([0, 4, 8], dtype=torch.int32),
        q_cu_seq_lens_host_values=[4, 8],
        q_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([6, 7], dtype=torch.int32),
        kv_seq_lens_host_values=[6, 7],
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=False,
        has_kv_shard=False,
    )

    prepared = backend.prepare_metadata(metadata, device_kv_lengths=True)

    assert prepared.actual_seq_q == [4, 8]


@pytest.mark.parametrize("use_fia_v2", [False, True])
@pytest.mark.parametrize(
    ("query_ends", "batch_size", "query_tokens"),
    [([7], 1, 7), ([4, 8], 2, 8), ([1, 2], 2, 2), ([], 2, 2)],
)
def test_graph_workspace_uses_packed_query_token_count(
    use_fia_v2: bool,
    query_ends: list[int],
    batch_size: int,
    query_tokens: int,
) -> None:
    backend = _ordinary_backend()
    backend._use_fia_v2 = use_fia_v2
    backend._actual_seq_q = list(query_ends)
    backend._actual_seq_kv = [6] * batch_size
    block_table = torch.zeros(batch_size, 2, dtype=torch.int32)
    kv_capacity = [256] * batch_size
    workspace = torch.empty(1, dtype=torch.uint8)
    helper_name = (
        "_npu_fused_infer_attention_score_v2_get_max_workspace"
        if use_fia_v2
        else "_npu_fused_infer_attention_score_get_max_workspace"
    )

    with patch(
        f"xllm.python.attention.npu_paged_attention.torch_npu.{helper_name}",
        return_value=workspace,
        create=True,
    ) as allocate:
        result = backend._allocate_graph_workspace(batch_size, block_table, actual_seq_kv=kv_capacity)

    assert result is workspace
    allocate.assert_called_once()
    kwargs = allocate.call_args.kwargs
    assert kwargs["query"].shape == (query_tokens, backend.num_heads, backend.head_dim)
    assert kwargs["block_table"] is block_table
    query_lengths_key = "actual_seq_qlen" if use_fia_v2 else "actual_seq_lengths"
    kv_lengths_key = "actual_seq_kvlen" if use_fia_v2 else "actual_seq_lengths_kv"
    assert kwargs[query_lengths_key] == query_ends
    assert kwargs[kv_lengths_key] == kv_capacity


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


def test_sparse_mtp_page_crossing_updates_without_host_lengths_or_query_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xllm.python.model_executor.forward_context import (
        AclGraphExecutionState,
        ForwardContext,
        forward_context,
    )

    backend = _mla_backend()
    cache = torch.empty(4, 128, 1, 512)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache, index=cache)])
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([127], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([128], dtype=torch.int32),
        q_cu_seq_lens=None,
        q_seq_lens=None,
        kv_seq_lens_host_values=[],
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    def reject_readback(*args: object, **kwargs: object) -> None:
        raise AssertionError("sparse MTP metadata must stay on Device")

    context = ForwardContext(backend, torch.device("cpu"), metadata, [], execution_state=AclGraphExecutionState({}))
    with monkeypatch.context() as no_readback, forward_context(context):
        no_readback.setattr(torch.Tensor, "cpu", reject_readback)
        no_readback.setattr(torch.Tensor, "item", reject_readback)
        backend.prepare_owned_graph_metadata(metadata)
        assert backend.graph_metadata_updated_in_place
        assert backend._mla_actual_seq_kv.data_ptr() == metadata.kv_seq_lens.data_ptr()
        destinations = tuple(
            (tensor.shape, tensor.dtype, tensor.device, tensor.data_ptr())
            for tensor in (metadata.slot_mapping, metadata.block_table, metadata.kv_seq_lens)
        )
        query_ptr = backend._mla_actual_seq_q.data_ptr()
        metadata.kv_seq_lens.fill_(129)
        metadata.slot_mapping.fill_(128)
        metadata.block_table.copy_(torch.tensor([[2, 3]], dtype=torch.int32))
    assert backend._causal_mask is None
    assert destinations == tuple(
        (tensor.shape, tensor.dtype, tensor.device, tensor.data_ptr())
        for tensor in (metadata.slot_mapping, metadata.block_table, metadata.kv_seq_lens)
    )
    assert backend._mla_actual_seq_q.data_ptr() == query_ptr
    assert backend._mla_actual_seq_q.tolist() == [1]
    assert backend._mla_actual_seq_kv.tolist() == [129]
    assert backend._block_table_i32.tolist() == [[2, 3]]
    backend.prepare(metadata)
    assert not backend.graph_metadata_updated_in_place


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


@pytest.mark.parametrize("chunked", [False, True])
def test_prepared_cp_rebuilds_segment_tables_when_slot_storage_is_reused(chunked: bool) -> None:
    backend = _mla_backend()
    metadata = _mla_metadata(prefill=not chunked, chunked=chunked)
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    backend.prepare(metadata)
    cp_context = SimpleNamespace(segment_seq_indices=torch.tensor([0, 1, 0], dtype=torch.int64))
    first = backend._segment_block_table(metadata.block_table, cp_context)
    assert backend._segment_block_table(metadata.block_table, cp_context) is first

    # A subsequent Task uses the same Slot views for different cache pages.
    # Host-only preparation must leave the active Task's cached table intact;
    # only Launch may replace the shared backend's per-forward state.
    metadata.block_table.copy_(torch.tensor([[2], [3]], dtype=torch.int32))
    state = backend.prepare_metadata(metadata)
    assert backend._segment_block_table(metadata.block_table, cp_context) is first
    torch.testing.assert_close(first, torch.tensor([[0], [1], [0]], dtype=torch.int32))

    metadata.prepared_attention_state = state
    backend.prepare(metadata)
    second = backend._segment_block_table(metadata.block_table, cp_context)
    assert second is not first
    torch.testing.assert_close(second, torch.tensor([[2], [3], [2]], dtype=torch.int32))
    assert backend._segment_block_table(metadata.block_table, cp_context) is second
    assert len(backend._mla_cp_block_tables) == 1

    decode = _mla_metadata()
    decode.prepared_attention_state = backend.prepare_metadata(decode)
    backend.prepare(decode, graph_mode=True)
    assert not backend._mla_cp_block_tables
    assert backend._block_table_i32 is decode.block_table


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
@pytest.mark.parametrize("prefill,chunked", [(True, False), (False, True)])
def test_prepared_graph_rejection_preserves_active_slot(is_mla: bool, prefill: bool, chunked: bool) -> None:
    backend = _mla_backend() if is_mla else _ordinary_backend()
    active = _mla_metadata()
    active.prepared_attention_state = backend.prepare_metadata(active)
    backend.prepare(active)
    active_state = backend._metadata
    backend._mla_quant_indexer_metadata["active_slot"] = object()
    # A warmed decode graph must not accept prepared prefill metadata.
    graph_state = object()
    backend._paged_graph_state = graph_state
    metadata = _mla_metadata(prefill=prefill, chunked=chunked)
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)

    with pytest.raises(ValueError, match="requires.*decode"):
        backend.prepare(metadata, graph_mode=True)

    assert backend._metadata is active_state
    assert backend._block_table_i32 is active.block_table
    assert "active_slot" in backend._mla_quant_indexer_metadata
    assert backend._paged_graph_state is graph_state


@pytest.mark.parametrize("is_mla", [False, True])
def test_prepared_decode_uses_warmed_graph_state(is_mla: bool) -> None:
    from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context

    backend = _mla_backend() if is_mla else _ordinary_backend()
    metadata = _mla_metadata()
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    state = PagedAttentionGraphState(
        torch.empty(1), torch.empty(2, 8, 64), torch.empty(0), metadata.block_table, [], []
    )
    execution = AclGraphExecutionState({}, {2: state})
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [], execution_state=execution)):
        backend.prepare(metadata, graph_mode=True)
    assert backend._block_table_i32 is metadata.block_table
    if is_mla:
        assert backend._metadata is metadata.prepared_attention_state
        assert backend._mla_actual_seq_q is metadata.q_cu_seq_lens
        assert backend._mla_actual_seq_kv is metadata.kv_seq_lens
        assert backend._mla_max_seqlen_k == metadata.block_table.shape[1] * backend.page_size
    else:
        assert backend._metadata is metadata
        assert backend._paged_graph_state is state
        assert state.kv == metadata.kv_seq_lens_host_values
        assert backend._actual_seq_lens is None


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.parametrize("window", [None, 32])
def test_block_attention_reads_accepted_device_lengths(width: int, window: int | None) -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    device = torch.device("npu:0")
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=0,
        is_mla=False,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(1234)
    cache = torch.randn(4, 128, 2, 128, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(cache)
    backend.bind_kv_caches([LayerCache(key=cache, value=values)])
    query = torch.randn(2 * width, 4, 128, device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        q_seq_lens=torch.full((2,), width, device=device, dtype=torch.int32),
        q_cu_seq_lens_host_values=[width, 2 * width],
        block_table=torch.tensor([[0, 1], [2, 3]], device=device, dtype=torch.int32),
        kv_seq_lens=torch.tensor([137, 143], device=device, dtype=torch.int32),
        kv_seq_lens_host_values=[137, 143],
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=False,
        has_kv_shard=False,
        expanded_decode_metadata=None,
    )
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    # Acceptance arrives after Prepare; the bound view must observe the new
    # device lengths while Host upper bounds remain unchanged.
    metadata.kv_seq_lens.copy_(torch.tensor([130, 139], device=device, dtype=torch.int32))
    assert metadata.prepared_attention_state.actual_seq_kv == [137, 143]
    layer = SimpleNamespace(
        causal=False,
        fia_use_attention_mask=False,
        fia_sparse_mode=None,
        fia_pre_tokens=2147483647,
        fia_next_tokens=0,
    )
    if window is not None:
        layer.fia_use_attention_mask = True
        layer.fia_sparse_mode = 4
        layer.fia_pre_tokens = window - 1
        layer.fia_next_tokens = width - 1
    backend.prepare(metadata)
    actual = backend._prefill(query, query, query, cache, values, metadata, 2 * width, layer)
    metadata.kv_seq_lens_host_values = [130, 139]
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    backend.prepare(metadata)
    expected = backend._prefill(query, query, query, cache, values, metadata, 2 * width, layer)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("window", [None, 64])
@torch.inference_mode()
def test_block_graph_replay_matches_eager_after_kv_length_changes(window: int | None) -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.model_executor.runners.block_draft_acl_graph import BlockDraftAclGraphRunner

    device = torch.device("npu:0")
    width, heads, head_dim = 7, 4, 192
    page_size, capacity = 128, 4096
    backend = NpuPagedAttentionBackend(heads, heads, head_dim, head_dim**-0.5, 0, False, device, torch.bfloat16)
    cache = torch.zeros(33, page_size, heads, head_dim, device=device, dtype=torch.bfloat16)
    token_values = torch.arange(33 * page_size, device=device, dtype=torch.float32) / 2048
    values = token_values.view(33, page_size, 1, 1).expand_as(cache).contiguous().to(cache.dtype)
    backend.bind_kv_caches([LayerCache(key=cache, value=values)])
    query = torch.zeros(width, heads, head_dim, device=device, dtype=cache.dtype)
    tokens = torch.zeros(width, device=device, dtype=torch.int32)
    metadata = SimpleNamespace(
        q_seq_lens=torch.tensor([width], device=device, dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, width], device=device, dtype=torch.int32),
        q_cu_seq_lens_host_values=[width],
        block_table=torch.arange(1, 33, device=device, dtype=torch.int32).view(1, -1),
        slot_mapping=torch.arange(width, device=device, dtype=torch.int32),
        kv_seq_lens=torch.tensor([23], device=device, dtype=torch.int32),
        kv_seq_lens_host_values=[23],
        kv_cu_seq_lens=None,
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=False,
    )
    layer = SimpleNamespace(
        causal=False,
        fia_use_attention_mask=window is not None,
        fia_sparse_mode=4 if window is not None else None,
        fia_pre_tokens=window - 1 if window is not None else 2147483647,
        fia_next_tokens=width - 1,
    )
    runner = BlockDraftAclGraphRunner(torch.nn.Identity(), backend, device, 1, capacity)
    entry = runner._allocate_entry(tokens, tokens, metadata)
    runner._fill_entry(entry, tokens, tokens, metadata)
    static = entry.static_metadata

    def _attend(current: SimpleNamespace) -> torch.Tensor:
        backend.prepare(current)
        return backend._prefill(query, query, query, cache, values, current, width, layer)

    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            _attend(static)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = _attend(static)
    torch.npu.synchronize()

    # Grow far beyond the short capture length, then shrink and grow again.
    for length in (1875, 130, 1987):
        metadata.kv_seq_lens.fill_(length)
        metadata.kv_seq_lens_host_values[:] = [length]
        runner._fill_entry(entry, tokens, tokens, metadata)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.synchronize()
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
        expected = _attend(metadata)
        torch.testing.assert_close(captured, expected, rtol=1e-2, atol=2e-3)


def test_block_attention_shares_masks_per_forward_and_refreshes_device_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _ordinary_backend()
    metadata = _ordinary_metadata(paged=True)
    metadata.q_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens_host_values = [2, 4]
    metadata.block_table = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    metadata.is_chunked_prefill = True
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    query = torch.zeros(4, 8, 64)
    cache = torch.zeros(4, 128, 2, 64)
    calls: list[dict[str, object]] = []

    def attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        return query.clone(), torch.empty(0)

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", attention)
    full = SimpleNamespace(causal=False, fia_use_attention_mask=False, fia_sparse_mode=None)
    window = SimpleNamespace(
        causal=False, fia_use_attention_mask=True, fia_sparse_mode=4, fia_pre_tokens=2, fia_next_tokens=1
    )
    metadata.kv_seq_lens.copy_(torch.tensor([5, 3], dtype=torch.int32))
    backend.prepare(metadata)
    backend._prefill(query, query, query, cache, cache, metadata, 4, full)
    # Preparing another Slot is Host-only and must not alter the active
    # forward's mask or block table while its decoder layers are launching.
    backend.prepare_metadata(_ordinary_metadata(paged=True))
    for layer in (full, window, window):
        backend._prefill(query, query, query, cache, cache, metadata, 4, layer)
    assert calls[0]["atten_mask"] is calls[1]["atten_mask"]
    assert calls[2]["atten_mask"] is calls[3]["atten_mask"]
    assert calls[0]["atten_mask"] is not calls[2]["atten_mask"]
    assert all(call["block_table"] is calls[0]["block_table"] for call in calls)
    torch.testing.assert_close(calls[0]["block_table"], torch.tensor([[0], [1]], dtype=torch.int32))
    assert (~calls[0]["atten_mask"]).sum(dim=-1).tolist() == [[[5, 5]], [[3, 3]]]
    assert torch.where(~calls[2]["atten_mask"][0, 0, 0])[0].tolist() == [1, 2, 3, 4]
    assert torch.where(~calls[2]["atten_mask"][0, 0, 1])[0].tolist() == [2, 3, 4]

    # The same Slot metadata retains its Host upper bounds, but a new forward
    # must use the latest accepted Device lengths rather than cached masks.
    metadata.kv_seq_lens.copy_(torch.tensor([4, 2], dtype=torch.int32))
    backend.prepare(metadata)
    for layer in (full, window):
        backend._prefill(query, query, query, cache, cache, metadata, 4, layer)
    assert calls[4]["atten_mask"] is not calls[0]["atten_mask"]
    assert calls[5]["atten_mask"] is not calls[2]["atten_mask"]
    assert (~calls[4]["atten_mask"]).sum(dim=-1).tolist() == [[[4, 4]], [[2, 2]]]
    assert torch.where(~calls[5]["atten_mask"][0, 0, 0])[0].tolist() == [0, 1, 2, 3]
    assert torch.where(~calls[5]["atten_mask"][0, 0, 1])[0].tolist() == [1, 2, 3]
    assert metadata.kv_seq_lens_host_values == [6, 4]


def test_paged_graph_requires_an_execution_owner() -> None:
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    backend = _ordinary_backend()
    metadata = _mla_metadata()
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    with (
        forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])),
        pytest.raises(RuntimeError, match="execution entry"),
    ):
        backend.prepare(metadata, graph_mode=True)


@pytest.mark.parametrize("share_workspace", [False, True])
def test_paged_graph_keeps_metadata_private_when_workspace_is_shared(
    share_workspace: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context

    backend = _ordinary_backend()
    monkeypatch.setattr(backend, "_allocate_graph_workspace", lambda batch, table, **kwargs: torch.empty(4))
    shared = {} if share_workspace else None
    executions = [AclGraphExecutionState({}, shared_persistent_buffers=shared) for _ in range(2)]
    metadata = [_mla_metadata(), _mla_metadata()]
    contexts = [
        ForwardContext(backend, torch.device("cpu"), view, [], execution_state=execution)
        for view, execution in zip(metadata, executions)
    ]
    for context in contexts:
        with forward_context(context):
            backend.prepare(context.metadata, graph_mode=True)

    first, second = [execution.paged_attention[2] for execution in executions]
    assert first is not second
    assert first.query is not second.query
    assert first.kv is not second.kv
    assert first.block_table.data_ptr() != second.block_table.data_ptr()
    assert (first.workspace.data_ptr() == second.workspace.data_ptr()) == share_workspace
    assert (first.output.data_ptr() == second.output.data_ptr()) == share_workspace

    # The first role's final metadata updates only its own captured storage.
    first.block_table.fill_(2)
    first.kv[:] = [7, 9]
    assert second.kv == [6, 4]
    assert first.block_table.tolist() == [[2], [2]]
    assert second.block_table.tolist() == [[0], [1]]


def test_paged_workspace_sharing_respects_kv_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context

    backend = _ordinary_backend()
    capacities = []

    def allocate(batch: int, table: torch.Tensor, *, actual_seq_kv: list[int]) -> torch.Tensor:
        capacities.append(actual_seq_kv)
        return torch.empty(max(actual_seq_kv))

    monkeypatch.setattr(backend, "_allocate_graph_workspace", allocate)
    shared = {}
    states = []
    for capacity in (128, 256, 128):
        execution = AclGraphExecutionState({}, shared_persistent_buffers=shared)
        metadata = _mla_metadata()
        backend.prepare(metadata)
        with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [], execution_state=execution)):
            backend._prepare_paged_graph(workspace_kv_length=capacity)
        states.append(execution.paged_attention[2])

    assert capacities == [[128, 128], [256, 256]]
    assert states[0].workspace.data_ptr() != states[1].workspace.data_ptr()
    assert states[0].workspace.data_ptr() == states[2].workspace.data_ptr()


def _sparse_attention_out_args(
    query: torch.Tensor, output: torch.Tensor, layout: str, *, rope: bool = False
) -> tuple[object, ...]:
    return (
        query,
        query,
        query,
        torch.zeros(1, 1, 1, dtype=torch.int32),
        None,
        None,
        None,
        query if rope else None,
        query if rope else None,
        0.5,
        1,
        layout,
        "PA_BSND",
        3,
        output,
    )


def _invalid_sparse_attention_output(query: torch.Tensor, invalid: str) -> torch.Tensor:
    if invalid == "shape":
        shape = (*query.shape[:-2], 1, query.shape[-1])
        return query.new_empty(shape)
    if invalid == "dtype":
        return torch.empty_like(query, dtype=torch.float32)
    if invalid == "device":
        return torch.empty_like(query, device="meta")
    shape = (*query.shape[:-1], query.shape[-1] * 2)
    return query.new_empty(shape)[..., ::2]


@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "contiguous"])
def test_sparse_attention_rejects_invalid_output_before_dispatch(layout: str, invalid: str) -> None:
    shape = (2, 2, 4) if layout == "TND" else (1, 2, 2, 4)
    query = torch.ones(shape, dtype=torch.bfloat16)
    output = _invalid_sparse_attention_output(query, invalid)
    with (
        patch.object(torch.ops.xllm_ops, "sparse_flash_attention_lse_out", create=True) as native,
        patch.object(torch.ops.npu, "npu_sparse_flash_attention", create=True) as fallback,
        pytest.raises(ValueError, match=invalid),
    ):
        sparse_attention.sparse_flash_attention_out(*_sparse_attention_out_args(query, output, layout))
    native.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize("rope", [False, True])
def test_sparse_attention_preserves_output_storage_across_dispatch(layout: str, rope: bool) -> None:
    shape = (2, 2, 4) if layout == "TND" else (1, 2, 2, 4)
    query = torch.ones(shape, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    expected = torch.full_like(query, 3)

    def write_native(*args: object) -> torch.Tensor:
        output.copy_(expected)
        return output

    with (
        patch.object(
            torch.ops.xllm_ops, "sparse_flash_attention_lse_out", side_effect=write_native, create=True
        ) as native,
        patch.object(
            torch.ops.npu, "npu_sparse_flash_attention", return_value=(expected, None, None), create=True
        ) as fallback,
        patch.object(
            torch.ops.xllm_ops, "sparse_flash_attention_lse", return_value=(expected, None, None), create=True
        ) as allocating,
    ):
        result = sparse_attention.sparse_flash_attention_out(
            *_sparse_attention_out_args(query, output, layout, rope=rope)
        )
    assert result is output
    torch.testing.assert_close(result, expected)
    if layout == "TND" and rope:
        native.assert_called_once()
        assert native.call_args.args[-1] is output
        fallback.assert_not_called()
        allocating.assert_not_called()
    elif rope:
        native.assert_not_called()
        fallback.assert_not_called()
        allocating.assert_called_once()
    else:
        native.assert_not_called()
        fallback.assert_called_once()
        allocating.assert_not_called()


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "contiguous"])
def test_direct_sparse_attention_matches_output_contract(fake: bool, invalid: str) -> None:
    query = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    output = _invalid_sparse_attention_output(query, invalid)
    args = _sparse_attention_out_args(query, output, "TND")
    with (
        patch.object(torch.ops.xllm_ops, "sparse_flash_attention_lse_out", create=True) as native,
        pytest.raises(ValueError, match=invalid),
    ):
        if fake:
            _custom_op._sparse_flash_attention_lse_out_fake(*args[:-1], 2**63 - 1, 2**63 - 1, 2, False, output)
        else:
            sparse_attention.sparse_flash_attention_lse_out(*args)
    native.assert_not_called()


@pytest.mark.usefixtures("causal_conv1d_reference")
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("activation", ["identity", "silu"])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_kda_dense_conv_dispatches_fused_activation(
    monkeypatch: pytest.MonkeyPatch, width: int, activation: str, is_prefill: bool
) -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    value = torch.arange(2 * 12 * width, dtype=torch.bfloat16).view(2, 12, width)
    weight = torch.ones(12, 1, 4, dtype=torch.float32)
    layer = SimpleNamespace(
        layer_id=0, activation=activation, conv_weight_t=weight.squeeze(1).t().to(value.dtype).contiguous()
    )
    state = torch.zeros(2, 3, 12, dtype=torch.bfloat16)
    expected_conv = torch.linspace(-4, 4, value.numel()).to(torch.bfloat16).view(2, width, 12)

    def native_conv(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        dense_state: torch.Tensor,
        query_start_loc: list[int],
        activation_mode: int,
        run_mode: int,
    ) -> torch.Tensor:
        assert inputs.is_contiguous()
        assert weights is layer.conv_weight_t
        torch.testing.assert_close(weights, weight.squeeze(1).t().to(value.dtype))
        assert dense_state is state
        assert query_start_loc == []
        assert activation_mode == (1 if activation == "silu" else 0)
        assert run_mode == (0 if is_prefill else 1)
        torch.testing.assert_close(inputs, value.transpose(1, 2))
        dense_state.fill_(7)
        return expected_conv.clone()

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d", native_conv)
    output = backend._causal_conv1d(value, state, layer.conv_weight_t, activation, is_prefill=is_prefill)
    torch.testing.assert_close(output, expected_conv.transpose(1, 2), rtol=0, atol=0)
    assert output.dtype == torch.bfloat16
    assert torch.all(state == 7)


@pytest.mark.parametrize("value_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_width", [2, 4])
def test_kda_verify_uses_native_conv(
    causal_conv1d_reference: list[dict],
    value_dtype: torch.dtype,
    kernel_width: int,
) -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    layer = SimpleNamespace(
        head_dim=1,
        num_heads_local=1,
        qkv_dim=1,
        conv_dim=3,
        conv_kernel_size=kernel_width,
        activation="silu",
        layer_id=0,
        conv_weight_t=torch.ones(kernel_width, 3, dtype=value_dtype),
    )
    metadata = SimpleNamespace(expanded_decode_metadata=None, kv_seq_lens=torch.tensor([1]))

    def recurrent(query: torch.Tensor, *arguments: object, **options: object) -> torch.Tensor:
        return query

    output = backend._spec_verify(
        torch.ones(1, 3, 1, dtype=value_dtype),
        torch.zeros(1, 1, 1, 1),
        torch.ones(1, 1, 1),
        layer.layer_id,
        layer.conv_weight_t,
        layer.activation,
        torch.tensor([0]),
        metadata,
        torch.zeros(2, kernel_width - 1, 3, dtype=value_dtype),
        torch.zeros(2, 1, 1, 1),
        recurrent,
    )
    assert len(causal_conv1d_reference) == 1
    assert causal_conv1d_reference[0]["activation_mode"] == 1
    assert causal_conv1d_reference[0]["run_mode"] == 1
    assert causal_conv1d_reference[0]["weight"].dtype == value_dtype
    assert torch.isfinite(output).all()


@pytest.mark.usefixtures("causal_conv1d_reference")
@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("sequence_count", [1, 4])
@pytest.mark.parametrize("state_length", [1, 3])
def test_kda_conv_tails_preserve_each_proposal_state(width: int, sequence_count: int, state_length: int) -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    capacity = sequence_count * 2 + 1
    conv_dim = 3
    token_count = sequence_count * width
    indices = torch.arange(sequence_count, dtype=torch.int64) * 2 + 1
    mixed = torch.arange(conv_dim * token_count, dtype=torch.float32).reshape(1, conv_dim, token_count)
    conv_cache = torch.arange(capacity * state_length * conv_dim, dtype=torch.float32).reshape(
        capacity, state_length, conv_dim
    )
    initial_cache = conv_cache.clone()
    layer = SimpleNamespace(
        head_dim=1,
        num_heads_local=1,
        qkv_dim=1,
        conv_dim=conv_dim,
        conv_kernel_size=state_length + 1,
        activation="identity",
        layer_id=0,
        conv_weight_t=torch.ones(state_length + 1, conv_dim),
    )
    metadata = SimpleNamespace(expanded_decode_metadata=None, kv_seq_lens=torch.ones(token_count))

    def recurrent(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        return value

    backend._spec_verify(
        mixed,
        torch.zeros(1, token_count, 1, 1),
        torch.ones(1, token_count, 1),
        layer.layer_id,
        layer.conv_weight_t,
        layer.activation,
        indices,
        metadata,
        conv_cache,
        torch.zeros(capacity, 1, 1, 1),
        recurrent,
    )
    expected = torch.zeros_like(backend._kda_state[0]["combined_conv"])
    values = mixed.reshape(conv_dim, sequence_count, width).permute(1, 2, 0)
    expected_cache = initial_cache.clone()
    for sequence, slot in enumerate(indices.tolist()):
        history = torch.cat([initial_cache[slot], values[sequence]], dim=0)
        for proposal in range(width):
            expected[slot + proposal * capacity] = history[proposal + 1 : proposal + 1 + state_length]
        expected_cache[slot] = expected[slot]
    torch.testing.assert_close(backend._kda_state[0]["combined_conv"], expected, rtol=0, atol=0)
    torch.testing.assert_close(conv_cache, expected_cache, rtol=0, atol=0)


def _run_kda_verify_step(
    backend: NpuPagedAttentionBackend,
    width: int,
    base_lengths: list[int],
    conv_cache: torch.Tensor,
    ssm_cache: torch.Tensor,
    metadata: SimpleNamespace | None = None,
    layer_id: int = 0,
) -> torch.Tensor:
    sequence_count = len(base_lengths)
    token_count = sequence_count * width
    state_indices = torch.arange(sequence_count, dtype=torch.int64) * 2
    layer = SimpleNamespace(
        head_dim=1,
        num_heads_local=1,
        qkv_dim=1,
        conv_dim=3,
        conv_kernel_size=2,
        activation="identity",
        layer_id=layer_id,
        conv_weight_t=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    if metadata is None:
        metadata = SimpleNamespace(
            expanded_decode_metadata=None,
            kv_seq_lens=(torch.tensor(base_lengths)[:, None] + torch.arange(width)).flatten(),
        )

    def recurrent_contract(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        offsets = kwargs["cu_seqlens"].tolist()
        assert offsets == list(range(0, token_count + 1, width))
        slots = kwargs["ssm_state_indices"].tolist()
        accepted_counts = kwargs["num_accepted_tokens"].tolist()
        state = kwargs["initial_state"]
        output = torch.empty_like(value)
        for sequence, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            initial_slot = slots[start + accepted_counts[sequence] - 1]
            running_state = state[initial_slot].clone()
            for row in range(start, end):
                running_state = running_state + value[row].unsqueeze(-1)
                state[slots[row]].copy_(running_state)
                output[row].copy_(running_state.squeeze(-1))
        return output

    return backend._spec_verify(
        torch.ones(1, 3, token_count),
        torch.zeros(1, token_count, 1, 1),
        torch.ones(1, token_count, 1),
        layer.layer_id,
        layer.conv_weight_t,
        layer.activation,
        state_indices,
        metadata,
        conv_cache,
        ssm_cache,
        recurrent_contract,
    )


@pytest.mark.usefixtures("causal_conv1d_reference")
def test_kda_verify_reads_live_kv_lengths_on_every_call() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_cache = torch.zeros(4, 1, 3)
    ssm_cache = torch.zeros(4, 1, 1, 1)
    lengths = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int32)
    metadata = SimpleNamespace(expanded_decode_metadata=None, kv_seq_lens=lengths)
    device = torch.device("cpu")
    original_address = lengths.data_ptr()
    with forward_context(ForwardContext(backend, device, metadata, [])):
        _run_kda_verify_step(backend, 4, [10, 20], conv_cache, ssm_cache, metadata=metadata)
        lengths.view(2, 4).add_(torch.tensor([[3], [1]], dtype=torch.int32))
        assert lengths.data_ptr() == original_address
        output = _run_kda_verify_step(backend, 4, [13, 21], conv_cache, ssm_cache, metadata=metadata)
        torch.testing.assert_close(output.view(2, 4)[:, 0], torch.tensor([4.0, 2.0], dtype=output.dtype))
        torch.testing.assert_close(backend._kda_state[0]["kv_prev"][[0, 2]], torch.tensor([13, 21]))
    lengths.add_(1)
    with forward_context(ForwardContext(backend, device, metadata, [])):
        output = _run_kda_verify_step(backend, 4, [14, 22], conv_cache, ssm_cache, metadata=metadata)
        torch.testing.assert_close(output.view(2, 4)[:, 0], torch.tensor([5.0, 3.0], dtype=output.dtype))


@pytest.mark.usefixtures("causal_conv1d_reference")
def test_kda_verify_reads_expanded_lengths_and_sequence_layout() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_cache = torch.zeros(8, 1, 3)
    ssm_cache = torch.zeros(8, 1, 1, 1)
    expanded = SimpleNamespace(
        enabled=True,
        kv_seq_lens=torch.tensor([40, 41, 42, 43, 60, 61, 62, 63], dtype=torch.int32),
        block_table=torch.zeros(8, 1, dtype=torch.int32),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
    )
    metadata = SimpleNamespace(
        expanded_decode_metadata=expanded,
        kv_seq_lens=torch.tensor([1, 2]),
        slot_mapping=torch.arange(8),
    )
    device = torch.device("cpu")
    with forward_context(ForwardContext(backend, device, metadata, [])):
        _run_kda_verify_step(backend, 4, [40, 60], conv_cache, ssm_cache, metadata=metadata)
        torch.testing.assert_close(backend._kda_state[0]["kv_prev"][[0, 2]], torch.tensor([40, 60]))
        _run_kda_verify_step(backend, 2, [40, 42, 60, 62], conv_cache, ssm_cache, metadata=metadata)
        torch.testing.assert_close(backend._kda_state[0]["kv_prev"][[0, 2, 4, 6]], torch.tensor([40, 42, 60, 62]))
    expanded.kv_seq_lens.add_(4)
    with forward_context(ForwardContext(backend, device, metadata, [])):
        _run_kda_verify_step(backend, 4, [44, 64], conv_cache, ssm_cache, metadata=metadata)
        torch.testing.assert_close(backend._kda_state[0]["kv_prev"][[0, 2]], torch.tensor([44, 64]))


@pytest.mark.usefixtures("causal_conv1d_reference")
def test_kda_verify_selects_boundaries_across_plain_and_verify_steps() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_cache = torch.zeros(4, 1, 3)
    ssm_cache = torch.zeros(4, 1, 1, 1)
    _run_kda_verify_step(backend, 1, [10, 20], conv_cache, ssm_cache)
    _run_kda_verify_step(backend, 4, [11, 21], conv_cache, ssm_cache)
    output = _run_kda_verify_step(backend, 1, [14, 25], conv_cache, ssm_cache)
    torch.testing.assert_close(output.flatten(), torch.tensor([5.0, 6.0], dtype=output.dtype))
    torch.testing.assert_close(ssm_cache[[0, 2]].flatten(), torch.tensor([5.0, 6.0]))


@pytest.mark.usefixtures("causal_conv1d_reference")
def test_kda_shared_metadata_preserves_per_layer_accepted_boundaries() -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_verify_width = 4
    conv_caches = [torch.zeros(4, 1, 3), torch.zeros(4, 1, 3)]
    ssm_caches = [torch.zeros(4, 1, 1, 1), torch.zeros(4, 1, 1, 1)]
    for layer_id, base_length in enumerate((10, 12)):
        _run_kda_verify_step(backend, 4, [base_length], conv_caches[layer_id], ssm_caches[layer_id], layer_id=layer_id)
    metadata = SimpleNamespace(expanded_decode_metadata=None, kv_seq_lens=torch.arange(14, 18, dtype=torch.int32))
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        for layer_id, expected in enumerate((5.0, 3.0)):
            output = _run_kda_verify_step(
                backend, 4, [14], conv_caches[layer_id], ssm_caches[layer_id], metadata=metadata, layer_id=layer_id
            )
            torch.testing.assert_close(output[0, 0], torch.full_like(output[0, 0], expected))
