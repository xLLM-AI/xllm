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
        is_chunked_prefill=False,
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


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("leading_zero", [False, True])
def test_prepared_host_lengths_remain_private_across_tasks(is_mla: bool, leading_zero: bool) -> None:
    backend = _mla_backend() if is_mla else _ordinary_backend()
    metadata = _mla_metadata(prefill=True)
    snapshots = []
    lengths = [([1, 2], [6, 4]), ([2, 5], [8, 9]), ([1, 3], [5, 7])]

    for query_ends, kv_lengths in lengths:
        metadata.q_cu_seq_lens_host_values[:] = [0, *query_ends] if leading_zero else query_ends
        metadata.kv_seq_lens_host_values[:] = kv_lengths
        metadata.q_cu_seq_lens.copy_(torch.tensor(query_ends, dtype=torch.int32))
        metadata.q_seq_lens.copy_(torch.tensor([query_ends[0], query_ends[1] - query_ends[0]], dtype=torch.int32))
        metadata.kv_seq_lens.copy_(torch.tensor(kv_lengths, dtype=torch.int32))
        metadata.slot_mapping = torch.arange(query_ends[-1], dtype=torch.int32)
        prepared = backend.prepare_metadata(metadata)
        snapshots.append(prepared)
        metadata.prepared_attention_state = prepared
        backend.prepare(metadata)

    metadata.q_cu_seq_lens_host_values[:] = [-99]
    metadata.kv_seq_lens_host_values[:] = [-99]
    for prepared, (query_ends, kv_lengths) in zip(snapshots, lengths):
        assert prepared.query_ends == query_ends
        if is_mla:
            assert prepared.kv_lengths == kv_lengths
            assert prepared.max_query_len == max(query_ends[0], query_ends[1] - query_ends[0])
            assert prepared.max_seq_len == max(kv_lengths)
        else:
            assert prepared.actual_seq_q == query_ends
            assert prepared.actual_seq_kv == kv_lengths
            assert prepared.actual_seq_q is not prepared.query_ends


@pytest.mark.parametrize("width", [2, 3, 4, 17])
@pytest.mark.parametrize("expanded", [False, True])
def test_prepared_mtp_keeps_live_device_lengths(width: int, expanded: bool) -> None:
    backend = _ordinary_backend()
    rows = 2 * width
    lengths = torch.arange(3, rows + 3, dtype=torch.int32)
    table = torch.zeros(rows, 1, dtype=torch.int32)
    expanded_view = SimpleNamespace(
        enabled=expanded,
        block_table=table,
        kv_seq_lens=lengths,
        kv_seq_lens_host_values=[100] * rows,
    )
    metadata = SimpleNamespace(
        q_seq_lens=torch.full((2,), width, dtype=torch.int32) if expanded else torch.ones(rows, dtype=torch.int32),
        q_cu_seq_lens_host_values=[0, width, rows] if expanded else list(range(rows + 1)),
        block_table=torch.zeros(2, 1, dtype=torch.int32) if expanded else table,
        kv_seq_lens=torch.ones(2, dtype=torch.int32) if expanded else lengths,
        kv_seq_lens_host_values=[100] * (2 if expanded else rows),
        slot_mapping=torch.zeros(rows, dtype=torch.int32),
        expanded_decode_metadata=expanded_view,
        linear_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1, width], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=expanded,
        is_spec_verify=True,
        has_kv_shard=False,
    )
    prepared = backend.prepare_metadata(metadata, device_kv_lengths=True)
    assert prepared.device_kv_lengths is lengths
    assert prepared.block_table is table
    assert prepared.query_lengths == [1] * rows
    assert prepared.query_ends == list(range(1, rows + 1))
    assert prepared.kv_capacity == 128


@pytest.mark.parametrize("width", [1, 4])
def test_device_length_graph_skips_unused_tnd_workspace(width: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.model_executor.forward_context import AclGraphExecutionState

    backend = _ordinary_backend()
    metadata = _ordinary_metadata(paged=True)
    metadata.q_seq_lens.fill_(width)
    metadata.q_cu_seq_lens_host_values = [width, 2 * width]
    metadata.is_chunked_prefill = width > 1
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    execution = AclGraphExecutionState({}, {})
    backend._paged_graph_state = object()

    def reject_tnd_workspace(*args: object, **kwargs: object) -> None:
        raise AssertionError("device-length BNSD graphs must not allocate unused TND .out workspace")

    monkeypatch.setattr(backend, "_prepare_paged_graph", reject_tnd_workspace)
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [], execution_state=execution)):
        backend.prepare(metadata, graph_mode=True)

    assert backend._paged_graph_state is None
    assert backend._block_table_i32 is metadata.block_table
    assert execution.paged_attention == {}
    assert execution.persistent_buffers == {}


def test_prepared_device_decode_masks_newly_rejected_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _ordinary_backend()
    metadata = _ordinary_metadata(paged=True)
    metadata.is_chunked_prefill = False
    metadata.kv_seq_lens_host_values = [100, 100]
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    backend.prepare(metadata)
    seen: list[torch.Tensor] = []

    def attention(query: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        seen.append(kwargs["atten_mask"].clone())
        assert kwargs["actual_seq_lengths_kv"] == [128, 128]
        assert kwargs["actual_seq_lengths"] == [1, 1]
        return query, None

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", attention, raising=False)
    cache = backend._kv_caches[0].key
    query = torch.zeros(2, 8, 64)
    backend._decode(query, cache, cache, metadata, 2)
    metadata.kv_seq_lens.copy_(torch.tensor([2, 9], dtype=torch.int32))
    backend._decode(query, cache, cache, metadata, 2)
    expected_first = torch.arange(128).view(1, 1, 1, -1) >= torch.tensor([6, 4]).view(2, 1, 1, 1)
    expected_second = torch.arange(128).view(1, 1, 1, -1) >= torch.tensor([2, 9]).view(2, 1, 1, 1)
    torch.testing.assert_close(seen[0], expected_first)
    torch.testing.assert_close(seen[1], expected_second)


@pytest.mark.parametrize("prepared", [False, True])
def test_verify_routes_only_prepared_device_lengths_to_single_query_decode(prepared: bool) -> None:
    backend = _ordinary_backend()
    metadata = _ordinary_metadata(paged=True)
    metadata.is_chunked_prefill = metadata.is_spec_verify = True
    metadata.slot_mapping = torch.zeros(2, dtype=torch.int32)
    metadata.linear_state_indices = torch.ones(1, dtype=torch.int32)
    metadata.num_accepted_tokens = torch.ones(1, dtype=torch.int32)
    if prepared:
        metadata.expanded_decode_metadata = SimpleNamespace(
            enabled=True,
            block_table=metadata.block_table,
            kv_seq_lens=metadata.kv_seq_lens,
            kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
        )
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    backend.prepare(metadata)
    query = torch.zeros(2, 8 * 64)
    key = value = torch.zeros(2, 2 * 64)
    layer = SimpleNamespace(layer_id=0, causal=True)
    context = ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)
    with (
        forward_context(context),
        patch("xllm.python.attention.npu_paged_attention.kernels.reshape_paged_cache"),
        patch.object(backend, "_prefill", return_value=query) as prefill,
        patch.object(backend, "_decode", return_value=query) as decode,
    ):
        assert backend.execute(query, key, value, layer) is query
    assert decode.call_count == int(prepared)
    assert prefill.call_count == int(not prepared)


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


def test_mla_block_mask_tracks_device_lengths_and_keeps_noncausal_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _mla_backend()
    backend._mla_block_draft = True
    backend._use_fia_v2 = True
    backend._mla_actual_seq_q_host = [3]
    backend._mla_actual_seq_kv = torch.tensor([5], dtype=torch.int32)
    query = torch.zeros(3, 2, 4)
    query_rope = torch.zeros(3, 2, 2)
    cache = torch.zeros(2, 4, 1, 4)
    rope_cache = torch.zeros(2, 4, 1, 2)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    calls = []

    def fia(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object) -> tuple:
        calls.append(kwargs)
        return query, torch.empty(0)

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score_v2", fia)
    layer = SimpleNamespace(num_heads=2, scale=0.5)
    backend._mla_block_fia_v2(query, query_rope, cache, rope_cache, table, layer)
    mask = calls[-1]["atten_mask"]
    assert mask.shape == (1, 1, 3, 8)
    assert not mask[..., :5].any()
    assert mask[..., 5:].all()
    assert calls[-1]["sparse_mode"] == 0
    assert calls[-1]["actual_seq_kvlen"] == [8]
    # Every query can see the complete draft block, not only its causal prefix.
    assert torch.equal(mask[0, 0, 0], mask[0, 0, 2])
    backend._mla_actual_seq_kv.fill_(3)
    backend._block_attention_masks.clear()
    backend._mla_block_fia_v2(query, query_rope, cache, rope_cache, table, layer)
    assert calls[-1]["atten_mask"][..., 3:].all()


def test_prepared_mla_block_capture_borrows_live_lengths() -> None:
    backend = _mla_backend()
    backend._mla_block_draft = True
    metadata = _mla_metadata(chunked=True)
    metadata.q_cu_seq_lens = torch.tensor([0, 3, 6], dtype=torch.int32)
    metadata.q_cu_seq_lens_host_values = [0, 3, 6]
    metadata.q_seq_lens = torch.tensor([3, 3], dtype=torch.int32)
    metadata.slot_mapping = torch.arange(6, dtype=torch.int32)
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    backend.prepare(metadata, graph_mode=True)
    assert backend._mla_actual_seq_kv is metadata.kv_seq_lens
    assert backend._mla_actual_seq_q.tolist() == [3, 6]


@torch.inference_mode()
def test_mla_block_graph_matches_cpu_attention_after_prefix_changes() -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata

    device = torch.device("npu:0")
    width, heads, latent_dim, rope_dim, page_size = 8, 4, 512, 64, 128
    backend = NpuPagedAttentionBackend(
        heads, 1, latent_dim + rope_dim, 256**-0.5, 0, True, device, torch.bfloat16, mla_block_draft=True
    )
    torch.manual_seed(83)
    latent = torch.randn(4, page_size, 1, latent_dim, dtype=torch.bfloat16, device=device) * 0.1
    rope = torch.randn(4, page_size, 1, rope_dim, dtype=torch.bfloat16, device=device) * 0.1
    query = torch.randn(2 * width, heads, latent_dim, dtype=torch.bfloat16, device=device) * 0.1
    query_rope = torch.randn(2 * width, heads, rope_dim, dtype=torch.bfloat16, device=device) * 0.1
    backend.bind_kv_caches([LayerCache(latent, rope)])
    metadata = StaticGraphAttentionMetadata(
        slot_mapping=torch.full((2 * width,), -1, dtype=torch.int32, device=device),
        paged_kv_indptr=None,
        paged_kv_indices=None,
        paged_kv_last_page_len=None,
        block_table=torch.arange(4, dtype=torch.int32, device=device).view(2, 2),
        q_seq_lens=torch.full((2,), width, dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, width, 2 * width], dtype=torch.int32, device=device),
        q_cu_seq_lens_host_values=[width, 2 * width],
        kv_seq_lens=torch.tensor([23, 130], dtype=torch.int32, device=device),
        kv_seq_lens_host_values=[256, 256],
        is_chunked_prefill=True,
    )
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    layer = SimpleNamespace(num_heads=heads, scale=256**-0.5)

    def attend() -> torch.Tensor:
        backend.prepare(metadata, graph_mode=True)
        return backend._mla_block_fia_v2(query, query_rope, latent, rope, metadata.block_table, layer)

    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            attend()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = attend()
    torch.npu.synchronize()

    q_cpu, q_rope_cpu = query.cpu().float(), query_rope.cpu().float()
    latent_cpu, rope_cpu = latent.cpu().float(), rope.cpu().float()
    for lengths in ([187, 130], [5, 72], [201, 199]):
        metadata.kv_seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph.replay()
        torch.npu.synchronize()
        expected = []
        for sequence, length in enumerate(lengths):
            keys = latent_cpu[2 * sequence : 2 * sequence + 2].flatten(0, 1)[:length, 0]
            key_rope = rope_cpu[2 * sequence : 2 * sequence + 2].flatten(0, 1)[:length, 0]
            rows = slice(sequence * width, (sequence + 1) * width)
            scores = (
                torch.einsum("thd,sd->hts", q_cpu[rows], keys) + torch.einsum("thd,sd->hts", q_rope_cpu[rows], key_rope)
            ) * layer.scale
            expected.append(torch.einsum("hts,sd->thd", scores.softmax(-1), keys))
        torch.testing.assert_close(captured.cpu().float(), torch.cat(expected), rtol=1e-2, atol=2e-3)


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


def test_prepared_decode_updates_captured_lists_without_mutating_snapshots() -> None:
    from xllm.python.model_executor.forward_context import AclGraphExecutionState

    backend = _ordinary_backend()
    metadata = _mla_metadata()
    state = PagedAttentionGraphState(
        torch.empty(1), torch.empty(2, 8, 64), torch.empty(0), metadata.block_table, [], []
    )
    execution = AclGraphExecutionState({}, {2: state})
    captured_query = state.query
    captured_kv = state.kv
    snapshots = []
    lengths = [[6, 4], [8, 9], [5, 7]]

    for kv_lengths in lengths:
        metadata.kv_seq_lens_host_values[:] = kv_lengths
        metadata.kv_seq_lens.copy_(torch.tensor(kv_lengths, dtype=torch.int32))
        prepared = backend.prepare_metadata(metadata)
        snapshots.append(prepared)
        metadata.prepared_attention_state = prepared
        with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [], execution_state=execution)):
            backend.prepare(metadata, graph_mode=True)
        assert state.query is captured_query
        assert state.kv is captured_kv
        assert state.query == [1, 2]
        assert state.kv == kv_lengths
        assert backend._actual_seq_q is captured_query
        assert backend._actual_seq_kv is captured_kv

    for prepared, kv_lengths in zip(snapshots, lengths):
        assert prepared.query_ends == [1, 2]
        assert prepared.actual_seq_q == [1, 2]
        assert prepared.actual_seq_kv == kv_lengths
        assert prepared.actual_seq_q is not captured_query
        assert prepared.actual_seq_kv is not captured_kv


def test_recurrent_graph_host_bounds_refresh_without_device_readback_or_slot_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _ordinary_backend()
    views = [_ordinary_metadata(paged=True), _ordinary_metadata(paged=True)]
    states = []
    for metadata in views:
        metadata.is_chunked_prefill = False
        metadata.linear_verify_graph = object()
        metadata.block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [256, 256]
        metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
        states.append(metadata.prepared_attention_state)

    def reject_readback(self: torch.Tensor, *args: object, **kwargs: object) -> None:
        raise AssertionError("graph Host bounds must not read Device values")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_readback)
    monkeypatch.setattr(torch.Tensor, "item", reject_readback)
    monkeypatch.setattr(torch.Tensor, "tolist", reject_readback)
    first_values, second_values = states[0].actual_seq_kv, states[1].actual_seq_kv
    for first, second in (([130, 197], [63, 45]), ([255, 256], [9, 7]), ([6, 4], [128, 129])):
        views[0].kv_seq_lens_host_values[:] = first
        previous_second = second_values.copy()
        backend.prepare(views[0], graph_mode=True)
        assert first_values == first
        assert second_values == previous_second
        views[1].kv_seq_lens_host_values[:] = second
        backend.prepare(views[1], graph_mode=True)
        assert first_values == first
        assert second_values == second
        assert states[0].kv_capacity == states[1].kv_capacity == 256
        assert states[0].actual_seq_kv is first_values
        assert states[1].actual_seq_kv is second_values
        assert first_values is not second_values


@pytest.mark.parametrize("width", [1, 4, 8])
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
        causal=width == 1,
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


@pytest.mark.parametrize("window,block_causal", [(None, None), (64, None), (64, False), (64, True)])
@torch.inference_mode()
def test_block_graph_replay_matches_eager_after_kv_length_changes(
    window: int | None, block_causal: bool | None
) -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    from xllm.python.model_executor.runners.block_draft_acl_graph import BlockDraftAclGraphRunner

    device = torch.device("npu:0")
    width, heads, head_dim = 7, 4, 192
    page_size, capacity = 128, 4096
    backend = NpuPagedAttentionBackend(heads, heads, head_dim, head_dim**-0.5, 0, False, device, torch.bfloat16)
    cache = torch.zeros(34, page_size, heads, head_dim, device=device, dtype=torch.bfloat16)
    token_values = torch.arange(34 * page_size, device=device, dtype=torch.float32) / 2048
    values = token_values.view(34, page_size, 1, 1).expand_as(cache).contiguous().to(cache.dtype)
    backend.bind_kv_caches([LayerCache(key=cache, value=values)])
    query = torch.zeros(width, heads, head_dim, device=device, dtype=cache.dtype)
    tokens = torch.zeros(width, device=device, dtype=torch.int32)
    metadata = SimpleNamespace(
        q_seq_lens=torch.tensor([width], device=device, dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, width], device=device, dtype=torch.int32),
        q_cu_seq_lens_host_values=[width],
        block_table=torch.arange(1, 34, device=device, dtype=torch.int32).view(1, -1),
        slot_mapping=torch.arange(width, device=device, dtype=torch.int32),
        kv_seq_lens=torch.tensor([23], device=device, dtype=torch.int32),
        kv_seq_lens_host_values=[23],
        kv_cu_seq_lens=None,
        # The source can retain reserved pages beyond the live graph bucket.
        paged_kv_indptr=torch.tensor([0, 33], device=device, dtype=torch.int32),
        paged_kv_indices=torch.arange(1, 34, device=device, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([page_size], device=device, dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=False,
    )
    layer = SimpleNamespace(
        causal=bool(block_causal),
        fia_use_attention_mask=window is not None,
        fia_sparse_mode=0 if block_causal is not None else (4 if window is not None else None),
        fia_pre_tokens=window - 1 if window is not None else 2147483647,
        fia_next_tokens=width - 1,
        sliding_window=window if block_causal is not None else 0,
    )
    runner = BlockDraftAclGraphRunner(torch.nn.Identity(), backend, device, 1, capacity)
    entry = runner._allocate_entry(tokens, tokens, metadata, sequence_count=1, page_table_capacity=32)
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
    assert calls[0]["block_table"] is calls[1]["block_table"]
    assert calls[2]["block_table"] is calls[3]["block_table"]
    assert calls[0]["block_table"] is not calls[2]["block_table"]
    torch.testing.assert_close(calls[0]["block_table"], torch.tensor([[0], [1]], dtype=torch.int32))
    torch.testing.assert_close(calls[2]["block_table"], torch.tensor([[0, 0], [1, 1]], dtype=torch.int32))
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


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("causal", [False, True])
def test_block_mask_matches_speculators_visibility(width: int, causal: bool) -> None:
    anchors = [0, 2047, 2048, 2049, 8192]
    capacity = max(anchors) + width + 7
    lengths = torch.tensor([anchor + width for anchor in anchors], dtype=torch.int32)
    prepared = SimpleNamespace(device_kv_lengths=lengths, query_lengths=[width] * len(anchors), kv_capacity=capacity)
    layer = SimpleNamespace(sliding_window=2048, causal=causal, fia_sparse_mode=0)
    mask = _ordinary_backend()._block_attention_mask(prepared, layer)
    expected = torch.ones(len(anchors), 1, width, capacity, dtype=torch.bool)
    for batch, anchor in enumerate(anchors):
        expected[batch, 0, :, max(0, anchor - 2048) : anchor] = False
        for row in range(width):
            block_end = anchor + row + 1 if causal else anchor + width
            expected[batch, 0, row, anchor:block_end] = False
    torch.testing.assert_close(mask, expected)
    assert mask.is_contiguous()


def test_windowed_draft_masks_are_cached_by_causality_and_refreshed_each_forward() -> None:
    backend = _ordinary_backend()
    metadata = _ordinary_metadata(paged=True)
    metadata.q_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens_host_values = [2, 4]
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    prepared = metadata.prepared_attention_state
    causal = SimpleNamespace(sliding_window=2, causal=True, fia_sparse_mode=0)
    non_causal = SimpleNamespace(sliding_window=2, causal=False, fia_sparse_mode=0)
    backend.prepare(metadata)
    first = backend._block_attention_mask(prepared, causal)
    assert backend._block_attention_mask(prepared, causal) is first
    assert backend._block_attention_mask(prepared, non_causal) is not first
    for row in range(2):
        assert torch.where(~first[0, 0, row])[0].tolist() == list(range(2, 5 + row))

    metadata.kv_seq_lens.copy_(torch.tensor([4, 3], dtype=torch.int32))
    backend.prepare(metadata)
    after = backend._block_attention_mask(prepared, causal)
    assert after is not first
    assert torch.where(~after[0, 0, 0])[0].tolist() == [0, 1, 2]
    assert after[0, :, :, 4:].all()


@pytest.mark.parametrize("prepared_lengths", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_windowed_draft_uses_explicit_mask_in_eager_and_prepared_paths(
    prepared_lengths: bool, causal: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _ordinary_backend()
    metadata = _ordinary_metadata(paged=True)
    metadata.q_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens_host_values = [2, 4]
    metadata.block_table = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    metadata.is_chunked_prefill = True
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=prepared_lengths)
    metadata.kv_seq_lens.copy_(torch.tensor([5, 3], dtype=torch.int32))
    backend.prepare(metadata)
    calls: list[dict] = []

    def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object) -> tuple:
        calls.append(kwargs)
        return query.clone(), torch.empty(0)

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", _attention)
    query = torch.zeros(4, 8, 64)
    cache = torch.zeros(4, 128, 2, 64)
    layer = SimpleNamespace(
        causal=causal,
        fia_use_attention_mask=False,
        fia_sparse_mode=0,
        sliding_window=2,
    )
    with patch.object(backend, "prepare_metadata", wraps=backend.prepare_metadata) as prepare:
        for _ in range(2):
            backend._prefill(query, query, query, cache, cache, metadata, 4, layer)
        assert prepare.call_count == (0 if prepared_lengths else 1)
        backend.prepare(metadata)
        backend._prefill(query, query, query, cache, cache, metadata, 4, layer)
        assert prepare.call_count == (0 if prepared_lengths else 2)
    assert calls[0]["input_layout"] == "BNSD"
    mask = calls[0]["atten_mask"]
    assert torch.where(~mask[0, 0, 0])[0].tolist() == ([1, 2, 3] if causal else [1, 2, 3, 4])
    assert torch.where(~mask[0, 0, 1])[0].tolist() == [1, 2, 3, 4]
    assert torch.where(~mask[1, 0, 0])[0].tolist() == ([0, 1] if causal else [0, 1, 2])
    assert mask[0, :, :, 5:].all()
    assert mask[1, :, :, 3:].all()


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
def test_kda_verify_uses_framework_native_conv(
    monkeypatch: pytest.MonkeyPatch, value_dtype: torch.dtype, kernel_width: int
) -> None:
    from xllm.python import kernels

    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    backend._kda_checkpoint_stride = 2
    backend._kda_query_start_locs = {}
    backend._kda_checkpoint_offsets = {}
    indices = torch.tensor([1])
    accepted = torch.tensor([2], dtype=torch.int32)
    metadata = SimpleNamespace(num_accepted_tokens=accepted)
    conv = torch.zeros(2, kernel_width, 3, dtype=value_dtype)
    ssm = torch.zeros(4, 1, 1, 1)
    calls = []

    def native_conv(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        state: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        width: int,
        accepted_tokens: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        assert weight.shape == (3, kernel_width)
        assert weight.dtype == value_dtype
        assert state is conv
        assert state_indices.tolist() == [1]
        assert query_start_loc.tolist() == [0, 2]
        assert width == 2
        assert accepted_tokens is accepted
        assert kwargs["activation"] is True
        calls.append(inputs)
        return inputs

    def recurrent(query: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        assert kwargs["initial_state"] is ssm
        assert kwargs["num_accepted_tokens"] is accepted
        assert kwargs["ssm_state_indices"].tolist() == [[2, 3]]
        return query

    monkeypatch.setattr(kernels, "causal_conv1d_update_v2", native_conv)
    for kv_length in (1, 50):
        metadata.kv_seq_lens = torch.tensor([kv_length])
        output = backend._spec_verify(
            torch.ones(1, 3, 2, dtype=value_dtype),
            torch.zeros(1, 2, 1, 1),
            torch.ones(1, 2, 1),
            torch.ones(kernel_width, 3, dtype=value_dtype),
            "silu",
            indices,
            metadata,
            conv,
            ssm,
            recurrent,
        )
        assert torch.isfinite(output).all()
    assert len(calls) == 2
