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

"""Call-contract tests for the DeepSeek-V4 CSA/HCA attention backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xllm.python import kernels
from xllm.python.attention import csa_attention as csa_attention_module
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa_attention import (
    COMPRESSED_SPARSE_ATTENTION,
    HEAVILY_COMPRESSED_ATTENTION,
    SLIDING_ATTENTION,
    CsaAttentionBackend,
    _attention_type_for_compress_ratio,
    _CompressedAttentionCacheMapping,
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.attention.dsa_metadata import build_cache_specs
from xllm.python.model_executor.v4_cp_context import (
    build_deepseek_v4_cp_context,
)


def _make_backend() -> CsaAttentionBackend:
    compress_ratios = [0, 4, 128]
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, 3)
    return CsaAttentionBackend(
        compress_ratios=compress_ratios,
        window_size=128,
        n_layers=3,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )


def test_compress_ratio_per_layer() -> None:
    b = _make_backend()
    assert b._layer_compress_ratio(0) == 1
    assert b._layer_compress_ratio(1) == 4
    assert b._layer_compress_ratio(2) == 128


@pytest.mark.parametrize(
    ("compress_ratio", "attention_type"),
    [
        (1, SLIDING_ATTENTION),
        (4, COMPRESSED_SPARSE_ATTENTION),
        (128, HEAVILY_COMPRESSED_ATTENTION),
    ],
)
def test_official_attention_type_names(compress_ratio: int, attention_type: str) -> None:
    assert _attention_type_for_compress_ratio(compress_ratio) == attention_type


def test_unsupported_compression_ratio_has_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported DeepSeek-V4 compression ratio"):
        _attention_type_for_compress_ratio(8)


def test_resolve_cache_mapping_csa() -> None:
    """CSA layer: compressed/index caches plus sliding-window state caches."""
    b = _make_backend()
    m = b._resolve_cache_mapping(1, 4)
    assert m.cmp_cache_idx == 0
    assert m.index_cache_idx == 1
    assert m.indexer_scale_cache_idx == 7
    assert m.ori_cache_idx == 2
    assert m.kv_state_cache_idx == 3
    assert m.score_state_cache_idx == 4
    assert m.index_kv_state_cache_idx == 5
    assert m.index_score_state_cache_idx == 6


def test_resolve_cache_mapping_hca() -> None:
    """HCA layer: compressed cache and sliding-window state caches."""
    b = _make_backend()
    m = b._resolve_cache_mapping(2, 128)
    assert m.cmp_cache_idx == 0
    assert m.ori_cache_idx == 1
    assert m.kv_state_cache_idx == 2
    assert m.score_state_cache_idx == 3
    assert m.index_cache_idx == -1  # HCA has no indexer
    assert m.indexer_scale_cache_idx == -1


def test_resolve_cache_mapping_sliding_attention() -> None:
    """Sliding-attention layer: one SWA cache and no compressed cache."""
    b = _make_backend()
    m = b._resolve_cache_mapping(0, 1)
    assert m.cmp_cache_idx == -1
    assert m.ori_cache_idx == 0
    assert m.index_cache_idx == -1


def test_get_layer_cache_tensor_bounds() -> None:
    tensors = [[torch.empty(0)], [torch.zeros(2), torch.zeros(3)]]
    assert _get_layer_cache_tensor(tensors, 0, 0).numel() == 0
    assert _get_layer_cache_tensor(tensors, 1, 1).numel() == 3
    assert _get_layer_cache_tensor(tensors, 5, 0) is None  # bad layer
    assert _get_layer_cache_tensor(tensors, 1, 9) is None  # bad cache idx


def test_scatter_by_slot_writes_rows() -> None:
    cache = torch.zeros(4, 3, dtype=torch.float32)
    # slot 0 -> row 0, slot 2 -> row 2, slot -1 skipped.
    slots = torch.tensor([0, -1, 2], dtype=torch.int32)
    value = torch.tensor([[1.0, 1.0, 1.0], [9.0, 9.0, 9.0], [2.0, 2.0, 2.0]])
    _scatter_by_slot(cache, slots, value)
    assert torch.equal(cache[0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(cache[2], torch.tensor([2.0, 2.0, 2.0]))
    # Row 1 untouched (slot -1 skipped); value row 1 dropped.
    assert torch.equal(cache[1], torch.zeros(3))


def test_scatter_by_slot_ignores_all_padded_rows() -> None:
    cache = torch.arange(12, dtype=torch.float32).view(4, 3)
    original = cache.clone()
    slots = torch.full((2,), -1, dtype=torch.int32)
    values = torch.full((2, 3), 99.0)

    _scatter_by_slot(cache, slots, values)

    assert torch.equal(cache, original)


def test_default_mapping_is_empty() -> None:
    m = _CompressedAttentionCacheMapping()
    assert m.cmp_cache_idx == -1
    assert m.ori_cache_idx == -1


def test_prepare_binds_compressed_metadata_to_current_forward(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda compressed_metadata: None)
    monkeypatch.setattr(
        backend,
        "_build_precomputed_metadata",
        lambda compressed_metadata: None,
    )

    def make_metadata(kv_len: int, is_prefill: bool) -> SimpleNamespace:
        q_len = kv_len if is_prefill else 1
        return SimpleNamespace(
            multi_block_tables=[],
            kv_seq_lens_host=torch.tensor([kv_len], dtype=torch.int32),
            q_seq_lens_host=torch.tensor([q_len], dtype=torch.int32),
            is_prefill=is_prefill,
            is_chunked_prefill=False,
            dsa_metadata=None,
            dsa_positions=None,
            dsa_cos_sin=None,
            dsa_c4_cos_sin=None,
            dsa_c128_cos_sin=None,
            dsa_graph_block_table_cols=0,
            dsa_graph_mode=False,
        )

    prefill = make_metadata(84, True)
    backend.prepare(prefill)
    backend.prepare_dsa_metadata_for_forward()
    prefill_compressed_metadata = prefill.dsa_metadata
    assert prefill_compressed_metadata.max_query_len == 84

    decode = make_metadata(85, False)
    backend.prepare(decode)
    backend.prepare_dsa_metadata_for_forward()
    assert decode.dsa_metadata is not prefill_compressed_metadata
    assert decode.dsa_metadata.max_query_len == 1
    assert prefill.dsa_metadata is prefill_compressed_metadata


def test_prepare_clamps_target_block_tables_to_draft_groups(monkeypatch) -> None:
    """An MTP draft backend consumes only its registered SWA manager table."""
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[
            torch.tensor([[10]], dtype=torch.int32),
            torch.tensor([[20]], dtype=torch.int32),
            torch.tensor([[30]], dtype=torch.int32),
        ],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        new_cache_slots_host_values=[1280],
        is_prefill=True,
        is_chunked_prefill=False,
        is_dummy=False,
        dsa_positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_graph_block_table_cols=0,
        dsa_graph_mode=False,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert dsa.block_tables[0][0].tolist() == [[10]]
    assert dsa.slot_mappings[0][0].tolist() == [1280]


@pytest.mark.parametrize(
    ("index_topk", "expected_c4_columns", "expected_c128_slot"),
    [
        (512, 1, 3),
        (1024, 2, 7),
        (2048, 4, 15),
    ],
)
def test_empty_dp_metadata_uses_safe_lengths_and_cache_tables(
    monkeypatch,
    index_topk: int,
    expected_c4_columns: int,
    expected_c128_slot: int,
) -> None:
    backend = _make_backend()
    backend.index_topk = index_topk
    backend.bind_kv_caches(
        [
            LayerCache(key=None, value=None, swa=torch.empty((2, 128, 1, 1))),
            LayerCache(
                key=torch.empty((4, 128, 1, 1)),
                value=None,
                swa=torch.empty((2, 128, 1, 1)),
            ),
            LayerCache(
                key=torch.empty((1, 128, 1, 1)),
                value=None,
                swa=torch.empty((2, 128, 1, 1)),
            ),
        ]
    )
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=True,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_metadata=None,
        dsa_positions=torch.tensor([0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=False,
        dsa_graph_block_table_cols=0,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert dsa.seq_lens.tolist() == [index_topk]
    assert dsa.seq_lens_q.tolist() == [1]
    assert dsa.max_seq_len == index_topk
    assert dsa.max_query_len == 1
    assert tuple(dsa.block_tables[1][0].shape) == (1, expected_c4_columns)
    assert dsa.slot_mappings[0][0].tolist() == [127]
    assert dsa.slot_mappings[1][0].tolist() == [127]
    assert dsa.slot_mappings[2][0].tolist() == [expected_c128_slot]


@pytest.mark.parametrize("index_topk", [512, 1024, 2048])
def test_empty_dp_graph_metadata_preserves_bucket_rows(monkeypatch, index_topk: int) -> None:
    backend = _make_backend()
    backend.index_topk = index_topk
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    graph_block_table_cols = max(8, index_topk // 128)
    metadata = SimpleNamespace(
        multi_block_tables=[
            torch.full((4, graph_block_table_cols), -1, dtype=torch.int32),
            torch.full((4, graph_block_table_cols), -1, dtype=torch.int32),
            torch.full((4, graph_block_table_cols), -1, dtype=torch.int32),
        ],
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[1, 0, 0, 0],
        q_seq_lens_host=None,
        q_seq_lens=torch.ones(4, dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=True,
        dsa_metadata=None,
        dsa_positions=torch.zeros(4, dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=True,
        dsa_graph_block_table_cols=graph_block_table_cols,
    )
    dummy_tables = backend._build_empty_dp_block_tables(
        list(metadata.multi_block_tables),
        4,
        index_topk,
        torch.device("cpu"),
        graph_mode=True,
        graph_block_table_capacity_cols=graph_block_table_cols,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert tuple(dummy_tables[0].shape) == (4, 1)
    assert dsa.seq_lens.tolist() == [index_topk] * 4
    assert dsa.seq_lens_q.tolist() == [1, 1, 1, 1]
    assert tuple(dsa.block_tables[0][0].shape) == (4, graph_block_table_cols)
    expected_swa_column = index_topk // 128 - 1
    assert dsa.block_tables[0][0][:, expected_swa_column].tolist() == [0, 0, 0, 0]
    assert tuple(dsa.block_tables[1][0].shape) == (4, graph_block_table_cols)
    assert dsa.block_tables[1][0][:, 0].tolist() == [0, 0, 0, 0]
    assert dsa.slot_mappings[0][0][:4].tolist() == [127, 127, 127, 127]
    assert dsa.slot_mappings[1][0][:4].tolist() == [127, 127, 127, 127]
    expected_c128_slot = index_topk // 128 - 1
    assert dsa.slot_mappings[2][0][:4].tolist() == [expected_c128_slot] * 4


def test_empty_dp_graph_rejects_insufficient_swa_capacity() -> None:
    backend = _make_backend()
    block_tables = [torch.full((4, 8), -1, dtype=torch.int32) for _ in range(3)]

    with pytest.raises(ValueError, match="SWA block table capacity is too small"):
        backend._build_empty_dp_block_tables(
            block_tables,
            4,
            2048,
            torch.device("cpu"),
            graph_mode=True,
            graph_block_table_capacity_cols=8,
        )


def test_prepare_clears_previous_forward_state() -> None:
    backend = _make_backend()
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_metadata=object(),
        dsa_positions=torch.ones(1),
        dsa_cos_sin=torch.ones(1),
        dsa_c4_cos_sin=torch.ones(1),
        dsa_c128_cos_sin=torch.ones(1),
        dsa_graph_mode=True,
    )
    backend.attach_compressor(lambda *_args: None)
    backend.attach_indexer(lambda *_args: None)
    backend.reset_forward(metadata)
    assert metadata.dsa_metadata is None
    assert metadata.dsa_positions is None
    assert metadata.dsa_cos_sin is None
    assert metadata.dsa_c4_cos_sin is None
    assert metadata.dsa_c128_cos_sin is None
    assert metadata.dsa_graph_mode is False
    assert not hasattr(backend, "_compressor_fn")
    assert not hasattr(backend, "_indexer_fn")


def test_dsa_api_aliases_are_equivalent(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_metadata=None,
        dsa_positions=torch.tensor([0]),
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_mode=False,
    )
    backend.prepare_dsa_metadata_for_forward(metadata)
    canonical = metadata.dsa_metadata
    metadata.dsa_metadata = None
    backend.prepare_csa_metadata_for_forward(metadata)
    assert metadata.dsa_metadata is not canonical


def test_graph_mode_is_recorded_on_metadata() -> None:
    backend = _make_backend()
    metadata = SimpleNamespace()
    backend.prepare(metadata, graph_mode=True)
    assert metadata.dsa_graph_mode is True


def test_decode_precomputed_metadata_matches_cpp_contract(monkeypatch) -> None:
    backend = _make_backend()
    sparse_calls: list[dict] = []
    qli_calls: list[dict] = []

    def fake_sparse_metadata(**kwargs):
        sparse_calls.append(kwargs)
        return torch.tensor([kwargs["cmp_ratio"]], dtype=torch.int32)

    def fake_qli_metadata(**kwargs):
        qli_calls.append(kwargs)
        return torch.tensor([4], dtype=torch.int32)

    monkeypatch.setattr(
        kernels,
        "sparse_attn_sharedkv_metadata",
        fake_sparse_metadata,
        raising=False,
    )
    monkeypatch.setattr(
        kernels,
        "quant_lightning_indexer_metadata",
        fake_qli_metadata,
        raising=False,
    )
    compressed_metadata = SimpleNamespace(
        actual_seq_lengths_query=torch.tensor([0, 1], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([85], dtype=torch.int32),
        seq_lens_q=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([85], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=85,
    )

    backend._build_precomputed_metadata(compressed_metadata)

    assert [call["cmp_ratio"] for call in sparse_calls] == [1, 4, 128]
    assert all(call["head_dim"] == 512 for call in sparse_calls)
    assert all(call["cu_seqlens_q"].tolist() == [0, 1] for call in sparse_calls)
    assert all(call["cu_seqlens_ori_kv"].numel() == 0 for call in sparse_calls)
    assert sparse_calls[1]["cmp_topk"] == 512
    assert qli_calls[0]["actual_seq_lengths_query"].tolist() == [1]
    assert qli_calls[0]["actual_seq_lengths_key"].tolist() == [85]
    assert qli_calls[0]["head_dim"] == 128
    assert compressed_metadata.precomputed_metadata_inputs[0] is compressed_metadata.actual_seq_lengths_query


def test_csa_execute_requires_model_compressor(monkeypatch) -> None:
    backend = _make_backend()
    empty_cache = LayerCache(key=None, value=None)
    cmp_cache = torch.zeros(2, 128, 1, 512)
    swa_cache = torch.zeros(2, 128, 1, 512)
    backend.bind_kv_caches(
        [
            empty_cache,
            LayerCache(key=cmp_cache, value=None, swa=swa_cache),
            empty_cache,
        ]
    )
    block_tables = [[], [torch.tensor([[1]], dtype=torch.int32) for _ in range(8)], []]
    slot_mappings = [[], [torch.tensor([128], dtype=torch.int32) for _ in range(8)], []]
    compressed_metadata = SimpleNamespace(
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        actual_seq_lengths_query=torch.tensor([0, 1], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([1], dtype=torch.int32),
        input_positions=torch.tensor([0], dtype=torch.int64),
        cos_table=torch.zeros(1, 64),
        sin_table=torch.zeros(1, 64),
        c4_cos=torch.zeros(1, 64),
        c4_sin=torch.zeros(1, 64),
        c128_cos=torch.zeros(1, 64),
        c128_sin=torch.zeros(1, 64),
        c4_metadata=torch.zeros(1, dtype=torch.int32),
    )
    backend._metadata = SimpleNamespace(
        dsa_metadata=compressed_metadata,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with pytest.raises(RuntimeError, match="compressor is required"):
        backend.execute(
            torch.zeros(1, 8, 512),
            torch.zeros(1, 1, 512),
            torch.zeros(1, 1, 512),
            SimpleNamespace(layer_id=1, attn_sink=None),
        )


def test_forward_rope_state_is_owned_by_each_metadata(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda compressed_metadata: None)
    monkeypatch.setattr(
        backend,
        "_build_precomputed_metadata",
        lambda compressed_metadata: None,
    )

    def make_metadata(kv_len: int, q_len: int) -> SimpleNamespace:
        return SimpleNamespace(
            multi_block_tables=[],
            kv_seq_lens_host=torch.tensor([kv_len], dtype=torch.int32),
            q_seq_lens_host=torch.tensor([q_len], dtype=torch.int32),
            is_prefill=q_len > 1,
            is_chunked_prefill=False,
            dsa_metadata=None,
            dsa_positions=None,
            dsa_cos_sin=None,
            dsa_c4_cos_sin=None,
            dsa_c128_cos_sin=None,
            dsa_graph_block_table_cols=0,
            dsa_graph_mode=False,
        )

    rope_cache = torch.arange(256 * 8, dtype=torch.float32).view(256, 8)
    prefill = make_metadata(84, 84)
    backend.prepare(prefill)
    backend.attach_rope_tables(
        torch.arange(84),
        rope_cache,
        csa_cos_sin=rope_cache,
        hca_cos_sin=rope_cache,
        metadata=prefill,
    )

    decode = make_metadata(85, 1)
    backend.prepare(decode)
    backend.attach_rope_tables(
        torch.tensor([84]),
        rope_cache,
        csa_cos_sin=rope_cache,
        hca_cos_sin=rope_cache,
        metadata=decode,
    )

    backend.prepare_csa_metadata_for_forward(prefill)

    assert prefill.dsa_metadata.input_positions.numel() == 84
    cos, sin = prefill.dsa_metadata.input_rope_by_ratio[1]
    assert torch.equal(cos, rope_cache[:84, :4])
    assert torch.equal(sin, rope_cache[:84, 4:])
    assert cos.is_contiguous() and sin.is_contiguous()
    assert decode.dsa_positions.numel() == 1
    assert prefill.dsa_positions.data_ptr() != decode.dsa_positions.data_ptr()


def test_cp_localization_keeps_runtime_metadata_read_only(monkeypatch) -> None:
    backend = _make_backend()
    dsa = backend._builder.build(
        multi_block_tables=[],
        kv_seq_lens=[8],
        q_seq_lens=[4],
        positions=torch.arange(4, dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=False,
    )

    class _ReadOnlyMetadata:
        def __init__(self) -> None:
            self.dsa_metadata = dsa
            self.dp_execution_token_counts = (4,)

        @property
        def q_seq_lens_host(self) -> torch.Tensor:
            return torch.tensor([4], dtype=torch.int32)

        @property
        def kv_seq_lens_host(self) -> torch.Tensor:
            return torch.tensor([8], dtype=torch.int32)

        @property
        def max_query_len(self) -> int:
            return 4

        @property
        def max_seq_len(self) -> int:
            return 8

    captured = []

    def capture_precomputed(**kwargs):
        captured.append(kwargs)
        return torch.empty(0, dtype=torch.int32)

    monkeypatch.setattr(
        backend,
        "_build_dsa_rope_metadata",
        lambda *_args: {1: (torch.zeros(2, 1), torch.zeros(2, 1))},
    )
    monkeypatch.setattr(kernels, "sparse_attn_sharedkv_metadata", capture_precomputed)
    monkeypatch.setattr(kernels, "quant_lightning_indexer_metadata", lambda **_kwargs: torch.empty(0))
    metadata = _ReadOnlyMetadata()
    cp_context = build_deepseek_v4_cp_context(
        2,
        0,
        [4],
        [8],
        torch.arange(4, dtype=torch.int64),
    )

    backend.localize_dsa_metadata_for_cp(cp_context, metadata)

    assert dsa.seq_lens_q.tolist() == [2]
    assert dsa.seq_lens.tolist() == [6]
    assert len(captured) == 3
    for inputs in captured:
        assert inputs["max_seqlen_q"] == 2
        assert inputs["max_seqlen_kv"] == 6
        assert inputs["cu_seqlens_q"].tolist() == [0, 2]
        assert inputs["cu_seqlens_ori_kv"].tolist() == [0, 6]
    assert metadata.q_seq_lens_host.tolist() == [4]
    assert metadata.kv_seq_lens_host.tolist() == [8]


def test_graph_dsa_refresh_preserves_tensor_addresses() -> None:
    def make_metadata(value: int, seq_rows: int) -> SimpleNamespace:
        return SimpleNamespace(
            seq_lens=torch.full((seq_rows,), value, dtype=torch.int32),
            block_tables=[[torch.full((seq_rows, 2), value, dtype=torch.int32)]],
            slot_mappings=[[torch.full((seq_rows,), value, dtype=torch.int32)]],
            input_rope_by_ratio={
                1: (
                    torch.full((2, 2), value, dtype=torch.float32),
                    torch.full((2, 2), value, dtype=torch.float32),
                )
            },
            max_query_len=value,
            max_seq_len=value,
            is_acl_graph=True,
            precomputed_metadata_inputs=(),
        )

    persistent = make_metadata(1, 4)
    refreshed = make_metadata(7, 2)
    seq_lens_ptr = persistent.seq_lens.data_ptr()
    block_table_ptr = persistent.block_tables[0][0].data_ptr()
    rope_ptr = persistent.input_rope_by_ratio[1][0].data_ptr()

    CsaAttentionBackend._copy_graph_dsa_metadata(persistent, refreshed)

    assert persistent.seq_lens.data_ptr() == seq_lens_ptr
    assert persistent.seq_lens.tolist() == [7, 7, 0, 0]
    assert persistent.block_tables[0][0].data_ptr() == block_table_ptr
    assert persistent.block_tables[0][0].tolist() == [[7, 7], [7, 7], [-1, -1], [-1, -1]]
    assert persistent.slot_mappings[0][0].tolist() == [7, 7, -1, -1]
    assert persistent.input_rope_by_ratio[1][0].data_ptr() == rope_ptr
    assert persistent.input_rope_by_ratio[1][0].tolist() == [[7.0, 7.0], [7.0, 7.0]]
    assert persistent.max_query_len == 7
    assert persistent.max_seq_len == 7


def test_graph_dsa_build_uses_stable_host_length_values(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_move_metadata_to_device", lambda _metadata: None)
    monkeypatch.setattr(backend, "_build_precomputed_metadata", lambda *_args: None)
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=[9, 1],
        q_seq_lens_host=None,
        q_seq_lens=None,
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_positions=torch.tensor([8, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        dsa_graph_mode=True,
        dsa_graph_block_table_cols=4,
    )

    dsa = backend._build_dsa_metadata_for_forward(metadata)

    assert dsa.seq_lens.tolist() == [9, 1]
    assert dsa.seq_lens_q.tolist() == [1, 1]
    assert dsa.start_pos.tolist() == [8, 0]


def test_prefill_persists_swa_for_decode_and_omits_ori_kv_cu_seqlens(
    monkeypatch,
) -> None:
    backend = CsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        index_topk=512,
        index_n_heads=64,
        index_head_dim=128,
        rope_head_dim=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    swa = torch.zeros(2, 128, 1, 512, dtype=torch.float32)
    backend.bind_kv_caches([LayerCache(key=None, value=None, swa=swa)])
    block_table = torch.tensor([[1]], dtype=torch.int32)
    layer = SimpleNamespace(
        layer_id=0,
        attn_sink=torch.tensor([1.0], dtype=torch.float32),
        attn_sink_loaded=False,
    )
    calls: list[dict] = []

    def fake_sparse_attn(**kwargs):
        calls.append(kwargs)
        return kwargs["q"].clone(), torch.empty(0)

    monkeypatch.setattr(csa_attention_module, "_sparse_attn_sharedkv", fake_sparse_attn)

    def prepare_step(kv_len: int, q_len: int, is_prefill: bool):
        compressed_metadata = backend._builder.build(
            multi_block_tables=[block_table],
            kv_seq_lens=[kv_len],
            q_seq_lens=[q_len],
            positions=torch.arange(kv_len - q_len, kv_len, dtype=torch.int64),
            is_prefill=is_prefill,
            is_chunked_prefill=False,
        )
        compressed_metadata.c1_metadata = torch.zeros(1, dtype=torch.int32)
        backend._metadata = SimpleNamespace(
            dsa_metadata=compressed_metadata,
            is_prefill=is_prefill,
            is_chunked_prefill=False,
        )
        return compressed_metadata

    prepare_step(kv_len=2, q_len=2, is_prefill=True)
    prefill_kv = torch.arange(2 * 512, dtype=torch.float32).view(2, 1, 512)
    backend.execute(torch.zeros(2, 8, 512), prefill_kv, prefill_kv, layer)
    assert torch.equal(swa[1, :2], prefill_kv)

    prepare_step(kv_len=3, q_len=1, is_prefill=False)
    decode_kv = torch.full((1, 1, 512), 7.0)
    backend.execute(torch.zeros(1, 8, 512), decode_kv, decode_kv, layer)
    assert torch.equal(swa[1, 2], decode_kv[0])
    assert calls[-1]["cu_seqlens_ori_kv"] is None
    assert calls[-1]["sinks"] is None

    layer.attn_sink_loaded = True
    prepare_step(kv_len=3, q_len=1, is_prefill=False)
    backend.execute(torch.zeros(1, 8, 512), decode_kv, decode_kv, layer)
    assert torch.equal(calls[-1]["sinks"], layer.attn_sink)
