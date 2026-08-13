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

"""Unit tests for the DSA attention backend (cache-mapping + slot scatter).

Pure-Python: does not load compiled operators.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from xllm.python.attention import dsa_attention as dsa_attention_module
from xllm.python.attention.dsa_attention import (
    DsaAttentionBackend,
    _DsaCacheMapping,
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.dsa_metadata import build_cache_specs


def _make_backend() -> DsaAttentionBackend:
    compress_ratios = [0, 4, 128]
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, 3)
    return DsaAttentionBackend(
        compress_ratios=compress_ratios,
        window_size=128,
        n_layers=3,
        num_heads=8,
        attn_head_dim=512,
        metadata_head_dim=1088,
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


def test_resolve_cache_mapping_c4() -> None:
    """C4 layer: cmp/index/indexer_scale (TOKEN, idx 0/1/7), ori..index_score (SWA, 2..6)."""
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


def test_resolve_cache_mapping_c128() -> None:
    """C128 layer: cmp (TOKEN, idx 0), ori..index_score (SWA, 1..3)."""
    b = _make_backend()
    m = b._resolve_cache_mapping(2, 128)
    assert m.cmp_cache_idx == 0
    assert m.ori_cache_idx == 1
    assert m.kv_state_cache_idx == 2
    assert m.score_state_cache_idx == 3
    assert m.index_cache_idx == -1  # C128 has no indexer
    assert m.indexer_scale_cache_idx == -1


def test_resolve_cache_mapping_c1() -> None:
    """C1 layer: only one SWA cache; compress_ratio==1 -> no cmp."""
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
    m = _DsaCacheMapping()
    assert m.cmp_cache_idx == -1
    assert m.ori_cache_idx == -1


def test_prepare_binds_dsa_metadata_to_current_forward(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_pack_metadata_to_device", lambda dsa: None)
    monkeypatch.setattr(
        backend, "_build_precomputed_metadata", lambda dsa, metadata: None
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
    prefill_dsa = prefill.dsa_metadata
    assert prefill_dsa is backend._dsa_metadata
    assert prefill_dsa.max_query_len == 84

    decode = make_metadata(85, False)
    backend.prepare(decode)
    backend.prepare_dsa_metadata_for_forward()
    assert decode.dsa_metadata is backend._dsa_metadata
    assert decode.dsa_metadata is not prefill_dsa
    assert decode.dsa_metadata.max_query_len == 1
    assert prefill.dsa_metadata is prefill_dsa


def test_graph_prepare_updates_persistent_metadata_in_place(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_pack_metadata_to_device", lambda dsa: None)
    monkeypatch.setattr(
        backend, "_build_precomputed_metadata", lambda dsa, metadata: None
    )
    metadata = SimpleNamespace(
        multi_block_tables=[],
        kv_seq_lens_host=torch.tensor([8, 10], dtype=torch.int32),
        q_seq_lens_host=torch.ones(2, dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=False,
        dsa_metadata=None,
        dsa_positions=None,
        dsa_cos_sin=None,
        dsa_c4_cos_sin=None,
        dsa_c128_cos_sin=None,
        dsa_graph_block_table_cols=0,
        dsa_graph_mode=False,
    )
    rope_cache = torch.arange(128 * 8, dtype=torch.float32).view(128, 8)
    backend.prepare(metadata, graph_mode=True)
    backend.attach_rope_tables(
        torch.tensor([7, 9]), rope_cache, c4_cos_sin=rope_cache,
        c128_cos_sin=rope_cache,
        metadata=metadata,
    )
    backend.prepare_graph_forward_metadata()
    persistent = metadata.dsa_metadata
    seq_ptr = persistent.seq_lens.data_ptr()
    c4_ptr = persistent.c4_cos.data_ptr()
    c4_before = persistent.c4_cos.clone()

    metadata.kv_seq_lens_host.copy_(torch.tensor([12, 11]))
    backend.attach_rope_tables(
        torch.tensor([11, 10]), rope_cache, c4_cos_sin=rope_cache,
        c128_cos_sin=rope_cache,
    )
    backend.prepare_graph_forward_metadata()

    assert metadata.dsa_metadata is persistent
    assert persistent.seq_lens.data_ptr() == seq_ptr
    assert persistent.c4_cos.data_ptr() == c4_ptr
    assert persistent.seq_lens.tolist() == [12, 11]
    assert persistent.start_pos.tolist() == [11, 10]
    assert not torch.equal(persistent.c4_cos, c4_before)


def test_forward_rope_state_is_owned_by_each_metadata(monkeypatch) -> None:
    backend = _make_backend()
    monkeypatch.setattr(backend, "_pack_metadata_to_device", lambda dsa: None)
    monkeypatch.setattr(
        backend, "_build_precomputed_metadata", lambda dsa, metadata: None
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
        c4_cos_sin=rope_cache,
        c128_cos_sin=rope_cache,
        metadata=prefill,
    )

    decode = make_metadata(85, 1)
    backend.prepare(decode, graph_mode=True)
    backend.attach_rope_tables(
        torch.tensor([84]),
        rope_cache,
        c4_cos_sin=rope_cache,
        c128_cos_sin=rope_cache,
        metadata=decode,
    )

    backend.prepare_dsa_metadata_for_forward(prefill)

    assert prefill.dsa_metadata.input_positions.numel() == 84
    assert decode.dsa_positions.numel() == 1
    assert prefill.dsa_positions.data_ptr() != decode.dsa_positions.data_ptr()


def test_prefill_persists_swa_for_decode_and_omits_ori_kv_cu_seqlens(
    monkeypatch,
) -> None:
    backend = DsaAttentionBackend(
        compress_ratios=[1],
        window_size=128,
        n_layers=1,
        num_heads=8,
        attn_head_dim=512,
        metadata_head_dim=1088,
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
    layer = SimpleNamespace(layer_id=0, attn_sink=None)
    calls: list[dict] = []

    def fake_sparse_attn(**kwargs):
        calls.append(kwargs)
        return kwargs["q"].clone(), torch.empty(0)

    monkeypatch.setattr(dsa_attention_module, "_sparse_attn_sharedkv", fake_sparse_attn)

    def prepare_step(kv_len: int, q_len: int, is_prefill: bool):
        dsa = backend._builder.build(
            multi_block_tables=[block_table],
            kv_seq_lens=[kv_len],
            q_seq_lens=[q_len],
            positions=torch.arange(kv_len - q_len, kv_len, dtype=torch.int64),
            dsa_cos_sin=None,
            is_prefill=is_prefill,
            is_chunked_prefill=False,
        )
        dsa.c1_metadata = torch.zeros(1, dtype=torch.int32)
        backend._metadata = SimpleNamespace(
            dsa_metadata=dsa,
            is_prefill=is_prefill,
            is_chunked_prefill=False,
        )
        return dsa

    prepare_step(kv_len=2, q_len=2, is_prefill=True)
    prefill_kv = torch.arange(2 * 512, dtype=torch.float32).view(2, 1, 512)
    backend.execute(
        torch.zeros(2, 8, 512), prefill_kv, prefill_kv, layer
    )
    assert torch.equal(swa[1, :2], prefill_kv)

    prepare_step(kv_len=3, q_len=1, is_prefill=False)
    decode_kv = torch.full((1, 1, 512), 7.0)
    backend.execute(torch.zeros(1, 8, 512), decode_kv, decode_kv, layer)
    assert torch.equal(swa[1, 2], decode_kv[0])
    assert calls[-1]["cu_seqlens_ori_kv"] is None
