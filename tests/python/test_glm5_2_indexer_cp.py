# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM indexer/backend contracts; accelerator kernels and transport use CPU fakes."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.model_executor.cp_utils import CpContext, cp_shard_positions, cp_shard_rows
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models import glm5_2


def _indexer_out(select):
    """Preserve the caller-owned indices and BF16 values kernel contract."""

    def write(*args):
        indices, values = args[-2:]
        assert indices.dtype == torch.int32
        assert values.dtype == torch.bfloat16
        assert values.shape == indices.shape
        indices.copy_(select(*args[:-2]))
        values.zero_()
        return indices

    return write


def _three_token_plan(rank: int) -> CpContext:
    shard = [0, -1] if rank == 0 else [1, 2]
    q_ends = [1] if rank == 0 else [1, 2]
    kv_lengths = [1] if rank == 0 else [2, 3]
    return CpContext(
        cp_size=2,
        cp_rank=rank,
        total_local=2,
        shard_index=torch.tensor(shard),
        shard_gather_index=torch.tensor(shard).clamp_min(0),
        shard_valid_mask=torch.tensor(shard) >= 0,
        restore_index=torch.tensor([0, 2, 3]),
        query_index=torch.tensor([0] if rank == 0 else [0, 1]),
        q_cu_seqlens=q_ends,
        q_cu_seqlens_tensor=torch.tensor(q_ends, dtype=torch.int32),
        kv_gather_index=torch.tensor([0] if rank == 0 else [0, 1, 0, 1, 2]),
        kv_cu_seqlens=[1] if rank == 0 else [2, 5],
        segment_seq_indices=torch.tensor([0] if rank == 0 else [0, 0]),
        segment_kv_seq_lens=kv_lengths,
        segment_kv_seq_lens_tensor=torch.tensor(kv_lengths, dtype=torch.int32),
        has_prefix=False,
    )


def _indexer() -> glm5_2.Glm52Indexer:
    cfg = glm5_2.Glm52Config(
        hidden_size=2,
        q_lora_rank=2,
        index_n_heads=1,
        index_head_dim=2,
        qk_rope_head_dim=2,
        index_topk=1,
        indexer_rope_interleave=False,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    indexer.wq_b._set_dynamic_activation(False)
    with torch.no_grad():
        indexer.wq_b.weight.copy_(torch.eye(2, dtype=torch.int8))
        indexer.wq_b.deq_scale.fill_(1)
        indexer.wq_b.quant_bias.zero_()
        indexer.wq_b.input_scale.fill_(1)
        indexer.wq_b.input_offset.zero_()
        indexer.wk.weight.copy_(torch.eye(2))
        indexer.k_norm.weight.zero_()
        indexer.k_norm.bias.copy_(torch.tensor([5.0, 7.0]))
        indexer.weights_proj.weight.fill_(1)
    return indexer


def _backend_and_metadata(
    quantized: bool, num_blocks: int = 2
) -> tuple[NpuPagedAttentionBackend, SimpleNamespace, LayerCache]:
    backend = NpuPagedAttentionBackend(1, 1, 2, 1.0, -1, True, torch.device("cpu"), torch.float32)
    cache = LayerCache(
        key=torch.zeros(num_blocks, 2, 1, 2),
        value=torch.zeros(num_blocks, 2, 1, 2),
        index=torch.full((num_blocks, 2, 1, 2), -9, dtype=torch.int8 if quantized else torch.float32),
        indexer_scale=torch.full((num_blocks, 2, 1, 1), -9.0, dtype=torch.float16) if quantized else None,
    )
    backend.bind_kv_caches([cache])
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2]),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([3]),
        kv_seq_lens_host_values=[3],
        q_cu_seq_lens=torch.tensor([0, 3]),
        q_seq_lens=torch.tensor([3]),
        expanded_decode_metadata=None,
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        has_kv_shard=False,
        kv_split_size=1,
    )
    return backend, metadata, cache


def _quantize(x: torch.Tensor, *_args: object) -> torch.Tensor:
    if x.shape[0] == 0:
        raise AssertionError("empty-query ranks must not launch the Q projection")
    return x.round().to(torch.int8)


def _quant_matmul(x: torch.Tensor, weight: torch.Tensor, *_args: object) -> torch.Tensor:
    return x.float() @ weight.float().T


def _dynamic_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x.round().to(torch.int8), torch.ones(x.shape[:-1], device=x.device)


def _reference_quant(rows: torch.Tensor) -> torch.Tensor:
    """Independent two-dimensional Hadamard rotation followed by our fake quant."""
    first, second = rows.unbind(-1)
    return (torch.stack((first + second, first - second), dim=-1) / 2**0.5).round().to(torch.int8)


def _scatter(cache: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
    for slot, value in zip(indices.flatten().tolist(), values, strict=True):
        if slot >= 0:
            cache[slot].copy_(value)


@pytest.mark.parametrize("quantized", [False, True])
def test_empty_query_rank_still_populates_index_cache(quantized: bool) -> None:
    empty = torch.empty(0, dtype=torch.int64)
    plan = replace(
        _three_token_plan(1),
        shard_index=torch.tensor([-1, -1]),
        shard_gather_index=torch.tensor([0, 0]),
        shard_valid_mask=torch.tensor([False, False]),
        restore_index=torch.tensor([0]),
        query_index=empty,
        q_cu_seqlens=[],
        q_cu_seqlens_tensor=empty.to(torch.int32),
        kv_gather_index=empty,
        kv_cu_seqlens=[],
        segment_seq_indices=empty,
        segment_kv_seq_lens=[],
        segment_kv_seq_lens_tensor=empty.to(torch.int32),
    )
    indexer = _indexer()
    indexer._q_stream = MagicMock()
    indexer._weights_stream = MagicMock()
    indexer._q_stream.wait_for_current.side_effect = AssertionError("empty Q must not fork")
    indexer._weights_stream.wait_for_current.side_effect = AssertionError("empty weights must not fork")
    backend, metadata, cache = _backend_and_metadata(quantized)
    metadata.slot_mapping = torch.tensor([0])
    metadata.kv_seq_lens = torch.tensor([1])
    metadata.kv_seq_lens_host_values = [1]
    metadata.q_cu_seq_lens = torch.tensor([0, 1])
    metadata.q_seq_lens = torch.tensor([1])

    def all_gather(local: torch.Tensor, dim: int, world_size: int, group: str) -> torch.Tensor:
        assert (dim, world_size, group) == (0, 2, "cp")
        assert local.shape == (2, 2)
        return local.new_tensor([[5, 7], [5, 7], [5, 7], [5, 7]])

    def select(*_args: object) -> torch.Tensor:
        raise AssertionError("empty-query ranks must not launch a lightning indexer")

    context = ForwardContext(backend, torch.device("cpu"), metadata, [cache], cp_context=plan)
    with (
        forward_context(context),
        patch.object(glm5_2.kernels, "quantize_per_tensor", side_effect=_quantize, create=True),
        patch.object(glm5_2.kernels, "quant_matmul", side_effect=_quant_matmul, create=True),
        patch.object(glm5_2.kernels, "dynamic_quant", side_effect=_dynamic_quant, create=True),
        patch.object(glm5_2.kernels, "scatter_nd_update", side_effect=_scatter, create=True),
        patch.object(glm5_2.kernels, "quant_lightning_indexer", side_effect=select, create=True),
        patch.object(glm5_2.kernels, "lightning_indexer_out", side_effect=_indexer_out(select), create=True),
        patch("xllm.python.model_executor.cp_utils.distributed.all_gather", side_effect=all_gather, create=True),
    ):
        backend.prepare(metadata)
        output = indexer.select_qli(
            torch.empty(0, 2),
            torch.empty(0, 2),
            backend.mla_index_context(SimpleNamespace(layer_id=0)),
            (torch.empty(0, 1), torch.empty(0, 1)),
            (torch.ones(2, 1), torch.zeros(2, 1)),
            cache_hidden=torch.zeros(2, 2),
        )

    assert output.dtype == torch.int32
    assert output.tolist() == [[[-1]], [[-1]]]
    assert cache.index.view(4, 2)[0].tolist() == ([8, -1] if quantized else [5.0, 7.0])
    assert cache.index.view(4, 2)[1:].tolist() == [[-9, -9]] * 3
    if quantized:
        assert cache.index_scale.view(-1).tolist() == [1.0, -9.0, -9.0, -9.0]


@pytest.mark.parametrize(
    "packed,rank",
    [(False, 0), (False, 1), (True, 0), (True, 3)],
    ids=["short-rank0", "short-rank1", "packed-rank0", "packed-rank3"],
)
@pytest.mark.parametrize("quantized", [False, True])
def test_cp_indexer_preserves_query_segments_and_cache_order(packed: bool, rank: int, quantized: bool) -> None:
    indexer = _indexer()
    backend, metadata, cache = _backend_and_metadata(quantized, num_blocks=8 if packed else 2)
    if packed:
        # Prefixes 3/2 extend by 5/1 tokens; PCP4 ranks 0/3 own tokens 0,5/3,4.
        shard = [0, -1, 5, -1] if rank == 0 else [3, 4, -1, -1]
        query_rows = [0, 2] if rank == 0 else [0, 1]
        lengths = [4, 3] if rank == 0 else [7, 8]
        sequences = [0, 1] if rank == 0 else [0, 0]
        plan = replace(
            _three_token_plan(0),
            cp_size=4,
            cp_rank=rank,
            total_local=4,
            shard_index=torch.tensor(shard),
            shard_gather_index=torch.tensor(shard).clamp_min(0),
            shard_valid_mask=torch.tensor(shard) >= 0,
            restore_index=torch.tensor([0, 4, 8, 12, 13, 2]),
            query_index=torch.tensor(query_rows),
            q_cu_seqlens=[1, 2],
            q_cu_seqlens_tensor=torch.tensor([1, 2], dtype=torch.int32),
            kv_gather_index=torch.tensor([0, 5] if rank == 0 else [0, 1, 2, 3, 0, 1, 2, 3, 4]),
            kv_cu_seqlens=[1, 2] if rank == 0 else [4, 9],
            segment_seq_indices=torch.tensor(sequences),
            segment_kv_seq_lens=lengths,
            segment_kv_seq_lens_tensor=torch.tensor(lengths, dtype=torch.int32),
            has_prefix=True,
        )
        slots = [3, 4, 5, 6, 7, 10]
        pages = [[0, 1, 2, 3], [4, 5, -1, -1]]
        metadata.slot_mapping = torch.tensor(slots)
        metadata.block_table = torch.tensor(pages, dtype=torch.int32)
        metadata.kv_seq_lens = torch.tensor([8, 3])
        metadata.kv_seq_lens_host_values = [8, 3]
        metadata.q_cu_seq_lens = torch.tensor([0, 5, 6])
        metadata.q_seq_lens = torch.tensor([5, 1])
        metadata.is_prefill = False
        metadata.is_chunked_prefill = True
        global_hidden = torch.arange(1, 7).float().unsqueeze(1).expand(-1, 2)
        global_positions = torch.tensor([3, 4, 5, 6, 7, 2])
        transport_positions = [3, 0, 2, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 7, 0, 0]
        owned_tokens = [0, 5] if rank == 0 else [3, 4]
        topk_rows = [3, 2] if rank == 0 else [6, 7]
        expected_topk = [3, -1, 2, -1] if rank == 0 else [6, 7, -1, -1]
    else:
        plan = _three_token_plan(rank)
        lengths = [1] if rank == 0 else [2, 3]
        sequences = [0] if rank == 0 else [0, 0]
        slots, pages = [0, 1, 2], [[0, 1]]
        global_hidden = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        global_positions = torch.arange(3)
        transport_positions = [0, 0, 1, 2]
        owned_tokens = [0] if rank == 0 else [1, 2]
        topk_rows = [0] if rank == 0 else [1, 2]
        expected_topk = [0, -1] if rank == 0 else [1, 2]

    hidden = cp_shard_rows(global_hidden, plan)
    positions = cp_shard_positions(global_positions, plan)
    key_cos_sin = ((positions + 1).float().unsqueeze(1), torch.zeros(len(positions), 1))
    query_cos_sin = tuple(coefficient.index_select(0, plan.query_index) for coefficient in key_cos_sin)
    # Original token order and write slots form the oracle, not restore_index.
    keys = (global_positions + 1).unsqueeze(1) * torch.tensor([5.0, 7.0])
    expected_query = global_hidden[owned_tokens] * (global_positions[owned_tokens] + 1).unsqueeze(1)
    if quantized:
        keys, expected_query = _reference_quant(keys), _reference_quant(expected_query)
    expected_cache = torch.full((cache.index.numel() // 2, 2), -9, dtype=keys.dtype)
    expected_cache[slots] = keys
    transport_keys = (torch.tensor(transport_positions) + 1).unsqueeze(1) * torch.tensor([5.0, 7.0])

    def all_gather(local: torch.Tensor, dim: int, world_size: int, group: str) -> torch.Tensor:
        assert (dim, world_size, group) == (0, plan.cp_size, "cp")
        start = rank * plan.total_local
        torch.testing.assert_close(local, transport_keys[start : start + plan.total_local])
        return transport_keys

    def indexer_metadata(*args: object) -> torch.Tensor:
        assert args[:3] == (64, 1, 2)
        assert args[3].tolist() == list(range(1, len(owned_tokens) + 1))
        assert args[4].tolist() == lengths
        assert args[5:] == (1, max(lengths), 1, 1)
        return torch.empty(0, dtype=torch.int32)

    def select(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor, *args: object) -> torch.Tensor:
        q_ends, kv_lengths, blocks = args[3:6] if quantized else args[:3]
        assert weights.shape[0] == len(owned_tokens)
        assert q_ends.tolist() == list(range(1, len(owned_tokens) + 1))
        assert kv_lengths.tolist() == lengths
        assert blocks.tolist() == [pages[sequence] for sequence in sequences]
        torch.testing.assert_close(query[:, 0], expected_query)
        torch.testing.assert_close(key.view_as(expected_cache), expected_cache)
        return torch.tensor(topk_rows, dtype=torch.int32).view(-1, 1, 1)

    with (
        forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [cache], cp_context=plan)),
        patch.object(glm5_2.kernels, "quantize_per_tensor", side_effect=_quantize, create=True),
        patch.object(glm5_2.kernels, "quant_matmul", side_effect=_quant_matmul, create=True),
        patch.object(glm5_2.kernels, "dynamic_quant", side_effect=_dynamic_quant, create=True),
        patch.object(glm5_2.kernels, "scatter_nd_update", side_effect=_scatter, create=True),
        patch.object(glm5_2.kernels, "quant_lightning_indexer_metadata", side_effect=indexer_metadata, create=True),
        patch.object(
            glm5_2.kernels,
            "quant_lightning_indexer" if quantized else "lightning_indexer_out",
            side_effect=select if quantized else _indexer_out(select),
            create=True,
        ),
        patch("xllm.python.model_executor.cp_utils.distributed.all_gather", side_effect=all_gather, create=True),
    ):
        backend.prepare(metadata)
        query_hidden = hidden.index_select(0, plan.query_index)
        output = indexer.select_qli(
            query_hidden,
            query_hidden,
            backend.mla_index_context(SimpleNamespace(layer_id=0)),
            query_cos_sin,
            key_cos_sin,
            cache_hidden=hidden,
        )

    assert output.view(-1).tolist() == expected_topk
    if quantized:
        expected_scale = torch.full((expected_cache.shape[0],), -9.0, dtype=torch.float16)
        expected_scale[slots] = 1.0
        torch.testing.assert_close(cache.index_scale.view(-1), expected_scale)


def test_indexer_fuses_k_and_weight_projections_after_loading() -> None:
    cfg = glm5_2.Glm52Config(
        hidden_size=3,
        q_lora_rank=2,
        index_n_heads=2,
        index_head_dim=2,
        qk_rope_head_dim=1,
        index_topk=1,
        indexer_rope_interleave=False,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    indexer.wq_b._set_dynamic_activation(False)
    with torch.no_grad():
        indexer.wk.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        indexer.weights_proj.weight.copy_(torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))

    hidden = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    with patch.object(
        glm5_2.kernels,
        "prepare_quant_weight",
        side_effect=lambda weight: weight,
        create=True,
    ):
        indexer.process_weights_after_loading()

    key, weights = indexer._project_index_inputs(hidden, hidden)
    expected = hidden @ torch.cat((indexer.wk.weight, indexer.weights_proj.weight), dim=0).T
    assert indexer._wk_weights_proj_ready
    assert weights.is_contiguous()
    torch.testing.assert_close(torch.cat((key, weights), dim=-1), expected)


def test_indexer_keeps_separate_projections_for_distinct_cache_rows() -> None:
    cfg = glm5_2.Glm52Config(
        hidden_size=3,
        q_lora_rank=2,
        index_n_heads=2,
        index_head_dim=2,
        qk_rope_head_dim=1,
        index_topk=1,
        indexer_rope_interleave=False,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        indexer.wk.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        indexer.weights_proj.weight.copy_(torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]))

    hidden = torch.tensor([[1.0, 2.0, 3.0]])
    cache_hidden = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    key, weights = indexer._project_index_inputs(hidden, cache_hidden)

    torch.testing.assert_close(key, cache_hidden @ indexer.wk.weight.T)
    torch.testing.assert_close(weights, hidden @ indexer.weights_proj.weight.T)


def test_indexer_invalidates_fused_projection_after_state_dict_load() -> None:
    cfg = glm5_2.Glm52Config(
        hidden_size=3,
        q_lora_rank=2,
        index_n_heads=2,
        index_head_dim=2,
        qk_rope_head_dim=1,
        index_topk=1,
        indexer_rope_interleave=False,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    indexer.wq_b._set_dynamic_activation(False)
    with patch.object(
        glm5_2.kernels,
        "prepare_quant_weight",
        side_effect=lambda weight: weight,
        create=True,
    ):
        indexer.process_weights_after_loading()
    assert indexer._wk_weights_proj_ready

    state_dict = {name: value.clone() for name, value in indexer.state_dict().items()}
    state_dict["wk.weight"] = torch.full_like(indexer.wk.weight, 2.0)
    state_dict["weights_proj.weight"] = torch.full_like(indexer.weights_proj.weight, 3.0)
    indexer.load_state_dict(state_dict)

    hidden = torch.ones((1, cfg.hidden_size))
    key, weights = indexer._project_index_inputs(hidden, hidden)
    assert not indexer._wk_weights_proj_ready
    torch.testing.assert_close(key, torch.full((1, cfg.index_head_dim), 6.0))
    torch.testing.assert_close(weights, torch.full((1, cfg.index_n_heads), 9.0))


def test_interleaved_indexer_rope_uses_inplace_partial_kernel() -> None:
    cfg = glm5_2.Glm52Config(
        hidden_size=4,
        q_lora_rank=4,
        index_n_heads=1,
        index_head_dim=4,
        qk_rope_head_dim=2,
        index_topk=1,
        indexer_rope_interleave=True,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    cos = torch.tensor([[[[2.0, 2.0]]], [[[3.0, 3.0]]]])
    sin = torch.tensor([[[[1.0, 1.0]]], [[[1.0, 1.0]]]])
    calls: list[tuple[torch.Size, torch.Size, torch.Size, int, int]] = []

    def inplace_rope(
        tensor: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        start: int,
        dim: int,
    ) -> torch.Tensor:
        calls.append((tensor.shape, cosine.shape, sine.shape, start, dim))
        first = tensor[..., 0].clone()
        second = tensor[..., 1].clone()
        cosine = cosine[:, 0].unsqueeze(1)
        sine = sine[:, 0].unsqueeze(1)
        tensor[..., 0] = first * cosine - second * sine
        tensor[..., 1] = second * cosine + first * sine
        return tensor

    with (
        patch.object(
            glm5_2.kernels,
            "npu_inplace_partial_rotary_mul",
            side_effect=inplace_rope,
            create=True,
        ),
    ):
        output = indexer._apply_interleaved_rope(value, (cos, sin))

    assert output.data_ptr() == value.data_ptr()
    assert calls == [((2, 1, 4), (2, 2), (2, 2), 0, 2)]
    torch.testing.assert_close(output[:, :2], torch.tensor([[0.0, 5.0], [9.0, 23.0]]))


@pytest.mark.parametrize("multi_stream", [False, True])
def test_indexer_consumes_explicit_distinct_query_and_key_cos_sin(multi_stream: bool) -> None:
    cfg = glm5_2.Glm52Config(
        hidden_size=2,
        q_lora_rank=2,
        index_n_heads=1,
        index_head_dim=4,
        qk_rope_head_dim=2,
        index_topk=1,
        indexer_rope_interleave=True,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.float32, torch.device("cpu"))
    index_cache = torch.zeros(1, 1, 1, 4)
    if multi_stream:
        indexer._weights_stream = MagicMock()
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    ctx = SimpleNamespace(
        actual_seq_q=torch.tensor([1]),
        actual_seq_kv=torch.tensor([1]),
        cp_context=None,
        index_cache=index_cache,
        index_cache_scale=None,
        materialize_index_cache=lambda: (index_cache, None, block_table),
        update_index_cache=MagicMock(),
    )
    hidden = torch.ones(1, 2)
    projected_k = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    query_cos_sin = (torch.full((1, 1, 1, 2), 2.0), torch.full((1, 1, 1, 2), 3.0))
    key_cos_sin = (torch.full((1, 1, 1, 2), 4.0), torch.full((1, 1, 1, 2), 5.0))

    def partial_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int, dim: int) -> None:
        first, second = value[..., 0].clone(), value[..., 1].clone()
        value[..., 0] = first * cos[:, :1] - second * sin[:, :1]
        value[..., 1] = second * cos[:, :1] + first * sin[:, :1]

    def select(query: torch.Tensor, *_args: object) -> torch.Tensor:
        torch.testing.assert_close(query, torch.tensor([[[1.0, 8.0, 6.0, 7.0]]]))
        return torch.zeros(1, 1, 1, dtype=torch.int32)

    with (
        patch.object(indexer, "_update_index_cache", wraps=indexer._update_index_cache) as update_cache,
        patch.object(indexer.wk, "forward", return_value=projected_k) as project_key,
        patch.object(indexer.k_norm, "forward", return_value=projected_k.clone()),
        patch.object(indexer.wq_b, "forward", return_value=torch.tensor([[2.0, 1.0, 6.0, 7.0]])),
        patch.object(
            glm5_2.kernels,
            "npu_inplace_partial_rotary_mul",
            side_effect=partial_rope,
            create=True,
        ),
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], execution_state=None)),
        patch.object(
            glm5_2.kernels,
            "lightning_indexer_out",
            side_effect=_indexer_out(select),
            create=True,
        ),
    ):
        indexer.select_qli(hidden, hidden, ctx, query_cos_sin, key_cos_sin, cache_hidden=hidden)

    project_key.assert_called_once_with(hidden)
    update_cache.assert_called_once_with(hidden, ctx, key_cos_sin, projected_k=projected_k)
    assert update_cache.call_args.kwargs["projected_k"] is projected_k
    ctx.update_index_cache.assert_called_once()
    key, scale = ctx.update_index_cache.call_args.args
    torch.testing.assert_close(key, torch.tensor([[-6.0, 13.0, 3.0, 4.0]]))
    assert scale is None
