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

"""Unit tests for the DeepSeek-V4 prefill CP context plan.

These tests pin the pure arithmetic of the Python port to the C++ reference so
CP sharding bugs cannot hide behind device collectives. They run on CPU.
"""

from __future__ import annotations

import pytest
import torch

from xllm.python.model_executor.v4_cp_context import (
    _compute_cp_local_seq_lens,
    _compute_cp_rows_by_rank,
    build_deepseek_v4_cp_context,
)


def _positions(num_tokens: int) -> torch.Tensor:
    return torch.arange(num_tokens, dtype=torch.int64)


def test_compute_cp_rows_by_rank_covers_each_row_once() -> None:
    rows_by_rank = _compute_cp_rows_by_rank(3, [5, 2, 6])
    assert len(rows_by_rank) == 3
    seen: list[int] = []
    for rows in rows_by_rank:
        assert rows == sorted(rows)
        seen.extend(rows)
    assert sorted(seen) == list(range(13))
    assert len(seen) == 13


def test_compute_cp_rows_by_rank_aggregates_multi_sequence_by_rank() -> None:
    # Do not flatten (rank, sequence) pairs: each outer entry must contain one
    # rank's rows across every sequence, matching the C++ reference.
    assert _compute_cp_rows_by_rank(3, [5, 2, 6]) == [
        [0, 1, 5, 7, 8],
        [2, 3, 6, 9, 10],
        [4, 11, 12],
    ]


def test_multi_sequence_cp_context_builds_rank_major_plan() -> None:
    ctx = build_deepseek_v4_cp_context(
        3,
        1,
        [5, 2, 6],
        [5, 2, 6],
        _positions(13),
    )
    assert ctx.global_q_cu_seq_lens.dtype == torch.int32
    assert ctx.local_kv_cu_seq_lens.dtype == torch.int32
    assert ctx.tokens_per_rank == [5, 5, 3]
    assert ctx.local_row_indices.tolist() == [2, 3, 6, 9, 10]
    # After rank-major gather: [0,1,5,7,8, 2,3,6,9,10, 4,11,12].
    assert ctx.restore_indices[2] == 5
    assert ctx.restore_indices[3] == 6
    assert ctx.restore_indices[6] == 7


def test_compute_cp_local_seq_lens_matches_cpp_contiguous_split() -> None:
    local_q, local_kv = _compute_cp_local_seq_lens(3, 1, [5, 2, 6], [8, 4, 9])
    # Sequence 0: segment=2, rank 1 owns [2, 4) -> q=2, kv=(8-5)+4=7.
    # Sequence 1: segment=1, rank 1 owns row 1 only -> q=1, kv=(4-2)+2=4.
    # Sequence 2: segment=2, rank 1 owns [2, 4) -> q=2, kv=(9-6)+4=7.
    assert local_q == [2, 1, 2]
    assert local_kv == [7, 4, 7]


def test_local_indices_and_positions_are_contiguous_and_true() -> None:
    global_positions = _positions(7)
    ctx = build_deepseek_v4_cp_context(
        2,
        1,
        [7],
        [7],
        global_positions,
    )
    assert ctx.enabled()
    assert ctx.local_row_indices.tolist() == [4, 5, 6]
    assert ctx.local_positions.tolist() == [4, 5, 6]
    assert ctx.global_positions.tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_global_rope_expands_built_half_width_pair() -> None:
    ctx = build_deepseek_v4_cp_context(2, 0, [4], [4], _positions(4))
    half_cos = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    half_sin = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100
    ctx.set_global_rope_pair(4, (half_cos, half_sin))
    cos, sin = ctx.global_rope(4)
    assert cos is not None and sin is not None
    assert tuple(cos.shape) == (4, 4)
    assert tuple(sin.shape) == (4, 4)
    assert cos.tolist() == [
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [4, 4, 5, 5],
        [6, 6, 7, 7],
    ]
    assert sin.tolist() == [
        [100, 100, 101, 101],
        [102, 102, 103, 103],
        [104, 104, 105, 105],
        [106, 106, 107, 107],
    ]
    # The compressor-owned input pair must not be mutated.
    assert tuple(half_cos.shape) == (4, 2)


def test_global_rope_pair_rejects_mismatched_widths() -> None:
    ctx = build_deepseek_v4_cp_context(2, 0, [2], [2], _positions(2))
    try:
        ctx.set_global_rope_pair(1, (torch.zeros(2, 3), torch.zeros(2, 4)))
    except RuntimeError as exc:
        assert "widths differ" in str(exc)
    else:
        raise AssertionError("expected mismatched RoPE widths to fail")


def test_gather_restore_permutation_is_rank_major() -> None:
    ctx = build_deepseek_v4_cp_context(2, 0, [5], [5], _positions(5))
    # cp=2 contiguous split of rows 0..4:
    #   rank0 owns [0,1,2], rank1 owns [3,4]
    # all_gather gives [0,1,2,3,4], so restore[0..4] = [0,1,2,3,4].
    assert ctx.restore_indices.tolist() == [0, 1, 2, 3, 4]

    ctx1 = build_deepseek_v4_cp_context(2, 0, [4], [4], _positions(4))
    # rank0 owns [0,1], rank1 owns [2,3], both identical here.
    assert ctx1.restore_indices.tolist() == [0, 1, 2, 3]


def test_global_rope_prefers_requested_ratio_and_falls_back() -> None:
    cache = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    ctx = build_deepseek_v4_cp_context(2, 0, [4], [4], _positions(4))
    ctx.set_global_rope_cache(1, cache)
    cos, sin = ctx.global_rope(4)
    assert cos is not None and sin is not None
    # [cos, sin] is split before interleave: the requested fallback uses ratio
    # 1's cache, whose half widths are 1 and 2 columns respectively.
    assert tuple(cos.shape) == (4, 2)
    assert tuple(sin.shape) == (4, 4)

    ctx.set_global_rope_pair(1, (torch.zeros(4, 1), torch.zeros(4, 1)))
    requested_cache = cache + 100
    ctx.set_global_rope_cache(4, requested_cache)
    cos, sin = ctx.global_rope(4)
    assert torch.equal(cos, requested_cache[:, :1].repeat_interleave(2, dim=-1))
    assert torch.equal(sin, requested_cache[:, 1:].repeat_interleave(2, dim=-1))


def test_gather_restore_uses_uneven_cp_ranks(monkeypatch) -> None:
    """Pin CP restore to the uneven ``tokens_per_rank`` collective path."""
    positions = _positions(5)
    ctx = build_deepseek_v4_cp_context(2, 1, [5], [5], positions)
    assert ctx.tokens_per_rank == [3, 2]

    captured: dict[str, object] = {}

    class _FakeXllmDistributed:
        @staticmethod
        def all_gather_variable(
            tensor: torch.Tensor,
            token_counts: list[int],
            rank: int,
            group_name: str,
        ) -> torch.Tensor:
            captured["tensor"] = tensor
            captured["token_counts"] = list(token_counts)
            captured["rank"] = rank
            captured["group_name"] = group_name
            # Simulate rank-major all_gather for [3, 2].
            return torch.arange(5, dtype=torch.int64).reshape(5, 1)

    import xllm.python.model_executor.v4_cp_context as cp_module

    fake = _FakeXllmDistributed()
    monkeypatch.setattr(cp_module, "distributed", fake)
    result = ctx.gather_restore(torch.arange(2, dtype=torch.int64).reshape(2, 1))
    assert captured["token_counts"] == [3, 2]
    assert captured["rank"] == 1
    assert captured["group_name"] == "cp"
    assert result.tolist() == [[0], [1], [2], [3], [4]]


@pytest.mark.parametrize("cp_rank", [-1, 2])
def test_cp_context_rejects_invalid_rank(cp_rank: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        build_deepseek_v4_cp_context(2, cp_rank, [4], [4], _positions(4))


@pytest.mark.parametrize("cp_size,q_lens", [(1, [3]), (2, [1]), (4, [5]), (4, [1, 2])])
def test_disabled_context_keeps_global_shape(cp_size: int, q_lens: list[int]) -> None:
    tensor = torch.zeros(sum(q_lens), 4)
    for rank in range(cp_size):
        ctx = build_deepseek_v4_cp_context(cp_size, rank, q_lens, q_lens, _positions(tensor.size(0)))
        assert not ctx.enabled()
        assert ctx.shard_rows(tensor) is tensor
        assert ctx.gather_restore(tensor) is tensor
