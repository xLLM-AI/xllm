# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for xllm.python.model_executor.cp_utils (zigzag CP).

Validates the pure index math with no NPU or live process group: the
shard/all-gather/merge round-trip is exact, sharding is load-balanced, and the
packed query/KV gather indices reproduce full causal attention. ``all_gather``
is emulated on CPU by stacking every rank's shard in rank-major order.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import torch

# cp_utils imports ``from xllm.python import ops`` for cp_all_gather at runtime;
# mock it so the module imports without the C++ binary. The tests emulate the
# collective directly instead of calling ops.
sys.modules.setdefault("xllm.python.ops", MagicMock())

from xllm.python.model_executor.cp_utils import (  # noqa: E402
    build_cp_context,
    cp_shard_rows,
)


def _emulate_all_gather(per_rank_shards):
    """Concatenate rank-major shards, mirroring ops.cp_all_gather(dim=0)."""
    return torch.cat(per_rank_shards, dim=0)


def _shard_all_ranks(x, seq_lens, cp_size):
    """Return each rank's local shard of the global packed tensor ``x``."""
    return [
        cp_shard_rows(x, build_cp_context(seq_lens, cp_size, r, x.device))
        for r in range(cp_size)
    ]


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [8],  # single, divisible by 2*cp_size for cp_size<=4
        [16, 24],  # multiple divisible
        [7],  # single, needs padding
        [5, 13, 2],  # mixed, all need padding
        [1],  # degenerate single token
    ],
)
def test_shard_merge_round_trip(seq_lens, cp_size):
    """merge(all_gather(shard(x))) == x for arbitrary lengths/ranks."""
    total = sum(seq_lens)
    x = torch.arange(total * 3, dtype=torch.float32).reshape(total, 3)

    shards = _shard_all_ranks(x, seq_lens, cp_size)
    gathered = _emulate_all_gather(shards)

    # Any rank's context restores the same global order (restore_index is
    # rank-independent).
    ctx0 = build_cp_context(seq_lens, cp_size, 0, x.device)
    restored = gathered.index_select(0, ctx0.restore_index)
    assert torch.equal(restored, x)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_shards_are_disjoint_and_complete(cp_size):
    """Every real global row is owned by exactly one rank."""
    seq_lens = [10, 7]
    owners = {}
    for r in range(cp_size):
        ctx = build_cp_context(seq_lens, cp_size, r, torch.device("cpu"))
        real = ctx.shard_index[ctx.shard_valid_mask].tolist()
        for g in real:
            assert g not in owners, f"row {g} owned by ranks {owners[g]} and {r}"
            owners[g] = r
    assert sorted(owners) == list(range(sum(seq_lens)))


@pytest.mark.parametrize("cp_size", [2, 4])
def test_query_load_is_balanced(cp_size):
    """Zigzag equalizes per-rank attention work (sum of causal prefixes).

    Contiguous sharding would give rank r a prefix mass of ~(r+1)/cp_size; the
    head-tail pairing makes every rank's total KV-prefix length equal for a
    length divisible by 2*cp_size.
    """
    seq_lens = [4 * cp_size * 2]  # divisible by 2*cp_size, no padding
    prefix_mass = []
    for r in range(cp_size):
        ctx = build_cp_context(seq_lens, cp_size, r, torch.device("cpu"))
        prefix_mass.append(ctx.kv_cu_seqlens[-1])
    assert len(set(prefix_mass)) == 1, f"imbalanced: {prefix_mass}"


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("seq_lens", [[8], [16, 24], [7], [5, 13, 2]])
def test_packed_attention_matches_reference(seq_lens, cp_size):
    """Gathering query rows + causal KV prefixes reproduces full attention.

    Emulates one FIA call per segment on CPU (softmax over the segment's causal
    prefix) and checks the reassembled global output equals a dense causal
    attention over the whole sequence.
    """
    torch.manual_seed(0)
    total = sum(seq_lens)
    dim = 4
    q = torch.randn(total, dim)
    k = torch.randn(total, dim)
    v = torch.randn(total, dim)

    # Reference: dense per-sequence causal attention in global order.
    ref = torch.zeros(total, dim)
    base = 0
    for length in seq_lens:
        for i in range(length):
            qi = q[base + i]
            kk = k[base : base + i + 1]
            vv = v[base : base + i + 1]
            w = torch.softmax(kk @ qi / (dim**0.5), dim=0)
            ref[base + i] = w @ vv
        base += length

    # CP path: each rank gathers its query rows and their causal KV prefixes,
    # runs segment-local causal attention, scatters back; then merge.
    out_shards = []
    for r in range(cp_size):
        ctx = build_cp_context(seq_lens, cp_size, r, q.device)
        # Mirror _prefill_cp: q is sharded to this rank's local rows, then the
        # real query rows are selected from that local layout. KV is replicated
        # to full global order (all-gather), so segment prefixes index global k/v.
        q_local = cp_shard_rows(q, ctx)
        q_real = q_local.index_select(0, ctx.query_index)
        # Walk segments via cu_seqlens to run per-segment causal softmax.
        out_real = torch.zeros(q_real.shape[0], dim)
        q_prev = 0
        kv_prev = 0
        for si in range(len(ctx.q_cu_seqlens)):
            q_end = ctx.q_cu_seqlens[si]
            kv_end = ctx.kv_cu_seqlens[si]
            seg_q = q_real[q_prev:q_end]
            seg_kv_idx = ctx.kv_gather_index[kv_prev:kv_end]
            seg_k = k.index_select(0, seg_kv_idx)
            seg_v = v.index_select(0, seg_kv_idx)
            qcount = q_end - q_prev
            prefix = kv_end - kv_prev
            start = prefix - qcount  # segment_start (right-aligned causal)
            for j in range(qcount):
                allowed = start + j + 1
                w = torch.softmax(
                    seg_k[:allowed] @ seg_q[j] / (dim**0.5), dim=0
                )
                out_real[q_prev + j] = w @ seg_v[:allowed]
            q_prev = q_end
            kv_prev = kv_end
        # Scatter into padded local layout.
        out_local = torch.zeros(ctx.total_local, dim)
        out_local.index_copy_(0, ctx.query_index, out_real)
        out_shards.append(out_local)

    gathered = _emulate_all_gather(out_shards)
    ctx0 = build_cp_context(seq_lens, cp_size, 0, q.device)
    merged = gathered.index_select(0, ctx0.restore_index)
    assert torch.allclose(merged, ref, atol=1e-5), (merged - ref).abs().max()


def test_cp_size_one_rejected():
    with pytest.raises(ValueError):
        build_cp_context([8], 1, 0, torch.device("cpu"))
