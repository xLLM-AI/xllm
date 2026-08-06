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
    cp_compact_slots,
    cp_decode_local_kv_lens,
    cp_shard_rows,
    cp_slot_owner,
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


# ---------------------------------------------------------------------------
# DCP: KV-storage sharding (cp_compact_slots / owner / decode local kv lens).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("page_size", [1, 4])
def test_compact_slots_disjoint_and_complete(cp_size, page_size):
    """Every logical slot is owned by exactly one rank; compaction is a bijection.

    Across all ranks the compacted physical slots must partition the logical
    space with no collisions, and each rank's owned physical slots stay within
    its physical cache range ``[0, n_blocks * page_size)``.
    """
    n_blocks = 3
    logical_total = n_blocks * cp_size * page_size
    logical = torch.arange(logical_total, dtype=torch.int64)

    seen = {}
    for r in range(cp_size):
        phys = cp_compact_slots(logical, cp_size, r, page_size)
        owned = phys >= 0
        # Each rank owns exactly 1/cp of the logical slots.
        assert int(owned.sum()) == logical_total // cp_size
        # Physical slots stay inside this rank's own cache.
        assert int(phys[owned].max()) < n_blocks * page_size
        for log_s, phy_s in zip(logical[owned].tolist(), phys[owned].tolist()):
            key = (r, phy_s)
            assert key not in seen, f"physical collision {key}"
            seen[key] = log_s

    # Ownership is exhaustive: every logical slot got exactly one owner.
    total_owned = sum(
        int((cp_compact_slots(logical, cp_size, r, page_size) >= 0).sum())
        for r in range(cp_size)
    )
    assert total_owned == logical_total


@pytest.mark.parametrize("cp_size", [2, 4])
def test_compact_slots_preserves_negative(cp_size):
    """Original ``-1`` (padding) slots stay ``-1`` for every rank."""
    page_size = 4
    logical = torch.tensor([-1, 0, 5, -1, 17], dtype=torch.int64)
    for r in range(cp_size):
        phys = cp_compact_slots(logical, cp_size, r, page_size)
        assert phys[0].item() == -1
        assert phys[3].item() == -1


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("page_size", [1, 4])
def test_slot_owner_matches_compact(cp_size, page_size):
    """cp_slot_owner agrees with which rank cp_compact_slots assigns a slot to."""
    logical = torch.arange(cp_size * page_size * 2, dtype=torch.int64)
    owner = cp_slot_owner(logical, cp_size, page_size)
    for r in range(cp_size):
        owned = cp_compact_slots(logical, cp_size, r, page_size) >= 0
        assert torch.equal(owner == r, owned)


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("page_size", [1, 4])
def test_decode_local_kv_lens_sum_to_total(cp_size, page_size):
    """Per-rank stored KV counts partition each sequence's context length."""
    kv_seq_lens = torch.tensor([1, 7, 16, 33, 100], dtype=torch.int64)
    per_rank = [
        cp_decode_local_kv_lens(kv_seq_lens, cp_size, r, page_size)
        for r in range(cp_size)
    ]
    total = torch.stack(per_rank, dim=0).sum(dim=0)
    assert torch.equal(total, kv_seq_lens)


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("page_size", [1, 4])
def test_decode_local_kv_lens_matches_owner_count(cp_size, page_size):
    """local_kv_len equals the number of positions this rank actually owns.

    Cross-check the closed-form count against explicitly labelling every
    context position by its block-granular owner.
    """
    for L in (1, 5, 8, 15, 40):
        positions = torch.arange(L, dtype=torch.int64)
        owner = cp_slot_owner(positions, cp_size, page_size)
        for r in range(cp_size):
            expect = int((owner == r).sum())
            got = int(
                cp_decode_local_kv_lens(
                    torch.tensor([L]), cp_size, r, page_size
                )[0]
            )
            assert got == expect, f"L={L} rank={r}: {got} != {expect}"


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("page_size", [1, 4])
def test_decode_shard_read_is_physically_contiguous(cp_size, page_size):
    """DCP decode read invariant: each rank's owned physical slots equal the
    first ``local_kv_len`` slots that FIA reads contiguously across the
    sequence's block table.

    ``_decode_cp`` runs FIA with ``block_table`` = the logical block ids,
    ``block_size`` = page_size, and ``actual_seq_lengths_kv`` = local_kv_len, so
    it reads token ``t`` from physical slot ``block_ids[t // page] * page + t %
    page``. That is only correct if the tokens this rank actually owns
    (``cp_compact_slots >= 0``) map to exactly those physical slots — i.e. this
    rank's shard sits at offsets ``[0, local_kv_len)`` within its physical
    pages. Verify it against explicit ownership labelling.
    """
    block_width = cp_size * page_size
    for L in (1, 5, 8, 15, 40):
        num_blocks = (L + block_width - 1) // block_width
        # Arbitrary but distinct logical block ids for this sequence.
        block_ids = [10 + b for b in range(num_blocks)]
        logical = torch.tensor(
            [block_ids[i // block_width] * block_width + i % block_width
             for i in range(L)],
            dtype=torch.int64,
        )
        for r in range(cp_size):
            phys = cp_compact_slots(logical, cp_size, r, page_size)
            owned_phys = sorted(int(s) for s in phys if int(s) >= 0)
            local_kv = int(
                cp_decode_local_kv_lens(
                    torch.tensor([L]), cp_size, r, page_size
                )[0]
            )
            fia_read = sorted(
                block_ids[t // page_size] * page_size + t % page_size
                for t in range(local_kv)
            )
            assert owned_phys == fia_read, (
                f"L={L} rank={r}: owned physical slots {owned_phys} do not "
                f"match FIA contiguous read {fia_read}"
            )

