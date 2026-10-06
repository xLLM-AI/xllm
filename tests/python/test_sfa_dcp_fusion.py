# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""SFA DCP remap precision: CPU golden and optional AOT kernel vs naive torch."""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers.sfa_dcp import _fill_wide_remap
from xllm.python.layers.sfa_dcp_ref import remap_sparse_indices

TOPK = 2048
# GLM-5.3: index_kpool=4 with always-select-tail, so the tail is index_kpool - 1.
KPOOL_TAIL_COLS = 3


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _aot_remap_available() -> bool:
    ops = getattr(torch.ops, "xllm_ops", None)
    return ops is not None and hasattr(ops, "sfa_dcp_remap_out")


def _make_logical_slots(
    num_tokens: int,
    layout: KVShardLayout,
    *,
    device: torch.device,
    width: int = TOPK,
) -> torch.Tensor:
    torch.manual_seed(0)
    slots = torch.randint(
        0,
        8 * layout.logical_block_size,
        (num_tokens, width),
        device=device,
        dtype=torch.int32,
    )
    mask = torch.rand((num_tokens, width), device=device) < 0.25
    return torch.where(mask, torch.full_like(slots, KVShardLayout.INVALID_SLOT), slots)


def _assert_valid_slots_form_one_run(slots: torch.Tensor) -> None:
    """SFA stops at the first ``-1``, so nothing valid may follow one."""
    for row in slots.tolist():
        invalid = [pos for pos, slot in enumerate(row) if slot < 0]
        if invalid:
            assert all(slot < 0 for slot in row[invalid[0] :]), f"valid slot behind the first -1: {row}"


@pytest.mark.parametrize(
    ("dcp_rank", "logical_slots", "expected"),
    [
        (0, [[0, 5, -1, 8], [1, 4, 9, -1]], [[0, 4, -1, -1], [1, 5, -1, -1]]),
        (1, [[13, 4, -1, 9], [6, 21, 12, -1]], [[5, 0, -1, -1], [2, 9, 4, -1]]),
    ],
    ids=["rank0", "rank1-cross-block-order"],
)
def test_remap_sparse_indices_packs_owned_slots(
    dcp_rank: int, logical_slots: list[list[int]], expected: list[list[int]]
) -> None:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=dcp_rank)
    slots = torch.tensor(logical_slots, dtype=torch.int32)

    remapped = remap_sparse_indices(slots, layout, index_topk=4)
    assert remapped.shape == slots.shape
    torch.testing.assert_close(remapped, torch.tensor(expected, dtype=torch.int32))


def test_remap_sparse_indices_accepts_kpool_tail_wider_than_index_topk() -> None:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0)
    slots = torch.tensor([[0, 5, -1, 8, 12, 1, 4]], dtype=torch.int32)

    remapped = remap_sparse_indices(slots, layout, index_topk=4)

    assert remapped.shape == slots.shape
    # Owned prefix slots stay first, owned tail slots continue the same run.
    assert remapped[remapped >= 0].tolist() == [0, 4, 1]
    _assert_valid_slots_form_one_run(remapped)


def test_fill_wide_remap_matches_the_whole_row_reference() -> None:
    """The fused wide assembly has to equal packing the whole row at once.

    The AOT kernel only packs the configured prefix, so the tail is joined by
    ``_fill_wide_remap``; on CPU the packed prefix comes from the layout.
    """
    layout = KVShardLayout(physical_block_size=128, dcp_size=4, dcp_rank=0)
    slots = _make_logical_slots(4, layout, device=torch.device("cpu"), width=TOPK + KPOOL_TAIL_COLS)

    out = torch.empty_like(slots)
    _fill_wide_remap(
        out=out,
        packed_prefix=layout.pack_owned_slots(slots[..., :TOPK]),
        tail_slots=slots[..., TOPK:],
        layout=layout,
    )

    assert torch.equal(out, remap_sparse_indices(slots, layout, index_topk=TOPK))
    assert int((out >= 0).sum()) == int((layout.localize_slots(slots) >= 0).sum())
    _assert_valid_slots_form_one_run(out)


def test_fill_wide_remap_keeps_a_fully_owned_tail_contiguous() -> None:
    """With every slot owned there is no ``-1`` padding to hide the tail behind."""
    layout = KVShardLayout(physical_block_size=128, dcp_size=1, dcp_rank=0)
    slots = torch.cat(
        [
            torch.arange(TOPK, dtype=torch.int32),
            torch.arange(TOPK, TOPK + KPOOL_TAIL_COLS, dtype=torch.int32),
        ]
    ).unsqueeze(0)

    out = torch.empty_like(slots)
    _fill_wide_remap(
        out=out,
        packed_prefix=layout.pack_owned_slots(slots[..., :TOPK]),
        tail_slots=slots[..., TOPK:],
        layout=layout,
    )

    assert torch.equal(out, slots)
    _assert_valid_slots_form_one_run(out)


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
@pytest.mark.skipif(
    not _aot_remap_available(),
    reason="TileLang AOT sfa_dcp_remap_out is not registered",
)
def test_fused_remap_matches_naive() -> None:
    """Both widths the indexer emits: exactly ``index_topk``, and the wider
    kPool tail (``topk + index_kpool - 1``, what GLM-5.3 ships). The fused path
    packs the prefix with the AOT kernel and joins the tail through
    ``_fill_wide_remap``, which is what the model feeds SFA.
    """
    device = torch.device("npu")
    cases = (
        (128, 4, 2, 1),
        (128, 4, 2, 8),
        (64, 2, 1, 8),
        (128, 32, 7, 8),
    )
    for physical_block_size, dcp_size, dcp_rank, num_tokens in cases:
        for width in (TOPK, TOPK + KPOOL_TAIL_COLS):
            layout = KVShardLayout(
                physical_block_size=physical_block_size,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
            slots = _make_logical_slots(num_tokens, layout, device=device, width=width)
            prefix = slots[..., :TOPK].contiguous()
            out = torch.empty_like(prefix)
            scratch = torch.empty(num_tokens * TOPK, dtype=torch.int32, device=device)
            fused = torch.ops.xllm_ops.sfa_dcp_remap_out(
                prefix,
                layout.physical_block_size,
                layout.dcp_size,
                layout.dcp_rank,
                out,
                scratch,
            )
            torch.npu.synchronize()
            if width > TOPK:
                wide = torch.empty_like(slots)
                _fill_wide_remap(
                    out=wide,
                    packed_prefix=fused,
                    tail_slots=slots[..., TOPK:],
                    layout=layout,
                )
                fused = wide
            naive = remap_sparse_indices(slots, layout, index_topk=TOPK)
            _assert_valid_slots_form_one_run(fused)
            assert torch.equal(fused, naive), (
                f"remap mismatch pb={physical_block_size} dcp={dcp_size} T={num_tokens} width={width}"
            )
