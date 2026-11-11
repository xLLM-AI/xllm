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

"""Device-only references for budgeted DSpark prefix allocation and packing.

These references define the dedicated kernel contract, not a serving fallback.
The host chooses the exact draft budget and a captured physical row capacity
before executing the draft. Tensor values never determine output shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DSparkVerifyBuffers:
    """Fixed-address verification inputs, allocated before graph capture."""

    tokens: torch.Tensor
    positions: torch.Tensor
    kv_seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    q_seq_lens: torch.Tensor
    owners: torch.Tensor
    columns: torch.Tensor


def create_verify_buffers(
    capacity: int,
    block_table_width: int,
    device: torch.device | str,
) -> DSparkVerifyBuffers:
    """Allocate maximum-capacity outputs during executor initialization."""
    if capacity < 0 or block_table_width < 1:
        raise ValueError("capacity must be nonnegative and block_table_width positive")

    def _vector() -> torch.Tensor:
        return torch.empty(capacity, dtype=torch.int32, device=device)

    return DSparkVerifyBuffers(
        tokens=_vector(),
        positions=_vector(),
        kv_seq_lens=_vector(),
        slot_mapping=_vector(),
        block_table=torch.empty((capacity, block_table_width), dtype=torch.int32, device=device),
        q_seq_lens=_vector(),
        owners=_vector(),
        columns=_vector(),
    )


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have shape={shape}, dtype={dtype}, device={device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_disjoint_outputs(inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]) -> None:
    """Reject overlapping live spans, while allowing disjoint arena views."""
    for output_index, output in enumerate(outputs):
        if output.numel() == 0:
            continue
        start = output.data_ptr()
        end = start + output.numel() * output.element_size()
        for other in (*inputs, *outputs[:output_index]):
            if other.numel() == 0:
                continue
            other_start = other.data_ptr()
            other_end = other_start + other.numel() * other.element_size()
            if max(start, other_start) < min(end, other_end):
                raise ValueError("DSpark output spans must not overlap live buffers")


def build_verify_offsets(widths: torch.Tensor, offsets: torch.Tensor) -> None:
    """Refresh offsets after the authoritative TP prefix widths are broadcast."""
    if widths.ndim != 1 or widths.dtype != torch.int32:
        raise ValueError("widths must be a one-dimensional int32 tensor")
    _check_tensor("widths", widths, (widths.shape[0],), torch.int32, widths.device)
    _check_tensor("offsets", offsets, (widths.shape[0] + 1,), torch.int32, widths.device)
    _check_disjoint_outputs((widths,), (offsets,))
    offsets[:1].zero_()
    torch.cumsum(widths, dim=0, dtype=torch.int32, out=offsets[1:])


def allocate_verify_prefixes(
    confidence: torch.Tensor,
    budget: int,
    widths: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    """Allocate exactly ``budget`` drafts, with one mandatory anchor per request.

    Input probabilities are calibrated conditional confidences, not raw head
    logits. Normalization is device-side: NaN/negative infinity become zero,
    positive infinity becomes one, then values are clamped to [0, 1]. No
    epsilon filter drops candidates: even all-zero confidences fill the exact
    host budget. Ties prefer earlier positions, then earlier request indices,
    so every request receives a continuous prefix. No input is modified.

    ``offsets`` must be rebuilt after any TP broadcast that changes ``widths``.
    """
    if confidence.ndim != 2 or confidence.dtype != torch.float32:
        raise ValueError("confidence must be a two-dimensional float32 tensor")
    batch_size, gamma = confidence.shape
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise ValueError("budget must be a host integer")
    if gamma < 1 or not 0 <= budget <= batch_size * gamma:
        raise ValueError("budget must be between zero and batch_size * gamma")
    _check_tensor("confidence", confidence, (batch_size, gamma), torch.float32, confidence.device)
    _check_tensor("widths", widths, (batch_size,), torch.int32, confidence.device)
    _check_tensor("offsets", offsets, (batch_size + 1,), torch.int32, confidence.device)
    _check_disjoint_outputs((confidence,), (widths, offsets))
    probabilities = torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    survival = torch.cumprod(probabilities, dim=1)
    # Position-major flattening makes stable score sorting resolve ties by
    # position first, then request, without perturbing the survival scores.
    order = torch.argsort(survival.transpose(0, 1).reshape(-1), descending=True, stable=True)
    widths.fill_(1)
    if budget:
        requests = torch.remainder(order[:budget], batch_size)
        widths.scatter_add_(0, requests, torch.ones(budget, dtype=torch.int32, device=confidence.device))
    build_verify_offsets(widths, offsets)


def pack_verify_inputs(
    *,
    anchors: torch.Tensor,
    proposals: torch.Tensor,
    base_positions: torch.Tensor,
    base_kv_lengths: torch.Tensor,
    base_block_table: torch.Tensor,
    widths: torch.Tensor,
    offsets: torch.Tensor,
    logical_rows: int,
    block_size: int,
    outputs: DSparkVerifyBuffers,
) -> None:
    """Pack per-request prefixes into a host-chosen fixed graph capacity.

    ``base_positions`` is the anchor's cache position; ``base_kv_lengths``
    already includes that anchor.
    Real row (request, column) uses position=base_position+column and
    kv_len=base_kv_length+column. KV slot addressing uses the cache position,
    not the anchor-inclusive attention length. Caller guarantees in-range
    allocated cache pages and widths in [1, gamma+1] summing to logical_rows.

    All padded rows are safe singleton queries: token=1, position=0,
    kv_len=1, q_len=1, slot=-1, reserved-zero block table, owner/column=-1.
    No padded or unverified row may enter the subsequent acceptance/commit.
    """
    if proposals.ndim != 2 or proposals.dtype != torch.int64:
        raise ValueError("proposals must be a two-dimensional int64 tensor")
    batch_size, gamma = proposals.shape
    device = proposals.device
    capacity = outputs.tokens.shape[0]
    if (
        not isinstance(logical_rows, int)
        or isinstance(logical_rows, bool)
        or not isinstance(block_size, int)
        or isinstance(block_size, bool)
    ):
        raise ValueError("logical_rows and block_size must be host integers")
    if block_size < 1 or gamma < 1:
        raise ValueError("block_size and gamma must be positive")
    if not batch_size <= logical_rows <= batch_size * (gamma + 1):
        raise ValueError("logical_rows must include anchors and fit the draft width")
    if capacity < logical_rows:
        raise ValueError("graph capacity must cover the host-planned logical rows")
    if base_block_table.ndim != 2 or base_block_table.shape[0] != batch_size:
        raise ValueError("base_block_table must have one row per request")
    table_width = base_block_table.shape[1]
    if table_width < 1:
        raise ValueError("base_block_table must have at least one cache page")
    _check_tensor("proposals", proposals, (batch_size, gamma), torch.int64, device)
    for name, tensor, dtype in (
        ("anchors", anchors, torch.int64),
        ("base_positions", base_positions, torch.int32),
        ("base_kv_lengths", base_kv_lengths, torch.int32),
        ("widths", widths, torch.int32),
    ):
        _check_tensor(name, tensor, (batch_size,), dtype, device)
    _check_tensor("base_block_table", base_block_table, (batch_size, table_width), torch.int32, device)
    _check_tensor("offsets", offsets, (batch_size + 1,), torch.int32, device)
    for name in (
        "tokens",
        "positions",
        "kv_seq_lens",
        "slot_mapping",
        "q_seq_lens",
        "owners",
        "columns",
    ):
        _check_tensor(name, getattr(outputs, name), (capacity,), torch.int32, device)
    _check_tensor("block_table", outputs.block_table, (capacity, table_width), torch.int32, device)
    _check_disjoint_outputs(
        (anchors, proposals, base_positions, base_kv_lengths, base_block_table, widths),
        (
            offsets,
            outputs.tokens,
            outputs.positions,
            outputs.kv_seq_lens,
            outputs.slot_mapping,
            outputs.block_table,
            outputs.q_seq_lens,
            outputs.owners,
            outputs.columns,
        ),
    )
    build_verify_offsets(widths, offsets)
    outputs.tokens.fill_(1)
    outputs.positions.zero_()
    outputs.kv_seq_lens.fill_(1)
    outputs.slot_mapping.fill_(-1)
    outputs.block_table.zero_()
    outputs.q_seq_lens.fill_(1)
    outputs.owners.fill_(-1)
    outputs.columns.fill_(-1)
    if batch_size == 0 or capacity == 0:
        return

    rows = torch.arange(capacity, dtype=torch.int32, device=device)
    valid = rows < offsets[-1]
    owners = torch.searchsorted(offsets[1:], rows, right=True)
    safe_owners = owners.clamp(max=batch_size - 1).to(torch.int64)
    columns = rows - offsets[safe_owners]
    safe_columns = columns.clamp(min=0, max=gamma).to(torch.int64)
    gathered_tokens = torch.where(
        safe_columns == 0,
        anchors[safe_owners],
        proposals[safe_owners, (safe_columns - 1).clamp_min(0)],
    ).to(torch.int32)
    positions = base_positions[safe_owners] + columns
    safe_positions = torch.where(valid, positions, torch.zeros_like(positions))
    page_columns = torch.div(safe_positions, block_size, rounding_mode="floor").to(torch.int64)
    slots = base_block_table[safe_owners, page_columns] * block_size + torch.remainder(safe_positions, block_size)
    outputs.tokens.copy_(torch.where(valid, gathered_tokens, 1))
    outputs.positions.copy_(safe_positions)
    outputs.kv_seq_lens.copy_(torch.where(valid, base_kv_lengths[safe_owners] + columns, 1))
    outputs.slot_mapping.copy_(torch.where(valid, slots, -1))
    outputs.block_table.copy_(torch.where(valid.unsqueeze(1), base_block_table[safe_owners], 0))
    outputs.owners.copy_(torch.where(valid, owners.to(torch.int32), -1))
    outputs.columns.copy_(torch.where(valid, columns, -1))
