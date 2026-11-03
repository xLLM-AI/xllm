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

"""Bind one sparse Unified variant's metadata to its final graph addresses."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

SparseMetadataFactory = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]


class MtpSparsePositionStorage:
    """Final int64 positions for K draft steps, target rows and first repair."""

    def __init__(self, batch_size: int, speculative_tokens: int, device: torch.device) -> None:
        if batch_size <= 0 or speculative_tokens <= 0:
            raise ValueError("sparse MTP position dimensions must be positive")
        self.batch_size = batch_size
        self.speculative_tokens = speculative_tokens
        count = (2 * speculative_tokens + 1) * batch_size
        self.arena = torch.empty(count + 2 * batch_size, dtype=torch.long, device=device)
        self.positions = self.arena[:count]
        self.first_draft = self.arena[count:]


class MtpSparseMetadataStorage:
    """Allocate final role storage before constructing its native metadata views.

    All views are created from this allocation. No caller-owned TensorImpl is
    rebound, and the fused updater initializes every field before capture.
    """

    def __init__(
        self,
        factory: SparseMetadataFactory,
        rows: Sequence[int],
        capacity: int,
        device: torch.device,
        *,
        share_block_tables: bool = False,
    ) -> None:
        if not rows or any(count <= 0 for count in rows) or capacity <= 0 or capacity % 32:
            raise ValueError("sparse MTP storage requires positive rows and aligned table capacity")
        shapes = [shape for count in rows for shape in ((count,), (count,), (count, capacity))]
        spans: list[tuple[int, int]] = []
        tables: dict[int, tuple[int, int]] = {}
        words = 0
        for index, shape in enumerate(shapes):
            size = shape[0] * (shape[1] if len(shape) == 2 else 1)
            if share_block_tables and index % 3 == 2 and shape[0] in tables:
                spans.append(tables[shape[0]])
                continue
            span = (words, size)
            spans.append(span)
            words += ((size + 127) // 128) * 128
            if share_block_tables and index % 3 == 2:
                tables[shape[0]] = span
        if words > (1 << 31) - 1:
            raise ValueError("sparse MTP arena offsets must fit int32")
        self.arena = torch.empty(words, dtype=torch.int32, device=device)
        views = [self.arena.narrow(0, offset, size).view(shape) for shape, (offset, size) in zip(shapes, spans)]

        self.metadata = tuple(
            factory(views[index + 2], views[index + 1], views[index]) for index in range(0, len(views), 3)
        )


def _role_layout(
    metadata: Sequence[object], arenas: Sequence[torch.Tensor], rows: Sequence[int], capacity: int
) -> tuple[torch.Tensor, list[int]]:
    """Validate the native graph clones once, before publishing a binding."""
    values = [getattr(item, field) for item in metadata for field in ("slot_mapping", "kv_seq_lens", "block_table")]
    if len(values) != len(rows) * 3 or not values:
        raise ValueError("sparse MTP metadata must have three fields per role step")
    storage = values[0].untyped_storage().data_ptr()
    matching = [arena for arena in arenas if arena.untyped_storage().data_ptr() == storage]
    if len(matching) != 1:
        raise ValueError("sparse MTP metadata must belong to one graph-owned int32 arena")
    arena = matching[0]
    if arena.dtype != torch.int32 or arena.ndim != 1 or not arena.is_contiguous():
        raise ValueError("sparse MTP arena must be a contiguous int32 vector")
    if arena.numel() > (1 << 31) - 1:
        raise ValueError("sparse MTP arena offsets must fit int32")
    offsets: list[int] = []
    spans: list[tuple[int, int, bool]] = []
    for index, value in enumerate(values):
        count = rows[index // 3]
        shape = (count, capacity) if index % 3 == 2 else (count,)
        if value.dtype != torch.int32 or value.device != arena.device or tuple(value.shape) != shape:
            raise ValueError("sparse MTP metadata dtype, device or shape differs from its fixed layout")
        if not value.is_contiguous() or value.untyped_storage().data_ptr() != storage:
            raise ValueError("sparse MTP metadata must be a contiguous view of its role arena")
        offset = value.storage_offset() - arena.storage_offset()
        end = offset + value.numel()
        if offset < 0 or end > arena.numel() or offset % 128:
            raise ValueError("sparse MTP metadata must fit an aligned, owned arena slice")
        is_table = index % 3 == 2
        for previous_start, previous_end, previous_is_table in spans:
            overlaps = offset < previous_end and previous_start < end
            same_table = is_table and previous_is_table and (offset, end) == (previous_start, previous_end)
            if overlaps and not same_table:
                raise ValueError("sparse MTP permits overlap only for identical block-table destinations")
        offsets.append(offset)
        spans.append((offset, end, is_table))
    return arena, offsets


class MtpSparseMetadataBinding:
    """Immutable destinations; only the fused updater writes them per replay.

    The runner owns the adapters and their arenas. This binding shares those
    allocations and lives in the same bounded variant entry, with no staging
    copy of the expanded tables or derived slots and lengths.
    """

    def __init__(
        self,
        draft_metadata: Sequence[object],
        draft_arenas: Sequence[torch.Tensor],
        target_metadata: Sequence[object],
        target_arenas: Sequence[torch.Tensor],
        *,
        position_storage: MtpSparsePositionStorage,
        batch_size: int,
        speculative_tokens: int,
        table_capacity: int,
        block_size: int,
        target_step_major: bool,
    ) -> None:
        if batch_size <= 0 or speculative_tokens <= 0 or block_size <= 0:
            raise ValueError("sparse MTP dimensions must be positive")
        if table_capacity <= 0 or table_capacity % 32:
            raise ValueError("sparse MTP table capacity must be a multiple of 32")
        if position_storage.batch_size != batch_size or position_storage.speculative_tokens != speculative_tokens:
            raise ValueError("sparse MTP position storage must match the variant")
        self._position_arena = position_storage.arena
        self._draft_arena, draft_offsets = _role_layout(
            draft_metadata, draft_arenas, [2 * batch_size] + [batch_size] * (speculative_tokens - 1), table_capacity
        )
        self._target_arena, target_offsets = _role_layout(
            target_metadata, target_arenas, [batch_size * (speculative_tokens + 1)], table_capacity
        )
        if self._draft_arena.device != self._target_arena.device:
            raise ValueError("sparse MTP roles must share a device")
        if self._position_arena.device != self._draft_arena.device:
            raise ValueError("sparse MTP positions must share the role device")
        if self._draft_arena.untyped_storage().data_ptr() == self._target_arena.untyped_storage().data_ptr():
            raise ValueError("sparse MTP draft and target must own separate arenas")
        self._layout = torch.tensor(draft_offsets + target_offsets, dtype=torch.int32, device=self._draft_arena.device)
        self._speculative_tokens = speculative_tokens
        self._table_capacity = table_capacity
        self._block_size = block_size
        self._target_step_major = target_step_major

    @property
    def table_capacity(self) -> int:
        return self._table_capacity

    def update(
        self,
        block_table: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        first_kv_seq_lens: torch.Tensor,
        first_slots: torch.Tensor,
    ) -> None:
        # Runs explicitly before replay on the caller's compute stream. The
        # Worker has already waited for the previous accepted state/input event.
        torch.ops.xllm_ops.mtp_sparse_metadata_update(
            block_table,
            base_positions,
            kv_seq_lens,
            first_kv_seq_lens,
            first_slots,
            self._layout,
            self._draft_arena,
            self._target_arena,
            self._position_arena,
            self._speculative_tokens,
            self._table_capacity,
            self._block_size,
            self._target_step_major,
        )


class MtpPagedMetadataBinding:
    """Qwen's verifier-conditioned W-row prefix and later one-row proposals."""

    def __init__(
        self,
        draft_metadata: Sequence[object],
        draft_arenas: Sequence[torch.Tensor],
        target_metadata: Sequence[object],
        target_arenas: Sequence[torch.Tensor],
        *,
        position_storage: MtpSparsePositionStorage,
        batch_size: int,
        speculative_tokens: int,
        table_capacity: int,
        block_size: int,
        target_step_major: bool,
    ) -> None:
        if target_step_major:
            raise ValueError("Qwen unified MTP requires request-major verification")
        width = speculative_tokens + 1
        _role_layout(
            draft_metadata, draft_arenas, [batch_size * width] + [batch_size] * (speculative_tokens - 1), table_capacity
        )
        _role_layout(target_metadata, target_arenas, [batch_size * width], table_capacity)
        self._draft = tuple(draft_metadata)
        self._target = target_metadata[0]
        self._positions = position_storage.positions
        self._batch = batch_size
        self._steps = speculative_tokens
        self._width = width
        self._table_capacity = table_capacity
        self._block_size = block_size

    @property
    def table_capacity(self) -> int:
        return self._table_capacity

    def update(
        self,
        block_table: torch.Tensor,
        base_positions: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        first_kv_seq_lens: torch.Tensor,
        first_slots: torch.Tensor,
    ) -> None:
        def copy_table(metadata: object, width: int) -> None:
            metadata.block_table.zero_()
            metadata.block_table[:, : block_table.shape[1]].copy_(block_table.repeat_interleave(width, dim=0))

        def slots(lengths: torch.Tensor) -> torch.Tensor:
            cache_positions = lengths - 1
            page_indices = torch.div(cache_positions, self._block_size, rounding_mode="floor").long()
            pages = block_table.gather(1, page_indices)
            return (pages * self._block_size + cache_positions.remainder(self._block_size)).flatten()

        first = self._draft[0]
        copy_table(first, self._width)
        first.kv_seq_lens.copy_(first_kv_seq_lens)
        first.slot_mapping.copy_(first_slots)
        for step, metadata in enumerate(self._draft[1:], start=1):
            copy_table(metadata, 1)
            lengths = kv_seq_lens + step - 1
            metadata.kv_seq_lens.copy_(lengths)
            metadata.slot_mapping.copy_(slots(lengths[:, None]))
        offsets = torch.arange(self._width, device=block_table.device)
        target_lengths = kv_seq_lens[:, None] + offsets
        copy_table(self._target, self._width)
        self._target.kv_seq_lens.copy_(target_lengths.flatten())
        self._target.slot_mapping.copy_(slots(target_lengths))
        draft_positions = base_positions[None, :] + torch.arange(self._steps, device=block_table.device)[:, None] - 1
        self._positions[: self._batch * self._steps].copy_(draft_positions.flatten())
        self._positions[self._batch * self._steps :].copy_((base_positions[:, None] + offsets).flatten())
