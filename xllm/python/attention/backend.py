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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from math import gcd, lcm
from typing import TYPE_CHECKING, Protocol, TypeGuard

import torch

from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadataLike,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.layers.kda import KdaForgetGate
    from xllm.python.model_executor.cp_utils import CpContext


def has_rope_dim(tensor: torch.Tensor | None) -> TypeGuard[torch.Tensor]:
    """Whether an MLA rope operand carries a rope dimension.

    NoPE models (``qk_rope_head_dim == 0``) surface the rope operand as ``None``
    or as a zero-width tensor; both mean "skip the rope path".
    """
    return tensor is not None and tensor.shape[-1] > 0


@dataclass(frozen=True, slots=True)
class LayerCache:
    """Every cache a layer may own, named rather than positional.

    A layer holds a subset: full attention uses ``key``/``value``, the MLA
    sparse indexer adds ``index`` and optionally ``kpool_tail``, and a
    linear-attention layer uses ``conv``/``ssm`` instead of K/V. An absent
    slot is ``None`` rather than an empty tensor, so a layer that reads the
    wrong slot fails loudly.
    """

    key: torch.Tensor | None
    value: torch.Tensor | None
    index: torch.Tensor | None = None
    conv: torch.Tensor | None = None
    ssm: torch.Tensor | None = None
    # DeepSeek-V4 compressed-attention cache slots. Generic models leave these
    # as None; the tuple order is shared with PyExecutorImpl::bind_kv_caches.
    swa: torch.Tensor | None = None
    compress_kv_state: torch.Tensor | None = None
    compress_score_state: torch.Tensor | None = None
    compress_index_kv_state: torch.Tensor | None = None
    compress_index_score_state: torch.Tensor | None = None
    indexer_scale: torch.Tensor | None = None
    kpool_tail: torch.Tensor | None = None
    # Shared slabs expose [2, logical capacity]: Conv rows and SSM checkpoint
    # starts for this layer. Legacy fixed storage exposes one group slot row.
    linear_state_slots: torch.Tensor | None = None
    # Slab storage maps logical Full blocks to this layer's paired K/V rows.
    full_block_rows: torch.Tensor | None = None
    shared_checkpoint_stride: int | None = None

    @property
    def index_scale(self) -> torch.Tensor | None:
        """Legacy MLA alias for the shared indexer scale cache."""
        return self.indexer_scale


@dataclass(frozen=True, slots=True)
class SharedKVCacheLayout:
    """Physical row geometry for a shared Full/Linear cache arena.

    Slab storage resolves logical ids with each layer's stable physical row
    maps. Full K/V allocations are adjacent, while Conv and SSM allocations
    are independent. Slab count and logical id capacity need not agree.

    Legacy fixed storage uses the strides below: a Full block id is one slot, and the
    layer reads its lane's page inside that slot.  A recurrent state id is its
    group-0 slot; layers of other groups first map the state id through their
    own ``LayerCache.linear_state_slots`` row.  Both then apply the
    role-specific row mapping below.  The descriptor is deliberately
    independent of the native cache object because the Python executor
    receives only tensor views.
    """

    page_size_tokens: int
    full_row_stride: int
    conv_row_stride: int | None
    ssm_row_stride: int
    checkpoint_stride: int
    num_slots: int
    slot_bytes: int
    pages_per_slot: int = 1
    num_linear_groups: int = 1
    conv_row_bytes: int = 0
    scratch_full_rows: int = 0
    scratch_conv_rows: int = 0
    scratch_ssm_rows: int = 0
    scratch_full_offset: int = 0
    scratch_conv_offset: int = 0
    scratch_ssm_offset: int = 0
    num_slabs: int = 0

    def __post_init__(self) -> None:
        values = (
            self.page_size_tokens,
            self.full_row_stride,
            self.ssm_row_stride,
            self.checkpoint_stride,
            self.num_slots,
            self.slot_bytes,
            self.pages_per_slot,
            self.num_linear_groups,
        )
        if any(value <= 0 for value in values):
            raise ValueError("shared KV cache layout dimensions must be positive")
        if self.slot_bytes % self.pages_per_slot:
            raise ValueError("shared slot bytes must hold whole physical pages")
        if self.conv_row_stride is None:
            if self.conv_row_bytes <= 0 or self.slot_bytes % self.conv_row_bytes == 0:
                raise ValueError("shared Conv ceiling mapping requires non-integral row geometry")
        elif self.conv_row_stride <= 0:
            raise ValueError("shared Conv row stride must be positive")
        if self.conv_row_bytes < 0:
            raise ValueError("shared Conv row bytes must be non-negative")
        if self.conv_row_bytes > 0 and self.conv_row_stride is not None:
            if self.slot_bytes % self.conv_row_bytes or self.conv_row_stride != self.slot_bytes // self.conv_row_bytes:
                raise ValueError("shared Conv row stride does not match its physical slot")
        scratch_values = (
            (self.scratch_full_offset, self.scratch_full_rows),
            (self.scratch_conv_offset, self.scratch_conv_rows),
            (self.scratch_ssm_offset, self.scratch_ssm_rows),
        )
        if any(offset < 0 or rows < 0 for offset, rows in scratch_values):
            raise ValueError("shared KV cache scratch extents must be non-negative")

    def full_block_table(self, block_table: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Translate logical Full page ids while preserving negative sentinels."""
        if rows is not None:
            physical = rows.index_select(0, block_table.clamp_min(0).reshape(-1).to(torch.int64)).reshape_as(
                block_table
            )
            return torch.where(block_table >= 0, physical.to(block_table.dtype), block_table)
        if self.num_slabs:
            raise ValueError("shared slab Full cache requires the reading layer's block rows")
        if block_table.numel() == 0 or self.full_row_stride == 1:
            return block_table
        return torch.where(
            block_table >= 0,
            block_table * self.full_row_stride,
            block_table,
        )

    def full_slot_mapping(self, slot_mapping: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Translate logical token slots and retain their in-page offsets."""
        if rows is not None:
            logical_rows = torch.div(slot_mapping, self.page_size_tokens, rounding_mode="floor")
            physical_rows = self.full_block_table(logical_rows, rows)
            translated = physical_rows * self.page_size_tokens + torch.remainder(slot_mapping, self.page_size_tokens)
            return torch.where(slot_mapping >= 0, translated, slot_mapping)
        if self.num_slabs:
            raise ValueError("shared slab Full cache requires the writing layer's block rows")
        if slot_mapping.numel() == 0 or self.full_row_stride == 1:
            return slot_mapping
        rows = torch.div(slot_mapping, self.page_size_tokens, rounding_mode="floor")
        offsets = torch.remainder(slot_mapping, self.page_size_tokens)
        translated = rows * self.full_row_stride * self.page_size_tokens + offsets
        return torch.where(slot_mapping >= 0, translated, slot_mapping)

    def linear_state_indices(
        self,
        state_indices: torch.Tensor,
        *,
        role: str,
        slots: torch.Tensor | None = None,
        checkpoint_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Translate Conv or SSM state ids to physical row indices.

        ``slots`` is the reading layer's ``LayerCache.linear_state_slots`` row
        and is required whenever the arena holds more than one Linear group.
        """
        if self.num_slabs:
            if role not in ("conv", "ssm"):
                raise ValueError(f"unsupported shared linear cache role: {role}")
            if slots is None:
                raise ValueError("shared slab Linear cache requires the reading layer's role rows")
            if role == "conv" and checkpoint_offsets is not None:
                raise ValueError("checkpoint offsets are valid only for SSM indices")
            role_rows = slots[0 if role == "conv" else 1]
            physical = role_rows.index_select(0, state_indices.clamp_min(0).to(torch.int64))
            if checkpoint_offsets is not None:
                physical = physical + checkpoint_offsets
            return torch.where(state_indices >= 0, physical.to(state_indices.dtype), state_indices)
        if slots is not None:
            physical = slots.index_select(0, state_indices.clamp_min(0).to(torch.int64))
            state_indices = torch.where(state_indices >= 0, physical.to(state_indices.dtype), state_indices)
        elif self.num_linear_groups > 1:
            raise ValueError("shared Linear groups require the reading layer's state slot row")
        if role == "conv":
            if checkpoint_offsets is not None:
                raise ValueError("checkpoint offsets are valid only for SSM indices")
            stride = self.conv_row_stride
            if stride is None:
                divisor = gcd(self.slot_bytes, self.conv_row_bytes)
                numerator = self.slot_bytes // divisor
                denominator = self.conv_row_bytes // divisor
                offsets = state_indices.to(torch.int64).clamp_min(0) * numerator
                rows = torch.div(offsets, denominator, rounding_mode="floor") + torch.remainder(
                    offsets, denominator
                ).ne(0)
                return torch.where(state_indices >= 0, rows, state_indices).to(state_indices.dtype)
        elif role == "ssm":
            stride = self.ssm_row_stride
        else:
            raise ValueError(f"unsupported shared linear cache role: {role}")
        rows = torch.where(state_indices >= 0, state_indices * stride, state_indices)
        if checkpoint_offsets is not None:
            rows = torch.where(state_indices >= 0, rows + checkpoint_offsets, rows)
        return rows


def _tensor_storage_span(tensor: torch.Tensor) -> tuple[int, int, int]:
    """Return storage identity, byte offset, and storage capacity."""
    storage = tensor.untyped_storage()
    storage_base = storage.data_ptr()
    offset = tensor.data_ptr() - storage_base
    return storage_base, offset, storage.nbytes()


def _tensor_row_bytes(tensor: torch.Tensor) -> int:
    if tensor.ndim < 2 or not tensor.is_contiguous() or tensor.shape[0] <= 0:
        raise ValueError("shared KV cache tensors must be contiguous row views")
    return tensor.element_size() * tensor[0].numel()


def _detect_shared_slab_layout(
    full: Sequence[LayerCache], linear: Sequence[LayerCache], arena_bytes: int
) -> SharedKVCacheLayout:
    key, conv, ssm = full[0].key, linear[0].conv, linear[0].ssm
    assert key is not None and conv is not None and ssm is not None
    stride = linear[0].shared_checkpoint_stride
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("shared slab caches require an explicit positive checkpoint stride")
    full_bytes, conv_bytes, ssm_bytes = (_tensor_row_bytes(tensor) for tensor in (key, conv, ssm))
    slab_bytes = lcm(2 * full_bytes, conv_bytes, stride * ssm_bytes)
    num_slabs, remainder = divmod(arena_bytes, slab_bytes)
    if remainder or num_slabs < 2:
        raise ValueError("shared slab arena must hold scratch and allocatable physical slabs")
    first_rows = full[0].full_block_rows
    if first_rows is None or first_rows.ndim != 1 or first_rows.numel() < 2:
        raise ValueError("shared slab Full caches require a logical block row map")
    num_slots = first_rows.numel()
    for cache in (*full, *linear):
        if cache.shared_checkpoint_stride != stride:
            raise ValueError("shared slab caches must use one checkpoint stride")
    for cache in full:
        assert cache.key is not None and cache.value is not None
        rows = cache.full_block_rows
        if rows is None or rows.shape != (num_slots,) or rows.dtype != torch.int32 or rows.device != key.device:
            raise ValueError("shared slab Full caches require device int32 block row maps")
        if cache.linear_state_slots is not None:
            raise ValueError("shared Full cache layers do not own recurrent state slots")
        for tensor, offset in ((cache.key, 0), (cache.value, full_bytes)):
            expected_rows = (arena_bytes - offset) // full_bytes
            if (
                tensor.shape[1:] != key.shape[1:]
                or tensor.shape[0] != expected_rows
                or tensor.dtype != key.dtype
                or _tensor_storage_span(tensor)[1] != offset
                or _tensor_row_bytes(tensor) != full_bytes
            ):
                raise ValueError("shared slab Full views must expose paired dense arena rows")
    for cache in linear:
        assert cache.conv is not None and cache.ssm is not None
        rows = cache.linear_state_slots
        if rows is None or rows.shape != (2, num_slots) or rows.dtype != torch.int32 or rows.device != conv.device:
            raise ValueError("shared slab Linear caches require device Conv and SSM row maps")
        if cache.full_block_rows is not None:
            raise ValueError("shared Linear cache layers do not own Full block rows")
        for tensor, prototype, row_bytes in ((cache.conv, conv, conv_bytes), (cache.ssm, ssm, ssm_bytes)):
            if (
                tensor.shape[1:] != prototype.shape[1:]
                or tensor.shape[0] != arena_bytes // row_bytes
                or tensor.dtype != prototype.dtype
                or _tensor_storage_span(tensor)[1] != 0
                or _tensor_row_bytes(tensor) != row_bytes
            ):
                raise ValueError("shared slab Linear views must expose dense zero-origin arena rows")
    return SharedKVCacheLayout(
        page_size_tokens=int(key.shape[1]),
        full_row_stride=slab_bytes // full_bytes,
        conv_row_stride=slab_bytes // conv_bytes,
        ssm_row_stride=slab_bytes // ssm_bytes,
        checkpoint_stride=stride,
        num_slots=num_slots,
        slot_bytes=slab_bytes,
        num_linear_groups=len(linear),
        conv_row_bytes=conv_bytes,
        scratch_full_rows=1,
        scratch_conv_rows=1,
        scratch_ssm_rows=stride,
        num_slabs=num_slabs,
    )


def detect_shared_kv_cache_layout(
    layer_caches: Sequence[LayerCache],
) -> SharedKVCacheLayout | None:
    """Detect and validate the static arena layout exposed as tensor views.

    The native bridge passes cache tensors and the shared checkpoint stride. A normal cache owns one
    storage per role and returns ``None``.  A shared arena is accepted only
    when all Full and Conv/SSM views share one storage and their dimensions
    match the slab allocator's dense views or the legacy static equations.
    Shape ratios alone are not used as a layout signal.
    """
    full = [cache for cache in layer_caches if cache.key is not None or cache.value is not None]
    linear = [cache for cache in layer_caches if cache.conv is not None or cache.ssm is not None]
    if not full or not linear:
        return None
    full_tensors = [tensor for cache in full for tensor in (cache.key, cache.value) if tensor is not None]
    linear_tensors = [tensor for cache in linear for tensor in (cache.conv, cache.ssm) if tensor is not None]
    full_storages = {(tensor.device, _tensor_storage_span(tensor)[0]) for tensor in full_tensors}
    linear_storages = {(tensor.device, _tensor_storage_span(tensor)[0]) for tensor in linear_tensors}
    if not full_storages.intersection(linear_storages):
        return None
    if len(full_storages.union(linear_storages)) != 1:
        raise ValueError("shared KV cache roles must expose one complete arena")
    if any(
        cache.key is None or cache.value is None or cache.conv is not None or cache.ssm is not None for cache in full
    ):
        raise ValueError("shared Full cache layers require only paired key/value roles")
    if any(
        cache.conv is None or cache.ssm is None or cache.key is not None or cache.value is not None for cache in linear
    ):
        raise ValueError("shared Linear cache layers require only paired Conv/SSM roles")
    unsupported = (
        "index",
        "swa",
        "compress_kv_state",
        "compress_score_state",
        "compress_index_kv_state",
        "compress_index_score_state",
        "indexer_scale",
        "kpool_tail",
    )
    if any(getattr(cache, field_name) is not None for cache in layer_caches for field_name in unsupported):
        raise ValueError("shared static KV cache supports only Full key/value and Linear Conv/SSM roles")
    role_tensors = full_tensors + linear_tensors
    spans = [_tensor_storage_span(tensor) for tensor in role_tensors]
    arena_bytes = spans[0][2]
    if arena_bytes <= 0 or any(span[2] != arena_bytes for span in spans):
        raise ValueError("shared KV cache views must expose one complete arena")

    key = full[0].key
    value = full[0].value
    conv = linear[0].conv
    ssm = linear[0].ssm
    assert key is not None and value is not None and conv is not None and ssm is not None
    full_row_bytes = _tensor_row_bytes(key)
    conv_row_bytes = _tensor_row_bytes(conv)
    ssm_row_bytes = _tensor_row_bytes(ssm)
    page_size_tokens = int(key.shape[1])
    if page_size_tokens <= 0 or min(full_row_bytes, conv_row_bytes, ssm_row_bytes) <= 0:
        raise ValueError("shared KV cache rows and Full page size must be positive")
    if any(cache.full_block_rows is not None or cache.shared_checkpoint_stride is not None for cache in full + linear):
        return _detect_shared_slab_layout(full, linear, arena_bytes)
    pages_per_slot = (len(full) + 3) // 4
    num_linear_groups = (len(linear) + pages_per_slot - 1) // pages_per_slot
    full_base = (int(key.shape[0]) - 1) * full_row_bytes
    slot_bytes = arena_bytes - full_base
    if slot_bytes <= 0:
        raise ValueError("shared Full cache view does not fit the arena")
    page_bytes, remainder = divmod(slot_bytes, pages_per_slot)
    if remainder or page_bytes <= 0:
        raise ValueError("cannot derive static physical page bytes from Full views")
    if slot_bytes % full_row_bytes or slot_bytes % ssm_row_bytes:
        raise ValueError("shared slot bytes must align with Full and SSM rows")
    num_slots, remainder = divmod(arena_bytes, slot_bytes)
    if remainder or num_slots < 2:
        raise ValueError("shared arena cannot fit padding and one allocatable slot")

    last_slot_bytes = (num_slots - 1) * slot_bytes
    conv_base = (int(conv.shape[0]) - 1) * conv_row_bytes
    expected_conv_start = (last_slot_bytes + conv_row_bytes - 1) // conv_row_bytes * conv_row_bytes
    if conv_base != expected_conv_start:
        raise ValueError("shared Conv cache view geometry is inconsistent")
    checkpoint_rows = int(ssm.shape[0]) - last_slot_bytes // ssm_row_bytes
    if checkpoint_rows <= 0:
        raise ValueError("shared SSM cache view geometry is inconsistent")
    linear_ssm_offset = (conv_row_bytes + ssm_row_bytes - 1) // ssm_row_bytes * ssm_row_bytes
    page_payload_bytes = max(8 * full_row_bytes, linear_ssm_offset + checkpoint_rows * ssm_row_bytes)
    row_alignment = lcm(full_row_bytes, ssm_row_bytes)
    candidate = (page_payload_bytes + row_alignment - 1) // row_alignment * row_alignment
    conv_gap = conv_row_bytes - gcd(candidate, conv_row_bytes)
    if linear_ssm_offset >= conv_row_bytes + conv_gap:
        expected_page_bytes = candidate
    else:
        row_alignment = lcm(row_alignment, conv_row_bytes)
        expected_page_bytes = (page_payload_bytes + row_alignment - 1) // row_alignment * row_alignment
    if page_bytes != expected_page_bytes:
        raise ValueError("shared physical page bytes do not match the static allocator layout")

    for tensor, (_, offset, capacity) in zip(role_tensors, spans, strict=True):
        row_bytes = _tensor_row_bytes(tensor)
        view_bytes = int(tensor.shape[0]) * row_bytes
        if offset < 0 or offset + view_bytes > capacity or offset % tensor.element_size():
            raise ValueError("shared KV cache role view is outside aligned arena storage")

    for layer_index, cache in enumerate(full):
        assert cache.key is not None and cache.value is not None
        for tensor in (cache.key, cache.value):
            if tensor.shape != key.shape or tensor.dtype != key.dtype:
                raise ValueError("shared Full cache views have inconsistent page geometry")
        expected = (layer_index // 4) * page_bytes + (layer_index % 4) * 2 * full_row_bytes
        if cache.linear_state_slots is not None:
            raise ValueError("shared Full cache layers do not own recurrent state slots")
        if (
            _tensor_storage_span(cache.key)[1] != expected
            or _tensor_storage_span(cache.value)[1] != expected + full_row_bytes
        ):
            raise ValueError("shared Full cache views do not match static page role offsets")
    for layer_index, cache in enumerate(linear):
        assert cache.conv is not None and cache.ssm is not None
        if cache.conv.shape != conv.shape or cache.conv.dtype != key.dtype:
            raise ValueError("shared Linear Conv views have inconsistent row geometry")
        if cache.ssm.shape != ssm.shape or cache.ssm.dtype != ssm.dtype:
            raise ValueError("shared Linear SSM views have inconsistent row geometry")
        expected = (layer_index % pages_per_slot) * page_bytes
        if (
            _tensor_storage_span(cache.conv)[1] != expected
            or _tensor_storage_span(cache.ssm)[1] != expected + linear_ssm_offset
        ):
            raise ValueError("shared Linear cache views do not match static page role offsets")
        slots = cache.linear_state_slots
        if slots is None or slots.shape != (num_slots,) or slots.dtype != torch.int32 or slots.device != conv.device:
            raise ValueError("shared Linear cache layers require a device state slot row per slot")

    return SharedKVCacheLayout(
        page_size_tokens=page_size_tokens,
        full_row_stride=slot_bytes // full_row_bytes,
        conv_row_stride=slot_bytes // conv_row_bytes if slot_bytes % conv_row_bytes == 0 else None,
        ssm_row_stride=slot_bytes // ssm_row_bytes,
        checkpoint_stride=checkpoint_rows,
        num_slots=num_slots,
        slot_bytes=slot_bytes,
        pages_per_slot=pages_per_slot,
        num_linear_groups=num_linear_groups,
        conv_row_bytes=conv_row_bytes,
        # StaticKVCacheBlockPool reserves slot 0 for padding in every group,
        # so each role's first rows are graph-owned scratch even though the
        # tensor views have different row strides.
        scratch_full_rows=1,
        scratch_conv_rows=1,
        scratch_ssm_rows=checkpoint_rows,
    )


#: Field order of the tuple form, which is what the C++ executor hands over.
_LAYER_CACHE_SLOTS = tuple(field.name for field in fields(LayerCache))

LayerCacheInput = LayerCache | tuple[torch.Tensor | int | None, ...]


def normalize_layer_caches(caches: Sequence[LayerCacheInput]) -> list[LayerCache]:
    """Accept the tuple form and return named caches with empty slots dropped."""
    normalized: list[LayerCache] = []
    for cache in caches:
        if isinstance(cache, LayerCache):
            normalized.append(cache)
            continue
        if not 2 <= len(cache) <= len(_LAYER_CACHE_SLOTS):
            raise ValueError(f"layer cache must hold between K/V and {'/'.join(_LAYER_CACHE_SLOTS)} fields")
        slots = [
            tensor if isinstance(tensor, int) else None if tensor is None or not tensor.numel() else tensor
            for tensor in cache
        ]
        slots.extend([None] * (len(_LAYER_CACHE_SLOTS) - len(slots)))
        normalized.append(LayerCache(*slots))
    return normalized


@dataclass(frozen=True, slots=True)
class LinearVerifyGraphInputs:
    """Stable request checkpoints and token layout for recurrent graphs.

    Packed verifiers consume cu_seqlens without dense token maps. Dense
    verifiers use the two maps, whose missing entries select appended zero
    rows. The graph refreshes values, never shapes or storage, before layers
    consume this descriptor.
    """

    state_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor
    query_width: int
    packed_to_dense: torch.Tensor | None = None
    dense_to_packed: torch.Tensor | None = None
    cu_seqlens: torch.Tensor | None = None


class ExecutionMetadataSource(Protocol):
    slot_mapping: torch.Tensor
    q_cu_seq_lens: torch.Tensor | None
    q_cu_seq_lens_host_values: list[int] | None
    q_seq_lens_host: torch.Tensor | None
    linear_state_indices: torch.Tensor | None
    linear_verify_graph: LinearVerifyGraphInputs | None
    linear_state_read_indices: torch.Tensor | None
    linear_state_write_indices: torch.Tensor | None
    num_accepted_tokens: torch.Tensor | None
    num_accepted_tokens_host_values: Sequence[int]
    has_initial_state: torch.Tensor | None
    dp_execution_token_counts: Sequence[int]
    is_prefill: bool
    is_chunked_prefill: bool
    is_mixed: bool
    is_spec_verify: bool
    is_dummy: bool


class AttentionMetadata(ExecutionMetadataSource, Protocol):
    prepared_attention_state: object | None
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None
    kv_cu_seq_lens: torch.Tensor | None
    kv_seq_lens_host: torch.Tensor | None
    kv_seq_lens_host_values: list[int] | None
    new_cache_slots_host_values: list[int] | None
    paged_kv_indptr_host: torch.Tensor | None
    paged_kv_last_page_len_host: torch.Tensor | None
    block_table: torch.Tensor | None
    kv_seq_lens: torch.Tensor | None
    max_query_len: int
    max_seq_len: int
    multi_block_tables: Sequence[torch.Tensor | None]
    # Legacy C++/pybind DSAMetadata field names retained for ABI compatibility.
    dsa_metadata: object | None
    dsa_positions: torch.Tensor | None
    dsa_cos_sin: torch.Tensor | None
    dsa_c4_cos_sin: torch.Tensor | None
    dsa_c128_cos_sin: torch.Tensor | None
    dsa_graph_block_table_cols: int
    dsa_graph_mode: bool
    kpool_query_lens: Sequence[int]
    raw_dp_execution_token_counts: Sequence[int]
    dp_global_sequence_nums: Sequence[int]
    # Host-planned logical KV lengths, including the query; empty ranks keep zero.
    dp_global_kv_max_seq_lens: Sequence[int]
    dp_is_decode: Sequence[int]
    q_seq_lens: torch.Tensor | None
    expanded_decode_metadata: ExpandedDecodeMetadataLike
    local_slot_mapping: torch.Tensor | None
    kv_split_size: int
    kv_split_rank: int
    has_kv_shard: bool


def resolve_linear_state_io_indices(
    metadata: ExecutionMetadataSource,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve linear-state read and live/write slots for one forward."""
    write_indices = getattr(metadata, "linear_state_write_indices", None)
    if write_indices is None:
        write_indices = getattr(metadata, "linear_state_indices", None)
    read_indices = getattr(metadata, "linear_state_read_indices", None)
    if read_indices is None or read_indices is write_indices:
        return write_indices, write_indices
    if write_indices is None:
        raise RuntimeError("linear-state read indices require write indices")
    if read_indices.shape != write_indices.shape:
        raise RuntimeError(
            "linear-state read/write indices must have the same shape: "
            f"read={tuple(read_indices.shape)}, write={tuple(write_indices.shape)}"
        )
    is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
    if not is_prefill or getattr(metadata, "is_spec_verify", False):
        if not torch.equal(read_indices, write_indices):
            raise RuntimeError("linear-state read/write separation is only supported for non-speculative prefill")
        # Canonicalize equal metadata to one object. Hot-path state users can
        # then skip an otherwise redundant snapshot and self-copy.
        return write_indices, write_indices
    return read_indices, write_indices


@dataclass(frozen=True)
class MlaIndexContext:
    """Public contract handed to an optional LightningIndexer.

    Replaces direct model access to ``backend._metadata`` / ``backend._kv_caches``
    for MLA layers. The backend owns the paged index cache (``LayerCache.index``)
    and prepares the paging / sequence-length metadata once per step; the indexer
    receives this view and produces ``topk``. ``materialize_index_cache`` returns
    the cache, optional scale, and their single matching block table together.
    """

    index_cache: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor | None
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor
    index_cache_scale: torch.Tensor | None
    get_quant_indexer_metadata: Callable[[int, int, int, int], torch.Tensor]
    update_index_cache: Callable[[torch.Tensor, torch.Tensor | None], None]
    materialize_index_cache: Callable[
        [],
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor],
    ]
    # Optional model-side CP plan for packed local query segments.
    cp_context: CpContext | None = None
    # Host-planned KV lengths, when the model needs a shape bound without
    # synchronizing the device sequence-length tensor.
    actual_seq_kv_host: Sequence[int] | None = None


@dataclass(frozen=True)
class MlaPreprocessContext:
    """Decode cache tensors consumed by fused MLA preprocessing."""

    kv_cache: torch.Tensor
    rope_cache: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass(frozen=True)
class CsaIndexContext:
    """Per-forward cache and metadata view consumed by the DSV4 indexer."""

    index_cache: torch.Tensor
    indexer_scale: torch.Tensor | None
    slot_mapping: torch.Tensor
    block_table: torch.Tensor | None
    cmp_block_table: torch.Tensor | None
    kv_state: torch.Tensor | None
    score_state: torch.Tensor | None
    kv_block_table: torch.Tensor | None
    score_block_table: torch.Tensor | None
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor
    start_pos: torch.Tensor | None
    qli_metadata: torch.Tensor | None


class AttentionBackend(ABC):
    def prepare_owned_graph_metadata(self, metadata: AttentionMetadata) -> None:
        """Prepare metadata whose storage is owned by this graph entry."""
        self.prepare(metadata, graph_mode=True)

    @property
    def uses_executor_cp_context(self) -> bool:
        return True

    def create_graph_block_tables(
        self,
        batch_size: int,
        max_model_len: int,
        max_block_columns: int,
    ) -> tuple[torch.Tensor, ...]:
        return ()

    @property
    def supports_linear_spec_verify_graph(self) -> bool:
        """Whether expanded speculative verification can capture linear state."""
        return False

    @property
    def supports_prepared_metadata(self) -> bool:
        return False

    def prepare_metadata(self, metadata: AttentionMetadata) -> object:
        """Build private Host metadata without changing an active forward."""
        raise NotImplementedError("attention backend does not support private prepared metadata")

    def prepare_graph_replay(self, metadata: AttentionMetadata) -> None:
        """Install live metadata for an already captured graph entry."""
        self.prepare(metadata, graph_mode=True)

    def reset_forward(self, metadata: AttentionMetadata | None = None) -> None:
        """Reset request-owned state before a model attaches current inputs."""
        del metadata

    @property
    def is_mla(self) -> bool:
        """Whether this backend implements MLA execution."""
        return False

    @abstractmethod
    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        pass

    @abstractmethod
    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_kv_blocks(self) -> int:
        pass

    @property
    @abstractmethod
    def page_size(self) -> int:
        pass

    @property
    def logical_page_size(self) -> int:
        """Number of global KV tokens represented by one block-table entry."""
        return self.page_size

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        layer: Attention,
        topk: torch.Tensor | None = None,
        cache_is_preprocessed: bool = False,
    ) -> torch.Tensor:
        """Absorbed-MLA attention over paged latent (nope) + rope caches.

        Returns ``[T, H, kv_lora]``; caller bmm's ``W_UV``. When ``topk`` is
        provided, dispatches to the sparse SFA path driven by an optional
        LightningIndexer; otherwise a dense MLA path is requested. Backends
        that do not implement MLA raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support MLA")

    def mla_preprocess_context(
        self,
        layer: Attention,
    ) -> MlaPreprocessContext | None:
        """Return decode cache tensors for a fused preprocessing region."""
        del layer
        return None

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        beta: torch.Tensor,
        layer_id: int,
        conv_weight: torch.Tensor,
        forget_gate: KdaForgetGate,
        activation: str,
        raw_gate_proj: torch.Tensor,
    ) -> torch.Tensor:
        """KDA linear attention (conv1d + delta-rule) over framework state.

        ``mixed_qkv`` is ``[B, 3*qkv_dim, S]`` (q/k/v concatenated, the local
        head-subset already sharded). ``raw_gate_proj`` is the pre-gate forget
        projection ``f_b(f_a(x))`` shaped ``[B, S, num_heads_local, head_dim]``;
        the plain decode/prefill kernels fuse the safe-gate from it in-kernel,
        while the MTP verify / cross-layer / mask paths materialize the gate
        from it on demand. ``beta`` is ``[B, S, num_heads_local]``. Returns the
        core attention output ``[B, S, num_heads_local, head_dim]`` for the
        caller to gate + project.

        The backend owns the per-layer conv/ssm state (the conv/ssm slots of
        ``LayerCache``) and reads/advances/writes it via the
        ``linear_state_indices`` / ``has_initial_state`` metadata view.
        Backends that do not implement linear attention raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support linear attention (KDA)")

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        """Public hook for an optional LightningIndexer.

        Hands out the paged index cache view (``LayerCache.index``) plus the
        paging / sequence-length metadata the indexer needs, so the model never
        touches ``backend._metadata`` / ``backend._kv_caches`` directly.
        Backends that do not support the sparse MLA indexer raise.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support MLA indexer")
