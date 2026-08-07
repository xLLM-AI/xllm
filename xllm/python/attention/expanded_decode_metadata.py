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

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class ExpandedDecodeMetadataLike(Protocol):
    enabled: bool
    kv_seq_lens: torch.Tensor | None
    block_table: torch.Tensor | None
    paged_kv_indptr: torch.Tensor | None
    paged_kv_indices: torch.Tensor | None
    paged_kv_last_page_len: torch.Tensor | None
    paged_attention_tiling_data: torch.Tensor | None
    kv_seq_lens_host: torch.Tensor | None
    kv_seq_lens_host_values: list[int] | None


@dataclass(frozen=True, slots=True)
class ExpandedDecodeMetadata:
    kv_seq_lens: torch.Tensor
    block_table: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    paged_attention_tiling_data: torch.Tensor | None
    kv_seq_lens_host: torch.Tensor | None
    kv_seq_lens_host_values: list[int] | None
    enabled: bool = True


def resolve_expanded_decode_metadata(
    metadata: object,
    *,
    block_size: int = 0,
) -> ExpandedDecodeMetadata | None:
    expanded = getattr(metadata, "expanded_decode_metadata", None)
    if expanded is None or not bool(getattr(expanded, "enabled", True)):
        return None

    required = {
        "kv_seq_lens": expanded.kv_seq_lens,
        "block_table": expanded.block_table,
        "paged_kv_indptr": expanded.paged_kv_indptr,
        "paged_kv_indices": expanded.paged_kv_indices,
        "paged_kv_last_page_len": expanded.paged_kv_last_page_len,
    }
    missing = [name for name, tensor in required.items() if tensor is None]
    if missing:
        raise RuntimeError(
            "expanded decode metadata is missing: " + ", ".join(missing)
        )

    host_values_source = getattr(expanded, "kv_seq_lens_host_values", None)
    host_values = (
        list(host_values_source) if host_values_source is not None else None
    )
    resolved = ExpandedDecodeMetadata(
        kv_seq_lens=expanded.kv_seq_lens.to(torch.int32),
        block_table=expanded.block_table.to(torch.int32),
        paged_kv_indptr=expanded.paged_kv_indptr,
        paged_kv_indices=expanded.paged_kv_indices,
        paged_kv_last_page_len=expanded.paged_kv_last_page_len,
        paged_attention_tiling_data=expanded.paged_attention_tiling_data,
        kv_seq_lens_host=expanded.kv_seq_lens_host,
        kv_seq_lens_host_values=host_values,
    )
    _validate_expanded_decode_metadata(
        resolved,
        slot_mapping=getattr(metadata, "slot_mapping", None),
        block_size=block_size,
    )
    return resolved


def _validate_expanded_decode_metadata(
    metadata: ExpandedDecodeMetadata,
    *,
    slot_mapping: torch.Tensor | None,
    block_size: int,
) -> None:
    if metadata.block_table.dim() != 2:
        raise RuntimeError("expanded decode block_table must be two-dimensional")
    sequence_count = metadata.block_table.shape[0]
    if (
        slot_mapping is None
        or slot_mapping.dim() != 1
        or slot_mapping.numel() != sequence_count
    ):
        raise RuntimeError(
            "expanded decode slot_mapping must contain one slot per token"
        )
    per_sequence_tensors = (
        ("kv_seq_lens", metadata.kv_seq_lens),
        ("paged_kv_last_page_len", metadata.paged_kv_last_page_len),
    )
    if metadata.kv_seq_lens_host is not None:
        per_sequence_tensors += (
            ("kv_seq_lens_host", metadata.kv_seq_lens_host),
        )
    for name, tensor in per_sequence_tensors:
        if tensor.dim() != 1 or tensor.numel() != sequence_count:
            raise RuntimeError(
                f"expanded decode {name} must contain one value per sequence"
            )
    if (
        metadata.kv_seq_lens_host_values is not None
        and len(metadata.kv_seq_lens_host_values) != sequence_count
    ):
        raise RuntimeError(
            "expanded decode kv_seq_lens_host_values must contain one value "
            "per sequence"
        )
    if (
        metadata.paged_kv_indptr.dim() != 1
        or metadata.paged_kv_indptr.numel() != sequence_count + 1
    ):
        raise RuntimeError(
            "expanded decode paged_kv_indptr must contain one offset per "
            "sequence plus the terminal offset"
        )
    if (
        metadata.paged_kv_indices.dim() != 1
        or metadata.paged_kv_indices.numel() == 0
    ):
        raise RuntimeError(
            "expanded decode paged_kv_indices must be a non-empty flat page list"
        )

    if metadata.paged_kv_indptr.device.type == "cpu":
        indptr = metadata.paged_kv_indptr
        if int(indptr[0]) != 0:
            raise RuntimeError("expanded decode paged_kv_indptr must start at zero")
        if not bool(torch.all(indptr[1:] >= indptr[:-1])):
            raise RuntimeError("expanded decode paged_kv_indptr must be monotonic")
        if int(indptr[-1]) != metadata.paged_kv_indices.numel():
            raise RuntimeError(
                "expanded decode terminal page offset must match page count"
            )
    if metadata.paged_kv_last_page_len.device.type == "cpu":
        last_page_lens = metadata.paged_kv_last_page_len
        if not bool(torch.all(last_page_lens >= 1)):
            raise RuntimeError(
                "expanded decode last-page lengths must be positive"
            )
        if block_size > 0 and not bool(torch.all(last_page_lens <= block_size)):
            raise RuntimeError(
                "expanded decode last-page lengths must not exceed block size"
            )
    if metadata.kv_seq_lens_host_values is not None and block_size > 0:
        block_table_capacity = metadata.block_table.shape[1]
        for kv_seq_len in metadata.kv_seq_lens_host_values:
            if kv_seq_len < 0:
                raise RuntimeError(
                    "expanded decode host KV lengths must be non-negative"
                )
            page_count = (max(kv_seq_len, 1) + block_size - 1) // block_size
            if page_count > block_table_capacity:
                raise RuntimeError(
                    "expanded decode page count exceeds block-table capacity"
                )
