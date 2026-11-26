# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Paged KV cache contracts and startup-bound APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

RESHAPE_PAGED_CACHE_CONTRACT = OpContract("reshape_paged_cache", 1, ("key_cache", "value_cache"), True, "copy")
RESHAPE_PAGED_CACHE_OPERATION = Operation(
    RESHAPE_PAGED_CACHE_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.kv_cache.npu",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.atb.reshape_paged_cache", "npu.aclnn.reshape_paged_cache"),
        ),
    ),
)
SHARED_FULL_INDICES_OUT_CONTRACT = OpContract(
    "shared_full_indices_out", 1, ("out_tables", "out_slots"), False, "paged_indices"
)
SHARED_FULL_INDICES_OUT_OPERATION = Operation(
    SHARED_FULL_INDICES_OUT_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.kv_cache.tilelang",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "tilelang", "triton"),
            implementations=("npu.tilelang.shared_full_indices_out",),
        ),
    ),
)
UPDATE_DECODE_GRAPH_METADATA_CONTRACT = OpContract(
    "update_decode_graph_metadata",
    1,
    (
        "dst_tokens",
        "dst_positions",
        "dst_slot_mapping",
        "dst_kv_seq_lens",
        "dst_kv_seq_lens_delta",
        "dst_paged_kv_indptr",
        "dst_paged_kv_indices",
        "dst_paged_kv_last_page_len",
    ),
    True,
    "decode_metadata",
)
UPDATE_DECODE_GRAPH_METADATA_OPERATION = Operation(
    UPDATE_DECODE_GRAPH_METADATA_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.kv_cache.decode_metadata",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.update_decode_graph_metadata",),
        ),
    ),
)
WRITE_FP8_KV_CONTRACT = OpContract("write_fp8_kv", 1, ("key_cache", "value_cache"), False, "static_fp8_e4m3_scatter")
CONVERT_XQA_PAGES_CONTRACT = OpContract("convert_xqa_pages", 1, ("table", "lengths"), False, "csr_to_xqa_pages")
WRITE_FP8_KV_OPERATION = Operation(
    WRITE_FP8_KV_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.kv_cache.native_kv_writer",
            platform=PlatformEnum.CUDA,
            requires=("torch", "triton"),
            implementations=("cuda.triton.write_fp8_kv",),
        ),
    ),
)
CONVERT_XQA_PAGES_OPERATION = Operation(
    CONVERT_XQA_PAGES_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.kv_cache.native_kv_writer",
            platform=PlatformEnum.CUDA,
            requires=("torch", "triton"),
            implementations=("cuda.triton.convert_xqa_pages",),
        ),
    ),
)
OPERATIONS = (
    RESHAPE_PAGED_CACHE_OPERATION,
    SHARED_FULL_INDICES_OUT_OPERATION,
    UPDATE_DECODE_GRAPH_METADATA_OPERATION,
    WRITE_FP8_KV_OPERATION,
    CONVERT_XQA_PAGES_OPERATION,
)


def has_shared_full_indices_out() -> bool:
    """Return whether the worker prepared the fused Full page and slot mapping."""
    return SHARED_FULL_INDICES_OUT_OPERATION.prepared is not None


def reshape_paged_cache(
    slot_mapping: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    """Write token-major [T,H,D] rows at flattened paged slots and return key_cache.

    ND caches use [blocks, page_size, H, D]. Slot mapping
    is a device int32 vector with one entry per source row. Negative slots are
    padding. Nonnegative slots must be in range and unique within each write.
    The two caches may coincide for a single MLA stream written as both K and V.
    ATB supports separate K and V widths, including MLA FRACTAL_NZ caches in
    [blocks, ceil(D/16), page_size, 16] layout. Ascend950 uses contiguous ND
    FP16/BF16 and matching K/V shapes. Sources are read only.
    """
    return RESHAPE_PAGED_CACHE_OPERATION.get_prepared().function(slot_mapping, keys, values, key_cache, value_cache)


def shared_full_indices_out(
    block_table: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    page_size: int,
    padding_row: int,
    out_tables: torch.Tensor,
    out_slots: torch.Tensor,
) -> None:
    """Resolve logical pages and write slots into caller-owned per-layer buffers.

    Table page 0 reads from padding_row. Slot 0 writes to the mapped warmup
    page. Negative entries pass through. Remap rows support a padded stride.
    """
    SHARED_FULL_INDICES_OUT_OPERATION.get_prepared().function(
        block_table, slots, rows, page_size, padding_row, out_tables, out_slots
    )


def update_decode_graph_metadata(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    dst_tokens: torch.Tensor,
    dst_positions: torch.Tensor,
    dst_slot_mapping: torch.Tensor,
    dst_kv_seq_lens: torch.Tensor,
    dst_kv_seq_lens_delta: torch.Tensor,
    dst_paged_kv_indptr: torch.Tensor,
    dst_paged_kv_indices: torch.Tensor,
    dst_paged_kv_last_page_len: torch.Tensor,
    padded_num_tokens: int,
) -> torch.Tensor:
    """Refresh static decode input buffers and return the original dst_tokens.

    Padding writes zero tokens and positions, slot -1 and last-page length 1.
    Cumulative lengths and CSR indptr repeat terminal values over padded rows.
    """
    return UPDATE_DECODE_GRAPH_METADATA_OPERATION.get_prepared().function(
        tokens,
        positions,
        slot_mapping,
        kv_seq_lens,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        dst_tokens,
        dst_positions,
        dst_slot_mapping,
        dst_kv_seq_lens,
        dst_kv_seq_lens_delta,
        dst_paged_kv_indptr,
        dst_paged_kv_indices,
        dst_paged_kv_last_page_len,
        padded_num_tokens,
    )


def write_fp8_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale: float,
    value_scale: float,
) -> None:
    """Scatter strided BF16/FP16 [T,H,D] rows into disjoint contiguous NHD E4M3 pages.

    Each K/V element is divided in FP32 by its positive fixed scale, clamped to
    [-448, 448], then cast to E4M3FN. The contiguous integer slot vector has T
    elements. In-range slots must be unique; out-of-range slots are ignored.
    Only the two destination caches are written.
    """
    WRITE_FP8_KV_OPERATION.get_prepared().function(key, value, slots, key_cache, value_cache, key_scale, value_scale)


def convert_xqa_pages(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    page_size: int,
) -> None:
    """Expand contiguous int32 CSR page indices into caller-owned XQA buffers.

    For B rows, indptr has B+1 entries and last has B final-page token counts.
    The contiguous int32 table [B,W] must fit the longest page list. Unused
    columns are zeroed. Each uint32 length is (pages-1)*page_size + last, or
    one for an empty padding row. Inputs and outputs must have disjoint storage.
    """
    CONVERT_XQA_PAGES_OPERATION.get_prepared().function(indptr, indices, last, table, lengths, page_size)
