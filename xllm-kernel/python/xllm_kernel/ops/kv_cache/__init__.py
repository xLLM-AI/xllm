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
OPERATIONS = (RESHAPE_PAGED_CACHE_OPERATION, WRITE_FP8_KV_OPERATION, CONVERT_XQA_PAGES_OPERATION)


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
