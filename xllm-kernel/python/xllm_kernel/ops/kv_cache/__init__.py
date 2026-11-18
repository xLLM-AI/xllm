# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Paged KV write contract and startup-bound API."""

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
OPERATIONS = (RESHAPE_PAGED_CACHE_OPERATION,)


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
