# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Launch the shared Full page and slot mapping on the caller's stream."""

import tilelang
import torch

from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    from xllm_kernel.ops._tilelang import VECTOR_CORES
    from xllm_kernel.ops.kv_cache import SHARED_FULL_INDICES_OUT_CONTRACT
    from xllm_kernel.ops.kv_cache._shared_full_indices import (
        SHARED_FULL_INDICES_PASS_CONFIGS,
        SHARED_FULL_INDICES_TILE,
        build_shared_full_indices_kernel,
    )

    _MAX_OFFSET = torch.iinfo(torch.int32).max
    _TILE = SHARED_FULL_INDICES_TILE
    _KERNEL = tilelang.compile(build_shared_full_indices_kernel(_TILE), pass_configs=SHARED_FULL_INDICES_PASS_CONFIGS)

    @register_kernel(
        name="npu.tilelang.shared_full_indices_out",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Int32 paged tables, slots and strided remap rows with caller-owned output buffers.",
        solution="tilelang",
        binding="python",
        contract=SHARED_FULL_INDICES_OUT_CONTRACT,
    )
    def _shared_full_indices_out(
        block_table: torch.Tensor,
        slots: torch.Tensor,
        rows: torch.Tensor,
        page_size: int,
        padding_row: int,
        out_tables: torch.Tensor,
        out_slots: torch.Tensor,
    ) -> None:
        device = block_table.device
        tensors = (block_table, slots, rows, out_tables, out_slots)
        if device.type != "npu" or any(tensor.device != device or tensor.dtype != torch.int32 for tensor in tensors):
            raise ValueError("shared Full mapping requires int32 tensors on one NPU")
        if block_table.dim() != 2 or slots.dim() != 1 or rows.dim() != 2:
            raise ValueError("expected a block-table matrix, a slot vector and a row-table matrix")
        if any(not tensor.is_contiguous() for tensor in (block_table, slots, out_tables, out_slots)):
            raise ValueError("shared Full block tables, slots and outputs must be contiguous")

        layers, capacity = rows.shape
        row_stride = rows.stride(0)
        if not layers or not capacity or rows.stride(1) != 1 or row_stride < capacity:
            raise ValueError("shared Full rows must be nonempty with unit inner stride")
        if out_tables.shape != (layers, *block_table.shape) or out_slots.shape != (layers, slots.numel()):
            raise ValueError("shared Full output shapes must match the per-layer tables and slots")
        if not 0 < page_size <= _MAX_OFFSET or not 0 <= padding_row <= _MAX_OFFSET:
            raise ValueError("invalid shared Full page size or padding row")
        row_span = (layers - 1) * row_stride + capacity
        if max(row_span, row_stride, out_tables.numel(), out_slots.numel()) > _MAX_OFFSET:
            raise ValueError("shared Full mapping exceeds the int32 offset range")

        table_elements = block_table.numel()
        slot_elements = slots.numel()
        table_tiles = (table_elements + _TILE - 1) // _TILE
        slot_tiles = (slot_elements + _TILE - 1) // _TILE
        tile_count = layers * (table_tiles + slot_tiles)
        if tile_count == 0:
            return

        launch_cores = min((tile_count + 1) // 2, VECTOR_CORES // 2)
        with torch.npu.device(device):
            _KERNEL(
                block_table.view(-1),
                slots,
                rows.as_strided((row_span,), (1,)),
                out_tables.view(-1),
                out_slots.view(-1),
                table_elements,
                slot_elements,
                layers,
                capacity,
                row_stride,
                page_size,
                padding_row,
                launch_cores,
            )
