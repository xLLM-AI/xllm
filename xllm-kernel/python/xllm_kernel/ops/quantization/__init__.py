# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""NPU static and dynamic quantization contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch


def _operation(name: str) -> Operation:
    return Operation(
        OpContract(name, 1, (), False, name),
        (
            BackendModule(
                module="xllm_kernel.ops.quantization.npu",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.aclnn.{name}",),
            ),
        ),
    )


QUANTIZE_PER_TENSOR_OPERATION = _operation("quantize_per_tensor")
DYNAMIC_QUANT_OPERATION = _operation("dynamic_quant")
OPERATIONS = (QUANTIZE_PER_TENSOR_OPERATION, DYNAMIC_QUANT_OPERATION)


def quantize_per_tensor(
    value: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
) -> torch.Tensor:
    """Quantize using explicit scale and zero point along the given axis.

    QInt8, QUInt8 and QInt32 use plain INT8, UINT8 and INT32 storage.
    QUInt4x2 packs eight values per INT32 word along the last dimension.
    """
    return QUANTIZE_PER_TENSOR_OPERATION.get_prepared().function(value, scales, zero_points, dtype, axis)


def dynamic_quant(
    value: torch.Tensor,
    smooth_scales: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    dst_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return quantized values and one FP32 scale per row.

    The default output is INT8. QUInt4x2 packs eight values per INT32 word.
    Optional smooth scales and group boundaries are passed to the quantizer.
    """
    return DYNAMIC_QUANT_OPERATION.get_prepared().function(value, smooth_scales, group_index, dst_type)
