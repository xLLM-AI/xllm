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

"""NPU matmul kernels and weight preparation for linear layers."""

from __future__ import annotations

import torch
import torch_npu

_FRACTAL_NZ_FORMAT = 29
_COLUMN_PARALLEL_NZ_DEVICES = frozenset({"Ascend910B2"})
_COLUMN_PARALLEL_NZ_DTYPES = frozenset({torch.bfloat16, torch.float16})


def grouped_linear(hidden: torch.Tensor, weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Project a shared input into separate contiguous outputs in one launch.

    The input has shape ``[tokens, in_features]``. Weights are transpose
    views in ``[in_features, out_features]`` order.
    Keeping the outputs separate avoids splitting and copying a concatenated
    projection before consumers that require contiguous tensors.
    """
    # Normalize a strided input once rather than once per grouped operand.
    hidden = hidden.contiguous()
    return torch_npu.npu_grouped_matmul([hidden] * len(weights), weights, split_item=0, group_type=-1)


def prepare_row_parallel_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Lay out a row-parallel weight for the NPU matmul kernels.

    The matmul kernels read ``[K, N]`` in the fractal-NZ format, so the
    checkpoint layout is transposed and cast. A weight still on the host is
    returned untouched: the format cast needs device memory and runs after the
    weight is moved.

    Args:
        weight: Row-parallel weight of shape ``[N, K]``.

    Returns:
        The weight and whether it was transposed to ``[K, N]``.
    """
    if weight.device.type == "cpu":
        return weight, False
    transposed = weight.transpose(0, 1).contiguous()
    return torch_npu.npu_format_cast(transposed, _FRACTAL_NZ_FORMAT), True


def prepare_column_parallel_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Lay out a supported column-parallel weight for NPU matmul.

    Column-parallel checkpoints store each local projection as ``[N, K]``.
    The selected matmul path consumes a single ``[K, N]`` FRACTAL_NZ tensor,
    so replacing the parameter avoids retaining an ND copy beside it. Keep ND
    for unverified devices, unsupported dtypes, meta tensors, and singleton
    dimensions that cannot use the weight-NZ matmul path.
    """
    if (
        weight.device.type != "npu"
        or weight.is_meta
        or weight.dtype not in _COLUMN_PARALLEL_NZ_DTYPES
        or weight.ndim < 2
        or weight.shape[-1] == 1
        or weight.shape[-2] == 1
    ):
        return weight, False
    device_name = torch_npu.npu.get_device_name(weight.device.index)
    if device_name not in _COLUMN_PARALLEL_NZ_DEVICES:
        return weight, False
    transposed = weight.transpose(0, 1).contiguous()
    return torch_npu.npu_format_cast(transposed, _FRACTAL_NZ_FORMAT), True


def prepare_quant_weight(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Pack an INT8 ``[out, in]`` weight for quantized matmul.

    ``torch_npu.npu_quant_matmul`` reads the right-hand operand as ``[K, N]``
    and benefits substantially from the NPU Fractal-NZ layout. Quantized Python
    linear layers use the same layout as the checkpoint before this helper is
    called, so keep the transpose and format conversion together here.  The
    CPU branch is intentionally a plain transpose for unit tests and weight
    loading without an initialized NPU.
    """
    transposed = weight.transpose(0, 1).contiguous()
    if weight.device.type == "cpu":
        return transposed
    return torch_npu.npu_format_cast(transposed, _FRACTAL_NZ_FORMAT)


def atb_matmul_ein_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run fixed ATB MATMUL_EIN_SUM: [T,H,D],[H,D,O]->[T,H,O].

    Both operands must be BF16 tensors on the same NPU in ND format. Weight
    must be contiguous; a strided input is copied to contiguous storage.
    This is ATB's fixed matmul mode, not a general einsum expression parser.
    """
    return torch.ops.xllm_ops.atb_matmul_ein_sum(x, weight)


def atb_matmul_ein_sum_out(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Run fixed ATB EIN_SUM directly into a caller-owned ND output.

    The output must be contiguous BF16 [T,H,O] on the operands' NPU and
    must not alias either operand. The math and input packing match the
    allocating variant.
    """
    return torch.ops.xllm_ops.atb_matmul_ein_sum_out(x, weight, out)


__all__ = [
    "grouped_linear",
    "atb_matmul_ein_sum",
    "atb_matmul_ein_sum_out",
    "prepare_row_parallel_weight",
    "prepare_column_parallel_weight",
    "prepare_quant_weight",
]
