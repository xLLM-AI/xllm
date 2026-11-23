# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free matrix product contracts and startup-bound public APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

MATMUL_EIN_SUM_CONTRACT = OpContract("matmul_ein_sum", 1, (), False, "bf16_matmul")
MATMUL_EIN_SUM_OUT_CONTRACT = OpContract("matmul_ein_sum_out", 1, ("out",), True, "bf16_matmul")


def _operation(contract: OpContract) -> Operation:
    return Operation(
        contract,
        (
            BackendModule(
                module="xllm_kernel.ops.gemm.atb",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=("npu.atb." + contract.operation,),
            ),
        ),
    )


MATMUL_EIN_SUM_OPERATION = _operation(MATMUL_EIN_SUM_CONTRACT)
MATMUL_EIN_SUM_OUT_OPERATION = _operation(MATMUL_EIN_SUM_OUT_CONTRACT)
QUANT_MATMUL_OUT_CONTRACT = OpContract("quant_matmul_out", 1, ("out",), True, "quant_matmul")
QUANT_MATMUL_OUT_OPERATION = Operation(
    QUANT_MATMUL_OUT_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.gemm.ascendc",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.ascendc.quant_matmul_out",),
        ),
    ),
)
QUANT_MATMUL_CONTRACT = OpContract("quant_matmul", 1, (), False, "quant_matmul")
QUANT_MATMUL_OPERATION = Operation(
    QUANT_MATMUL_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.gemm.ascendc",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.aclnn.quant_matmul",),
        ),
    ),
)
GROUPED_MATMUL_CONTRACT = OpContract("grouped_matmul", 1, (), False, "grouped_matmul")
GROUPED_MATMUL_OUT_CONTRACT = OpContract("grouped_matmul_out", 1, ("output",), True, "grouped_matmul")


def _grouped_operation(contract: OpContract, implementation: str) -> Operation:
    return Operation(
        contract,
        (
            BackendModule(
                module="xllm_kernel.ops.gemm.npu",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(implementation,),
            ),
        ),
    )


GROUPED_MATMUL_OPERATION = _grouped_operation(GROUPED_MATMUL_CONTRACT, "npu.torch.grouped_matmul")
GROUPED_MATMUL_OUT_OPERATION = _grouped_operation(GROUPED_MATMUL_OUT_CONTRACT, "npu.aclnn.grouped_matmul_out")
OPERATIONS = (
    MATMUL_EIN_SUM_OPERATION,
    MATMUL_EIN_SUM_OUT_OPERATION,
    QUANT_MATMUL_OPERATION,
    QUANT_MATMUL_OUT_OPERATION,
    GROUPED_MATMUL_OPERATION,
    GROUPED_MATMUL_OUT_OPERATION,
)


def has_grouped_matmul_out() -> bool:
    """Return whether CANN provides the grouped out kernel on this host."""
    from xllm_kernel.ops.gemm.npu import HAS_GROUPED_MATMUL_OUT

    return HAS_GROUPED_MATMUL_OUT


def matmul_ein_sum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Contract BF16 NPU [T,H,D] and [H,D,O] into fresh contiguous [T,H,O].

    Dimensions must be nonempty, operands must share a device, and weight must
    be contiguous. Both operands use ND/base format. Strided input (including
    storage-offset views) is packed without modifying it. This is the fixed
    MLA projection, not a general einsum parser. The ATB implementation retains
    the existing BF16 math and supports eager and Python ACLGraph execution.
    """
    return MATMUL_EIN_SUM_OPERATION.get_prepared().function(x, weight)


def matmul_ein_sum_out(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Write the same contraction into out and return that exact tensor.

    Out is contiguous ND/base-format BF16 [T,H,O] on the operands' NPU and must
    not overlap either operand. Input and weight remain unchanged; every out
    element is overwritten. Its address stays fixed across capture and replay.
    """
    return MATMUL_EIN_SUM_OUT_OPERATION.get_prepared().function(x, weight, out)


def quant_matmul_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    transpose2: bool,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
    pertoken_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write the quantized product into out and return the same tensor."""
    return QUANT_MATMUL_OUT_OPERATION.get_prepared().function(
        input, weight, transpose2, scale, offset, pertoken_scale, bias, output_dtype, out
    )


def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    transpose2: bool,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
    pertoken_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Return a quantized product with weight stored as [N, K] when transpose2 is true."""
    return QUANT_MATMUL_OPERATION.get_prepared().function(
        x1, x2, transpose2, scale, offset, pertoken_scale, bias, output_dtype
    )


def grouped_matmul(
    x: list[torch.Tensor],
    weight: list[torch.Tensor],
    *,
    bias: list[torch.Tensor] | None = None,
    scale: list[torch.Tensor] | None = None,
    per_token_scale: list[torch.Tensor] | None = None,
    split_item: int,
    group_type: int,
    group_list_type: int = 0,
    group_list: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """Return TorchNPU grouped products with optional bias and quantization scales."""
    return GROUPED_MATMUL_OPERATION.get_prepared().function(
        x=x,
        weight=weight,
        bias=bias,
        scale=scale,
        per_token_scale=per_token_scale,
        split_item=split_item,
        group_type=group_type,
        group_list_type=group_list_type,
        group_list=group_list,
        output_dtype=output_dtype,
    )


def grouped_matmul_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_list: torch.Tensor,
    *,
    split_item: int,
    group_type: int,
    group_list_type: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write an [M, K] by [E, K, N] grouped product into contiguous output and return it."""
    return GROUPED_MATMUL_OUT_OPERATION.get_prepared().function(
        x, weight, scale, per_token_scale, group_list, split_item, group_type, group_list_type, output
    )
