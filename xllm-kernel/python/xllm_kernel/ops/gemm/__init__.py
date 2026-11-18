# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free MLA projection contracts and startup-bound public APIs."""

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
OPERATIONS = (MATMUL_EIN_SUM_OPERATION, MATMUL_EIN_SUM_OUT_OPERATION)


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
