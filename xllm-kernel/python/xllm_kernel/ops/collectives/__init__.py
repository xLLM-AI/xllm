# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Collectives using a caller-owned communicator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

OPERATIONS = tuple(
    Operation(
        OpContract(name, 1, mutates, False, name),
        (
            BackendModule(
                module="xllm_kernel.ops.collectives.npu",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.{provider}.{name}",),
            ),
        ),
    )
    for name, mutates, provider in (
        ("all_reduce", ("x",), "hccl"),
        ("all_gather", ("output",), "hccl"),
        ("reduce_scatter", ("output",), "hccl"),
        ("matmul_reduce_scatter", (), "torch"),
    )
)
_ALL_REDUCE, _ALL_GATHER, _REDUCE_SCATTER, _MATMUL_REDUCE_SCATTER = OPERATIONS
HCCL_OPERATIONS = tuple(operation.contract.operation for operation in OPERATIONS[:3])


def all_reduce(x: torch.Tensor, comm: int) -> None:
    """Sum in place on the current stream."""
    _ALL_REDUCE.get_prepared().function(x, comm)


def all_gather(input: torch.Tensor, output: torch.Tensor, comm: int) -> None:
    """Gather equal blocks into caller-owned storage in rank order on the current stream."""
    _ALL_GATHER.get_prepared().function(input, output, comm)


def reduce_scatter(input: torch.Tensor, output: torch.Tensor, comm: int) -> None:
    """Sum rank-ordered blocks and scatter into caller-owned storage on the current stream."""
    _REDUCE_SCATTER.get_prepared().function(input, output, comm)


def has_matmul_reduce_scatter() -> bool:
    """Return whether TorchNPU exposes the fused matmul and reduce scatter."""
    import torch_npu

    return hasattr(torch_npu, "npu_mm_reduce_scatter_base")


def matmul_reduce_scatter(
    input: torch.Tensor, weight: torch.Tensor, hcom: str, world_size: int, comm_mode: str | None
) -> torch.Tensor:
    """Fuse matmul and SUM reduce scatter, preserving a transposed ND weight view."""
    return _MATMUL_REDUCE_SCATTER.get_prepared().function(
        input, weight, hcom, world_size, reduce_op="sum", comm_mode=comm_mode
    )
