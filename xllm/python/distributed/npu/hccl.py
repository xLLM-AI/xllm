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

"""HCCL collectives with a bounded current-stream all-reduce fast path.

Buffers must be contiguous and nonempty. AllGather also accepts aligned,
whole 2D FRACTAL_NZ storage; other collectives require ND. The caller must retain
the group and buffers through completion and graph replay. Direct capture
requires ``HCCL_OP_EXPANSION_MODE=AIV``. Large all-reduces use c10d's managed
communication stream because HCCL can select multi-stream AI CPU expansion
despite that environment setting.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from xllm_kernel.ops.collectives import (
    all_gather,
    all_reduce,
    matmul_reduce_scatter,
    reduce_scatter,
)

# Keep the inline optimization conservative, rather than assuming HCCL's
# topology- and version-dependent AIV-to-AI-CPU threshold. c10d owns the stream
# fork/join and buffer lifetimes for larger transfers, including graph capture.
_CURRENT_STREAM_ALL_REDUCE_MAX_BYTES = 512 * 1024


def get_hccl_comm(
    group: dist.ProcessGroup | None,
    device: torch.device,
    *,
    initialize: bool = False,
) -> int:
    if device.type != "npu":
        raise RuntimeError(f"HCCL requires an NPU tensor, got {device}")
    if not dist.is_initialized():
        raise RuntimeError("distributed is not initialized")
    group = dist.distributed_c10d._get_default_group() if group is None else group
    device_index = device.index if device.index is not None else torch.npu.current_device()
    device = torch.device("npu", device_index)
    backend = group._get_backend(device)
    if initialize:
        # Use the backend's idempotent initialization rather than inserting a
        # collective conditionally on rank-local communicator cache state.
        with torch.npu.device(device):
            backend.eager_connect_single_device(device)
    comm = backend.get_hccl_comm(device_index)
    if not comm:
        raise RuntimeError(f"group {group.group_name} has no HCCL communicator on {device}")
    return comm


def all_reduce_on_current_stream(x: torch.Tensor, group: dist.ProcessGroup) -> None:
    """In-place SUM, preserving the input dtype."""
    comm = get_hccl_comm(group, x.device)
    if x.numel() * x.element_size() > _CURRENT_STREAM_ALL_REDUCE_MAX_BYTES:
        dist.all_reduce(x, group=group)
        return
    all_reduce(x, comm)


def all_gather_on_current_stream(
    input: torch.Tensor,
    output: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    initialize: bool = False,
) -> None:
    """Gather equal input blocks in rank order into a nonoverlapping output.

    ``output.numel()`` must equal ``group.size() * input.numel()``.
    """
    all_gather(input, output, get_hccl_comm(group, input.device, initialize=initialize))


def reduce_scatter_on_current_stream(input: torch.Tensor, output: torch.Tensor, group: dist.ProcessGroup) -> None:
    """SUM rank-ordered blocks and scatter into a nonoverlapping output.

    ``input.numel()`` must equal ``group.size() * output.numel()``.
    """
    reduce_scatter(input, output, get_hccl_comm(group, input.device))


def matmul_reduce_scatter_on_current_stream(
    input: torch.Tensor, weight: torch.Tensor, group: dist.ProcessGroup, comm_mode: str
) -> torch.Tensor:
    """Fuse matmul with SUM ReduceScatter on the active TP communicator.

    The input is contiguous ND; the weight may be a transposed ND view.
    Bias is handled by the row-parallel layer after the collective.
    """
    device_index = input.device.index if input.device.index is not None else torch.npu.current_device()
    backend = group._get_backend(input.device)
    hcom = backend.get_hccl_comm_name(device_index)
    if not hcom:
        raise RuntimeError(f"group {group.group_name} has no HCCL communicator name on {input.device}")
    return matmul_reduce_scatter(
        input,
        weight,
        hcom,
        group.size(),
        comm_mode=None if comm_mode in ("", "none") else comm_mode,
    )
