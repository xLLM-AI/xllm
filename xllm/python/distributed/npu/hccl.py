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

"""HCCL collectives on the caller's current NPU stream.

Buffers must be contiguous, nonempty and in ND storage format. The caller must
retain the group and buffers through completion and graph replay. Capture requires
``HCCL_OP_EXPANSION_MODE=AIV``; unsupported execution fails without fallback.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def _hccl_comm(group: dist.ProcessGroup, x: torch.Tensor) -> int:
    if x.device.type != "npu":
        raise RuntimeError(f"HCCL requires an NPU tensor, got {x.device}")
    # torch_npu indexes local communicators by device, not group rank.
    comm = group._get_backend(x.device).get_hccl_comm(x.device.index)
    if not comm:
        raise RuntimeError(f"group {group.group_name} has no HCCL communicator on {x.device}")
    return comm


def all_reduce_on_current_stream(x: torch.Tensor, group: dist.ProcessGroup) -> None:
    """In-place SUM, preserving the input dtype."""
    torch.ops.xllm_ops.npu_all_reduce(x, _hccl_comm(group, x))


def all_gather_on_current_stream(input: torch.Tensor, output: torch.Tensor, group: dist.ProcessGroup) -> None:
    """Gather equal input blocks in rank order into a nonoverlapping output.

    ``output.numel()`` must equal ``group.size() * input.numel()``.
    """
    torch.ops.xllm_ops.npu_all_gather(input, output, _hccl_comm(group, input))


def reduce_scatter_on_current_stream(input: torch.Tensor, output: torch.Tensor, group: dist.ProcessGroup) -> None:
    """SUM rank-ordered blocks and scatter into a nonoverlapping output.

    ``input.numel()`` must equal ``group.size() * output.numel()``.
    """
    torch.ops.xllm_ops.npu_reduce_scatter(input, output, _hccl_comm(group, input))
