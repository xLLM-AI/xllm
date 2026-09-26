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

"""CUDA all-reduce with bounded symmetric-memory staging for decode shapes."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import ProcessGroup

# A one-shot symmetric-memory reduction is an ordinary kernel on the current
# stream, so a captured graph runs it inline. NCCL runs collectives on its own
# stream, which costs a fork/join per call -- measured at ~32us of device idle
# before every all-reduce in the decode graph, and there are dozens per step.
# Past this size the staging copy stops paying for itself and NCCL's ring wins,
# and the bound also keeps the buffer cache to decode-sized shapes rather than
# one entry per distinct prefill length.
_SYMM_MEM_MAX_BYTES = 512 * 1024
# Distinct shapes still accumulate one buffer each; stop allocating rather than
# grow without limit when a workload sweeps many small shapes.
_SYMM_MEM_MAX_BUFFERS = 64
_SYMM_MEM_DTYPES = frozenset((torch.float32, torch.bfloat16))

_symm_eligible: dict[tuple[ProcessGroup, str], bool] = {}
_symm_buffers: dict[tuple[ProcessGroup, str, torch.dtype, int], torch.Tensor] = {}


def _supports_symmetric_memory(
    device: torch.device,
    ranks: list[int],
    world_topology: list[dict[str, object]] | None,
) -> bool:
    if device.type != "cuda" or world_topology is None:
        return False
    topology = [world_topology[rank] for rank in ranks]
    hostnames = {entry["hostname"] for entry in topology}
    if len(hostnames) != 1:
        return False
    device_indices = [entry["device_index"] for entry in topology]
    if any(not isinstance(index, int) or index < 0 for index in device_indices):
        return False
    if len(set(device_indices)) != len(device_indices):
        return False
    return all(
        torch.cuda.can_device_access_peer(source, destination)
        for source in device_indices
        for destination in device_indices
        if source != destination
    )


def init_group(
    group: ProcessGroup,
    device: torch.device,
    ranks: list[int],
    world_topology: list[dict[str, object]] | None,
) -> None:
    """Record symmetric-memory eligibility after the group's rendezvous."""
    _symm_eligible[(group, str(device))] = _supports_symmetric_memory(device, ranks, world_topology)


def _symm_buffer(group: ProcessGroup, x: torch.Tensor) -> torch.Tensor | None:
    """Return a symmetric-memory staging buffer for ``x``, or None.

    Allocation and rendezvous are collective and cannot run inside a graph
    capture, so a shape first seen during capture falls back to NCCL. The
    capture path is warmed up eagerly beforehand, which is where the decode
    shapes get their buffers.
    """
    group_key = (group, str(x.device))
    if not _symm_eligible.get(group_key, False):
        return None
    if not x.is_contiguous() or x.dtype not in _SYMM_MEM_DTYPES:
        return None
    if x.numel() * x.element_size() > _SYMM_MEM_MAX_BYTES:
        return None

    key = (group, str(x.device), x.dtype, x.numel())
    buffer = _symm_buffers.get(key)
    if buffer is not None:
        return buffer
    if torch.cuda.is_current_stream_capturing():
        return None
    if len(_symm_buffers) >= _SYMM_MEM_MAX_BUFFERS:
        return None

    buffer = symm_mem.empty(x.numel(), dtype=x.dtype, device=x.device)
    symm_mem.rendezvous(buffer, group.group_name)
    _symm_buffers[key] = buffer
    return buffer


def all_reduce(x: torch.Tensor, group: ProcessGroup) -> None:
    """In-place SUM via symmetric memory when eligible, otherwise NCCL."""
    buffer = _symm_buffer(group, x)
    if buffer is None:
        dist.all_reduce(x, group=group)
        return
    flat = x.view(-1)
    buffer.copy_(flat)
    torch.ops.symm_mem.one_shot_all_reduce_out(buffer, "sum", group.group_name, flat)
