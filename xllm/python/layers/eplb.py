# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""EPLB (Expert Parallel Load Balancing) utilities.

Ports the core logic from the C++ implementation in
``xllm/core/layers/npu_torch/deepseek_v4_eplb_utils.h``.
"""

from __future__ import annotations

import torch


def local_physical_experts_num(
    num_total_experts: int,
    ep_size: int,
    redundant_experts_num: int,
) -> int:
    """Number of physical expert slots per device (routed + redundant)."""
    return num_total_experts // ep_size + redundant_experts_num


def build_initial_expert_ids(
    num_total_experts: int,
    ep_size: int,
    device_experts_num: int,
    redundant_experts_num: int,
) -> list[int]:
    """Build the initial flat expert-id distribution across all EP ranks.

    Returns a list of length ``ep_size * device_experts_num``. Each segment
    of ``device_experts_num`` corresponds to one rank. Routed slots hold
    sequential expert ids; redundant slots duplicate the last routed expert
    of that rank.
    """
    if num_total_experts <= 0 or ep_size <= 0 or device_experts_num <= redundant_experts_num:
        return []
    routed_per_rank = device_experts_num - redundant_experts_num
    expert_ids: list[int] = []
    for rank in range(ep_size):
        base = rank * routed_per_rank
        for slot in range(routed_per_rank):
            expert_ids.append(base + slot)
        duplicate_expert = base + routed_per_rank - 1
        for _ in range(redundant_experts_num):
            expert_ids.append(duplicate_expert)
    return expert_ids


def slice_rank_expert_ids(
    expert_ids: list[int],
    ep_rank: int,
    device_experts_num: int,
) -> list[int]:
    """Extract the expert-id segment for a single EP rank."""
    if ep_rank < 0 or device_experts_num <= 0:
        return []
    begin = ep_rank * device_experts_num
    end = begin + device_experts_num
    if end > len(expert_ids):
        return []
    return expert_ids[begin:end]


def build_log2phy_map(
    expert_ids: list[int],
    num_total_experts: int,
    ep_rank: int,
    moe_tp_rank_in_group: int = 0,
) -> list[int]:
    """Build logical-expert-id to physical-slot-id mapping.

    For each logical expert id (0..num_total_experts-1), resolves which
    physical slot index (in the global flat expert_ids list) this consumer
    should route to. When an expert has multiple duplicates, the pick is
    rotated by ``(ep_rank + moe_tp_rank_in_group)`` so that ranks sharing
    the same EP position but different moe_tp positions land on different
    duplicates.
    """
    log2phy: list[int] = [-1] * num_total_experts
    if num_total_experts <= 0 or not expert_ids:
        return log2phy

    rotation_base = max(ep_rank, 0) + max(moe_tp_rank_in_group, 0)

    duplicate_counts = [0] * num_total_experts
    for eid in expert_ids:
        if 0 <= eid < num_total_experts:
            duplicate_counts[eid] += 1

    selected_duplicates = [-1] * num_total_experts
    for eid in range(num_total_experts):
        cnt = duplicate_counts[eid]
        if cnt > 0:
            selected_duplicates[eid] = rotation_base % cnt

    duplicate_indices = [0] * num_total_experts
    for physical_id, eid in enumerate(expert_ids):
        if eid < 0 or eid >= num_total_experts:
            continue
        if duplicate_indices[eid] == selected_duplicates[eid]:
            log2phy[eid] = physical_id
        duplicate_indices[eid] += 1

    return log2phy


def remap_expert_ids(
    topk_ids: torch.Tensor,
    log2phy_map: torch.Tensor,
) -> torch.Tensor:
    """Remap logical expert ids to physical slot ids via gather."""
    flat = topk_ids.reshape(-1).long()
    remapped = log2phy_map.index_select(0, flat).reshape(topk_ids.shape)
    return remapped.to(torch.int32)


def expand_redundant_weight_storage(
    tensor: torch.Tensor,
    num_local_experts: int,
    device_experts_num: int,
) -> None:
    """Fill redundant slots by copying the last routed expert (in-place).

    After weight loading fills slots [0, num_local_experts), this copies the
    last loaded expert into slots [num_local_experts, device_experts_num).
    """
    if tensor is None or tensor.dim() == 0 or device_experts_num <= num_local_experts:
        return
    for slot in range(num_local_experts, device_experts_num):
        tensor[slot].copy_(tensor[num_local_experts - 1])
