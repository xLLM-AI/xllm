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

"""Shared helpers for NZ AllGather torchrun tests."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu

from scripts.logger import logger
from xllm import xllm_export  # noqa: F401
from xllm.python.distributed.all_gather_nz_bytes import ACL_FORMAT_FRACTAL_NZ

ACL_FORMAT_ND = int(torch_npu.Format.ND)


def npu_device() -> torch.device:
    return torch.device(f"npu:{int(os.environ['LOCAL_RANK'])}")


def seed_nd(
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    int8_low: int = -8,
    int8_high: int = 8,
) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    if dtype == torch.int8:
        return torch.randint(int8_low, int8_high, (n, k), generator=g, dtype=torch.int32).to(dtype).to(device)
    return torch.randn(n, k, generator=g, dtype=torch.float32).to(dtype).to(device)


def as_nz(nd: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_format_cast(nd.contiguous(), ACL_FORMAT_FRACTAL_NZ)


def to_nd(nz: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_format_cast(nz, ACL_FORMAT_ND)


def maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def print_rank0(rank: int, msg: str) -> None:
    if rank == 0:
        logger.info(msg)


def init_hccl(timeout_s: int = 180) -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    torch.npu.config.allow_internal_format = True
    dist.init_process_group(
        backend="hccl",
        timeout=timedelta(seconds=timeout_s),
        device_id=torch.device(f"npu:{local_rank}"),
    )
    return rank, world, local_rank
