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

"""2-rank: K-axis AllGather NZ weight goes straight into matmul vs full NZ matmul.

torchrun --nproc_per_node=2 --master_port=29701 tests/python/test_matmul_nz_bytes.py
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

if __name__ != "__main__":
    pytest.skip(
        "run via torchrun --nproc_per_node=N tests/python/test_matmul_nz_bytes.py",
        allow_module_level=True,
    )

import torch_npu

from tests.python.nz_test_utils import as_nz, init_hccl, maxabs, npu_device, print_rank0
from xllm.python.distributed.all_gather_nz_bytes import (
    all_gather_nz_bytes,
    is_fractal_nz,
    raw_storage_uint8,
    shard_nz,
)

_M, _N, _K = 32, 64, 128


def _one(rank: int, world: int) -> None:
    torch.manual_seed(20260910)
    dev = npu_device()
    x = torch.randn(_M, _K, dtype=torch.float16, device=dev)
    full_nd = torch.randn(_N, _K, dtype=torch.float16, device=dev)
    full_nz = as_nz(full_nd)
    torch.npu.synchronize()

    local_nz = shard_nz(full_nz, rank_world=(rank, world))
    gathered_nz = all_gather_nz_bytes(local_nz)
    torch.npu.synchronize()

    if not is_fractal_nz(gathered_nz) or not is_fractal_nz(full_nz):
        raise RuntimeError("expected FRACTAL_NZ weights")
    if tuple(gathered_nz.shape) != tuple(full_nz.shape):
        raise RuntimeError(f"gathered {tuple(gathered_nz.shape)} != full {tuple(full_nz.shape)}")

    st_eq = bool(torch.equal(raw_storage_uint8(gathered_nz), raw_storage_uint8(full_nz)))

    y_full = F.linear(x, full_nz)
    y_ag = F.linear(x, gathered_nz)
    torch.npu.synchronize()
    y_nd = F.linear(x, full_nd)
    torch.npu.synchronize()

    err_ag_full = maxabs(y_ag, y_full)
    err_ag_nd = maxabs(y_ag, y_nd)
    err_full_nd = maxabs(y_full, y_nd)
    eq_ag_full = bool(torch.equal(y_ag, y_full))
    print_rank0(
        rank,
        f"x={tuple(x.shape)} W={tuple(full_nz.shape)} "
        f"W.fmt={int(torch_npu.get_npu_format(gathered_nz))} "
        f"storage_eq={st_eq} "
        f"y_ag==y_full {eq_ag_full} maxabs={err_ag_full} "
        f"y_ag vs ND-W maxabs={err_ag_nd} "
        f"y_full vs ND-W maxabs={err_full_nd}",
    )
    if err_ag_full != 0.0:
        raise RuntimeError(f"gathered-NZ matmul != full-NZ matmul maxabs={err_ag_full}")


def main() -> None:
    rank, world, _ = init_hccl(timeout_s=120)
    _one(rank, world)
    dist.barrier()
    dist.destroy_process_group()
    print_rank0(rank, f"matmul NZ allgather vs full-NZ passed world={world} MNK={(_M, _N, _K)}")


if __name__ == "__main__":
    main()
