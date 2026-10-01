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

"""Correctness: byte AllGather returns FRACTAL_NZ (K-axis storage concat).

torchrun --nproc_per_node=2 --master_port=29671 tests/python/test_all_gather_nz_bytes.py
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

if __name__ != "__main__":
    pytest.skip(
        "run via torchrun --nproc_per_node=N tests/python/test_all_gather_nz_bytes.py",
        allow_module_level=True,
    )

import torch_npu

from tests.python.nz_test_utils import as_nz, init_hccl, maxabs, npu_device, print_rank0, seed_nd, to_nd
from xllm.python.distributed.all_gather_nz_bytes import (
    ACL_FORMAT_FRACTAL_NZ,
    all_gather_nz_bytes,
    empty_nz,
    fractal_k0,
    is_fractal_nz,
)

_DTYPES = (torch.float16, torch.bfloat16, torch.int8, torch.float32)


def test_returns_nz(rank: int, world: int, dtype: torch.dtype) -> None:
    n, k = 64, 128
    local_nz = as_nz(seed_nd(n, k, dtype, npu_device(), seed=1234))
    gathered = all_gather_nz_bytes(local_nz)
    torch.npu.synchronize()
    if not is_fractal_nz(gathered):
        raise RuntimeError(f"output format={torch_npu.get_npu_format(gathered)}, want FRACTAL_NZ")
    if gathered.dtype != dtype:
        raise RuntimeError(f"output dtype {gathered.dtype} != {dtype}")
    if tuple(gathered.shape) != (n, k * world):
        raise RuntimeError(f"output shape {tuple(gathered.shape)} != {(n, k * world)}")
    print_rank0(rank, f"PASS returns NZ {dtype} shape={[n, k * world]}")


def test_k_concat_is_full_nz(rank: int, world: int, dtype: torch.dtype) -> None:
    """K-shard NZ storages concatenated are the full-weight NZ tensor."""
    n, k = 64, 128
    k0 = fractal_k0(dtype)
    assert k % world == 0 and k % k0 == 0
    full_nd = seed_nd(n, k, dtype, npu_device(), seed=1234)
    kl = k // world
    local_nd = full_nd[:, rank * kl : (rank + 1) * kl].contiguous()
    local_nz = as_nz(local_nd)
    torch.npu.synchronize()
    gathered = all_gather_nz_bytes(local_nz)
    torch.npu.synchronize()
    if not is_fractal_nz(gathered) or tuple(gathered.shape) != (n, k):
        raise RuntimeError(f"got format={torch_npu.get_npu_format(gathered)} shape={tuple(gathered.shape)}")
    err = maxabs(to_nd(gathered), full_nd)
    print_rank0(rank, f"{'PASS' if err == 0.0 else 'FAIL'} K-concat NZ {dtype} maxabs={err}")
    if err != 0.0:
        raise RuntimeError(f"K concat reconstruct failed {dtype} maxabs={err}")


def test_out_reuse(rank: int, world: int) -> None:
    n, k = 32, 64
    local_nz = as_nz(torch.randn(n, k, dtype=torch.float16, device=npu_device()))
    out = empty_nz(n, k * world, torch.float16, npu_device())
    ptr = out.data_ptr()
    got = all_gather_nz_bytes(local_nz, out=out)
    torch.npu.synchronize()
    if got.data_ptr() != ptr or not is_fractal_nz(got):
        raise RuntimeError("out= was not reused as NZ recv")
    print_rank0(rank, "PASS out= reuses NZ buffer")


def test_no_format_change_on_input(rank: int, world: int) -> None:
    x = as_nz(torch.randn(32, 64, dtype=torch.float16, device=npu_device()))
    ptr0 = x.data_ptr()
    fmt0 = int(torch_npu.get_npu_format(x))
    _ = all_gather_nz_bytes(x)
    torch.npu.synchronize()
    fmt1 = int(torch_npu.get_npu_format(x))
    if fmt0 != ACL_FORMAT_FRACTAL_NZ or fmt1 != ACL_FORMAT_FRACTAL_NZ:
        raise RuntimeError(f"input format changed {fmt0} -> {fmt1}")
    if x.data_ptr() != ptr0:
        raise RuntimeError("input storage replaced (possible Transdata)")
    print_rank0(rank, "PASS input stays FRACTAL_NZ at the same data_ptr")


def main() -> None:
    rank, world, _ = init_hccl(timeout_s=180)
    for dtype in _DTYPES:
        test_returns_nz(rank, world, dtype)
        dist.barrier()
        test_k_concat_is_full_nz(rank, world, dtype)
        dist.barrier()
    test_out_reuse(rank, world)
    dist.barrier()
    test_no_format_change_on_input(rank, world)
    dist.barrier()
    dist.destroy_process_group()
    print_rank0(rank, f"all byte-allgather NZ cases passed world={world}")


if __name__ == "__main__":
    main()
