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

"""Bitwise precision: NZ byte AllGather vs ND AllGather + format_cast.

Compares:
  1. NZ storage bytes vs format_cast(full_nd) storage (bit exact)
  2. NZ storage bytes vs format_cast(ND-AllGather) storage (bit exact)
  3. Transdata-back ND vs full_nd / ND-AllGather (torch.equal)

  torchrun --nproc_per_node=2 --master_port=29681 tests/python/test_bitwise_nz_bytes.py
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
import torch.distributed as dist

if __name__ != "__main__":
    pytest.skip(
        "run via torchrun --nproc_per_node=N tests/python/test_bitwise_nz_bytes.py",
        allow_module_level=True,
    )

import torch_npu

from tests.python.nz_test_utils import as_nz, init_hccl, npu_device, print_rank0, seed_nd, to_nd
from xllm.python.distributed.all_gather_nz_bytes import (
    ACL_FORMAT_FRACTAL_NZ,
    all_gather_nz_bytes,
    fractal_k0,
    is_fractal_nz,
    nz_storage_nbytes,
    raw_storage_uint8,
    shard_nz,
)

_DTYPES = (torch.float16, torch.bfloat16, torch.int8, torch.float32)
_SHAPES = ((32, 64), (64, 128), (256, 256), (1024, 1024))
_SEED = 20260910


class _Row(NamedTuple):
    ok: bool
    line: str


def _cell(eq: bool, bad: int = 0) -> str:
    return "EQUAL" if eq else f"DIFF:{bad}"


def _nd_allgather_k(local_nd: torch.Tensor, world: int) -> torch.Tensor:
    n, kl = local_nd.shape
    buf = torch.empty(world, n, kl, dtype=local_nd.dtype, device=local_nd.device)
    dist.all_gather_into_tensor(buf.reshape(-1), local_nd.reshape(-1))
    return buf.permute(1, 0, 2).contiguous().reshape(n, world * kl)


def _bit_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, int, int]:
    """Return (equal, nbytes, n_mismatch). Tensors must be same numel uint8."""
    if a.shape != b.shape or a.dtype != torch.uint8 or b.dtype != torch.uint8:
        return False, int(a.numel()), int(a.numel())
    neq = a != b
    n_bad = int(neq.sum().item())
    return n_bad == 0, int(a.numel()), n_bad


def _one_case(rank: int, world: int, n: int, k: int, dtype: torch.dtype) -> _Row | None:
    k0 = fractal_k0(dtype)
    if k % world != 0 or n % 16 != 0:
        return None
    kl = k // world
    if kl % k0 != 0:
        return None
    full_nd = seed_nd(n, k, dtype, npu_device(), seed=_SEED, int8_low=-127, int8_high=128)
    local_nd = full_nd[:, rank * kl : (rank + 1) * kl].contiguous()
    local_nz = as_nz(local_nd)
    torch.npu.synchronize()

    got_nz = all_gather_nz_bytes(local_nz)
    nd_ag = _nd_allgather_k(local_nd, world)
    torch.npu.synchronize()
    if not torch.equal(nd_ag, full_nd):
        raise RuntimeError("ND AllGather != sliced full_nd (test setup)")
    nd_ag_copy = torch.empty_like(full_nd)
    nd_ag_copy.copy_(nd_ag)
    full_nz = as_nz(full_nd)
    nd_ag_nz = as_nz(nd_ag_copy)
    torch.npu.synchronize()
    fmt_full = int(torch_npu.get_npu_format(full_nz))
    fmt_ndag = int(torch_npu.get_npu_format(nd_ag_nz))
    fmt_got = int(torch_npu.get_npu_format(got_nz))
    if fmt_full != ACL_FORMAT_FRACTAL_NZ or fmt_ndag != ACL_FORMAT_FRACTAL_NZ:
        raise RuntimeError(f"golden not NZ: full={fmt_full} ndag={fmt_ndag}")

    got_bytes = raw_storage_uint8(got_nz)
    full_bytes = raw_storage_uint8(full_nz)
    ndag_bytes = raw_storage_uint8(nd_ag_nz)
    torch.npu.synchronize()

    eq_full_st, nbytes, bad_full = _bit_diff(got_bytes, full_bytes)
    eq_ndag_st, _, bad_ndag = _bit_diff(got_bytes, ndag_bytes)
    got_nd = to_nd(got_nz)
    torch.npu.synchronize()
    eq_full_nd = bool(torch.equal(got_nd, full_nd))
    eq_ndag_nd = bool(torch.equal(got_nd, nd_ag))
    ok = (
        is_fractal_nz(got_nz)
        and tuple(got_nz.shape) == (n, k)
        and eq_full_st
        and eq_ndag_st
        and eq_full_nd
        and eq_ndag_nd
        and bool(torch.equal(nd_ag, full_nd))
        and nz_storage_nbytes(got_nz) == nbytes
    )
    dtype_s = str(dtype).replace("torch.", "")
    line = (
        f"{dtype_s:<8} {f'{n}x{k}':>12} {nbytes:>10} "
        f"{_cell(eq_full_st, bad_full):>12} {_cell(eq_ndag_st, bad_ndag):>12} "
        f"{'EQUAL' if eq_full_nd else 'DIFF':>11} {'EQUAL' if eq_ndag_nd else 'DIFF':>9} "
        f"{fmt_got:>5}  {'PASS' if ok else 'FAIL'}"
    )
    return _Row(ok, line)


def _one_roundtrip(rank: int, world: int, n: int, k: int, dtype: torch.dtype) -> _Row | None:
    """full NZ → K-shard → allgather → full NZ, bitwise on storage and ND values."""
    k0 = fractal_k0(dtype)
    if k % world != 0 or (k // world) % k0 != 0 or n % 16 != 0:
        return None
    expect_local = (n, k // world)
    full_nd = seed_nd(n, k, dtype, npu_device(), seed=_SEED, int8_low=-127, int8_high=128)
    full_nz = as_nz(full_nd)
    torch.npu.synchronize()
    local_nz = shard_nz(full_nz, rank_world=(rank, world))
    torch.npu.synchronize()
    if not is_fractal_nz(local_nz) or tuple(local_nz.shape) != expect_local:
        raise RuntimeError(f"shard K got {tuple(local_nz.shape)} fmt={torch_npu.get_npu_format(local_nz)}")
    local_nd_ref = full_nd[:, rank * (k // world) : (rank + 1) * (k // world)].contiguous()
    shard_ref_nz = as_nz(local_nd_ref)
    torch.npu.synchronize()
    eq_shard_st, _, _ = _bit_diff(raw_storage_uint8(local_nz), raw_storage_uint8(shard_ref_nz))
    eq_shard_nd = bool(torch.equal(to_nd(local_nz), local_nd_ref))

    got_nz = all_gather_nz_bytes(local_nz)
    torch.npu.synchronize()
    eq_full_st, _, bad_f = _bit_diff(raw_storage_uint8(got_nz), raw_storage_uint8(full_nz))
    eq_full_nd = bool(torch.equal(to_nd(got_nz), full_nd))
    ok = (
        is_fractal_nz(got_nz)
        and tuple(got_nz.shape) == (n, k)
        and eq_shard_nd
        and eq_full_st
        and eq_full_nd
        and eq_shard_st
    )
    dtype_s = str(dtype).replace("torch.", "")
    line = (
        f"{dtype_s:<8} {f'{n}x{k}':>12} {f'{expect_local[0]}x{expect_local[1]}':>12} "
        f"{'EQUAL' if eq_shard_nd else 'DIFF':>10} {_cell(eq_full_st, bad_f):>10} "
        f"{'EQUAL' if eq_full_nd else 'DIFF':>9}  {'PASS' if ok else 'FAIL'}"
    )
    return _Row(ok, line)


def _run_table(rank: int, world: int, header: str, cases: list) -> tuple[int, int]:
    print_rank0(rank, header)
    passed = 0
    total = 0
    for fn, dtype, n, k in cases:
        rec = fn(rank, world, n, k, dtype)
        dist.barrier()
        if rec is None:
            continue
        total += 1
        passed += int(rec.ok)
        print_rank0(rank, rec.line)
    return passed, total


def main() -> None:
    rank, world, _ = init_hccl(timeout_s=180)
    header1 = (
        f"{'dtype':<8} {'shape':>12} {'bytes':>10} "
        f"{'stor@fullNZ':>12} {'stor@NDagNZ':>12} "
        f"{'val@fullND':>11} {'val@NDag':>9} {'fmt':>5}  result"
    )
    cases1 = [(_one_case, dtype, n, k) for dtype in _DTYPES for n, k in _SHAPES]
    p1, t1 = _run_table(rank, world, header1, cases1)
    dist.barrier()
    header2 = f"\n{'dtype':<8} {'full':>12} {'shard':>12} {'val@shard':>10} {'stor@full':>10} {'val@full':>9}  result"
    cases2 = [(_one_roundtrip, dtype, n, k) for dtype in _DTYPES for n, k in _SHAPES]
    p2, t2 = _run_table(rank, world, header2, cases2)
    dist.barrier()
    dist.destroy_process_group()
    passed, total = p1 + p2, t1 + t2
    print_rank0(
        rank,
        f"bitwise: {passed}/{total} passed world={world} (storage bits vs full-NZ and vs ND-AllGather+cast)",
    )
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
