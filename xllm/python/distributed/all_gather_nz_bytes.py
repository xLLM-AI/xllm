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

"""Custom AllGather for NZ tensors as a raw byte array.

Payload path: HcclAllGather(send=data_ptr, recv=data_ptr,
count=storage_nbytes/itemsize, dtype). No Transdata / npu_reshape on the
communicated bytes.

Only K-axis sharding (local K = full_K / world) is supported. Storage is
``[K1, N, C0]``; HCCL concat is already valid NZ ``[N, K * world]``.

Reuses the existing ProcessGroupHCCL communicator (no new HCCL communicator setup).
"""

from __future__ import annotations

import ctypes

import torch
import torch.distributed as dist
import torch_npu

from xllm.python.distributed.npu.hccl import all_gather_on_current_stream

ACL_FORMAT_FRACTAL_NZ = int(torch_npu.Format.FRACTAL_NZ)
FRACTAL_N0 = 16
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
# FRACTAL_NZ uses C0=32 for int8 and C0=16 for floating-point types.

_NZ_DTYPES = frozenset({torch.int8, torch.float16, torch.float32, torch.bfloat16})

_ACL: ctypes.CDLL | None = None


def _load_acl() -> ctypes.CDLL:
    global _ACL
    if _ACL is None:
        _ACL = ctypes.CDLL("libascendcl.so")
        _ACL.aclrtMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        _ACL.aclrtMemcpyAsync.restype = ctypes.c_int32
    return _ACL


def fractal_k0(dtype: torch.dtype) -> int:
    if dtype not in _NZ_DTYPES:
        raise ValueError(f"unsupported NZ dtype {dtype}")
    return 32 if dtype == torch.int8 else 16


def is_fractal_nz(t: torch.Tensor) -> bool:
    return int(torch_npu.get_npu_format(t)) == ACL_FORMAT_FRACTAL_NZ


def nz_storage_nbytes(t: torch.Tensor) -> int:
    """Physical NZ storage bytes from aligned logical shape (no NZ padding)."""
    return int(t.shape[0]) * int(t.shape[1]) * t.dtype.itemsize


def _internal_format_allowed() -> bool:
    # torch.npu.config.allow_internal_format is write-only (no __getattr__).
    opt = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    return opt is not None and opt.decode() == "enable"


def empty_nz(n: int, k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Allocate an uninitialized 2D FRACTAL_NZ buffer (no ND→NZ Transdata).

    Does not change ``torch.npu.config.allow_internal_format``; the caller must
    already have set it to True (required by ``empty_with_format`` for NZ).
    """
    if not _internal_format_allowed():
        raise RuntimeError("torch.npu.config.allow_internal_format must be True to allocate FRACTAL_NZ")
    return torch_npu.empty_with_format((n, k), dtype=dtype, device=device, acl_format=ACL_FORMAT_FRACTAL_NZ)


def _require_nz2d(x: torch.Tensor) -> tuple[int, int, int]:
    if not x.is_npu:
        raise ValueError("expected an NPU tensor")
    if x.dim() != 2 or not is_fractal_nz(x):
        raise ValueError(f"expected 2D FRACTAL_NZ, got dim={x.dim()} format={int(torch_npu.get_npu_format(x))}")
    if x.storage_offset() != 0 or not x.is_contiguous():
        raise ValueError("expected contiguous whole NZ storage, not a view")
    n, k = int(x.shape[0]), int(x.shape[1])
    k0 = fractal_k0(x.dtype)
    if n % FRACTAL_N0 != 0 or k % k0 != 0:
        raise ValueError(f"N={n} must be divisible by {FRACTAL_N0} and K={k} divisible by C0={k0} (no NZ padding)")
    if x.untyped_storage().nbytes() != nz_storage_nbytes(x):
        raise ValueError("expected whole NZ storage without padding")
    return n, k, k0


def _require_out_nz(
    out: torch.Tensor,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    src: torch.Tensor,
) -> torch.Tensor:
    _require_nz2d(out)
    if out.dtype != dtype or out.device != device:
        raise ValueError("out dtype/device must match x")
    if tuple(out.shape) != (n, k):
        raise ValueError(f"out shape {tuple(out.shape)} != ({n}, {k})")
    if out.data_ptr() == src.data_ptr():
        raise ValueError("out must not alias x")
    return out


def _resolve_rank_world(
    rank_world: tuple[int, int] | None,
    group: dist.ProcessGroup | None,
) -> tuple[int, int]:
    if rank_world is None:
        if not dist.is_initialized():
            raise RuntimeError("distributed is not initialized; pass rank_world")
        rank, world = dist.get_rank(group), dist.get_world_size(group)
    else:
        rank, world = rank_world
    if world <= 0 or rank < 0 or rank >= world:
        raise ValueError(f"invalid rank/world: rank={rank} world={world}")
    return int(rank), int(world)


def _memcpy_d2d(
    dst: torch.Tensor,
    src: torch.Tensor,
    nbytes: int,
    src_off: int = 0,
) -> None:
    """Raw D2D copy of ``nbytes`` from ``src`` at byte offset ``src_off``."""
    current_stream = torch.npu.current_stream(src.device)
    src.record_stream(current_stream)
    dst.record_stream(current_stream)
    acl = _load_acl()
    ret = acl.aclrtMemcpyAsync(
        ctypes.c_void_p(dst.data_ptr()),
        nbytes,
        ctypes.c_void_p(src.data_ptr() + src_off),
        nbytes,
        ACL_MEMCPY_DEVICE_TO_DEVICE,
        ctypes.c_void_p(int(current_stream.npu_stream)),
    )
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpyAsync failed, ret={ret}")


def shard_nz(
    x: torch.Tensor,
    rank_world: tuple[int, int] | None = None,
    group: dist.ProcessGroup | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slice a full FRACTAL_NZ weight along K into a local NZ shard (no Transdata).

    Storage is ``[K1, N, C0]``; copy this rank's ``K1/world`` tiles → NZ ``[N, K/world]``.
    When ``out is None``, returns a buffer distinct from ``x`` (including world==1).
    Pass ``rank_world=(rank, world)`` or omit it to use ``dist.get_rank/get_world_size``.
    """
    n, k, k0 = _require_nz2d(x)
    rank, world = _resolve_rank_world(rank_world, group)
    k1 = k // k0
    if k1 % world != 0:
        raise ValueError(f"K1={k1} must be divisible by world={world} for K-shard")
    local_k = k // world
    shard_nb = n * local_k * x.dtype.itemsize
    with torch.npu.device(x.device):
        if out is None:
            out = empty_nz(n, local_k, x.dtype, x.device)
        else:
            _require_out_nz(out, n, local_k, x.dtype, x.device, src=x)
        _memcpy_d2d(out, x, shard_nb, src_off=rank * shard_nb)
        return out


def all_gather_nz_bytes(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """AllGather NZ storage as bytes along K; return FRACTAL_NZ ``[N, K*world]``.

    Pass ``out`` (FRACTAL_NZ ``[N, K*world]``, distinct from ``x``) to skip allocation.
    world==1 copies into a distinct buffer (or ``out``).
    """
    if not dist.is_initialized():
        raise RuntimeError("distributed is not initialized")
    n, k, _ = _require_nz2d(x)
    world = dist.get_world_size(group)
    send_nb = nz_storage_nbytes(x)
    out_n, out_k = n, k * world
    with torch.npu.device(x.device):
        recv = (
            empty_nz(out_n, out_k, x.dtype, x.device)
            if out is None
            else _require_out_nz(out, out_n, out_k, x.dtype, x.device, src=x)
        )
        if world == 1:
            _memcpy_d2d(recv, x, send_nb)
            return recv
        current_stream = torch.npu.current_stream(x.device)
        x.record_stream(current_stream)
        recv.record_stream(current_stream)
        all_gather_on_current_stream(x, recv, group, initialize=True)
        return recv


def raw_storage_uint8(t: torch.Tensor) -> torch.Tensor:
    """ND uint8 copy of physical NZ storage (no Transdata)."""
    _require_nz2d(t)
    nbytes = nz_storage_nbytes(t)
    with torch.npu.device(t.device):
        buf = torch.empty(nbytes, dtype=torch.uint8, device=t.device)
        _memcpy_d2d(buf, t, nbytes)
        return buf
