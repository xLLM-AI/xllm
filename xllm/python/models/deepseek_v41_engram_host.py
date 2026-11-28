# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Node-local backing for read-only Engram INT8 shards."""

from __future__ import annotations

import atexit
import ctypes
import mmap
import os
import tempfile
from datetime import timedelta
from functools import cache

import torch

from xllm.python.distributed.collectives import node_local_group_store
from xllm.python.kernels_npu.triton.engram import lookup_int8
from xllm.python.models.deepseek_v41_engram import _validate_ue8m0_scale_bytes

_ACL_HOST_REG_MAPPED_PINNED = 0x10000002
_LOAD_TIMEOUT = timedelta(minutes=30)


@cache
def _ascend_runtime() -> ctypes.CDLL:
    runtime = ctypes.CDLL("libascendcl.so")
    runtime.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    runtime.aclrtHostRegisterV2.restype = ctypes.c_int
    runtime.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    runtime.aclrtHostGetDevicePointer.restype = ctypes.c_int
    runtime.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    runtime.aclrtHostUnregister.restype = ctypes.c_int
    return runtime


class EngramHostTable:
    """One TP shard, published by one DP rank and mapped by its local peers."""

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        device: torch.device,
        dp_size: int,
        layer: int,
    ) -> None:
        self._device = device
        self._runtime = _ascend_runtime()
        self._mapping: mmap.mmap | None = None
        self._registered_address: int | None = None
        self._e8m0 = scales.dtype == torch.uint8
        code_bytes = codes.numel()
        size = code_bytes + scales.numel() * scales.element_size()
        size = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        store, rank, peers = node_local_group_store("dp", device) if dp_size > 1 else (None, 0, (0,))
        leader = rank == peers[0]
        key = f"v41/engram/{layer}/{peers[0]}"
        path = None
        try:
            if leader:
                if self._e8m0:
                    _validate_ue8m0_scale_bytes(scales)
                fd, path = tempfile.mkstemp(prefix="xllm-engram-", dir="/dev/shm")
                with os.fdopen(fd, "r+b") as backing:
                    os.ftruncate(backing.fileno(), size)
                    self._mapping = mmap.mmap(backing.fileno(), size)
            else:
                store.wait([key], _LOAD_TIMEOUT)
                with open(os.fsdecode(store.get(key)), "r+b") as backing:
                    self._mapping = mmap.mmap(backing.fileno(), size)

            # Pinning faults in the leader's pages with the huge-page hint active.
            self._mapping.madvise(mmap.MADV_HUGEPAGE)
            self._register(size, code_bytes)
            if leader:
                torch.frombuffer(self._mapping, dtype=torch.int8, count=codes.numel()).view_as(codes).copy_(codes)
                torch.frombuffer(self._mapping, dtype=scales.dtype, count=scales.numel(), offset=code_bytes).view_as(
                    scales
                ).copy_(scales)
                if store is not None:
                    store.set(key, path)
            if store is not None:
                store.set(f"{key}/mapped/{rank}", "1")
                if leader:
                    store.wait([f"{key}/mapped/{peer}" for peer in peers], _LOAD_TIMEOUT)
        except (OSError, RuntimeError):
            self._close()
            raise
        finally:
            if path is not None:
                os.unlink(path)
        # Captured graphs borrow these addresses. Release only after execution
        # has stopped, before torch_npu destroys the device context.
        atexit.register(self._close)

    def _register(self, size: int, code_bytes: int) -> None:
        address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        with torch.npu.device(self._device):
            rc = self._runtime.aclrtHostRegisterV2(address, size, _ACL_HOST_REG_MAPPED_PINNED)
            if rc:
                raise RuntimeError(f"Engram Host registration failed for {size} bytes: {rc}")
            self._registered_address = address
            mapped = ctypes.c_void_p()
            rc = self._runtime.aclrtHostGetDevicePointer(address, ctypes.byref(mapped), 0)
            if rc or mapped.value is None:
                raise RuntimeError(f"Engram device address resolution failed: {rc}")
            self._codes = torch.tensor(mapped.value, dtype=torch.int64, device=self._device)
            self._scales = torch.tensor(mapped.value + code_bytes, dtype=torch.int64, device=self._device)

    def _close(self) -> None:
        if self._registered_address is not None:
            with torch.npu.device(self._device):
                torch.npu.synchronize(self._device)
                rc = self._runtime.aclrtHostUnregister(self._registered_address)
                if rc:
                    raise RuntimeError(f"Engram Host unregister failed: {rc}")
            self._registered_address = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None

    def lookup(
        self,
        indices: torch.Tensor,
        head_start: int,
        local_heads: int,
        vocab_start: int,
        vocab_end: int,
        width: int,
        quant_block: int,
    ) -> torch.Tensor:
        return lookup_int8(
            indices,
            self._codes,
            self._scales,
            head_start,
            local_heads,
            vocab_start,
            vocab_end,
            width,
            quant_block,
            mapped_host=True,
            e8m0=self._e8m0,
        )
