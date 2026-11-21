# -*- coding: utf-8 -*-
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
"""Packed page indices: fused valid-prefix packing + per-layer page translation.

Device paths (selected once in prepare from the output element count):

* fused Triton kernel (small outputs): one program per (layer, request, tile)
  reads cu on device, gathers M, applies the negative-page rule and stores.
* hybrid path (large outputs): a Triton kernel writes the packed translated
  logical-page index (``0`` for negative pages, otherwise ``x + 1``); the
  native NPU Gather then selects from an augmented mapping whose column 0 is
  ``-1``.  The index buffer is allocated per call so concurrent streams cannot
  race.

Host-side work (metadata, output allocation, mapping augmentation/refresh and
the native Gather dispatch) lives in the C++ extension under ``csrc/``.
``prepared.run`` never copies inputs to the host and always uses the current
NPU stream; it returns a newly allocated, contiguous int32 tensor.
"""
import os
import sys
from typing import Sequence

import torch
import torch_npu  # noqa: F401  (registers the NPU device)
import triton
import triton.language as tl

_I32 = 2 ** 31 - 1
_HYBRID_MIN_ELEMS = 32768

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENV_BIN = os.path.dirname(sys.executable)
if _VENV_BIN and _VENV_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _VENV_BIN + os.pathsep + os.environ.get("PATH", "")

_ext = None
_ext_error = None
try:
    from torch.utils.cpp_extension import load as _load_ext

    _build_dir = os.path.join(_HERE, "build_ext")
    os.makedirs(_build_dir, exist_ok=True)
    _ext = _load_ext(
        name="packed_page_ext",
        sources=[os.path.join(_HERE, "csrc", "packed_ext.cpp")],
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=False,
        build_directory=_build_dir,
        verbose=False,
    )
except Exception as exc:  # keep a documented Python fallback
    _ext_error = repr(exc)


def _alloc(x, layers: int, total: int, single: bool):
    if _ext is not None:
        return _ext.packed_empty(x, layers, total, single)
    shape = (total,) if single else (layers, total)
    return torch.empty(shape, dtype=torch.int32, device=x.device)


def _make_aug(m):
    if _ext is not None:
        return _ext.packed_make_aug(m)
    if m.dim() == 1:
        out = torch.empty(m.numel() + 1, dtype=torch.int32, device=m.device)
        out[0] = -1
        out[1:].copy_(m)
        return out
    out = torch.empty((m.shape[0], m.shape[1] + 1), dtype=torch.int32, device=m.device)
    out[:, 0] = -1
    out[:, 1:].copy_(m)
    return out


def _refresh_aug(aug, m):
    if _ext is not None:
        _ext.packed_refresh_aug(aug, m)
        return
    if m.dim() == 1:
        aug[1:].copy_(m)
    else:
        aug[:, 1:].copy_(m)


def _gather(m_aug, idx, single: bool):
    if _ext is not None:
        return _ext.packed_gather(m_aug, idx, single)
    return m_aug.index_select(0 if single else 1, idx)


@triton.jit
def _fused_kernel(
    X, M, CU, OUT,
    C, B, P,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    r = tl.program_id(1)
    l = tl.program_id(2)

    start = tl.load(CU + r)
    end = tl.load(CU + r + 1)
    count = end - start

    j = tile * BLOCK + tl.arange(0, BLOCK)
    mask = j < count
    x = tl.load(X + r * C + j, mask=mask, other=0)
    idx = tl.where(x < 0, 0, x)
    m = tl.load(M + l * B + idx, mask=mask, other=0)
    value = tl.where(x < 0, -1, m)
    tl.store(OUT + l * P + start + j, value, mask=mask)


@triton.jit
def _pack_index_kernel(
    X, CU, IDX,
    C,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    r = tl.program_id(1)

    start = tl.load(CU + r)
    end = tl.load(CU + r + 1)
    count = end - start

    j = tile * BLOCK + tl.arange(0, BLOCK)
    mask = j < count
    x = tl.load(X + r * C + j, mask=mask, other=0)
    # Augmented mapping: column 0 of M_aug is -1, column x+1 is M[x].
    idx = tl.where(x < 0, 0, x + 1)
    tl.store(IDX + start + j, idx, mask=mask)


def _choose_block(columns: int) -> int:
    if columns <= 0:
        return 16
    block = 16
    while block < columns and block < 2048:
        block *= 2
    return block


def _validate(inputs, attributes):
    if not isinstance(inputs, (tuple, list)) or not isinstance(attributes, dict):
        raise TypeError("inputs must be a sequence; attributes must be a dict")
    if len(inputs) != 3:
        raise ValueError("expected X, M, cu")
    if set(attributes) != set():
        raise ValueError("attributes missing or unknown")

    for t in inputs:
        if not isinstance(t, torch.Tensor) or t.dtype != torch.int32:
            raise TypeError("all inputs must be int32 tensors")
        if t.device.type != "npu":
            raise ValueError("NPU inputs required")
        if not t.is_contiguous():
            raise ValueError("contiguous inputs required; storage offset is supported")
        if t.device != inputs[0].device:
            raise ValueError("all inputs must be on the same device")
        if t.numel() > _I32:
            raise ValueError("input indexing exceeds int32")

    # One-time preflight; prepared.run never copies inputs to the host.
    values = [t.detach().cpu().tolist() for t in inputs]
    x, m, cu = inputs
    xv, mv, cv = values

    if x.dim() != 2:
        raise ValueError("X shape must be [R, C]")
    if m.dim() not in (1, 2) or m.shape[-1] < 1:
        raise ValueError("M shape must be [B] or [L, B] with B >= 1")
    if m.dim() == 2 and m.shape[0] < 1:
        raise ValueError("L >= 1")
    rows = [mv] if m.dim() == 1 else mv
    if any(v < 0 for row in rows for v in row):
        raise ValueError("M must be nonnegative")
    if cu.dim() != 1 or len(cv) != x.shape[0] + 1 or cv[0] != 0:
        raise ValueError("invalid cu shape/start")
    if any(cv[i] > cv[i + 1] or cv[i + 1] - cv[i] > x.shape[1] for i in range(x.shape[0])):
        raise ValueError("invalid cu counts")

    active = [row[:cv[r + 1] - cv[r]] for r, row in enumerate(xv)]
    if any(v >= m.shape[-1] for row in active for v in row):
        raise ValueError("logical block out of range")

    output_count = len(rows) * cv[-1]
    if output_count > _I32:
        raise ValueError("output indexing exceeds int32")

    return {
        "values": values,
        "layers": len(rows),
        "blocks": m.shape[-1],
        "single": m.dim() == 1,
    }


class _Prepared:
    def __init__(self, inputs, attributes, info):
        self.inputs = list(inputs)
        self.attributes = dict(attributes)
        self.info = info
        self.single = info["single"]
        self.layers = info["layers"]
        self.blocks = info["blocks"]
        self.requests = inputs[0].shape[0]
        self.columns = inputs[0].shape[1]
        self.n = info["values"][2][-1]
        self.shape = (self.n,) if self.single else (self.layers, self.n)
        self.device = inputs[0].device
        self.block = _choose_block(self.columns)
        self._tiles = (self.columns + self.block - 1) // self.block if self.columns > 0 else 0
        self._use_hybrid = self.layers * self.n >= _HYBRID_MIN_ELEMS
        self._compiled = None
        self._m_aug = None
        self._m_key = None
        # Host-side metadata validation, one-time.
        self.cxx_meta = None
        if _ext is not None:
            self.cxx_meta = _ext.packed_meta(inputs[0], inputs[1], inputs[2])
        self._prepare_path()

    # ------------------------------------------------------------------ paths
    def _prepare_path(self):
        if self.n == 0 or self.columns == 0 or self.requests == 0:
            return
        torch.npu.set_device(self.device)
        x, m, cu = self.inputs
        if self._use_hybrid:
            self._prepare_hybrid(x, m, cu)
        else:
            self._prepare_fused(x, m, cu)

    def _prepare_fused(self, x, m, cu):
        out = _alloc(x, self.layers, self.n, self.single)
        grid = (self._tiles, self.requests, self.layers)
        kernel = _fused_kernel.warmup(
            x, m, cu, out,
            self.columns, self.blocks, self.n,
            BLOCK=self.block,
            grid=grid,
        )
        kernel._init_handles()
        self._compiled = kernel
        self._grid = grid
        self._run_fn = kernel.run
        self._function = kernel.function
        self._packed_metadata = kernel.packed_metadata
        self._run_fused(out)  # warm device/allocator caches outside timing

    def _prepare_hybrid(self, x, m, cu):
        self._m_aug = _make_aug(m)
        self._m_key = (m.data_ptr(), m._version)

        idx = _alloc(x, 1, self.n, True)
        grid = (self._tiles, self.requests)
        kernel = _pack_index_kernel.warmup(
            x, cu, idx,
            self.columns,
            BLOCK=self.block,
            grid=grid,
        )
        kernel._init_handles()
        self._compiled = kernel
        self._grid = grid
        self._run_fn = kernel.run
        self._function = kernel.function
        self._packed_metadata = kernel.packed_metadata
        self._run_pack(idx)
        _gather(self._m_aug, idx, self.single)  # warm the native gather path

    # ------------------------------------------------------------- launches
    def _stream(self):
        return triton.runtime.driver.active.get_current_stream()

    def _run_fused(self, out):
        self._run_fn(
            self._grid[0], self._grid[1], self._grid[2], self._stream(),
            self._function, self._packed_metadata, None, None, None,
            self.inputs[0], self.inputs[1], self.inputs[2], out,
            self.columns, self.blocks, self.n,
        )

    def _run_pack(self, idx):
        self._run_fn(
            self._grid[0], self._grid[1], 1, self._stream(),
            self._function, self._packed_metadata, None, None, None,
            self.inputs[0], self.inputs[2], idx,
            self.columns,
        )

    def _refresh_mapping(self):
        m = self.inputs[1]
        key = (m.data_ptr(), m._version)
        if key == self._m_key:
            return
        _refresh_aug(self._m_aug, m)
        self._m_key = key

    # --------------------------------------------------------------- public
    def run(self) -> torch.Tensor:
        if self._compiled is None:
            return _alloc(self.inputs[0], self.layers, self.n, self.single)
        if not self._use_hybrid:
            out = _alloc(self.inputs[0], self.layers, self.n, self.single)
            self._run_fused(out)
            return out
        idx = _alloc(self.inputs[0], 1, self.n, True)
        self._run_pack(idx)
        self._refresh_mapping()
        return _gather(self._m_aug, idx, self.single)


def prepare(inputs: Sequence[torch.Tensor], attributes: dict) -> _Prepared:
    info = _validate(inputs, attributes)
    return _Prepared(inputs, attributes, info)


def run(inputs: Sequence[torch.Tensor], attributes: dict) -> torch.Tensor:
    return prepare(inputs, attributes).run()
