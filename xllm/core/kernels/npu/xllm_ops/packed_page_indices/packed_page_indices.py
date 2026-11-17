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

"""Pack ragged page tables and translate them into physical page indices.

For request ``r`` only the first ``cu[r + 1] - cu[r]`` entries of ``X[r]`` are
valid. The kernel copies those entries, in request order, into one contiguous
output and replaces every logical page number with its physical page number
``M[layer, page]``. Negative entries become ``-1`` and never index ``M``.

``prepare`` validates the inputs once, derives the output shape and compiles the
kernel. ``PackedPageIndicesPlan.run`` reads the current contents of the inputs
and the current NPU stream, then returns a freshly allocated output. Inputs are
never modified and an output never aliases an input.

A Triton tensor holds at most ``2**20`` elements, so the mapping row is kept in
one tensor only when ``B <= 2**20``. Inside that path the row is split into
16-bit halves: both halves stay below ``2**16`` and are therefore exact in
float32, and the int32 page number is reassembled with integer arithmetic.
Larger mappings take one exact int32 load per valid element. Neither path ever
rounds a page number through a float.
"""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver

_ROWS_PER_PROGRAM = 8
_COLUMN_BLOCK_LIMIT = 2048
# The gather path keeps two float32 copies of the mapping row (high and low
# 16-bit halves) live across all row iterations. On Ascend910_9362 the
# unified buffer holds 1572864 bits: B=16384 compiles at about two thirds of
# that budget and B=32768 overflows it. 8192 keeps a 2x margin so another
# CANN or triton-ascend build cannot turn a legal mapping into a compile
# failure. Wider mappings use the exact pointer-load path.
_ROW_GATHER_LIMIT = 1 << 13
_ACL_FORMAT_ND = "ND"
_ACL_FORMAT_ND_CODE = 2


def require_nd_storage(tensor: torch.Tensor, name: str) -> None:
    """Reject tensors whose storage is not plain ND.

    ``is_contiguous()`` stays true for a FRACTAL_NZ tensor while the kernel
    walks row-major ND addresses, so a format check is the only way to keep
    the pointer arithmetic honest. ``npu_format_cast`` to ND is the caller's
    job because a conversion would change the caller's storage identity.
    """
    import torch_npu

    getter = getattr(torch_npu, "get_npu_format", None)
    if getter is None:
        raise RuntimeError(
            "torch_npu.get_npu_format is required to verify ND storage"
        )
    value = getter(tensor)
    if value not in (_ACL_FORMAT_ND, _ACL_FORMAT_ND_CODE):
        raise ValueError(f"{name} must use ND storage, got {value}")


@triton.jit
def _packed_page_indices_kernel(
    x_ptr,
    m_ptr,
    cu_ptr,
    out_ptr,
    columns,
    blocks,
    total,
    rows,
    SINGLE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    ROW_GATHER: tl.constexpr,
):
    """One program handles one layer and a row block of requests.

    When ``ROW_GATHER`` is set the mapping row is loaded once per program and
    reused by every request in the row block, so each translation is a pair of
    register gathers over the 16-bit halves. Otherwise every valid element
    performs one exact int32 load, which supports mappings that cannot fit in a
    Triton tensor.
    """
    layer = tl.program_id(0)
    row_block = tl.program_id(1)
    column_block = tl.program_id(2)
    layer_offset = 0 if SINGLE else layer * blocks
    output_offset = 0 if SINGLE else layer * total

    column = column_block * BLOCK + tl.arange(0, BLOCK)
    if ROW_GATHER:
        page_index = tl.arange(0, BLOCK_BLOCKS)
        page_values = tl.load(
            m_ptr + layer_offset + page_index,
            mask=page_index < blocks,
            other=0,
        ).to(tl.int32)
        page_high = (page_values >> 16).to(tl.float32)
        page_low = (page_values & 0xFFFF).to(tl.float32)

    for step in tl.static_range(ROWS_PER_PROGRAM):
        row = row_block * ROWS_PER_PROGRAM + step
        row_valid = row < rows
        start = tl.load(cu_ptr + row, mask=row_valid, other=0)
        end = tl.load(cu_ptr + row + 1, mask=row_valid, other=0)
        valid = row_valid & (column < end - start)
        logical = tl.load(x_ptr + row * columns + column, mask=valid, other=-1)
        safe_page = tl.where(logical < 0, 0, logical)
        if ROW_GATHER:
            high = tl.gather(page_high, safe_page, 0).to(tl.int32)
            low = tl.gather(page_low, safe_page, 0).to(tl.int32)
            physical = high * 65536 + low
        else:
            physical = tl.load(
                m_ptr + layer_offset + safe_page,
                mask=valid & (logical >= 0),
                other=0,
            )
        value = tl.where(logical < 0, -1, physical)
        tl.store(
            out_ptr + output_offset + start + column,
            value,
            mask=valid,
        )


class PackedPageIndicesPlan:
    """Reusable compiled plan returned by :func:`prepare`.

    Contract:
        Preconditions: ``inputs`` passed the checks in :func:`prepare`.
        Postconditions: ``run`` returns a contiguous int32 tensor of the frozen
            shape, on the frozen device, holding the translation of the current
            input contents on the current stream.
        Invariants: the plan reads ``inputs`` and ``attributes`` afresh on every
            call, so in-place updates with unchanged shape, dtype, device and
            total ``P`` are observed.
        Violations: construction raises ``TypeError`` or ``ValueError``; a
            missing input, a changed device or a launcher mismatch raises
            ``RuntimeError`` at call time rather than producing stale output.
    """

    def __init__(
        self,
        inputs: list[torch.Tensor],
        attributes: dict[str, int],
        shape: Sequence[int],
        single: bool,
        layers: int,
        blocks: int,
        columns: int,
        total: int,
    ) -> None:
        self.inputs = inputs
        self.attributes = dict(attributes)
        self.x_shape = tuple(inputs[0].shape)
        self.m_shape = tuple(inputs[1].shape)
        self.cu_length = inputs[2].numel()
        self.shape = tuple(shape)
        self.single = single
        self.layers = layers
        self.blocks = blocks
        self.columns = columns
        self.total = total
        self.n = total
        self.dtype = torch.int32
        self.device = inputs[0].device
        self.rows = _ROWS_PER_PROGRAM
        self.block = min(
            _COLUMN_BLOCK_LIMIT,
            triton.next_power_of_2(max(1, columns)),
        )
        self.row_gather = blocks <= _ROW_GATHER_LIMIT
        self.block_blocks = triton.next_power_of_2(max(1, blocks)) if self.row_gather else 1
        self.grid = (
            1 if single else layers,
            triton.cdiv(max(1, inputs[0].shape[0]), self.rows),
            triton.cdiv(max(1, columns), self.block),
        )

        x, m, cu = inputs
        with torch.npu.device(self.device.index):
            probe = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            compiled = _packed_page_indices_kernel.warmup(
                x,
                m,
                cu,
                probe,
                columns,
                blocks,
                total,
                x.shape[0],
                SINGLE=single,
                ROWS_PER_PROGRAM=self.rows,
                BLOCK=self.block,
                BLOCK_BLOCKS=self.block_blocks,
                ROW_GATHER=self.row_gather,
                grid=self.grid,
            )
        for attribute in (
            "run",
            "function",
            "packed_metadata",
            "launch_enter_hook",
            "launch_exit_hook",
        ):
            if not hasattr(compiled, attribute):
                raise RuntimeError(
                    "the installed triton-ascend CompiledKernel does not expose "
                    f"{attribute!r}; packed_page_indices requires the direct "
                    "launcher introduced in triton-ascend 3.2"
                )
        self._launch = compiled.run
        self._function = compiled.function
        self._packed_metadata = compiled.packed_metadata
        self._enter_hook = compiled.launch_enter_hook
        self._exit_hook = compiled.launch_exit_hook

    def run(self) -> torch.Tensor:
        out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        if self.total == 0:
            return out
        x, m, cu = self.inputs
        for name, tensor in (("X", x), ("M", m), ("cu", cu)):
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"{name} must be a tensor, got {type(tensor)!r}")
            if tensor.device != self.device:
                raise RuntimeError(f"{name} moved from {self.device} to {tensor.device}")
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} is no longer contiguous")
            if tensor.dtype != torch.int32:
                raise RuntimeError(f"{name} changed dtype to {tensor.dtype}")
        if tuple(x.shape) != self.x_shape:
            raise RuntimeError(f"X shape changed from {self.x_shape} to {tuple(x.shape)}")
        if cu.numel() != self.cu_length:
            raise RuntimeError(f"cu length changed from {self.cu_length} to {cu.numel()}")
        if tuple(m.shape) != self.m_shape:
            raise RuntimeError(f"M shape changed from {self.m_shape} to {tuple(m.shape)}")
        if torch.npu.current_device() == self.device.index:
            self._dispatch(x, m, cu, out)
        else:
            with torch.npu.device(self.device.index):
                self._dispatch(x, m, cu, out)
        return out

    def _dispatch(self, x, m, cu, out) -> None:
        grid0, grid1, grid2 = self.grid
        stream = driver.active.get_current_stream(self.device.index)
        self._launch(
            grid0,
            grid1,
            grid2,
            stream,
            self._function,
            self._packed_metadata,
            None,
            self._enter_hook,
            self._exit_hook,
            x,
            m,
            cu,
            out,
            self.columns,
            self.blocks,
            self.total,
            x.shape[0],
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prepare(
    inputs: Sequence[torch.Tensor],
    attributes: dict[str, int],
) -> PackedPageIndicesPlan:
    """Validate ``X``, ``M`` and ``cu`` and compile the packing kernel.

    Args:
        inputs: ``[X, M, cu]``. ``X`` is ``[R, C]`` int32, ``M`` is ``[B]`` or
            ``[L, B]`` int32, ``cu`` is ``[R + 1]`` int32. All three are
            contiguous, on the same NPU, and may carry a nonzero storage
            offset.
        attributes: must be an empty dict.

    Returns:
        A plan whose ``run`` returns ``[P]`` or ``[L, P]`` int32 with
        ``P = cu[-1]``. The shape, dtype and device are frozen, while the
        total ``P`` and the per-request counts may change in place as long as
        the total stays the same.

    Raises:
        TypeError: an input is not a tensor or ``attributes`` is not a dict.
        ValueError: shape, dtype, device, contiguity, ``cu`` monotonicity,
            mapping values, page range or element count violates the contract.
    """
    if not isinstance(attributes, dict):
        raise TypeError(f"attributes must be a dict, got {type(attributes)!r}")
    if attributes:
        raise ValueError("attributes must be empty")
    if len(inputs) != 3:
        raise ValueError(f"expected [X, M, cu], got {len(inputs)} inputs")
    x, m, cu = inputs
    for name, tensor in (("X", x), ("M", m), ("cu", cu)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    for name, tensor in (("X", x), ("M", m), ("cu", cu)):
        if tensor.device.type != "npu":
            raise ValueError(f"{name} must be on an NPU, got {tensor.device}")
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        require_nd_storage(tensor, name)
    if x.dim() != 2:
        raise ValueError(f"X must be 2-D, got {tuple(x.shape)}")
    if cu.dim() != 1:
        raise ValueError(f"cu must be 1-D, got {tuple(cu.shape)}")
    if m.dim() not in (1, 2):
        raise ValueError(f"M must be 1-D or 2-D, got {tuple(m.shape)}")
    if not (x.device == m.device == cu.device):
        raise ValueError("X, M and cu must be on the same device")

    rows, columns = x.shape
    if cu.numel() != rows + 1:
        raise ValueError(f"cu must have R + 1 = {rows + 1} entries, got {cu.numel()}")

    single = m.dim() == 1
    if single:
        layers, blocks = 1, m.shape[0]
    else:
        layers, blocks = m.shape
    if layers < 1 or blocks < 1:
        raise ValueError(f"M must have at least one layer and one block, got {tuple(m.shape)}")

    cu_host = cu.detach().to("cpu").tolist()
    if cu_host[0] != 0:
        raise ValueError(f"cu[0] must be 0, got {cu_host[0]}")
    for row in range(rows):
        count = cu_host[row + 1] - cu_host[row]
        if not 0 <= count <= columns:
            raise ValueError(
                f"cu must be non-decreasing with counts in [0, {columns}], "
                f"got cu[{row + 1}] - cu[{row}] = {count}"
            )
    total = cu_host[-1]

    m_host = m.detach().to("cpu")
    if not bool((m_host >= 0).all()):
        raise ValueError("M must not contain negative page numbers")
    x_host = x.detach().to("cpu")
    for row in range(rows):
        count = cu_host[row + 1] - cu_host[row]
        if count == 0:
            continue
        prefix = x_host[row, :count]
        if not bool((prefix < blocks).all()):
            raise ValueError(
                f"X[{row}, :{count}] must be smaller than the {blocks} logical pages"
            )

    elements = max(total * layers, rows * columns, layers * blocks)
    if elements > 2147483647:
        raise ValueError(f"{elements} elements exceed the int32 index limit")

    return PackedPageIndicesPlan(
        [x, m, cu],
        attributes,
        (total,) if single else (layers, total),
        single,
        layers,
        blocks,
        columns,
        total,
    )


def run(
    inputs: Sequence[torch.Tensor],
    attributes: dict[str, int],
) -> torch.Tensor:
    """Prepare and execute in one call, including the preparation cost."""
    return prepare(inputs, attributes).run()
