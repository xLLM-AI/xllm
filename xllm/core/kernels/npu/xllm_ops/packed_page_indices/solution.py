# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare and run the PackedPageIndices standalone NPU component."""

from collections.abc import Sequence
from typing import Any

import torch
import torch_npu
import triton
from _packed_page_indices_host import HostPlan

from kernels import _pack_pages, _translate_pages

_INT32_MAX = 2147483647


def _validate(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> dict[str, Any]:
    if not isinstance(inputs, (list, tuple)) or not isinstance(attributes, dict):
        raise TypeError("inputs must be a list or tuple and attributes a dict")
    if len(inputs) != 3 or attributes:
        raise ValueError("expected exactly X, M, cu and empty attributes")
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            raise TypeError("all inputs must be INT32 tensors")
        if tensor.device.type != "npu" or tensor.device != inputs[0].device:
            raise ValueError("all inputs must be on the same NPU")
        if not tensor.is_contiguous() or tensor.numel() > _INT32_MAX:
            raise ValueError("inputs must be contiguous with INT32-sized indexing")
    x, mapping, cu = inputs
    if x.ndim != 2 or mapping.ndim not in (1, 2):
        raise ValueError("expected X[R,C] and M[B] or M[L,B]")
    if mapping.shape[-1] < 1 or (mapping.ndim == 2 and mapping.shape[0] < 1):
        raise ValueError("B and L must be positive")
    if cu.ndim != 1 or cu.numel() != x.shape[0] + 1:
        raise ValueError("cu must have shape [R+1]")
    counts = cu.detach().cpu().tolist()
    if counts[0] != 0 or any(b < a or b - a > x.shape[1] for a, b in zip(counts, counts[1:])):
        raise ValueError("cu must start at zero with counts in [0,C]")
    layers = 1 if mapping.ndim == 1 else mapping.shape[0]
    if layers * counts[-1] > _INT32_MAX:
        raise ValueError("output indexing exceeds INT32")
    mapping_values = mapping.detach().cpu().reshape(-1).tolist()
    if any(value < 0 for value in mapping_values):
        raise ValueError("M must be nonnegative")
    # Only active prefixes are transferred for validation. Inactive tails may be invalid.
    prefixes = [x[r, : counts[r + 1] - counts[r]].detach().cpu().tolist() for r in range(x.shape[0])]
    if any(value >= mapping.shape[-1] for row in prefixes for value in row):
        raise ValueError("active logical page exceeds mapping capacity")
    return {
        "layers": layers,
        "blocks": mapping.shape[-1],
        "single": mapping.ndim == 1,
        "values": [prefixes, mapping_values, counts],
    }


class Prepared:
    """Inputs retain their tensor identity; legal set_ changes are read on every run."""

    def __init__(self, inputs: Sequence[torch.Tensor], info: dict[str, Any]) -> None:
        self.inputs = list(inputs)
        self.attributes: dict[str, int] = {}
        self.info = info
        self.n = info["values"][2][-1]
        self.shape = (self.n,) if info["single"] else (info["layers"], self.n)
        self._host = HostPlan(self.inputs[0], list(self.shape), info["layers"], info["blocks"])
        self._columns = self.inputs[0].shape[1]
        self._rows = self.inputs[0].shape[0]
        self._column_tile = min(256, triton.next_power_of_2(max(1, self._columns)))
        self._layer_tile = 1
        self._mode = "gather" if self.n * info["layers"] >= 32768 else "fused"
        self._configure()

    def _configure(self) -> None:
        if self.n == 0:
            return
        x, mapping, cu = self.inputs
        output = self._host.allocate()
        grid = (
            self._rows,
            triton.cdiv(self.info["layers"], self._layer_tile),
            triton.cdiv(self._columns, self._column_tile),
        )
        compiled = _translate_pages.warmup(
            x,
            mapping,
            cu,
            output,
            self._columns,
            self.info["layers"],
            self.info["blocks"],
            self.n,
            self._column_tile,
            self._layer_tile,
            grid=grid,
        )
        self._launch_fused = compiled[grid]
        pages = torch.empty((self.n,), dtype=torch.int32, device=x.device)
        padded_blocks = triton.cdiv(self.info["blocks"] + 1, 32) * 32
        padded = torch.empty((self.info["layers"], padded_blocks), dtype=torch.int32, device=x.device)
        pack_grid = (
            max(
                self._rows * triton.cdiv(self._columns, self._column_tile),
                self.info["layers"] * triton.cdiv(padded_blocks, 4096),
            ),
            1,
            1,
        )
        packed = _pack_pages.warmup(
            x,
            mapping,
            cu,
            pages,
            padded,
            self._rows,
            self._columns,
            self.info["layers"],
            self.info["blocks"],
            self._column_tile,
            grid=pack_grid,
        )
        self._launch_pack = packed[pack_grid]

    def run(self) -> torch.Tensor:
        x, mapping, cu = self.inputs
        if self.n == 0:
            return self._host.allocate()
        if self._mode == "gather":
            pages, padded = self._host.allocate_workspace()
            self._launch_pack(x, mapping, cu, pages, padded)
            return self._host.gather(padded, pages)
        output = self._host.allocate()
        self._launch_fused(x, mapping, cu, output)
        return output


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> Prepared:
    return Prepared(inputs, _validate(inputs, attributes))


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    return prepare(inputs, attributes).run()
