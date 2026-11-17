"""PackedPageIndices submission entry point.

``prepare`` validates the inputs in the host-side C++ extension, freezes the
output shape and compiles the Triton kernel. ``prepared.run()`` reads the
current input contents on the current NPU stream and returns a new output.

The C++ extension is built on first use; run ``python3 build.py`` to pay that
cost once before a verification session. See ``DESIGN.md`` for the contract and
``README.md`` for the reproduction steps.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

import packed_page_indices as _kernel
from host.build_host import load_host


class Prepared(Protocol):
    def run(self) -> torch.Tensor: ...


_HOST = None


def _host():
    global _HOST
    if _HOST is None:
        _HOST = load_host()
    return _HOST


class _Prepared:
    def __init__(self, plan: _kernel.PackedPageIndicesPlan, info: dict) -> None:
        self._plan = plan
        self.inputs = plan.inputs
        self.attributes = plan.attributes
        self.shape = plan.shape
        self.n = plan.n
        self.info = info

    def run(self) -> torch.Tensor:
        return self._plan.run()


def prepare(
    inputs: Sequence[torch.Tensor],
    attributes: dict[str, int],
) -> Prepared:
    """Validate the inputs and freeze the launch plan.

    Preconditions: ``inputs`` is ``[X, M, cu]`` with contiguous int32 NPU
    tensors on one device, and ``attributes`` is an empty dict.
    Postconditions: the returned object reports the frozen shape ``[P]`` or
    ``[L, P]`` and its ``run`` observes later in-place value updates with the
    same shapes and total ``P``.
    Violations: a contract violation raises ``TypeError`` or ``ValueError``
    before any device work is launched.
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
        _kernel.require_nd_storage(tensor, name)
    if x.dim() != 2:
        raise ValueError(f"X must be 2-D, got {tuple(x.shape)}")
    if m.dim() not in (1, 2):
        raise ValueError(f"M must be 1-D or 2-D, got {tuple(m.shape)}")
    if not (x.device == m.device == cu.device):
        raise ValueError("X, M and cu must be on the same device")

    x_cpu = x.detach().to("cpu")
    m_cpu = m.detach().to("cpu")
    cu_cpu = cu.detach().to("cpu")
    host = _host()
    try:
        total, layers, blocks, single = host.plan(x_cpu, m_cpu, cu_cpu)
    except RuntimeError as error:
        raise ValueError(str(error)) from error

    plan = _kernel.PackedPageIndicesPlan(
        [x, m, cu],
        {},
        (total,) if single else (layers, total),
        single,
        layers,
        blocks,
        x.shape[1],
        total,
    )
    info = {
        "layers": layers,
        "blocks": blocks,
        "single": single,
        "values": [x_cpu.tolist(), m_cpu.tolist(), cu_cpu.tolist()],
    }
    return _Prepared(plan, info)


def run(
    inputs: Sequence[torch.Tensor],
    attributes: dict[str, int],
) -> torch.Tensor:
    """Prepare and execute in one call, including the preparation cost."""
    return prepare(inputs, attributes).run()
