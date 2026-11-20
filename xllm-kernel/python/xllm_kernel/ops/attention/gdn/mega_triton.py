# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the shared GDN state-index gather before capture."""

import torch
import triton
from triton.runtime.jit import MockTensor

from xllm_kernel.ops.attention.gdn.mega import SHARED_LINEAR_STATE_INDICES_OUT_CONTRACT
from xllm_kernel.ops.attention.gdn.npu._shared_indices import _shared_linear_state_indices_kernel
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

_BLOCK = 128
_MAX_OFFSET = torch.iinfo(torch.int32).max
_NUM_VECTOR_CORES = triton.runtime.driver.active.utils.get_device_properties(device=torch.npu.current_device())[
    "num_vectorcore"
]
if _NUM_VECTOR_CORES <= 0:
    raise RuntimeError("Failed to detect NPU vector cores")
_INT32 = MockTensor(dtype=torch.int32)

_shared_linear_state_indices_kernel.warmup(
    ids=_INT32,
    rows=_INT32,
    out=_INT32,
    batch_size=1,
    total=2,
    capacity=2,
    row_stride=2,
    BLOCK=_BLOCK,
    grid=(1,),
)._init_handles()


@register_kernel(
    name="npu.triton.shared_linear_state_indices_out",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Int32 logical IDs, strided Conv/SSM table rows and a caller-owned contiguous output.",
    solution="triton",
    binding="python",
    contract=SHARED_LINEAR_STATE_INDICES_OUT_CONTRACT,
)
def _shared_linear_state_indices_out(ids: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> None:
    if ids.device.type != "npu" or ids.device != rows.device or ids.device != out.device:
        raise ValueError("shared GDN state indices require tensors on the same NPU")
    if ids.dtype != torch.int32 or rows.dtype != torch.int32 or out.dtype != torch.int32:
        raise ValueError("shared GDN state indices require int32 tensors")
    if ids.dim() != 1 or rows.dim() != 2 or rows.size(0) % 2 or rows.stride(1) != 1 or rows.stride(0) < rows.size(1):
        raise ValueError("expected one-dimensional IDs and paired table rows with unit inner stride")
    if out.shape != (rows.size(0) // 2, ids.numel(), 2) or not out.is_contiguous() or not ids.is_contiguous():
        raise ValueError("expected contiguous IDs and output of shape [groups, batch, 2]")
    if not rows.size(0) or not rows.size(1):
        raise ValueError("shared GDN state row tables must be nonempty")
    if out.numel() > _MAX_OFFSET or (rows.size(0) - 1) * rows.stride(0) + rows.size(1) > _MAX_OFFSET:
        raise ValueError("shared GDN state indices exceed the int32 offset range")
    if out.numel() == 0:
        return
    with torch.npu.device(ids.device):
        _shared_linear_state_indices_kernel[(min(_NUM_VECTOR_CORES, triton.cdiv(out.numel(), _BLOCK)),)](
            ids=ids,
            rows=rows,
            out=out,
            batch_size=ids.numel(),
            total=out.numel(),
            capacity=rows.size(1),
            row_stride=rows.stride(0),
            BLOCK=_BLOCK,
        )
