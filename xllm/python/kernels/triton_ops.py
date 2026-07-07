# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementations of select xLLM ops.

Each kernel is wrapped as a ``torch.library.custom_op`` (opaque to Dynamo, with
a ``register_fake`` for shape inference) so it stays a single node in the
compiled graph. Fixed BLOCK size (no autotune) ensures CUDA-graph capturability.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Fixed block width (no autotune) — required for CUDA-graph capturability.
_SILU_BLOCK = 1024


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    d,
    x_row_stride,
    out_row_stride,
    BLOCK: tl.constexpr,
):
    """out[row, :d] = silu(x[row, :d]) * x[row, d:2d], computed in fp32."""
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < d
    gate = tl.load(x_ptr + row * x_row_stride + col, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(x_ptr + row * x_row_stride + d + col, mask=mask, other=0.0).to(
        tl.float32
    )
    silu = gate * tl.sigmoid(gate)
    tl.store(out_ptr + row * out_row_stride + col, silu * up, mask=mask)


@torch.library.custom_op("xllm_triton::silu_and_mul", mutates_args=())
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Gated SiLU: input ``[..., 2d]`` -> output ``[..., d]`` (SwiGLU)."""
    assert x.shape[-1] % 2 == 0, "silu_and_mul expects an even last dim"
    d = x.shape[-1] // 2
    x2d = x.contiguous().reshape(-1, x.shape[-1])
    n_rows = x2d.shape[0]
    out = torch.empty((n_rows, d), dtype=x.dtype, device=x.device)
    grid = (n_rows, triton.cdiv(d, _SILU_BLOCK))
    _silu_and_mul_kernel[grid](
        x2d, out, d, x2d.stride(0), out.stride(0), BLOCK=_SILU_BLOCK
    )
    return out.reshape(*x.shape[:-1], d)


@silu_and_mul.register_fake
def _silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    shape = list(x.shape)
    shape[-1] //= 2
    return x.new_empty(shape)
