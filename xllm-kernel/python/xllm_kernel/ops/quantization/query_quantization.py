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

"""One-pass static-scale FP8 query conversion for native FA3 attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_query_kernel(
    source: tl.tensor,
    destination: tl.tensor,
    count: tl.constexpr,
    heads: tl.constexpr,
    dimension: tl.constexpr,
    token_stride: tl.constexpr,
    head_stride: tl.constexpr,
    channel_stride: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    channel = offsets % dimension
    head = offsets // dimension % heads
    token = offsets // (heads * dimension)
    source_offsets = token * token_stride + head * head_stride + channel * channel_stride
    values = tl.load(source + source_offsets, offsets < count, other=0).to(tl.float32) / scale
    values = tl.minimum(tl.maximum(values, -448.0), 448.0)
    tl.store(destination + offsets, values, offsets < count)


def quantize_query_fp8(query: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize inside graph capture; the immutable scalar never requires a host sync."""
    if query.device.type != "cuda" or query.dtype not in (torch.bfloat16, torch.float16) or query.ndim != 3:
        raise ValueError("Native FP8 query conversion requires CUDA BF16/FP16 [token, head, channel] input")
    # Read projection/rotary strides in the conversion kernel. A separate
    # contiguous BF16 copy can otherwise precede the actual FP8 conversion.
    output = torch.empty(query.shape, dtype=torch.float8_e4m3fn, device=query.device)
    if query.numel():
        _quantize_query_kernel[(triton.cdiv(query.numel(), 256),)](
            query,
            output,
            query.numel(),
            query.shape[1],
            query.shape[2],
            *query.stride(),
            scale,
            256,
        )
    return output
