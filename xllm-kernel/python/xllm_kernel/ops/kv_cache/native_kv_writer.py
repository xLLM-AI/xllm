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

"""Fused FP8 quantize/scatter; attention stays in FlashInfer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from xllm_kernel.ops.kv_cache import CONVERT_XQA_PAGES_CONTRACT, WRITE_FP8_KV_CONTRACT
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel


@triton.jit
def _write_fp8(
    key: tl.tensor,
    value: tl.tensor,
    slots: tl.tensor,
    key_cache: tl.tensor,
    value_cache: tl.tensor,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_channel_stride: tl.constexpr,
    value_token_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_channel_stride: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    capacity: tl.constexpr,
    key_scale: tl.constexpr,
    value_scale: tl.constexpr,
    block: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slots + token).to(tl.int64)
    if slot >= 0 and slot < capacity:
        channel = tl.arange(0, block)
        k = tl.load(
            key + token * key_token_stride + head * key_head_stride + channel * key_channel_stride,
            mask=channel < dim,
            other=0,
        ).to(tl.float32)
        v = tl.load(
            value + token * value_token_stride + head * value_head_stride + channel * value_channel_stride,
            mask=channel < dim,
            other=0,
        ).to(tl.float32)
        offset = (slot * heads + head) * dim + channel
        tl.store(key_cache + offset, tl.clamp(k / key_scale, -448.0, 448.0), mask=channel < dim)
        tl.store(value_cache + offset, tl.clamp(v / value_scale, -448.0, 448.0), mask=channel < dim)


@register_kernel(
    name="cuda.triton.write_fp8_kv",
    contract=WRITE_FP8_KV_CONTRACT,
    platform=PlatformEnum.CUDA,
    execution_modes=("eager", "cudagraph"),
    input_domain="SM90 strided BF16/FP16 [T,H,D], contiguous int32/int64 slots and contiguous NHD E4M3FN caches.",
    solution="triton",
    binding="python",
)
def _write_fp8_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale: float,
    value_scale: float,
) -> None:
    # The model and bound cache establish shapes/dtypes once, before serving.
    # All input strides are explicit, including compiler-selected head layouts.
    _write_fp8[(key.shape[0], key.shape[1])](
        key,
        value,
        slots,
        key_cache,
        value_cache,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key.shape[1],
        key.shape[2],
        key_cache.shape[0] * key_cache.shape[1],
        key_scale,
        value_scale,
        block=triton.next_power_of_2(key.shape[2]),
        num_warps=4,
    )


@triton.jit
def _convert_xqa_pages_kernel(
    indptr: tl.tensor,
    indices: tl.tensor,
    last: tl.tensor,
    table: tl.tensor,
    lengths: tl.tensor,
    width: tl.constexpr,
    page_size: tl.constexpr,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    column = tl.arange(0, block)
    begin, end = tl.load(indptr + row), tl.load(indptr + row + 1)
    pages = tl.load(indices + begin + column, column < end - begin, other=0)
    tl.store(table + row * width + column, pages, column < width)
    # Existing graph buckets encode padding with no pages. XQA uses a safe
    # one-token read from initialized reserved page0; padded outputs are discarded.
    length = (end - begin - 1) * page_size + tl.load(last + row)
    tl.store(lengths + row, tl.where(end > begin, length, 1))


@register_kernel(
    name="cuda.triton.convert_xqa_pages",
    contract=CONVERT_XQA_PAGES_CONTRACT,
    platform=PlatformEnum.CUDA,
    execution_modes=("eager", "cudagraph"),
    input_domain="SM90 contiguous int32 CSR inputs and page table, contiguous uint32 lengths, page size 16/32/64/128.",
    solution="triton",
    binding="python",
)
def _convert_xqa_pages(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    page_size: int,
) -> None:
    rows, width = table.shape
    _convert_xqa_pages_kernel[(rows,)](
        indptr, indices, last, table, lengths, width, page_size, triton.next_power_of_2(width)
    )
