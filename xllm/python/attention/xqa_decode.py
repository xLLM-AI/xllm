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

"""Explicit SM90 XQA decode over resident pages and stable graph metadata."""

from __future__ import annotations

import torch

from xllm.python.attention.backend import AttentionMetadata

_XQA_PAGE_SIZES = (16, 32, 64, 128)


def _validate_xqa_lengths(indptr: list[int], last: list[int], page_size: int, maximum: int) -> None:
    """Reject unsupported lengths before changing the device plan."""
    if len(indptr) != len(last) + 1 or not last or indptr[0] != 0:
        raise ValueError("XQA host paging shape/origin differs")
    for begin, end, tail in zip(indptr, indptr[1:], last):
        if any(type(value) is not int for value in (begin, end, tail)):
            raise ValueError("XQA host page metadata must be integral")
        if end < begin or not 1 <= tail <= page_size or (end == begin and tail != 1):
            raise ValueError("XQA requires monotone pages and a valid partial tail/padding row")
        if (end - begin - 1) * page_size + tail > maximum:
            raise ValueError("XQA live context exceeds the declared model context")


def _convert_pages(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    page_size: int,
) -> None:
    import triton
    from xllm_kernel.ops.kv_cache.native_kv_writer import convert_xqa_pages_kernel

    rows, width = table.shape
    convert_xqa_pages_kernel[(rows,)](
        indptr, indices, last, table, lengths, width, page_size, triton.next_power_of_2(width)
    )


class XQADecodePlan:
    """One prepared bucket; no BF16 KV copy, host sync, or planner inside replay."""

    def __init__(self, batch_size: int, page_size: int, maximum: int, device: torch.device) -> None:
        if batch_size <= 0:
            raise ValueError("XQA requires a positive batch size")
        if page_size not in _XQA_PAGE_SIZES or maximum < page_size or maximum % page_size:
            raise ValueError("XQA requires a supported page size and a page-aligned maximum context")
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("This explicit production XQA profile is validated only for SM90")
        self._page_size, self._maximum = page_size, maximum
        self._batch_size = batch_size
        self._table = torch.empty((batch_size, maximum // page_size), dtype=torch.int32, device=device)
        self._lengths = torch.empty(batch_size, dtype=torch.uint32, device=device)
        self._metadata_addresses: tuple[int, int, int] | None = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool) -> None:
        indptr, last = metadata.paged_kv_indptr_host, metadata.paged_kv_last_page_len_host
        if (
            indptr is None
            or last is None
            or indptr.device.type != "cpu"
            or last.device.type != "cpu"
            or indptr.ndim != 1
            or last.ndim != 1
        ):
            raise ValueError("XQA requires one-dimensional CPU planner metadata without a device sync")
        indptr_values = indptr.tolist()
        _validate_xqa_lengths(indptr_values, last.tolist(), self._page_size, self._maximum)
        if len(last) != self._batch_size:
            raise ValueError("XQA planned batch differs from its stable bucket")
        inputs = (metadata.paged_kv_indptr, metadata.paged_kv_indices, metadata.paged_kv_last_page_len)
        if inputs[0].numel() != len(indptr) or inputs[2].numel() != len(last) or indptr_values[-1] > inputs[1].numel():
            raise ValueError("XQA paging buffers do not cover the host plan")
        if any(
            tensor.device != self._table.device
            or tensor.dtype != torch.int32
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            for tensor in inputs
        ):
            raise ValueError("XQA paged input must be a contiguous 1D int32 vector on the planned CUDA device")
        addresses = tuple(tensor.data_ptr() for tensor in inputs)
        if graph_mode and self._metadata_addresses is not None and addresses != self._metadata_addresses:
            raise RuntimeError("XQA graph paging input addresses changed")
        self._metadata_addresses = addresses
        _convert_pages(*inputs, self._table, self._lengths, self._page_size)

    def run(
        self,
        query: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
        workspace: torch.Tensor,
        softmax_scale: float,
        key_scale: float,
        value_scale: float,
    ) -> torch.Tensor:
        from flashinfer.decode import xqa_batch_decode_with_kv_cache

        if self._metadata_addresses is None:
            raise RuntimeError("XQA decode must be prepared before execution")
        # Immutable cache geometry/device/dtype are validated once by the
        # native backend's constructor/bind boundary, rather than every layer.
        # The SM90 wrapper forwards Q as a raw packed pointer without Q strides.
        # Fused QKV projections can leave token-stride gaps; materialize inside
        # capture rather than allowing reads into the adjacent K/V projections.
        query = query.contiguous()
        return xqa_batch_decode_with_kv_cache(
            query,
            cache,
            workspace,
            self._table,
            self._lengths,
            self._maximum,
            bmm1_scale=softmax_scale * key_scale,
            bmm2_scale=value_scale,
            kv_layout="NHD",
            enable_pdl=True,
        )
