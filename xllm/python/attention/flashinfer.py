# Copyright 2026 The xLLM Authors.
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

"""FlashInfer attention backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import flashinfer
import torch
from flashinfer.decode import fast_decode_plan

from xllm.python import ops
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    KVCache,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention

_WORKSPACE_SIZE = 128 * 1024 * 1024


class FlashInferBackend(AttentionBackend):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.dtype = dtype

        self._decode_workspace = torch.empty(
            _WORKSPACE_SIZE, dtype=torch.uint8, device=device
        )
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._decode_workspace, "NHD"
        )
        self._prefill_ragged_wrapper = (
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=device),
                "NHD",
            )
        )
        self._prefill_paged_wrapper = (
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=device),
                "NHD",
            )
        )

        self._kv_caches: list[KVCache] = []
        self._metadata: AttentionMetadata | None = None
        self._active_decode_wrapper = self._decode_wrapper
        self._graph_decode_wrappers: dict[
            int, flashinfer.BatchDecodeWithPagedKVCacheWrapper
        ] = {}
        self._graph_decode_buffer_ptrs: dict[int, tuple[int, int, int]] = {}
        self._planned_graph_batches: set[int] = set()

    def bind_kv_caches(self, kv_caches: list[KVCache]) -> None:
        if not kv_caches:
            raise ValueError("FlashInferBackend requires at least one KV cache")
        self._kv_caches = kv_caches

    @property
    def num_kv_blocks(self) -> int:
        if not self._kv_caches:
            raise RuntimeError("KV caches are not bound")
        return self._kv_caches[0][0].size(0)

    @property
    def page_size(self) -> int:
        if not self._kv_caches:
            raise RuntimeError("KV caches are not bound")
        k_cache = self._kv_caches[0][0]
        return k_cache.size(1) if k_cache.dim() >= 2 else 1

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        if not self._kv_caches:
            raise RuntimeError("KV caches are not bound")

        self._metadata = metadata
        # FlashInfer's window_left is the count of preceding tokens attended to,
        # so a sliding window of size W maps to window_left = W - 1 (matching the
        # native attention convention). -1 disables the sliding window.
        window_left = self.sliding_window - 1 if self.sliding_window > 0 else -1

        if metadata.is_prefill:
            self._prefill_ragged_wrapper.plan(
                metadata.q_cu_seq_lens,
                metadata.kv_cu_seq_lens,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                causal=True,
                sm_scale=self.scale,
                window_left=window_left,
                q_data_type=self.dtype,
            )
        elif metadata.is_chunked_prefill:
            self._prefill_paged_wrapper.plan(
                metadata.qo_indptr,
                metadata.paged_kv_indptr,
                metadata.paged_kv_indices,
                metadata.paged_kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                sm_scale=self.scale,
                window_left=window_left,
                q_data_type=self.dtype,
            )
        else:
            self._prepare_decode(metadata, graph_mode, window_left)

    def _prepare_decode(
        self,
        metadata: AttentionMetadata,
        graph_mode: bool,
        window_left: int,
    ) -> None:
        if not graph_mode:
            self._active_decode_wrapper = self._decode_wrapper
            self._decode_wrapper.plan(
                metadata.paged_kv_indptr,
                metadata.paged_kv_indices,
                metadata.paged_kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                sm_scale=self.scale,
                window_left=window_left,
                q_data_type=self.dtype,
            )
            return

        indptr_host = metadata.paged_kv_indptr_host
        last_page_len_host = metadata.paged_kv_last_page_len_host
        if indptr_host is None or last_page_len_host is None:
            raise RuntimeError("decode CUDA graph requires host planner metadata")

        batch_size = metadata.paged_kv_last_page_len.numel()
        wrapper = self._get_graph_decode_wrapper(metadata, batch_size)
        self._active_decode_wrapper = wrapper
        if batch_size not in self._planned_graph_batches:
            wrapper.plan(
                indptr_host,
                metadata.paged_kv_indices,
                last_page_len_host,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                sm_scale=self.scale,
                window_left=window_left,
                q_data_type=self.dtype,
            )
            self._planned_graph_batches.add(batch_size)
            return

        fast_decode_plan(
            wrapper,
            indptr_host,
            metadata.paged_kv_indices,
            last_page_len_host,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            sm_scale=self.scale,
            window_left=window_left,
            q_data_type=self.dtype,
            global_override_indptr_cpu=indptr_host,
        )

    def _get_graph_decode_wrapper(
        self, metadata: AttentionMetadata, batch_size: int
    ):
        wrapper = self._graph_decode_wrappers.get(batch_size)
        if wrapper is None:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self._decode_workspace,
                "NHD",
                use_cuda_graph=True,
                paged_kv_indptr_buffer=metadata.paged_kv_indptr,
                paged_kv_indices_buffer=metadata.paged_kv_indices,
                paged_kv_last_page_len_buffer=metadata.paged_kv_last_page_len,
            )
            self._graph_decode_wrappers[batch_size] = wrapper
            self._graph_decode_buffer_ptrs[batch_size] = self._buffer_ptrs(
                metadata
            )
            return wrapper

        if self._graph_decode_buffer_ptrs[batch_size] != self._buffer_ptrs(
            metadata
        ):
            raise RuntimeError("decode CUDA graph metadata address changed")
        return wrapper

    @staticmethod
    def _buffer_ptrs(metadata: AttentionMetadata) -> tuple[int, int, int]:
        return (
            metadata.paged_kv_indptr.data_ptr(),
            metadata.paged_kv_indices.data_ptr(),
            metadata.paged_kv_last_page_len.data_ptr(),
        )

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        metadata = self._metadata
        if metadata is None:
            raise RuntimeError("FlashInferBackend.prepare() was not called")

        k_cache, v_cache = self._kv_caches[layer.layer_id]
        q_3d = q.view(-1, layer.num_heads, layer.head_dim)
        k_3d = k.view(-1, layer.num_kv_heads, layer.head_dim)
        v_3d = v.view(-1, layer.num_kv_heads, layer.head_dim)

        ops.reshape_paged_cache(
            metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        if metadata.is_prefill:
            output = self._prefill_ragged_wrapper.run(q_3d, k_3d, v_3d)
        elif metadata.is_chunked_prefill:
            output = self._prefill_paged_wrapper.run(q_3d, (k_cache, v_cache))
        else:
            output = self._active_decode_wrapper.run(q_3d, (k_cache, v_cache))

        return output.view(-1, layer.num_heads * layer.head_dim)
