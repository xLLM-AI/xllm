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

"""Resident FP8 KV pages on the official FlashInfer execution contract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import flashinfer
import torch
from xllm_kernel.ops.kv_cache import write_fp8_kv
from xllm_kernel.ops.quantization import quantize_query_fp8

from scripts.logger import logger
from xllm.python.attention.backend import AttentionMetadata, LayerCache
from xllm.python.attention.flashinfer import FlashInferBackend
from xllm.python.attention.xqa_decode import _XQA_PAGE_SIZES, XQADecodePlan

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


class FlashInferNativeKVBackend(FlashInferBackend):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_dtype: str,
        fixed_scales: tuple[float, float] = (1.0, 1.0),
        max_context_tokens: int = 32768,
        virtual_memory: bool = False,
    ) -> None:
        if cache_dtype != "fp8_e4m3":
            raise ValueError("Native FP8 KV requires fp8_e4m3")
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("This native compressed KV implementation targets SM90")
        if dtype not in (torch.bfloat16, torch.float16) or head_dim not in (128, 256):
            raise ValueError("Native SM90 KV requires BF16/FP16 activations and head_dim 128/256")
        if num_kv_heads <= 0 or num_heads % num_kv_heads or num_heads // num_kv_heads < 4:
            raise ValueError("Native KV currently requires a tensor-core decode GQA group of at least 4")
        if virtual_memory:
            raise ValueError("Native FP8 KV requires initialized resident pages; VMM is not qualified")
        if sliding_window > 0:
            raise ValueError(
                "This XQA integration requires non-VMM full attention with zero-initialized resident pages"
            )
        self._max_context_tokens = max_context_tokens
        self._graph_xqa_plans: dict[int, XQADecodePlan] = {}
        self._eager_xqa_plan: XQADecodePlan | None = None
        self._active_xqa_plan: XQADecodePlan | None = None
        self._fixed_scales = fixed_scales
        self._pages: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        super().__init__(num_heads, num_kv_heads, head_dim, scale, sliding_window, device, dtype)
        # FlashInfer requires a zeroed workspace on first use. In
        # particular, its first 8MiB hold atomic CTA completion counters.
        # Initialize once, before capture, without per-step clearing.
        self._decode_workspace.zero_()
        # Reuse the official workspace allocation; only the quantized planner
        # selection changes. FlashInfer's FP8 FA3 path consumes FP8 Q explicitly.
        workspace = self._prefill_paged_wrapper._float_workspace_buffer
        self._prefill_paged_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            backend="fa3",
        )

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        geometry: tuple[int, int] | None = None
        pages: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_id, cache in enumerate(kv_caches):
            if cache.key is None:
                if cache.value is not None:
                    raise ValueError("Full-attention K/V must be paired")
                continue
            expected_shape = (*cache.key.shape[:2], self.num_kv_heads, self.head_dim)
            for tensor in (cache.key, cache.value):
                if (
                    tensor is None
                    or tensor.dtype != torch.float8_e4m3fn
                    or tensor.device != self._decode_workspace.device
                    or tensor.ndim != 4
                    or tuple(tensor.shape) != expected_shape
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "Native KV requires contiguous resident FP8 E4M3 pages with the format's physical head dimension"
                    )
            layer_geometry = (cache.key.size(0), cache.key.size(1))
            if geometry is None:
                if min(layer_geometry) <= 0:
                    raise ValueError("Native FP8 KV requires nonempty resident pages")
                if layer_geometry[1] not in _XQA_PAGE_SIZES:
                    raise ValueError(f"XQA page size must be one of {_XQA_PAGE_SIZES}")
                geometry = layer_geometry
            elif layer_geometry != geometry:
                raise ValueError(
                    f"Native FP8 KV layers must use the same page geometry: {geometry} vs {layer_geometry}"
                )
            pages[layer_id] = (cache.key, cache.value)
        if not pages:
            raise ValueError("Native KV requires full-attention resident pages")
        self._kv_caches = kv_caches
        self._pages = pages
        logger.info("Native KV backend bound: %s", json.dumps(self.native_kv_runtime_info(), sort_keys=True))

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        self._metadata = metadata
        if metadata.is_prefill or metadata.is_chunked_prefill:
            query_indptr = metadata.qo_indptr if metadata.qo_indptr is not None else metadata.q_cu_seq_lens
            self._prefill_paged_wrapper.plan(
                query_indptr,
                metadata.paged_kv_indptr,
                metadata.paged_kv_indices,
                metadata.paged_kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                sm_scale=self.scale,
                window_left=-1,
                q_data_type=torch.float8_e4m3fn,
                kv_data_type=torch.float8_e4m3fn,
                o_data_type=self.dtype,
            )
            return

        batch_size = metadata.paged_kv_last_page_len.numel()
        plan = self._graph_xqa_plans.get(batch_size) if graph_mode else self._eager_xqa_plan
        if plan is None or plan.batch_size != batch_size:
            page_size = self.page_size
            maximum = ((self._max_context_tokens + page_size - 1) // page_size) * page_size
            plan = XQADecodePlan(batch_size, page_size, maximum, self._decode_workspace.device)
            if graph_mode:
                self._graph_xqa_plans[batch_size] = plan
            else:
                self._eager_xqa_plan = plan
        plan.prepare(metadata, graph_mode=graph_mode)
        self._active_xqa_plan = plan

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        metadata = self._metadata
        if metadata is None:
            raise RuntimeError("Native KV prepare() was not called")
        prefill = metadata.is_prefill or metadata.is_chunked_prefill
        decode_plan = None if prefill else self._active_xqa_plan
        if not prefill and decode_plan is None:
            raise RuntimeError("XQA decode plan must be prepared before writing KV")
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view(-1, self.num_kv_heads, self.head_dim)
        pages = self._pages[layer.layer_id]
        write_fp8_kv(key, value, metadata.slot_mapping, *pages, *self._fixed_scales)
        query = q.view(-1, self.num_heads, self.head_dim)
        if decode_plan is not None:
            output = decode_plan.run(query, pages, self._decode_workspace, self.scale, *self._fixed_scales)
        else:
            query = quantize_query_fp8(query, 1.0)
            output = self._prefill_paged_wrapper.run(
                query, pages, k_scale=self._fixed_scales[0], v_scale=self._fixed_scales[1], q_scale=1.0
            )
        return output.view(-1, self.num_heads * self.head_dim)

    def native_kv_runtime_info(self) -> dict[str, object]:
        resident_bytes = sum(key.nbytes + value.nbytes for key, value in self._pages.values())
        return {
            "kv_cache_dtype": "fp8_e4m3",
            "resident_layout": "NHD",
            "k_scale": self._fixed_scales[0],
            "v_scale": self._fixed_scales[1],
            "writer": "fused_triton_quantize_scatter",
            "prefill_backend": "fa3",
            "decode_backend": "xqa",
            "prefill_query_dtype": str(torch.float8_e4m3fn),
            "decode_query_dtype": str(self.dtype),
            "max_context_tokens": self._max_context_tokens,
            "xqa_workspace_initialized": True,
            "query_dtype": str(self.dtype),
            "mixed_FP8_decode": None,
            "full_BF16_KV_shadow": False,
            "KV_tensor_address_bytes": resident_bytes,
            "VMM_virtual_address_capacity_not_residency": False,
            "resident_KV_bytes": resident_bytes,
            "resident_global_scale_bytes": 0,
        }
