"""Paged attention layer — Python owns the full attention lifecycle.

Equivalent to SGLang's BatchDecodeWithPagedKVCacheWrapper: this layer
manages plan computation (via update_*_plan ops) and kernel dispatch
(via batch_prefill/decode/chunked_prefill ops).

Workspace buffers live in the C++ FlashinferWorkspace singleton (thread-local,
128MB float + 8MB int + 8MB page-locked). The ops access it internally —
Python never allocates or passes workspace.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..attn_metadata import AttentionMetadata
from ..forward_context import get_forward_context


class PagedAttention(nn.Module):
    """Python owns attention: plan, dispatch. Workspace is C++ singleton."""

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
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.dtype_str = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }[dtype]

        # Plan state: fixed-address buffer for cudagraph compatibility.
        self._plan_info_buf: torch.Tensor = torch.zeros(16, dtype=torch.int64)
        self._plan_info_len: int = 0
        self.uri: str = ""

    @property
    def plan_info(self) -> torch.Tensor:
        return self._plan_info_buf[: self._plan_info_len]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        k_cache, v_cache = ctx.kv_caches[layer_id]

        q_3d = q.view(-1, self.num_heads, self.head_dim)
        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)
        output = torch.empty_like(q_3d)

        torch.ops.xllm_ops.reshape_paged_cache(
            attn_metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        if attn_metadata.is_prefill:
            torch.ops.xllm_ops.batch_prefill(
                q_3d, k_3d, v_3d,
                attn_metadata.q_cu_seq_lens,
                attn_metadata.kv_cu_seq_lens,
                self.plan_info, self.uri,
                self.scale, self.sliding_window,
                output,
            )
        elif attn_metadata.is_chunked_prefill:
            torch.ops.xllm_ops.batch_chunked_prefill(
                q_3d, k_cache, v_cache,
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                attn_metadata.paged_kv_last_page_len,
                self.plan_info, self.uri,
                self.scale, self.sliding_window,
                attn_metadata.qo_indptr,
                output,
            )
        else:
            torch.ops.xllm_ops.batch_decode(
                q_3d, k_cache, v_cache,
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                attn_metadata.paged_kv_last_page_len,
                self.plan_info, self.uri,
                self.scale, self.sliding_window,
                attn_metadata.use_tensor_core,
                attn_metadata.qo_indptr,
                output,
            )

        return output.view(-1, self.num_heads * self.head_dim)

    def plan(
        self,
        attn_metadata: AttentionMetadata,
        k_cache: torch.Tensor,
        enable_cuda_graph: bool = False,
    ) -> None:
        """Compute the attention plan for this batch."""
        block_size = k_cache.size(1) if k_cache.dim() >= 2 else 1
        if attn_metadata.is_prefill:
            new_info, self.uri = torch.ops.xllm_ops.update_prefill_plan(
                attn_metadata.q_cu_seq_lens,
                attn_metadata.kv_cu_seq_lens,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.dtype_str,
                self.dtype_str,
                self.dtype_str,
            )
        elif attn_metadata.is_chunked_prefill:
            new_info, self.uri = (
                torch.ops.xllm_ops.update_chunked_prefill_plan(
                    attn_metadata.paged_kv_indptr,
                    attn_metadata.paged_kv_last_page_len,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    block_size,
                    self.sliding_window,
                    self.dtype_str,
                    self.dtype_str,
                    self.dtype_str,
                )
            )
        else:
            new_info, self.uri = torch.ops.xllm_ops.update_decode_plan(
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                block_size,
                self.sliding_window,
                attn_metadata.use_tensor_core,
                enable_cuda_graph,
                self.dtype_str,
                self.dtype_str,
                self.dtype_str,
            )
        n = new_info.numel()
        self._plan_info_buf[:n].copy_(new_info)
        self._plan_info_len = n
