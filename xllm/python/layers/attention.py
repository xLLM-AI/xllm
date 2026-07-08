"""Paged attention layer — uses flashinfer Python API directly.

Wraps flashinfer's BatchDecodeWithPagedKVCacheWrapper and
BatchPrefillWithRaggedKVCacheWrapper / BatchPrefillWithPagedKVCacheWrapper.
Plan is called once per batch; run is called once per layer.

KV cache layout: [num_blocks, page_size=1, num_kv_heads, head_dim] (NHD).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import flashinfer

from .. import ops
from ..attn_metadata import AttentionMetadata
from ..forward_context import get_forward_context

_WORKSPACE_SIZE = 128 * 1024 * 1024  # 128 MB


class PagedAttention(nn.Module):
    """Flashinfer-backed paged attention. Plan once per batch, run per layer."""

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
        self.dtype = dtype

        workspace = torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=device)

        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD"
        )
        self._prefill_ragged_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=device), "NHD"
        )
        self._prefill_paged_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=device), "NHD"
        )

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

        ops.reshape_paged_cache(
            attn_metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        if attn_metadata.is_prefill:
            output = self._prefill_ragged_wrapper.run(q_3d, k_3d, v_3d)
        elif attn_metadata.is_chunked_prefill:
            output = self._prefill_paged_wrapper.run(q_3d, (k_cache, v_cache))
        else:
            output = self._decode_wrapper.run(q_3d, (k_cache, v_cache))

        return output.view(-1, self.num_heads * self.head_dim)

    def plan(
        self,
        attn_metadata: AttentionMetadata,
        k_cache: torch.Tensor,
        enable_cuda_graph: bool = False,
    ) -> None:
        """Compute the attention plan for this batch."""
        page_size = k_cache.size(1) if k_cache.dim() >= 2 else 1
        window_left = self.sliding_window if self.sliding_window > 0 else -1

        if attn_metadata.is_prefill:
            self._prefill_ragged_wrapper.plan(
                attn_metadata.q_cu_seq_lens,
                attn_metadata.kv_cu_seq_lens,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                causal=True,
                sm_scale=self.scale,
                window_left=window_left,
                q_data_type=self.dtype,
            )
        elif attn_metadata.is_chunked_prefill:
            self._prefill_paged_wrapper.plan(
                attn_metadata.qo_indptr,
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                attn_metadata.paged_kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                page_size,
                causal=True,
                sm_scale=self.scale,
                window_left=window_left,
                q_data_type=self.dtype,
            )
        else:
            self._decode_wrapper.plan(
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                attn_metadata.paged_kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                page_size,
                sm_scale=self.scale,
                window_left=window_left,
                q_data_type=self.dtype,
            )
