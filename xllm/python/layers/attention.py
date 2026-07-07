"""Paged attention layer — Python owns the full attention lifecycle.

Equivalent to SGLang's BatchDecodeWithPagedKVCacheWrapper: this layer
manages plan computation (via update_*_plan ops) and kernel dispatch
(via batch_prefill/decode/chunked_prefill ops).

Workspace buffers live in the C++ FlashinferWorkspace singleton (thread-local,
128MB float + 8MB int + 8MB page-locked). The ops access it internally —
Python never allocates or passes workspace.

The dispatch function ``_attention_dispatch`` is decorated with ``@eager_break``
from the breakable CUDA graph infrastructure. During piecewise capture, this
ends the current graph segment, runs attention eagerly (KV cache write + kernel
dispatch), and starts the next segment. On replay, the closure re-executes with
volatile metadata re-read from forward_context. Outside capture, it's a no-op
wrapper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..attn_metadata import AttentionMetadata
from ..forward_context import get_forward_context
from ..model_runner.breakable_cuda_graph import eager_break


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
        # plan_info is a CPU int64 tensor whose ADDRESS is stable across calls;
        # plan() updates its content in-place so captured CUDA graphs that
        # reference this tensor see fresh data on replay without address change.
        # Max size: prefill plan_info = 15 int64, decode = 10. Allocate 16.
        self._plan_info_buf: torch.Tensor = torch.zeros(16, dtype=torch.int64)
        self._plan_info_len: int = 0
        self.uri: str = ""

    @property
    def plan_info(self) -> torch.Tensor:
        return self._plan_info_buf[: self._plan_info_len]

    @torch._dynamo.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        ctx = get_forward_context()
        k_cache, v_cache = ctx.kv_caches[layer_id]

        q_3d = q.view(-1, self.num_heads, self.head_dim)
        k_3d = k.view(-1, self.num_kv_heads, self.head_dim)
        v_3d = v.view(-1, self.num_kv_heads, self.head_dim)

        # Pre-allocate output INSIDE the current cudagraph segment (fixed
        # address from the shared graph pool). The attention break function
        # writes into it in-place; the next segment reads it at the same addr.
        output = torch.empty_like(q_3d)

        # The actual attention dispatch is a graph-break point: during
        # piecewise capture this ends the current segment, runs eagerly
        # (KV write + kernel), then starts the next segment.
        # Only captures stable-address tensors and constants in the closure;
        # volatile attn_metadata is re-read from forward_context each call.
        _attention_dispatch(
            q_3d, k_3d, v_3d, output, layer_id,
            self.plan_info, self.uri,
            self.scale, self.sliding_window,
        )

        return output.view(-1, self.num_heads * self.head_dim)

    def plan(
        self,
        attn_metadata: AttentionMetadata,
        k_cache: torch.Tensor,
        enable_cuda_graph: bool = False,
    ) -> None:
        """Compute the attention plan for this batch. Called ONCE per batch,
        OUTSIDE the compiled/captured region (host-side scheduling, per-batch
        varying). Updates self._plan_info_buf IN-PLACE at a fixed address so
        cudagraph-captured kernels see fresh data on replay.

        enable_cuda_graph only affects the decode branch: when True, flashinfer
        emits a fixed-layout plan (constant grid/tiling), required so a single
        captured decode graph replays correctly as sequence length grows. The
        per-step seqlen tiling lands in the fixed-address int workspace buffer,
        refreshed by re-planning before each replay.
        """
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
        # In-place copy into fixed-address buffer.
        n = new_info.numel()
        self._plan_info_buf[:n].copy_(new_info)
        self._plan_info_len = n


@eager_break
def _attention_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
    plan_info: torch.Tensor,
    uri: str,
    scale: float,
    sliding_window: int,
) -> None:
    """Graph-break point: KV cache write + attention kernel dispatch.

    During piecewise capture, @eager_break ends the current cudagraph segment
    before this runs, then starts a new segment after. On replay, the closure
    re-runs this function — volatile attn_metadata is re-read from
    forward_context (refreshed before each replay), while q/k/v/output have
    stable addresses from the shared graph pool.
    """
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    k_cache, v_cache = ctx.kv_caches[layer_id]

    torch.ops.xllm_ops.reshape_paged_cache(
        attn_metadata.slot_mapping, k, v, k_cache, v_cache
    )

    if attn_metadata.is_prefill:
        torch.ops.xllm_ops.batch_prefill(
            q, k, v,
            attn_metadata.q_cu_seq_lens,
            attn_metadata.kv_cu_seq_lens,
            plan_info, uri, scale, sliding_window,
            output,
        )
    elif attn_metadata.is_chunked_prefill:
        torch.ops.xllm_ops.batch_chunked_prefill(
            q, k_cache, v_cache,
            attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices,
            attn_metadata.paged_kv_last_page_len,
            plan_info, uri, scale, sliding_window,
            attn_metadata.qo_indptr,
            output,
        )
    else:
        torch.ops.xllm_ops.batch_decode(
            q, k_cache, v_cache,
            attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices,
            attn_metadata.paged_kv_last_page_len,
            plan_info, uri, scale, sliding_window,
            attn_metadata.use_tensor_core,
            attn_metadata.qo_indptr,
            output,
        )
