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

"""NPU gated-delta-network kernels (PyTorch small-op implementation).

Implements the same semantics as the CUDA Triton references in
``kernels_cuda/triton/gdn_prefill.py`` and ``kernels_cuda/triton/gated_delta_net.py``
using only standard PyTorch operations. Performance is not optimized;
correctness and precision alignment are the goals.
"""

from __future__ import annotations

import torch


def fused_gdn_gating(
    a_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute decay gate g and beta from raw projections via TileLang kernel."""
    from .tilelang.fused_gdn_gating import fused_gdn_gating_kernel_jit

    num_batches, num_heads = a.shape
    kernel = fused_gdn_gating_kernel_jit(
        num_batches=num_batches,
        compile_max_batch=num_batches,
        num_heads=num_heads,
    )

    g_out = torch.empty(1, num_batches, num_heads, dtype=torch.float32, device=a.device)
    beta_out = torch.empty(1, num_batches, num_heads, dtype=a.dtype, device=a.device)
    kernel(
        a_log.to(torch.float32).contiguous(),
        a.contiguous(),
        b.contiguous(),
        dt_bias.to(torch.float32).contiguous(),
        g_out.squeeze(0),
        beta_out.squeeze(0),
        num_batches,
        1.0,  # softplus_beta
        20.0,  # softplus_threshold
    )
    return g_out, beta_out


def fused_sigmoid_gating_delta_rule_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Advance the recurrent state by one token per sequence.

    Args:
        mixed_qkv: Packed projection of shape ``[batch_size, qkv_size]``.
        a: Gate projection of shape ``[batch_size, num_value_heads]``.
        b: Beta projection of shape ``[batch_size, num_value_heads]``.
        a_log: Per-head log decay of shape ``[num_value_heads]``.
        dt_bias: Per-head timestep bias of shape ``[num_value_heads]``.
        initial_state: Recurrent state pool, updated in place.
            Shape ``[num_slots, num_value_heads, value_dim, key_dim]``.
        state_indices: State slot of every sequence.
        scale: Query scale.

    Returns:
        Output of shape ``[batch_size, 1, num_value_heads, value_head_dim]``.
    """
    batch = mixed_qkv.shape[0]
    num_value_heads, value_dim, key_dim = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    query_key_dim = qkv_dim - num_value_heads * value_dim
    query_dim = query_key_dim // 2
    num_key_heads = query_dim // key_dim

    # Split mixed_qkv
    q_flat = mixed_qkv[:, :query_dim]
    k_flat = mixed_qkv[:, query_dim : 2 * query_dim]
    v_flat = mixed_qkv[:, 2 * query_dim :]

    q = q_flat.view(batch, num_key_heads, key_dim)
    k = k_flat.view(batch, num_key_heads, key_dim)
    v = v_flat.view(batch, num_value_heads, value_dim)

    # Kernel expects [batch, seq_len, heads, dim] — add seq dim for decode
    q = q.unsqueeze(1)  # [batch, 1, num_key_heads, key_dim]
    k = k.unsqueeze(1)  # [batch, 1, num_key_heads, key_dim]
    v = v.unsqueeze(1)  # [batch, 1, num_value_heads, value_dim]

    # Kernel does l2norm, gating, GQA expansion, and recurrence internally.
    # cu_seqlens for decode: each seq has 1 token.
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=mixed_qkv.device)

    output = torch.ops.xllm_ops.fused_sigmoid_gating_delta_rule_decode(
        a_log,
        a.unsqueeze(1),
        dt_bias,
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        b.unsqueeze(1),
        initial_state,
        state_indices,
        cu_seqlens,
        scale,
    )
    return output.unsqueeze(1)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the chunked delta rule over a variable-length batch.

    Args:
        q: Query of shape ``[num_tokens, num_key_heads, key_head_dim]``.
        k: Key with the shape of ``q``.
        v: Value of shape ``[num_tokens, num_value_heads, value_head_dim]``.
        g: Decay gate of shape ``[num_tokens, num_value_heads]``.
        beta: Beta with the shape of ``g``.
        initial_state: Recurrent state each sequence starts from.
            Shape ``[batch, num_value_heads, value_dim, key_dim]``.
        cu_seqlens: Cumulative sequence lengths.
    Returns:
        The output with the shape of ``v`` and the final recurrent state.
    """
    # npu_mega_chunk_gdn expects [B, T, H, D] layout with B=1 for packed input
    # Cast g and beta to match C++ layer behavior (bf16 round-trip)
    g_input = g.to(v.dtype)
    beta_input = beta.to(v.dtype)
    output, final_state = torch.ops.xllm_ops.chunk_gated_delta_rule(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        g_input.unsqueeze(0),
        beta_input.unsqueeze(0),
        initial_state,
        cu_seqlens,
    )
    return output.squeeze(0), final_state


__all__ = [
    "fused_gdn_gating",
    "fused_sigmoid_gating_delta_rule_decode",
    "chunk_gated_delta_rule",
]
