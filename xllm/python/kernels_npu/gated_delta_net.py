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

"""NPU gated-delta-network kernel entry points.

Prefill uses the production MegaGdn operator. The remaining helpers preserve
the existing Python composition for paths not migrated to MegaGdn yet.
"""

from __future__ import annotations

import torch


def mega_gdn_prefill(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    z: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state_read_indices: torch.Tensor,
    conv_state_write_indices: torch.Tensor,
    ssm_state_read_indices: torch.Tensor,
    ssm_state_write_indices: torch.Tensor,
    ssm_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    norm_weight: torch.Tensor,
    num_matrices: int,
) -> torch.Tensor:
    """Run the fused Qwen3.5 prefill GDN and update both state caches."""
    return torch.ops.xllm_ops.mega_gdn_prefill(
        mixed_qkv,
        b,
        a,
        z,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        conv_state_read_indices,
        conv_state_write_indices,
        ssm_state_read_indices,
        ssm_state_write_indices,
        ssm_cache,
        cu_seqlens,
        norm_weight,
        num_matrices,
    )


def mega_gdn_decode(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    fla_ssm_state_layout: bool = True,
) -> torch.Tensor:
    """Run fused single-token Qwen3.5 decode and update both state caches.

    State indices are either legacy common slots ``[batch]`` or independent
    ``[batch, 2]`` Conv slots and SSM rows. A negative pair disables a row.
    """
    return torch.ops.xllm_ops.mega_gdn_decode(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        read_state_indices,
        write_state_indices,
        norm_weight,
        fla_ssm_state_layout,
    )


def mega_gdn_mtp_decode(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
    fla_ssm_state_layout: bool = True,
) -> torch.Tensor:
    """Restore the accepted checkpoint and record each verify-token state.

    Paired ``[batch, 2]`` indices name the Conv slot and the first SSM
    checkpoint row independently; legacy ``[batch]`` slots retain their stride.
    """
    return torch.ops.xllm_ops.mega_gdn_mtp_decode(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        read_state_indices,
        write_state_indices,
        num_accepted_tokens,
        norm_weight,
        fla_ssm_state_layout,
    )


def mega_gdn_mtp_decode_varlen(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    fla_ssm_state_layout: bool = True,
) -> None:
    """Verify packed prefixes using common slots or independent role rows."""
    torch.ops.xllm_ops.mega_gdn_mtp_decode_varlen(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        state_indices,
        num_accepted_tokens,
        norm_weight,
        cu_seqlens,
        out,
        fla_ssm_state_layout,
    )


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
    "mega_gdn_decode",
    "mega_gdn_mtp_decode",
    "mega_gdn_mtp_decode_varlen",
    "mega_gdn_prefill",
    "fused_gdn_gating",
    "chunk_gated_delta_rule",
]
