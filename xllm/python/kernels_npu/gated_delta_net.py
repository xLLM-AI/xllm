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

"""NPU gated-delta-network kernels.

None has an NPU kernel yet. The signatures are the contract an NPU
implementation has to meet; see ``kernels_cuda/gated_delta_net.py`` and the
Triton launchers it calls for the reference behaviour.
"""

from __future__ import annotations

from typing import Literal

import torch

GdnPrefillBackend = Literal["flashinfer", "triton"]


def resolve_gdn_prefill_backend(
    capability: tuple[int, int] | None = None,
) -> GdnPrefillBackend:
    """Select the prefill backend of the active device.

    Args:
        capability: Device capability to resolve for; ``None`` reads it from
            the current device.

    Returns:
        The name to pass as ``backend`` to :func:`chunk_gated_delta_rule`.
    """
    del capability
    raise NotImplementedError(
        "resolve_gdn_prefill_backend has no NPU implementation; gated delta "
        "networks are not supported on NPU yet"
    )


def fused_gdn_prefill_post_conv(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_key_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Split the post-convolution projection and build the recurrence gates.

    Args:
        mixed_qkv: Packed projection of shape ``[num_tokens, qkv_size]``.
        a: Gate projection of shape ``[num_tokens, num_value_heads]``.
        b: Beta projection of shape ``[num_tokens, num_value_heads]``.
        a_log: Per-head log decay of shape ``[num_value_heads]``.
        dt_bias: Per-head timestep bias of shape ``[num_value_heads]``.
        num_key_heads: Key heads on this rank.
        key_head_dim: Size of one key head.
        value_head_dim: Size of one value head.

    Returns:
        Query, key, value, the decay gate and beta.
    """
    del mixed_qkv, a, b, a_log, dt_bias
    del num_key_heads, key_head_dim, value_head_dim
    raise NotImplementedError(
        "fused_gdn_prefill_post_conv has no NPU kernel; see "
        "kernels_cuda/triton/gdn_prefill.py for the reference implementation"
    )


def fused_recurrent_gated_delta_rule_packed_decode(
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
        state_indices: State slot of every sequence.
        scale: Query scale.

    Returns:
        Output of shape ``[batch_size, 1, num_value_heads, value_head_dim]``.
    """
    del mixed_qkv, a, b, a_log, dt_bias, initial_state, state_indices, scale
    raise NotImplementedError(
        "fused_recurrent_gated_delta_rule_packed_decode has no NPU kernel; see "
        "kernels_cuda/triton/gated_delta_net.py for the reference implementation"
    )


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the chunked delta rule over a variable-length batch.

    Args:
        q: Query of shape ``[num_tokens, num_key_heads, key_head_dim]``.
        k: Key with the shape of ``q``.
        v: Value of shape ``[num_tokens, num_value_heads, value_head_dim]``.
        g: Decay gate of shape ``[num_tokens, num_value_heads]``.
        beta: Beta with the shape of ``g``.
        initial_state: Recurrent state each sequence starts from.
        cu_seqlens: Cumulative sequence lengths.
        backend: Name returned by :func:`resolve_gdn_prefill_backend`.

    Returns:
        The output with the shape of ``v`` and the final recurrent state.
    """
    del q, k, v, g, beta, initial_state, cu_seqlens, backend
    raise NotImplementedError(
        "chunk_gated_delta_rule has no NPU kernel; see kernels_cuda/triton/fla/ "
        "for the reference implementation"
    )


__all__ = [
    "GdnPrefillBackend",
    "resolve_gdn_prefill_backend",
    "fused_gdn_prefill_post_conv",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "chunk_gated_delta_rule",
]
