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

"""CUDA gated-delta-network kernels."""

from __future__ import annotations

from typing import Literal

import torch

GdnPrefillBackend = Literal["flashinfer", "triton"]
_FLASHINFER_ARCHITECTURES = frozenset((9, 10, 12))


def resolve_gdn_prefill_backend(
    capability: tuple[int, int] | None = None,
) -> GdnPrefillBackend:
    """Select a GDN prefill backend for a CUDA architecture."""
    if capability is None:
        capability = torch.cuda.get_device_capability()
    return "flashinfer" if capability[0] in _FLASHINFER_ARCHITECTURES else "triton"


@torch.library.custom_op("xllm_triton::fused_gdn_prefill_post_conv", mutates_args=())
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
    """Prepare normalized Q/K and GDN gates as one graph node."""
    from .triton.gdn_prefill import (
        fused_gdn_prefill_post_conv as triton_gdn_post_conv,
    )

    return triton_gdn_post_conv(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        num_key_heads,
        key_head_dim,
        value_head_dim,
    )


@fused_gdn_prefill_post_conv.register_fake
def _fused_gdn_prefill_post_conv_fake(
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
    del b, dt_bias
    num_tokens = mixed_qkv.shape[0]
    num_value_heads = a_log.numel()
    q = mixed_qkv.new_empty(num_tokens, num_key_heads, key_head_dim)
    k = torch.empty_like(q)
    v = mixed_qkv.new_empty(num_tokens, num_value_heads, value_head_dim)
    g = a.new_empty(num_tokens, num_value_heads, dtype=torch.float32)
    beta = torch.empty_like(g)
    return q, k, v, g, beta


@torch.library.custom_op(
    "xllm_triton::fused_recurrent_gdn_packed_decode",
    mutates_args=("initial_state",),
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
    """Run a packed recurrent GDN update as one graph node."""
    from .triton.gated_delta_net import (
        fused_recurrent_gated_delta_rule_packed_decode as triton_gdn_decode,
    )

    return triton_gdn_decode(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        initial_state,
        state_indices,
        scale,
    )


@fused_recurrent_gated_delta_rule_packed_decode.register_fake
def _fused_recurrent_gated_delta_rule_packed_decode_fake(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    del a, b, a_log, dt_bias, state_indices, scale
    return torch.empty(
        mixed_qkv.shape[0],
        1,
        initial_state.shape[-3],
        initial_state.shape[-2],
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )


@torch.library.custom_op("xllm_kernels::chunk_gated_delta_rule", mutates_args=())
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
    """Run a backend-selected chunked gated delta rule as one graph node."""
    if backend == "flashinfer":
        from .flashinfer.gated_delta_net import (
            chunk_gated_delta_rule as flashinfer_gdn,
        )

        return flashinfer_gdn(q, k, v, g, beta, initial_state, cu_seqlens)
    if backend == "triton":
        from .triton.fla import (
            chunk_gated_delta_rule as triton_gdn,
        )

        output, final_state = triton_gdn(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=initial_state,
            cu_seqlens=cu_seqlens.to(torch.int32),
        )
        return output.squeeze(0), final_state
    raise ValueError(f"unsupported GDN prefill backend: {backend}")


@chunk_gated_delta_rule.register_fake
def _chunk_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q, k, g, beta, cu_seqlens, backend
    return torch.empty_like(v), torch.empty_like(initial_state)


__all__ = [
    "GdnPrefillBackend",
    "resolve_gdn_prefill_backend",
    "fused_gdn_prefill_post_conv",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "chunk_gated_delta_rule",
]
