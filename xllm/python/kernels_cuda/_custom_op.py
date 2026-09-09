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

"""FakeTensor implementations for the CUDA ``xllm_ops`` operators.

The schemas live in ``xllm/core/kernels/cuda/cuda_ops_library.cpp``. Every
operator that a compiled graph may contain needs its shape and dtype contract
declared here, otherwise tracing fails when it reaches the call.

Importing this module registers all of them; the package ``__init__`` does so
before exposing any kernel.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _is_registered(qualname: str) -> bool:
    namespace, op_name = qualname.split("::", 1)
    library = getattr(torch.ops, namespace, None)
    return library is not None and hasattr(library, op_name)


def register_fake(qualname: str, fake_impl: Callable) -> None:
    """Register the FakeTensor implementation of a C++ operator.

    Raises when the operator is missing, so that a schema present in
    ``TORCH_LIBRARY`` but absent from the loaded library fails at import time
    rather than during graph capture.
    """
    if not _is_registered(qualname):
        raise RuntimeError(
            f"operator '{qualname}' is not registered; "
            "xllm/core/kernels/cuda/cuda_ops_library.cpp must define it before "
            "its fake implementation can be attached"
        )
    torch.library.register_fake(qualname)(fake_impl)


def register_fake_if_available(qualname: str, fake_impl: Callable) -> None:
    """Register a FakeTensor implementation only when the operator exists.

    Used for operators whose C++ registration is conditional on a build flag.
    """
    if _is_registered(qualname):
        torch.library.register_fake(qualname)(fake_impl)


def _rms_norm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, eps
    return torch.empty_like(input)


def _fused_add_rms_norm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, eps
    return input, residual


def _silu_and_mul_fake(input: torch.Tensor) -> torch.Tensor:
    shape = list(input.shape)
    shape[-1] //= 2
    return input.new_empty(shape)


def _fused_qk_norm_rope_fake(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    interleaved: bool,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    del (
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        interleaved,
        position_ids,
    )
    return qkv


def _reshape_paged_cache_fake(
    slot_mapping: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    del slot_mapping, keys, values, value_cache
    return key_cache


def _update_decode_graph_metadata_fake(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    dst_tokens: torch.Tensor,
    dst_positions: torch.Tensor,
    dst_slot_mapping: torch.Tensor,
    dst_kv_seq_lens: torch.Tensor,
    dst_kv_seq_lens_delta: torch.Tensor,
    dst_paged_kv_indptr: torch.Tensor,
    dst_paged_kv_indices: torch.Tensor,
    dst_paged_kv_last_page_len: torch.Tensor,
    padded_num_tokens: int,
) -> torch.Tensor:
    del (
        tokens,
        positions,
        slot_mapping,
        kv_seq_lens,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        dst_positions,
        dst_slot_mapping,
        dst_kv_seq_lens,
        dst_kv_seq_lens_delta,
        dst_paged_kv_indptr,
        dst_paged_kv_indices,
        dst_paged_kv_last_page_len,
        padded_num_tokens,
    )
    return dst_tokens


def _moe_fused_topk_fake(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    del renormalize, scoring_func
    shape = (gating_output.shape[0], topk)
    return (
        gating_output.new_empty(shape, dtype=torch.float32),
        gating_output.new_empty(shape, dtype=torch.int32),
    )


def _cutlass_fused_moe_fake(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    ep_size: int,
    ep_rank: int,
) -> torch.Tensor:
    del (
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
    )
    return input.new_empty((input.shape[0], fc2_expert_weights.shape[1]))


register_fake("xllm_ops::rms_norm", _rms_norm_fake)
register_fake("xllm_ops::fused_add_rms_norm", _fused_add_rms_norm_fake)
register_fake("xllm_ops::silu_and_mul", _silu_and_mul_fake)
register_fake("xllm_ops::fused_qk_norm_rope", _fused_qk_norm_rope_fake)
register_fake("xllm_ops::reshape_paged_cache", _reshape_paged_cache_fake)
register_fake("xllm_ops::update_decode_graph_metadata", _update_decode_graph_metadata_fake)

# Registered by a separate translation unit that is compiled only for the
# architectures with native expert GEMMs.
register_fake_if_available("xllm_ops::moe_fused_topk", _moe_fused_topk_fake)
register_fake_if_available("xllm_ops::cutlass_fused_moe", _cutlass_fused_moe_fake)
