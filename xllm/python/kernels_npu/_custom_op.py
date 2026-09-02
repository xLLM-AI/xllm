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

"""FakeTensor implementations for the NPU ``xllm_ops`` operators.

The schemas live in ``xllm/core/kernels/npu/npu_ops_library.cpp``. Every
operator that a compiled graph may contain needs its shape and dtype contract
declared here, otherwise tracing fails when it reaches the call.

Two schemas in that file have no fake here on purpose:

* ``fused_qk_norm_rope`` is declared but has no ``TORCH_LIBRARY_IMPL`` entry
  for ``PrivateUse1``; NPU runs the Triton kernel in ``rotary_embedding.py``.
* ``apply_rotary_embedding`` is reached from C++ only.

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
            "xllm/core/kernels/npu/npu_ops_library.cpp must define it before "
            "its fake implementation can be attached"
        )
    torch.library.register_fake(qualname)(fake_impl)


def _rms_norm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, eps
    return torch.empty_like(input)


def _rms_norm_gated_fake(
    input: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del gate, weight, eps
    return torch.empty_like(input)


def _l2_norm_fake(
    input: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del eps
    return torch.empty_like(input)


def _chunk_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q, k, g, beta, cu_seqlens
    num_seqs = initial_state.shape[0]
    return torch.empty_like(v), torch.empty_like(initial_state)


def _causal_conv1d_qkv_prefill_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_qk_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del weight, conv_state, state_indices, has_initial_state, query_start_loc
    num_tokens = x.shape[0]
    opts = x.options().dtype(torch.bfloat16)
    q = torch.empty(1, num_tokens, num_qk_heads, head_k_dim, **opts)
    k = torch.empty(1, num_tokens, num_qk_heads, head_k_dim, **opts)
    v = torch.empty(1, num_tokens, num_v_heads, head_v_dim, **opts)
    return q, k, v


def _fused_sigmoid_gating_delta_rule_decode_fake(
    a_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    del a_log, a, dt_bias, k, b, ssm_state, state_indices, cu_seqlens, scale
    # Output shape matches v: [num_tokens, num_value_heads, value_dim]
    return torch.empty_like(v)


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


def _reshape_paged_cache_fake(
    slot_mapping: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    del slot_mapping, keys, values, value_cache
    return key_cache


def _mla_preprocess_v2_fake(
    input: torch.Tensor,
    gamma0: torch.Tensor,
    beta0: torch.Tensor,
    quant_scale0: torch.Tensor,
    quant_offset0: torch.Tensor,
    wdqkv: torch.Tensor,
    descale0: torch.Tensor,
    bias0: torch.Tensor,
    gamma1: torch.Tensor,
    beta1: torch.Tensor,
    quant_scale1: torch.Tensor,
    quant_offset1: torch.Tensor,
    wuq: torch.Tensor,
    descale1: torch.Tensor,
    bias1: torch.Tensor,
    gamma2: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wuk: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    slot_mapping: torch.Tensor,
    ctkv_scale: torch.Tensor,
    q_nope_scale: torch.Tensor,
    wdq_dim: int,
    q_rope_dim: int,
    k_rope_dim: int,
    epsilon: float,
    q_rotary_coeff: int,
    k_rotary_coeff: int,
    transpose_wdq: bool,
    transpose_wuq: bool,
    transpose_wuk: bool,
    cache_mode: int,
    quant_mode: int,
    do_rms_norm: bool,
    wdkv_split_count: int,
    q_down_out_flag: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del (
        gamma0,
        beta0,
        quant_scale0,
        quant_offset0,
        wdqkv,
        descale0,
        bias0,
        beta1,
        quant_scale1,
        quant_offset1,
        wuq,
        descale1,
        bias1,
        gamma2,
        cos,
        sin,
        slot_mapping,
        ctkv_scale,
        q_nope_scale,
        wdq_dim,
        k_rope_dim,
        epsilon,
        q_rotary_coeff,
        k_rotary_coeff,
        transpose_wdq,
        transpose_wuq,
        transpose_wuk,
        quant_mode,
        do_rms_norm,
        wdkv_split_count,
        q_down_out_flag,
    )
    q_out_dim = 576 if cache_mode == 0 else 512
    q_out = kv_cache.new_empty((input.shape[0], wuk.shape[0], q_out_dim))
    q_rope_out = input.new_empty((input.shape[0], wuk.shape[0], q_rope_dim))
    q_down_out = input.new_empty((input.shape[0], gamma1.numel()))
    return q_out, kv_cache, q_rope_out, kv_cache_rope, q_down_out


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


def _quant_matmul_fake(
    x1: torch.Tensor,
    x2: torch.Tensor,
    transpose2: bool,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
    pertoken_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype | None,
) -> torch.Tensor:
    del scale, offset, pertoken_scale, bias
    out_last = x2.size(0) if transpose2 else x2.size(1)
    out_shape = list(x1.shape[:-1]) + [out_last]
    dtype = output_dtype if output_dtype is not None else torch.int8
    return x1.new_empty(out_shape, dtype=dtype)


def _quantize_per_tensor_fake(
    self: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
) -> torch.Tensor:
    del scales, zero_points, axis
    storage_dtype = {
        torch.qint8: torch.int8,
        torch.quint8: torch.uint8,
        torch.qint32: torch.int32,
    }.get(dtype, dtype)
    return self.new_empty(self.shape, dtype=storage_dtype)


def _dynamic_quant_fake(
    input: torch.Tensor,
    smooth_scales: torch.Tensor | None,
    group_index: torch.Tensor | None,
    dst_type: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del smooth_scales, group_index
    if dst_type == torch.quint4x2:
        if input.shape[-1] % 8:
            raise ValueError("dynamic_quant int4 input's last dimension must be divisible by 8")
        output_shape = (*input.shape[:-1], input.shape[-1] // 8)
        output_dtype = torch.int32
    else:
        output_shape = input.shape
        output_dtype = torch.int8
    output = input.new_empty(output_shape, dtype=output_dtype)
    scale = input.new_empty(input.shape[:-1], dtype=torch.float32)
    return output, scale


def _inplace_partial_rotary_mul_fake(
    input: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    rotary_mode: str,
    partial_slice: list[int],
) -> None:
    del input, cosine, sine, rotary_mode, partial_slice


def _quant_lightning_indexer_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    query_quant_mode: int,
    key_quant_mode: int,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_key: torch.Tensor | None,
    block_table: torch.Tensor | None,
    metadata: torch.Tensor | None,
    layout_query: str,
    layout_key: str,
    sparse_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    cmp_ratio: int,
    return_value: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        weights,
        query_dequant_scale,
        key_dequant_scale,
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        metadata,
        sparse_mode,
        pre_tokens,
        next_tokens,
        cmp_ratio,
    )
    key_head_num = key.size(1) if layout_key == "TND" else key.size(2)
    if layout_query == "BSND":
        output_shape = (
            query.size(0),
            query.size(1),
            key_head_num,
            sparse_count,
        )
    else:
        output_shape = (query.size(0), key_head_num, sparse_count)
    indices = query.new_zeros(output_shape, dtype=torch.int32)
    values_shape = output_shape if return_value else (0,)
    values = query.new_zeros(values_shape, dtype=torch.float32)
    return indices, values


def _quant_lightning_indexer_metadata_fake(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    query_quant_mode: int,
    key_quant_mode: int,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_key: torch.Tensor | None,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    layout_query: str,
    layout_key: str,
    sparse_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    cmp_ratio: int,
    device: str,
) -> torch.Tensor:
    del (
        num_heads_q,
        num_heads_k,
        head_dim,
        query_quant_mode,
        key_quant_mode,
        actual_seq_lengths_key,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        cmp_ratio,
        device,
    )
    assert actual_seq_lengths_query is not None
    return actual_seq_lengths_query.new_empty((1024,), dtype=torch.int32)


def _lightning_indexer_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_seq_lengths: torch.Tensor | None,
    key_seq_lengths: torch.Tensor | None,
    block_table: torch.Tensor | None,
    layout_query: str,
    layout_key: str,
    selected_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    return_value: bool,
) -> torch.Tensor:
    del (
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
    )
    key_head_num = key.size(1) if layout_key == "TND" else key.size(2)
    if layout_query == "BSND":
        out_shape = (query.size(0), query.size(1), key_head_num, selected_count)
    else:
        out_shape = (query.size(0), key_head_num, selected_count)
    return query.new_zeros(out_shape, dtype=torch.int32)


def _lightning_indexer_out_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_seq_lengths: torch.Tensor | None,
    key_seq_lengths: torch.Tensor | None,
    block_table: torch.Tensor | None,
    layout_query: str,
    layout_key: str,
    selected_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    return_value: bool,
    sparse_indices_out: torch.Tensor,
    sparse_values_out: torch.Tensor,
) -> torch.Tensor:
    del (
        query,
        key,
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        layout_query,
        layout_key,
        selected_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
        sparse_values_out,
    )
    return sparse_indices_out


def _scatter_nd_update_fake(
    var: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    del var, indices, updates


def _sparse_flash_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor | None,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_kv: torch.Tensor | None,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    scale_value: float,
    sparse_block_size: int,
    layout_query: str,
    layout_kv: str,
    sparse_mode: int,
) -> torch.Tensor:
    del (
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
    )
    return query.new_empty(query.shape, dtype=query.dtype)


def _sparse_flash_attention_out_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor | None,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_kv: torch.Tensor | None,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    scale_value: float,
    sparse_block_size: int,
    layout_query: str,
    layout_kv: str,
    sparse_mode: int,
    output: torch.Tensor,
) -> torch.Tensor:
    del (
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
    )
    return output


# ---------------------------------------------------------------------------
# DeepSeek-V4 DSA kernel fakes
# ---------------------------------------------------------------------------

# Matches kDsaMetadataBufferElements in xllm_ops_api.h.
_DSA_METADATA_BUFFER_ELEMENTS = 1024


def _rms_norm_dynamic_quant_fake(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight, eps
    return input.new_empty(input.shape, dtype=torch.int8), input.new_empty(input.shape[:-1], dtype=torch.float32)


def _npu_inplace_partial_rotary_mul_fake(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    rotary_mode: str,
    partial_slice: list[int],
) -> None:
    del x, r1, r2, rotary_mode, partial_slice


def _hc_pre_fake(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del hc_fn, hc_scale, hc_base, hc_sinkhorn_iters, norm_eps, hc_eps
    if x.dim() == 4:
        y_shape = (x.size(0), x.size(1), x.size(3))
        post_shape = (x.size(0), x.size(1), hc_mult)
        comb_shape = (x.size(0), x.size(1), hc_mult, hc_mult)
    else:
        y_shape = (x.size(0), x.size(2))
        post_shape = (x.size(0), hc_mult)
        comb_shape = (x.size(0), hc_mult, hc_mult)
    attn_input = x.new_empty(y_shape, dtype=x.dtype)
    post = x.new_empty(post_shape, dtype=torch.float32)
    comb = x.new_empty(comb_shape, dtype=torch.float32)
    return attn_input, post, comb


def _hc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    del post, comb
    # hc_post returns [T, hc_mult, hidden] (the merged residual streams).
    return residual.new_empty(residual.shape, dtype=residual.dtype)


def _compressor_fake(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    kv_block_table: torch.Tensor | None,
    score_block_table: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    seqused: torch.Tensor | None,
    start_pos: torch.Tensor | None,
    rope_head_dim: int,
    cmp_ratio: int,
    coff: int,
    norm_eps: float,
    rotary_mode: int,
    enable_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        wkv,
        wgate,
        kv_state,
        score_state,
        ape,
        rope_cos,
        kv_block_table,
        score_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        rope_head_dim,
        norm_eps,
        rotary_mode,
    )
    head_dim = norm_weight.size(0)
    if x.dim() == 3:
        compressed_seq = (x.size(1) + cmp_ratio - 1) // cmp_ratio
        cmp_kv_shape = (x.size(0), compressed_seq, head_dim)
        grad_shapes = (
            (x.size(0), x.size(1), coff * head_dim),
            (x.size(0), compressed_seq, coff * cmp_ratio, head_dim),
            (x.size(0), compressed_seq, head_dim),
            (x.size(0), compressed_seq),
        )
    else:
        compressed_seq = rope_sin.size(0)
        cmp_kv_shape = (compressed_seq, head_dim)
        grad_shapes = (
            (x.size(0), coff * head_dim),
            (compressed_seq, coff * cmp_ratio, head_dim),
            (compressed_seq, head_dim),
            (compressed_seq,),
        )
    outputs = [x.new_empty(cmp_kv_shape, dtype=x.dtype)]
    if enable_grad:
        outputs.extend(x.new_empty(shape, dtype=x.dtype) for shape in grad_shapes)
    else:
        outputs.extend(x.new_empty((0,), dtype=x.dtype) for _ in grad_shapes)
    return tuple(outputs)  # type: ignore[return-value]


def _sparse_attn_sharedkv_fake(
    q: torch.Tensor,
    ori_kv: torch.Tensor | None,
    cmp_kv: torch.Tensor | None,
    ori_sparse_indices: torch.Tensor | None,
    cmp_sparse_indices: torch.Tensor | None,
    ori_block_table: torch.Tensor | None,
    cmp_block_table: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
    cu_seqlens_cmp_kv: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_kv: torch.Tensor | None,
    sinks: torch.Tensor | None,
    metadata: torch.Tensor | None,
    softmax_scale: float,
    cmp_ratio: int,
    ori_mask_mode: int,
    cmp_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
    layout_q: str,
    layout_kv: str,
    return_softmax_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
    )
    out = q.new_empty(q.shape, dtype=q.dtype)
    lse_shape = (*q.shape[:-1], 1) if return_softmax_lse else (0,)
    lse = q.new_empty(lse_shape, dtype=torch.float32)
    return out, lse


def _sparse_attn_sharedkv_metadata_fake(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
    cu_seqlens_cmp_kv: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    seqused_kv: torch.Tensor | None,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    ori_topk: int,
    cmp_topk: int,
    cmp_ratio: int,
    ori_mask_mode: int,
    cmp_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
    layout_q: str,
    layout_kv: str,
    has_ori_kv: bool,
    has_cmp_kv: bool,
) -> torch.Tensor:
    del (
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        ori_topk,
        cmp_topk,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        has_ori_kv,
        has_cmp_kv,
    )
    for tensor in (
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_kv,
    ):
        if tensor is not None:
            return tensor.new_empty((_DSA_METADATA_BUFFER_ELEMENTS,), dtype=torch.int32)
    return torch.empty((_DSA_METADATA_BUFFER_ELEMENTS,), dtype=torch.int32, device="npu")


def _quant_lightning_indexer_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    query_quant_mode: int,
    key_quant_mode: int,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_key: torch.Tensor | None,
    block_table: torch.Tensor | None,
    metadata: torch.Tensor | None,
    layout_query: str,
    layout_key: str,
    sparse_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    cmp_ratio: int,
    return_value: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        weights,
        query_dequant_scale,
        key_dequant_scale,
        query_quant_mode,
        key_quant_mode,
        block_table,
        metadata,
        sparse_mode,
        pre_tokens,
        next_tokens,
        cmp_ratio,
    )
    key_head_num = key.size(1) if layout_key == "TND" else key.size(2)
    if layout_query == "BSND":
        out_shape = (query.size(0), query.size(1), key_head_num, sparse_count)
    else:
        out_shape = (query.size(0), key_head_num, sparse_count)
    out = query.new_zeros(out_shape, dtype=torch.int32)
    val = (
        query.new_empty(out_shape, dtype=torch.float32) if return_value else query.new_empty((0,), dtype=torch.float32)
    )
    return out, val


def _quant_lightning_indexer_metadata_fake(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    query_quant_mode: int,
    key_quant_mode: int,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_key: torch.Tensor | None,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    layout_query: str,
    layout_key: str,
    sparse_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    cmp_ratio: int,
    device: str,
) -> torch.Tensor:
    del (
        num_heads_q,
        num_heads_k,
        head_dim,
        query_quant_mode,
        key_quant_mode,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        cmp_ratio,
    )
    for tensor in (actual_seq_lengths_query, actual_seq_lengths_key):
        if tensor is not None:
            return tensor.new_empty((_DSA_METADATA_BUFFER_ELEMENTS,), dtype=torch.int32)
    return torch.empty((_DSA_METADATA_BUFFER_ELEMENTS,), dtype=torch.int32, device=device)


def _moe_gating_top_k_hash_fake(
    x: torch.Tensor,
    k: int,
    bias: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    tid2eid: torch.Tensor | None,
    k_group: int,
    group_count: int,
    routed_scaling_factor: float,
    eps: float,
    group_select_mode: int,
    renorm: int,
    norm_type: int,
    out_flag: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del bias, input_ids, tid2eid, k_group, group_count, routed_scaling_factor
    del eps, group_select_mode, renorm, norm_type, out_flag
    y_shape = (*x.shape[:-1], k)
    y = x.new_empty(y_shape, dtype=x.dtype)
    expert_idx = x.new_empty(y_shape, dtype=torch.int32)
    out = x.new_empty(x.shape, dtype=torch.float32)
    return y, expert_idx, out


def _dequant_swiglu_quant_fake(
    x: torch.Tensor,
    weight_scale: torch.Tensor | None,
    activation_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    quant_scale: torch.Tensor | None,
    quant_offset: torch.Tensor | None,
    group_index: torch.Tensor | None,
    activate_left: bool,
    quant_mode: int,
    swiglu_mode: int,
    clamp_limit: float,
    glu_alpha: float,
    glu_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del weight_scale, activation_scale, bias, quant_scale, quant_offset
    del group_index, activate_left, quant_mode, swiglu_mode
    del clamp_limit, glu_alpha, glu_bias
    # Output is half of input's last dim (SwiGLU splits gate/up).
    out_dim = x.size(-1) // 2
    act_quantized = x.new_empty((*x.shape[:-1], out_dim), dtype=torch.int8)
    act_scale = x.new_empty(x.shape[:-1], dtype=torch.float32)
    return act_quantized, act_scale


def _sparse_flash_attention_lse_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor | None,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_kv: torch.Tensor | None,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    scale_value: float,
    sparse_block_size: int,
    layout_query: str,
    layout_kv: str,
    sparse_mode: int,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    attention_mode: int = 2,
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
    )
    if return_softmax_lse:
        if query.dim() == 3:
            kv_head_num = key.size(2) if layout_kv == "PA_BSND" else key.size(1)
            softmax_size = (
                kv_head_num,
                query.size(0),
                query.size(1) // kv_head_num,
            )
        else:
            softmax_size = (
                query.size(0),
                key.size(2),
                query.size(1),
                query.size(2) // key.size(2),
            )
    else:
        softmax_size = (0,)
        del key, layout_kv
    softmax_max = query.new_empty(softmax_size, dtype=torch.float32)
    softmax_sum = query.new_empty(softmax_size, dtype=torch.float32)
    return query.new_empty(query.shape, dtype=query.dtype), softmax_max, softmax_sum


register_fake("xllm_ops::rms_norm", _rms_norm_fake)
register_fake("xllm_ops::rms_norm_gated", _rms_norm_gated_fake)
register_fake("xllm_ops::l2_norm", _l2_norm_fake)
register_fake("xllm_ops::chunk_gated_delta_rule", _chunk_gated_delta_rule_fake)
register_fake("xllm_ops::causal_conv1d_qkv_prefill", _causal_conv1d_qkv_prefill_fake)
register_fake(
    "xllm_ops::fused_sigmoid_gating_delta_rule_decode",
    _fused_sigmoid_gating_delta_rule_decode_fake,
)
register_fake("xllm_ops::fused_add_rms_norm", _fused_add_rms_norm_fake)
register_fake("xllm_ops::silu_and_mul", _silu_and_mul_fake)
register_fake("xllm_ops::reshape_paged_cache", _reshape_paged_cache_fake)
register_fake("xllm_ops::mla_preprocess_v2", _mla_preprocess_v2_fake)
register_fake("xllm_ops::update_decode_graph_metadata", _update_decode_graph_metadata_fake)
register_fake("xllm_ops::quant_matmul", _quant_matmul_fake)
register_fake("xllm_ops::quantize_per_tensor", _quantize_per_tensor_fake)
register_fake("xllm_ops::dynamic_quant", _dynamic_quant_fake)
register_fake(
    "xllm_ops::inplace_partial_rotary_mul",
    _inplace_partial_rotary_mul_fake,
)
register_fake(
    "xllm_ops::quant_lightning_indexer",
    _quant_lightning_indexer_fake,
)
register_fake(
    "xllm_ops::quant_lightning_indexer_metadata",
    _quant_lightning_indexer_metadata_fake,
)
register_fake("xllm_ops::lightning_indexer", _lightning_indexer_fake)
register_fake("xllm_ops::lightning_indexer_out", _lightning_indexer_out_fake)
register_fake("xllm_ops::scatter_nd_update", _scatter_nd_update_fake)
register_fake("xllm_ops::sparse_flash_attention", _sparse_flash_attention_fake)
register_fake("xllm_ops::sparse_flash_attention_out", _sparse_flash_attention_out_fake)
register_fake("xllm_ops::rms_norm_dynamic_quant", _rms_norm_dynamic_quant_fake)
register_fake(
    "xllm_ops::npu_inplace_partial_rotary_mul",
    _npu_inplace_partial_rotary_mul_fake,
)
register_fake("xllm_ops::compressor", _compressor_fake)
register_fake("xllm_ops::moe_gating_top_k_hash", _moe_gating_top_k_hash_fake)
register_fake("xllm_ops::dequant_swiglu_quant", _dequant_swiglu_quant_fake)
register_fake("xllm_ops::hc_pre", _hc_pre_fake)
register_fake("xllm_ops::hc_post", _hc_post_fake)
register_fake("xllm_ops::sparse_attn_sharedkv", _sparse_attn_sharedkv_fake)
register_fake("xllm_ops::sparse_attn_sharedkv_metadata", _sparse_attn_sharedkv_metadata_fake)
register_fake("xllm_ops::sparse_flash_attention_lse", _sparse_flash_attention_lse_fake)
