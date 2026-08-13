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
    return self.new_empty(self.shape, dtype=dtype)


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


register_fake("xllm_ops::rms_norm", _rms_norm_fake)
register_fake("xllm_ops::fused_add_rms_norm", _fused_add_rms_norm_fake)
register_fake("xllm_ops::silu_and_mul", _silu_and_mul_fake)
register_fake("xllm_ops::reshape_paged_cache", _reshape_paged_cache_fake)
register_fake("xllm_ops::update_decode_graph_metadata", _update_decode_graph_metadata_fake)
register_fake("xllm_ops::quant_matmul", _quant_matmul_fake)
register_fake("xllm_ops::quantize_per_tensor", _quantize_per_tensor_fake)
register_fake("xllm_ops::dynamic_quant", _dynamic_quant_fake)
register_fake("xllm_ops::lightning_indexer", _lightning_indexer_fake)
register_fake("xllm_ops::lightning_indexer_out", _lightning_indexer_out_fake)
register_fake("xllm_ops::scatter_nd_update", _scatter_nd_update_fake)
register_fake("xllm_ops::sparse_flash_attention", _sparse_flash_attention_fake)
register_fake("xllm_ops::sparse_flash_attention_out", _sparse_flash_attention_out_fake)
