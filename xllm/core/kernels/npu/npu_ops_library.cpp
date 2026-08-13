/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// xllm_ops: NPU (PrivateUse1) dispatch registration for the Python model
// executor. Mirrors the schema defined in cuda_ops_library.cpp (which is only
// compiled for USE_CUDA builds). Under USE_NPU the schema must be declared here
// since the CUDA source is never compiled.
//
// Each wrapper is a thin adapter between the torch.ops schema and the
// underlying NPU kernel API. Data preparation (reshaping, dtype alignment)
// belongs in the Python caller, not here.

#include <torch/library.h>
#include <torch/torch.h>

#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#include "npu_ops_api.h"

namespace xllm {

namespace {

torch::Tensor rms_norm_npu(const torch::Tensor& input,
                           const torch::Tensor& weight,
                           double eps) {
  return xllm::kernel::npu::rms_norm(input, weight, eps, "rmsnorm");
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rms_norm_npu(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  auto [normed, rstd, residual_sum] =
      xllm::kernel::npu::add_rms_norm(input, residual, weight, eps);
  return std::make_tuple(normed, residual_sum);
}

torch::Tensor silu_and_mul_npu(const torch::Tensor& input) {
  return xllm::kernel::npu::active(input, "swiglu");
}

torch::Tensor reshape_paged_cache_npu(const torch::Tensor& slot_mapping,
                                      torch::Tensor& keys,
                                      torch::Tensor& values,
                                      torch::Tensor& key_cache,
                                      torch::Tensor& value_cache) {
  std::optional<torch::Tensor> v = values;
  std::optional<torch::Tensor> vc = value_cache;
  xllm::kernel::npu::reshape_paged_cache(keys, v, key_cache, vc, slot_mapping);
  return key_cache;
}

void apply_rotary_embedding_npu(torch::Tensor& q,
                                torch::Tensor& k,
                                const torch::Tensor& cos_sin_cache,
                                const torch::Tensor& positions) {
  xllm::kernel::npu::apply_rotary(q, k, cos_sin_cache, positions);
}

// Graph-mode decode metadata update. Copies real data into the head of
// pre-allocated static buffers and fills padding slots with safe defaults
// (zero tokens/slot mapping/indptr, 1 last-page-len) so that the captured
// graph operates on valid data for every padded position.
torch::Tensor update_decode_graph_metadata_npu(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices,
    const torch::Tensor& paged_kv_last_page_len,
    torch::Tensor& dst_tokens,
    torch::Tensor& dst_positions,
    torch::Tensor& dst_slot_mapping,
    torch::Tensor& dst_kv_seq_lens,
    torch::Tensor& dst_kv_seq_lens_delta,
    torch::Tensor& dst_paged_kv_indptr,
    torch::Tensor& dst_paged_kv_indices,
    torch::Tensor& dst_paged_kv_last_page_len,
    int64_t padded_num_tokens) {
  const int64_t n = tokens.size(0);
  const int64_t p = padded_num_tokens;

  dst_tokens.slice(0, 0, n).copy_(tokens);
  dst_positions.slice(0, 0, n).copy_(positions);
  if (p > n) {
    dst_tokens.slice(0, n, p).zero_();
    dst_positions.slice(0, n, p).zero_();
  }

  // ModelInputParams::attention.device.new_cache_slots is optional during
  // native ACL graph profile/warmup. Match GraphPersistentParam::update:
  // restore the zero default, then copy only the source prefix when present.
  dst_slot_mapping.slice(0, 0, p).zero_();
  const int64_t slot_len = std::min<int64_t>(slot_mapping.numel(), n);
  if (slot_len > 0) {
    dst_slot_mapping.slice(0, 0, slot_len)
        .copy_(slot_mapping.slice(0, 0, slot_len));
  }

  const int64_t src_len = std::min<int64_t>(kv_seq_lens.size(0), n + 1);
  dst_kv_seq_lens.slice(0, 0, p + 1).zero_();
  if (src_len > 0) {
    dst_kv_seq_lens.slice(0, 0, src_len)
        .copy_(kv_seq_lens.slice(0, 0, src_len));
    dst_kv_seq_lens.slice(0, src_len, p + 1)
        .copy_(kv_seq_lens.slice(0, src_len - 1, src_len));
  }
  dst_kv_seq_lens_delta.slice(0, 0, p).copy_(
      dst_kv_seq_lens.slice(0, 1, p + 1) - dst_kv_seq_lens.slice(0, 0, p));

  const int64_t indptr_len = std::min<int64_t>(paged_kv_indptr.size(0), n + 1);
  dst_paged_kv_indptr.slice(0, 0, p + 1).zero_();
  if (indptr_len > 0) {
    dst_paged_kv_indptr.slice(0, 0, indptr_len)
        .copy_(paged_kv_indptr.slice(0, 0, indptr_len));
    dst_paged_kv_indptr.slice(0, indptr_len, p + 1)
        .copy_(paged_kv_indptr.slice(0, indptr_len - 1, indptr_len));
  }

  dst_paged_kv_last_page_len.slice(0, 0, p).fill_(1);
  const int64_t last_page_len =
      std::min<int64_t>(paged_kv_last_page_len.numel(), n);
  if (last_page_len > 0) {
    dst_paged_kv_last_page_len.slice(0, 0, last_page_len)
        .copy_(paged_kv_last_page_len.slice(0, 0, last_page_len));
  }

  const int64_t num_pages =
      std::min<int64_t>(paged_kv_indices.numel(), dst_paged_kv_indices.numel());
  if (num_pages > 0) {
    dst_paged_kv_indices.slice(0, 0, num_pages)
        .copy_(paged_kv_indices.slice(0, 0, num_pages));
  }

  return dst_tokens;
}

}  // namespace

void ensure_xllm_ops_registered() {
  // Intentionally empty — referencing this symbol prevents the linker from
  // stripping the TORCH_LIBRARY static initializers below.
}

}  // namespace xllm

// Schema declarations (device-agnostic). Identical to cuda_ops_library.cpp —
// compiled only under USE_NPU (mutually exclusive with USE_CUDA).
TORCH_LIBRARY(xllm_ops, m) {
  m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
  m.def(
      "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor "
      "weight, "
      "float eps) -> (Tensor, Tensor)");
  // Fused RMSNorm + dynamic per-token int8 quant (W8A8 query preprocess).
  // Returns (qr_int8, qr_pertoken_scale) matching C++ rms_norm_dynamic_quant
  // (npu_ops_api.h:122), used by the DSV4 indexer build_query path.
  m.def(
      "rms_norm_dynamic_quant(Tensor input, Tensor weight, float eps) -> "
      "(Tensor, Tensor)");
  // In-place partial rotary embedding (interleaved). x is 4D [B,N,S,D], r1/r2
  // are cos/sin [B,1,1,rope_head_dim]; partial_slice=[rope_start, rope_head_dim].
  // Mirrors C++ apply_partial_rope (deepseek_sparse_attention.cpp:151) used by
  // the DSV4 indexer build_query.
  m.def(
      "npu_inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, "
      "str rotary_mode, int[] partial_slice) -> ()");
  m.def("silu_and_mul(Tensor input) -> Tensor");
  m.def(
      "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
      "int "
      "num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, "
      "Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> Tensor");
  m.def(
      "reshape_paged_cache(Tensor slot_mapping, Tensor(c!) keys, Tensor(d!) "
      "values, "
      "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor");
  m.def(
      "apply_rotary_embedding(Tensor(a!) q, Tensor(b!) k, Tensor cos_sin_cache,"
      " Tensor positions) -> ()");
  m.def(
      "update_decode_graph_metadata(Tensor tokens, Tensor positions, Tensor "
      "slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, Tensor "
      "paged_kv_indices, Tensor paged_kv_last_page_len, Tensor(a!) dst_tokens, "
      "Tensor(b!) dst_positions, Tensor(c!) dst_slot_mapping, Tensor(d!) "
      "dst_kv_seq_lens, Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) "
      "dst_paged_kv_indptr, Tensor(g!) dst_paged_kv_indices, Tensor(h!) "
      "dst_paged_kv_last_page_len, int padded_num_tokens) -> Tensor");
  m.def(
      "quant_matmul(Tensor x1, Tensor x2, bool transpose2, Tensor scale, "
      "Tensor? offset, Tensor? pertoken_scale, Tensor? bias, ScalarType? "
      "output_dtype) -> Tensor");
  m.def(
      "quantize_per_tensor(Tensor self, Tensor scales, Tensor zero_points, "
      "ScalarType dtype, int axis) -> Tensor");
  m.def(
      "dynamic_quant(Tensor input, Tensor? smooth_scales, Tensor? group_index, "
      "ScalarType? dst_type) -> (Tensor, Tensor?)");
  m.def(
      "lightning_indexer(Tensor query, Tensor key, Tensor weights, "
      "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? "
      "block_table, str layout_query, str layout_key, int selected_count, int "
      "sparse_mode, int pre_tokens, int next_tokens, bool return_value) -> "
      "Tensor");
  m.def(
      "scatter_nd_update(Tensor(a!) var, Tensor indices, Tensor updates) -> "
      "()");
  m.def(
      "sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor "
      "sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, "
      "Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, "
      "float scale_value, int sparse_block_size, str layout_query, str "
      "layout_kv, int sparse_mode) -> Tensor");
  // ---- DeepSeek-V4 DSA kernels ----
  // MoE hash routing gate (returns routed output, expert_idx, token_unpermute).
  m.def(
      "moe_gating_top_k_hash(Tensor x, int k, Tensor? bias, Tensor? input_ids, "
      "Tensor? tid2eid, int k_group, int group_count, float routed_scaling_factor, "
      "float eps, int group_select_mode, int renorm, int norm_type, bool "
      "out_flag) -> (Tensor, Tensor, Tensor)");
  // Dequant + SwiGLU + quant (fused, replaces manual dequant loop).
  m.def(
      "dequant_swiglu_quant(Tensor x, Tensor? weight_scale, Tensor? "
      "activation_scale, Tensor? bias, Tensor? quant_scale, Tensor? "
      "quant_offset, Tensor? group_index, bool activate_left, int quant_mode, "
      "int swiglu_mode, float clamp_limit, float glu_alpha, float glu_bias) "
      "-> (Tensor, Tensor)");
  // HyperConnection pre/post (hc_pre returns attn_input, post, comb).
  m.def(
      "hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, "
      "int hc_mult, int hc_sinkhorn_iters, float norm_eps, float hc_eps) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> "
      "Tensor");
  // Compressor: NSA-style KV pooling. kv_state/score_state are in-place (Ref).
  // Returns (cmp_kv, wkv_proj, softmax_res, norm_x, norm_rstd).
  m.def(
      "compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) kv_state, "
      "Tensor(b!) score_state, Tensor ape, Tensor norm_weight, Tensor "
      "rope_sin, Tensor rope_cos, Tensor? kv_block_table, Tensor? "
      "score_block_table, Tensor? cu_seqlens, Tensor? seqused, Tensor? "
      "start_pos, int rope_head_dim, int cmp_ratio, int coff, float "
      "norm_eps, int rotary_mode, bool enable_grad) -> (Tensor, Tensor, "
      "Tensor, Tensor, Tensor)");
  // Probe variant for DSV4 debugging: same implementation as compressor, but
  // without alias annotations on kv_state/score_state. This isolates whether
  // Python dispatcher alias handling affects aclnnCompressor inputs.
  m.def(
      "compressor_noalias(Tensor x, Tensor wkv, Tensor wgate, Tensor kv_state, "
      "Tensor score_state, Tensor ape, Tensor norm_weight, Tensor rope_sin, "
      "Tensor rope_cos, Tensor? kv_block_table, Tensor? score_block_table, "
      "Tensor? cu_seqlens, Tensor? seqused, Tensor? start_pos, int "
      "rope_head_dim, int cmp_ratio, int coff, float norm_eps, int "
      "rotary_mode, bool enable_grad) -> (Tensor, Tensor, Tensor, Tensor, "
      "Tensor)");
  // Two-stage sparse attention over original + compressed KV.
  m.def(
      "sparse_attn_sharedkv(Tensor q, Tensor? ori_kv, Tensor? cmp_kv, "
      "Tensor? ori_sparse_indices, Tensor? cmp_sparse_indices, Tensor? "
      "ori_block_table, Tensor? cmp_block_table, Tensor? cu_seqlens_q, "
      "Tensor? cu_seqlens_ori_kv, Tensor? cu_seqlens_cmp_kv, Tensor? "
      "seqused_q, Tensor? seqused_kv, Tensor? sinks, Tensor? metadata, "
      "float softmax_scale, int cmp_ratio, int ori_mask_mode, int "
      "cmp_mask_mode, int ori_win_left, int ori_win_right, str layout_q, "
      "str layout_kv, bool return_softmax_lse) -> (Tensor, Tensor)");
  // AICPU tiling metadata builder for sparse_attn_sharedkv.
  m.def(
      "sparse_attn_sharedkv_metadata(int num_heads_q, int num_heads_kv, int "
      "head_dim, Tensor? cu_seqlens_q, Tensor? cu_seqlens_ori_kv, Tensor? "
      "cu_seqlens_cmp_kv, Tensor? seqused_q, Tensor? seqused_kv, int "
      "batch_size, int max_seqlen_q, int max_seqlen_kv, int ori_topk, int "
      "cmp_topk, int cmp_ratio, int ori_mask_mode, int cmp_mask_mode, int "
      "ori_win_left, int ori_win_right, str layout_q, str layout_kv, bool "
      "has_ori_kv, bool has_cmp_kv) -> Tensor");
  // Quantized lightning indexer: int8 q/k top-k selection with cmp_ratio.
  m.def(
      "quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, "
      "Tensor query_dequant_scale, Tensor key_dequant_scale, int "
      "query_quant_mode, int key_quant_mode, Tensor? actual_seq_lengths_query, "
      "Tensor? actual_seq_lengths_key, Tensor? block_table, Tensor? metadata, "
      "str layout_query, str layout_key, int sparse_count, int sparse_mode, "
      "int pre_tokens, int next_tokens, int cmp_ratio, bool return_value) -> "
      "(Tensor, Tensor)");
  // AICPU tiling metadata builder for quant_lightning_indexer.
  m.def(
      "quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, int "
      "head_dim, int query_quant_mode, int key_quant_mode, Tensor? "
      "actual_seq_lengths_query, Tensor? actual_seq_lengths_key, int "
      "batch_size, int max_seqlen_q, int max_seqlen_k, str layout_query, str "
      "layout_key, int sparse_count, int sparse_mode, int pre_tokens, int "
      "next_tokens, int cmp_ratio, str device) -> Tensor");
  // W8A8 MoE expert grouped GEMM wrapper. Routes through the same
  // xllm::kernel::group_gemm -> apply_npu_grouped_matmul path as the native
  // C++ FusedMoE (fused_moe.cpp:1028-1073), so an omitted per_token_scale
  // becomes a genuinely empty TensorList instead of a None-shaped tensor.
  m.def(
      "group_gemm(Tensor x, Tensor weight, Tensor? scale, Tensor? "
      "per_token_scale, Tensor group_list, int split_item, int group_type, "
      "int group_list_type, ScalarType? output_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m) {
  m.impl("rms_norm", TORCH_FN(xllm::rms_norm_npu));
  m.impl("fused_add_rms_norm", TORCH_FN(xllm::fused_add_rms_norm_npu));
  m.impl("rms_norm_dynamic_quant",
         TORCH_FN(xllm::kernel::npu::rms_norm_dynamic_quant));
  m.impl("npu_inplace_partial_rotary_mul",
         TORCH_FN(xllm::kernel::npu::npu_inplace_partial_rotary_mul));
  m.impl("silu_and_mul", TORCH_FN(xllm::silu_and_mul_npu));
  m.impl("reshape_paged_cache", TORCH_FN(xllm::reshape_paged_cache_npu));
  m.impl("apply_rotary_embedding", TORCH_FN(xllm::apply_rotary_embedding_npu));
  m.impl("update_decode_graph_metadata",
         TORCH_FN(xllm::update_decode_graph_metadata_npu));
  m.impl("quant_matmul", TORCH_FN(xllm::kernel::npu::quant_matmul));
  m.impl("quantize_per_tensor",
         TORCH_FN(xllm::kernel::npu::quantize_per_tensor));
  m.impl("dynamic_quant", TORCH_FN(xllm::kernel::npu::dynamic_quant));
  m.impl("lightning_indexer", TORCH_FN(xllm::kernel::npu::lightning_indexer));
  m.impl("scatter_nd_update", TORCH_FN(xllm::kernel::npu::scatter_nd_update));
  m.impl("sparse_flash_attention",
         TORCH_FN(xllm::kernel::npu::sparse_flash_attention));
  // ---- DeepSeek-V4 DSA kernels ----
  m.impl("moe_gating_top_k_hash",
         TORCH_FN(xllm::kernel::npu::moe_gating_top_k_hash));
  m.impl("dequant_swiglu_quant",
         TORCH_FN(xllm::kernel::npu::dequant_swiglu_quant));
  m.impl("hc_pre", TORCH_FN(xllm::kernel::npu::hc_pre));
  m.impl("hc_post", TORCH_FN(xllm::kernel::npu::hc_post));
  m.impl("compressor", TORCH_FN(xllm::kernel::npu::compressor));
  m.impl("compressor_noalias", TORCH_FN(xllm::kernel::npu::compressor));
  m.impl("sparse_attn_sharedkv",
         TORCH_FN(xllm::kernel::npu::sparse_attn_sharedkv));
  m.impl("sparse_attn_sharedkv_metadata",
         TORCH_FN(xllm::kernel::npu::sparse_attn_sharedkv_metadata));
  m.impl("quant_lightning_indexer",
         TORCH_FN(xllm::kernel::npu::quant_lightning_indexer));
  m.impl("quant_lightning_indexer_metadata",
         TORCH_FN(xllm::kernel::npu::quant_lightning_indexer_metadata));
  m.impl("group_gemm", TORCH_FN(xllm::kernel::npu::group_gemm));
}
