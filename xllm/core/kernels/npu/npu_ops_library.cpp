/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

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

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#include "npu_ops_api.h"
#include "triton_npu/torch_api/triton_ops_api.h"

namespace xllm {

namespace {

torch::Tensor rms_norm_npu(const torch::Tensor& input,
                           const torch::Tensor& weight,
                           double eps) {
  return xllm::kernel::npu::rms_norm(input, weight, eps, "rmsnorm");
}

torch::Tensor rms_norm_gated_npu(const torch::Tensor& input,
                                 const torch::Tensor& gate,
                                 const torch::Tensor& weight,
                                 double eps) {
  return xllm::kernel::npu::layer_norm_fwd_aclnn(input,
                                                 weight,
                                                 /*bias=*/torch::Tensor(),
                                                 eps,
                                                 /*z=*/gate,
                                                 /*group_size=*/input.size(-1),
                                                 /*norm_before_gate=*/true,
                                                 /*is_rms_norm=*/true);
}

torch::Tensor l2_norm_npu(torch::Tensor input, double eps) {
  return xllm::kernel::npu::npu_l2norm_last_dim(input, eps);
}

torch::Tensor causal_conv1d_prefill_npu(torch::Tensor x,
                                        torch::Tensor weight,
                                        torch::Tensor conv_state,
                                        torch::Tensor state_indices,
                                        torch::Tensor has_initial_state,
                                        torch::Tensor query_start_loc) {
  // Python layer stores weight as [dim, kernel_width]; CANN expects
  // [kernel_width, dim].
  if (weight.size(0) > weight.size(1)) {
    weight = weight.t().contiguous();
  }

  // Convert device tensors to host vectors for IntArrayRef parameters.
  auto qsl_cpu = query_start_loc.to(torch::kCPU, torch::kInt64).contiguous();
  auto si_cpu = state_indices.to(torch::kCPU, torch::kInt64).contiguous();
  auto ism_cpu = has_initial_state.to(torch::kCPU, torch::kInt64).contiguous();

  std::vector<int64_t> qsl_vec(qsl_cpu.data_ptr<int64_t>(),
                               qsl_cpu.data_ptr<int64_t>() + qsl_cpu.numel());
  std::vector<int64_t> si_vec(si_cpu.data_ptr<int64_t>(),
                              si_cpu.data_ptr<int64_t>() + si_cpu.numel());
  std::vector<int64_t> ism_vec(ism_cpu.data_ptr<int64_t>(),
                               ism_cpu.data_ptr<int64_t>() + ism_cpu.numel());

  constexpr int64_t kActivationSilu = 1;
  constexpr int64_t kPadSlotId = -1;
  constexpr int64_t kRunModeForward = 0;

  return xllm::kernel::npu::causal_conv1d(
      x,
      weight,
      conv_state,
      /*bias_opt=*/std::nullopt,
      torch::IntArrayRef(qsl_vec),
      torch::IntArrayRef(si_vec),
      torch::IntArrayRef(ism_vec),
      /*num_accepted_tokens_opt=*/torch::IntArrayRef{},
      kActivationSilu,
      kPadSlotId,
      kRunModeForward);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
causal_conv1d_qkv_prefill_npu(torch::Tensor x,
                              torch::Tensor weight,
                              torch::Tensor conv_state,
                              torch::Tensor state_indices,
                              torch::Tensor has_initial_state,
                              torch::Tensor query_start_loc,
                              int64_t num_qk_heads,
                              int64_t num_v_heads,
                              int64_t head_k_dim,
                              int64_t head_v_dim) {
  // Python layer stores weight as [dim, kernel_width]; CANN expects
  // [kernel_width, dim].
  if (weight.size(0) > weight.size(1)) {
    weight = weight.t().contiguous();
  }

  auto qsl_cpu = query_start_loc.to(torch::kCPU, torch::kInt64).contiguous();
  auto si_cpu = state_indices.to(torch::kCPU, torch::kInt64).contiguous();
  auto ism_cpu = has_initial_state.to(torch::kCPU, torch::kInt64).contiguous();

  std::vector<int64_t> qsl_vec(qsl_cpu.data_ptr<int64_t>(),
                               qsl_cpu.data_ptr<int64_t>() + qsl_cpu.numel());
  std::vector<int64_t> si_vec(si_cpu.data_ptr<int64_t>(),
                              si_cpu.data_ptr<int64_t>() + si_cpu.numel());
  std::vector<int64_t> ism_vec(ism_cpu.data_ptr<int64_t>(),
                               ism_cpu.data_ptr<int64_t>() + ism_cpu.numel());

  return xllm::kernel::npu::causal_conv1d_qkv(x,
                                              weight,
                                              conv_state,
                                              torch::IntArrayRef(qsl_vec),
                                              torch::IntArrayRef(si_vec),
                                              torch::IntArrayRef(ism_vec),
                                              num_qk_heads,
                                              num_v_heads,
                                              head_k_dim,
                                              head_v_dim);
}

std::tuple<torch::Tensor, torch::Tensor> chunk_gated_delta_rule_npu(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor initial_state,
    torch::Tensor cu_seqlens) {
  return xllm::kernel::npu::npu_mega_chunk_gdn(
      q,
      k,
      v,
      g,
      beta,
      /*scale=*/std::nullopt,
      /*initial_state=*/initial_state,
      /*output_final_state=*/true,
      /*cu_seqlens=*/cu_seqlens,
      /*q_seq_lens=*/{},
      /*use_qk_l2norm_in_kernel=*/true);
}

torch::Tensor fused_sigmoid_gating_delta_rule_decode_npu(
    torch::Tensor a_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor b,
    torch::Tensor ssm_state,
    torch::Tensor state_indices,
    torch::Tensor cu_seqlens,
    double scale) {
  auto a_log_f32 = a_log.to(torch::kFloat32);
  auto dt_bias_f32 = dt_bias.to(torch::kFloat32);
  return xllm::kernel::npu::npu_fused_sigmoid_gating_delta_rule_update(
      a_log_f32,
      a,
      dt_bias_f32,
      q,
      k,
      v,
      b,
      ssm_state,
      state_indices,
      cu_seqlens,
      /*scale=*/static_cast<float>(scale),
      /*use_qk_l2norm_in_kernel=*/true,
      /*softplus_beta=*/1.0f,
      /*softplus_threshold=*/20.0f);
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

void inplace_partial_rotary_mul_npu(torch::Tensor& input,
                                    const torch::Tensor& cosine,
                                    const torch::Tensor& sine,
                                    const std::string& rotary_mode,
                                    const std::vector<int64_t>& partial_slice) {
  xllm::kernel::npu::npu_inplace_partial_rotary_mul(
      input, cosine, sine, rotary_mode, partial_slice);
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
// (zero tokens, -1 slot mapping, 1 last-page-len) so that the captured
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
  dst_slot_mapping.slice(0, 0, n).copy_(slot_mapping);
  if (p > n) {
    dst_tokens.slice(0, n, p).zero_();
    dst_positions.slice(0, n, p).zero_();
    dst_slot_mapping.slice(0, n, p).fill_(-1);
  }

  const int64_t src_len = std::min<int64_t>(kv_seq_lens.size(0), n + 1);
  dst_kv_seq_lens.slice(0, 0, src_len).copy_(kv_seq_lens.slice(0, 0, src_len));
  if (p >= n) {
    dst_kv_seq_lens.slice(0, src_len, p + 1)
        .copy_(kv_seq_lens.slice(0, src_len - 1, src_len));
  }
  dst_kv_seq_lens_delta.slice(0, 0, p).copy_(
      dst_kv_seq_lens.slice(0, 1, p + 1) - dst_kv_seq_lens.slice(0, 0, p));

  const int64_t indptr_len = std::min<int64_t>(paged_kv_indptr.size(0), n + 1);
  dst_paged_kv_indptr.slice(0, 0, indptr_len)
      .copy_(paged_kv_indptr.slice(0, 0, indptr_len));
  if (p >= n) {
    dst_paged_kv_indptr.slice(0, indptr_len, p + 1)
        .copy_(paged_kv_indptr.slice(0, indptr_len - 1, indptr_len));
  }

  dst_paged_kv_last_page_len.slice(0, 0, n).copy_(
      paged_kv_last_page_len.slice(0, 0, n));
  if (p > n) {
    dst_paged_kv_last_page_len.slice(0, n, p).fill_(1);
  }

  const int64_t num_pages = paged_kv_indices.size(0);
  dst_paged_kv_indices.slice(0, 0, num_pages).copy_(paged_kv_indices);

  return dst_tokens;
}

// Zigzag Context-Parallel sharding plan for the Python model executor. Pure
// host index math (no NPU kernel): turns per-sequence query lengths into the
// shard/restore indices plus the packed query/KV gather indices that one FIA
// call consumes. This is the C++ lowering of the host scalar loops previously
// in xllm/python/model_executor/cp_utils.py; both passes below mirror that
// reference exactly. Each sequence is padded up to a multiple of 2*cp_size and
// cut into 2*cp_size chunks; rank r owns chunk r paired with chunk
// 2*cp_size-1-r (head-tail balanced). See cp_utils.CpContext for field docs.
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::vector<int64_t>,
           std::vector<int64_t>,
           int64_t>
build_cp_context_npu(const std::vector<int64_t>& seq_lens,
                     int64_t cp_size,
                     int64_t cp_rank,
                     c10::Device device) {
  TORCH_CHECK(cp_size > 1, "build_cp_context requires cp_size > 1");

  const int64_t num_chunks = cp_size * 2;
  // The two chunk ids this rank owns: an early one and its mirror.
  const int64_t first_chunk = cp_rank;
  const int64_t second_chunk = num_chunks - 1 - cp_rank;

  std::vector<int64_t> shard_index;
  std::vector<int64_t> query_index;
  std::vector<int64_t> q_cu_seqlens;
  std::vector<int64_t> kv_gather_index;
  std::vector<int64_t> kv_cu_seqlens;
  // restore_index needs the per-seq local segment offset (same on every rank)
  // and the ownership map, so accumulate it in a second pass below.
  std::vector<int64_t> chunk_lens;
  std::vector<int64_t> local_seg_offsets;
  chunk_lens.reserve(seq_lens.size());
  local_seg_offsets.reserve(seq_lens.size());

  int64_t global_offset = 0;
  int64_t local_offset = 0;
  int64_t q_run = 0;
  int64_t kv_run = 0;
  for (const int64_t length : seq_lens) {
    const int64_t padded =
        ((length + num_chunks - 1) / num_chunks) * num_chunks;
    const int64_t chunk_len = padded / num_chunks;
    chunk_lens.push_back(chunk_len);
    local_seg_offsets.push_back(local_offset);

    // Emit the two owned segments in local order: first half then second half.
    // For each, the real rows sit at the front (small j) because real position
    // grows with j, so query_index stays front-packed per segment.
    const int64_t halves[2][2] = {{0, first_chunk}, {1, second_chunk}};
    for (const auto& half_chunk : halves) {
      const int64_t half = half_chunk[0];
      const int64_t chunk_id = half_chunk[1];
      const int64_t seg_local_base = local_offset + half * chunk_len;
      const int64_t seg_start = chunk_id * chunk_len;  // first real position
      int64_t real_count = 0;
      for (int64_t j = 0; j < chunk_len; ++j) {
        const int64_t pos_in_seq = seg_start + j;
        if (pos_in_seq < length) {
          shard_index.push_back(global_offset + pos_in_seq);
          query_index.push_back(seg_local_base + j);
          ++real_count;
        } else {
          shard_index.push_back(-1);
        }
      }
      if (real_count > 0) {
        // Causal prefix ends exactly at the last real query position + 1
        // = seg_start + real_count (segment is a contiguous real range).
        const int64_t prefix_len = seg_start + real_count;
        q_run += real_count;
        q_cu_seqlens.push_back(q_run);
        for (int64_t p = 0; p < prefix_len; ++p) {
          kv_gather_index.push_back(global_offset + p);
        }
        kv_run += prefix_len;
        kv_cu_seqlens.push_back(kv_run);
      }
    }

    global_offset += length;
    local_offset += 2 * chunk_len;
  }

  const int64_t total_local = local_offset;

  // restore_index: for every global (real) row, where it lands in the
  // rank-major all-gather output [cp_size * total_local]. Its final size is the
  // total real-token count, which the first pass accumulated into
  // global_offset.
  std::vector<int64_t> restore_index;
  restore_index.reserve(global_offset);
  for (size_t s = 0; s < seq_lens.size(); ++s) {
    const int64_t length = seq_lens[s];
    const int64_t chunk_len = chunk_lens[s];
    const int64_t seg_offset = local_seg_offsets[s];
    for (int64_t pos_in_seq = 0; pos_in_seq < length; ++pos_in_seq) {
      const int64_t chunk_id = pos_in_seq / chunk_len;
      const int64_t row_in_chunk = pos_in_seq % chunk_len;
      int64_t owner_rank;
      int64_t local_pos;
      if (chunk_id < cp_size) {
        owner_rank = chunk_id;
        local_pos = seg_offset + row_in_chunk;
      } else {
        owner_rank = num_chunks - 1 - chunk_id;
        local_pos = seg_offset + chunk_len + row_in_chunk;
      }
      restore_index.push_back(owner_rank * total_local + local_pos);
    }
  }

  const auto cpu_int64 = torch::dtype(torch::kInt64).device(torch::kCPU);
  auto shard_tensor = torch::tensor(shard_index, cpu_int64);
  auto valid_mask = shard_tensor >= 0;
  auto gather_index =
      torch::where(valid_mask, shard_tensor, torch::zeros_like(shard_tensor));

  return std::make_tuple(shard_tensor.to(device),
                         gather_index.to(device),
                         valid_mask.to(device),
                         torch::tensor(restore_index, cpu_int64).to(device),
                         torch::tensor(query_index, cpu_int64).to(device),
                         torch::tensor(kv_gather_index, cpu_int64).to(device),
                         q_cu_seqlens,
                         kv_cu_seqlens,
                         total_local);
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
      "rms_norm_gated(Tensor input, Tensor gate, Tensor weight, float eps) -> "
      "Tensor");
  m.def("l2_norm(Tensor input, float eps) -> Tensor");
  m.def(
      "chunk_gated_delta_rule(Tensor q, Tensor k, Tensor v, Tensor g, "
      "Tensor beta, Tensor initial_state, Tensor cu_seqlens) -> "
      "(Tensor, Tensor)");
  m.def(
      "causal_conv1d_prefill(Tensor x, Tensor weight, Tensor(a!) conv_state, "
      "Tensor state_indices, Tensor has_initial_state, "
      "Tensor query_start_loc) -> Tensor");
  m.def(
      "causal_conv1d_qkv_prefill(Tensor x, Tensor weight, "
      "Tensor(a!) conv_state, Tensor state_indices, "
      "Tensor has_initial_state, Tensor query_start_loc, "
      "int num_qk_heads, int num_v_heads, "
      "int head_k_dim, int head_v_dim) -> (Tensor, Tensor, Tensor)");
  m.def(
      "fused_sigmoid_gating_delta_rule_decode(Tensor a_log, Tensor a, "
      "Tensor dt_bias, Tensor q, Tensor k, Tensor v, Tensor b, "
      "Tensor(a!) ssm_state, Tensor state_indices, Tensor cu_seqlens, "
      "float scale) -> Tensor");
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
  // are cos/sin [B,1,1,rope_head_dim]; partial_slice=[rope_start,
  // rope_head_dim]. Mirrors C++ apply_partial_rope
  // (deepseek_sparse_attention.cpp:151) used by the DSV4 indexer build_query.
  m.def(
      "npu_inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, "
      "str rotary_mode, int[] partial_slice) -> ()");
  m.def("silu_and_mul(Tensor input) -> Tensor");
  m.def(
      "inplace_partial_rotary_mul(Tensor(a!) input, Tensor cosine, Tensor "
      "sine, str rotary_mode, int[] partial_slice) -> ()");
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
      "quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, "
      "Tensor query_dequant_scale, Tensor key_dequant_scale, int "
      "query_quant_mode, int key_quant_mode, Tensor? "
      "actual_seq_lengths_query, Tensor? actual_seq_lengths_key, Tensor? "
      "block_table, Tensor? metadata, str layout_query, str layout_key, int "
      "sparse_count, int sparse_mode, int pre_tokens, int next_tokens, int "
      "cmp_ratio, bool return_value) -> (Tensor, Tensor)");
  m.def(
      "quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, "
      "int head_dim, int query_quant_mode, int key_quant_mode, Tensor? "
      "actual_seq_lengths_query, Tensor? actual_seq_lengths_key, int "
      "batch_size, int max_seqlen_q, int max_seqlen_k, str layout_query, "
      "str layout_key, int sparse_count, int sparse_mode, int pre_tokens, "
      "int next_tokens, int cmp_ratio, str device) -> Tensor");
  m.def(
      "lightning_indexer(Tensor query, Tensor key, Tensor weights, "
      "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? "
      "block_table, str layout_query, str layout_key, int selected_count, int "
      "sparse_mode, int pre_tokens, int next_tokens, bool return_value) -> "
      "Tensor");
  m.def(
      "lightning_indexer_out(Tensor query, Tensor key, Tensor weights, "
      "Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? "
      "block_table, str layout_query, str layout_key, int selected_count, "
      "int sparse_mode, int pre_tokens, int next_tokens, bool return_value, "
      "Tensor(a!) sparse_indices_out, Tensor(b!) sparse_values_out) -> "
      "Tensor(a!)");
  m.def(
      "scatter_nd_update(Tensor(a!) var, Tensor indices, Tensor updates) -> "
      "()");
  m.def(
      "sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor "
      "sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, "
      "Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, "
      "float scale_value, int sparse_block_size, str layout_query, str "
      "layout_kv, int sparse_mode) -> Tensor");
  m.def(
      "sparse_flash_attention_out(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_indices, Tensor? block_table, Tensor? "
      "actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? "
      "query_rope, Tensor? key_rope, float scale_value, int "
      "sparse_block_size, str layout_query, str layout_kv, int sparse_mode, "
      "Tensor(a!) output) -> Tensor(a!)");
  m.def(
      "sparse_flash_attention_lse(Tensor query, Tensor key, Tensor value, "
      "Tensor sparse_indices, Tensor? block_table, Tensor? "
      "actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? "
      "query_rope, Tensor? key_rope, float scale_value, int sparse_block_size, "
      "str layout_query, str layout_kv, int sparse_mode, int "
      "pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
      "int attention_mode=2, bool return_softmax_lse=False) -> (Tensor, "
      "Tensor, Tensor)");
  m.def(
      "build_cp_context(int[] seq_lens, int cp_size, int cp_rank, Device "
      "device) -> (Tensor shard_index, Tensor shard_gather_index, Tensor "
      "shard_valid_mask, Tensor restore_index, Tensor query_index, Tensor "
      "kv_gather_index, int[] q_cu_seqlens, int[] kv_cu_seqlens, int "
      "total_local)");
  m.def(
      "mla_preprocess_v2(Tensor input, Tensor gamma0, Tensor beta0, Tensor "
      "quant_scale0, Tensor quant_offset0, Tensor wdqkv, Tensor descale0, "
      "Tensor bias0, Tensor gamma1, Tensor beta1, Tensor quant_scale1, Tensor "
      "quant_offset1, Tensor wuq, Tensor descale1, Tensor bias1, Tensor "
      "gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor(a!) kv_cache, "
      "Tensor(b!) kv_cache_rope, Tensor slot_mapping, Tensor ctkv_scale, "
      "Tensor q_nope_scale, int wdq_dim, int q_rope_dim, int k_rope_dim, "
      "float epsilon, int q_rotary_coeff, int k_rotary_coeff, bool "
      "transpose_wdq, bool transpose_wuq, bool transpose_wuk, int "
      "cache_mode, int quant_mode, bool do_rms_norm, int "
      "wdkv_split_count, bool q_down_out_flag) -> (Tensor, Tensor(a!), "
      "Tensor, Tensor(b!), Tensor)");
  // ---- DeepSeek-V4 DSA kernels ----
  // MoE hash routing gate (returns routed output, expert_idx, token_unpermute).
  m.def(
      "moe_gating_top_k_hash(Tensor x, int k, Tensor? bias, Tensor? input_ids, "
      "Tensor? tid2eid, int k_group, int group_count, float "
      "routed_scaling_factor, "
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
}

TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m) {
  m.impl("rms_norm", TORCH_FN(xllm::rms_norm_npu));
  m.impl("rms_norm_gated", TORCH_FN(xllm::rms_norm_gated_npu));
  m.impl("l2_norm", TORCH_FN(xllm::l2_norm_npu));
  m.impl("chunk_gated_delta_rule", TORCH_FN(xllm::chunk_gated_delta_rule_npu));
  m.impl("causal_conv1d_prefill", TORCH_FN(xllm::causal_conv1d_prefill_npu));
  m.impl("causal_conv1d_qkv_prefill",
         TORCH_FN(xllm::causal_conv1d_qkv_prefill_npu));
  m.impl("fused_sigmoid_gating_delta_rule_decode",
         TORCH_FN(xllm::fused_sigmoid_gating_delta_rule_decode_npu));
  m.impl("fused_add_rms_norm", TORCH_FN(xllm::fused_add_rms_norm_npu));
  m.impl("rms_norm_dynamic_quant",
         TORCH_FN(xllm::kernel::npu::rms_norm_dynamic_quant));
  m.impl("npu_inplace_partial_rotary_mul",
         TORCH_FN(xllm::kernel::npu::npu_inplace_partial_rotary_mul));
  m.impl("silu_and_mul", TORCH_FN(xllm::silu_and_mul_npu));
  m.impl("inplace_partial_rotary_mul",
         TORCH_FN(xllm::inplace_partial_rotary_mul_npu));
  m.impl("reshape_paged_cache", TORCH_FN(xllm::reshape_paged_cache_npu));
  m.impl("apply_rotary_embedding", TORCH_FN(xllm::apply_rotary_embedding_npu));
  m.impl("update_decode_graph_metadata",
         TORCH_FN(xllm::update_decode_graph_metadata_npu));
  m.impl("quant_matmul", TORCH_FN(xllm::kernel::npu::quant_matmul));
  m.impl("quantize_per_tensor",
         TORCH_FN(xllm::kernel::npu::quantize_per_tensor));
  m.impl("dynamic_quant", TORCH_FN(xllm::kernel::npu::dynamic_quant));
  m.impl("quant_lightning_indexer",
         TORCH_FN(xllm::kernel::npu::quant_lightning_indexer));
  m.impl("quant_lightning_indexer_metadata",
         TORCH_FN(xllm::kernel::npu::quant_lightning_indexer_metadata));
  m.impl("lightning_indexer", TORCH_FN(xllm::kernel::npu::lightning_indexer));
  m.impl("lightning_indexer_out",
         TORCH_FN(xllm::kernel::npu::lightning_indexer_out));
  m.impl("scatter_nd_update", TORCH_FN(xllm::kernel::npu::scatter_nd_update));
  m.impl("sparse_flash_attention",
         TORCH_FN(xllm::kernel::npu::sparse_flash_attention));
  m.impl("sparse_flash_attention_out",
         TORCH_FN(xllm::kernel::npu::sparse_flash_attention_out));
  m.impl("mla_preprocess_v2", TORCH_FN(xllm::kernel::npu::mla_preprocess_v2));
  m.impl("moe_gating_top_k_hash",
         TORCH_FN(xllm::kernel::npu::moe_gating_top_k_hash));
  m.impl("dequant_swiglu_quant",
         TORCH_FN(xllm::kernel::npu::dequant_swiglu_quant));
  m.impl("hc_pre", TORCH_FN(xllm::kernel::npu::hc_pre));
  m.impl("hc_post", TORCH_FN(xllm::kernel::npu::hc_post));
  m.impl("compressor", TORCH_FN(xllm::kernel::npu::compressor));
  m.impl("sparse_attn_sharedkv",
         TORCH_FN(xllm::kernel::npu::sparse_attn_sharedkv));
  m.impl("sparse_flash_attention_lse",
         TORCH_FN(xllm::kernel::npu::sparse_flash_attention_lse));
}

// build_cp_context is pure host index math with no Tensor input, so the
// dispatcher cannot route it by tensor device; register it backend-agnostically
// (the target device is an explicit argument). It is eager-only (CP disables
// graph capture), so it needs no fake/meta registration.
TORCH_LIBRARY_IMPL(xllm_ops, CompositeExplicitAutograd, m) {
  m.impl("build_cp_context", TORCH_FN(xllm::build_cp_context_npu));
  // These metadata factories allow every Tensor argument to be omitted, so
  // there may be no device key to dispatch on. Their implementations select
  // the output NPU device explicitly (or inherit it from an optional Tensor).
  m.impl("sparse_attn_sharedkv_metadata",
         TORCH_FN(xllm::kernel::npu::sparse_attn_sharedkv_metadata));
  m.impl("quant_lightning_indexer_metadata",
         TORCH_FN(xllm::kernel::npu::quant_lightning_indexer_metadata));
}
