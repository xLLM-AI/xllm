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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

namespace {

constexpr int64_t kMaxBlockHv = 32;
constexpr int64_t kMaxBlockN = 4;
constexpr int64_t kBlockQueryLen = 4;
constexpr int64_t kSplitBlockV = 64;

int64_t choose_block_hv(int64_t num_k_heads,
                        int64_t num_v_heads,
                        int64_t max_block_hv) {
  CHECK_GT(num_k_heads, 0);
  CHECK_GE(num_v_heads, num_k_heads);
  CHECK_EQ(num_v_heads % num_k_heads, 0) << "HV must be divisible by H";

  int64_t heads_per_query = num_v_heads / num_k_heads;
  int64_t candidate = std::min<int64_t>(
      (max_block_hv / heads_per_query) * heads_per_query, num_v_heads);
  for (; candidate >= heads_per_query; candidate -= heads_per_query) {
    if (num_v_heads % candidate == 0) {
      return candidate;
    }
  }
  LOG(FATAL) << "Failed to select BLOCK_HV for H=" << num_k_heads
             << ", HV=" << num_v_heads;
  return heads_per_query;
}

int64_t choose_kda_head_group(int64_t num_sequences,
                              int64_t num_heads,
                              int64_t core_count) {
  int64_t best_group = num_heads;
  int64_t best_work =
      ((num_sequences + core_count - 1) / core_count) * num_heads;
  // Equal work prefers larger contiguous head tiles, reducing fragmented IO.
  for (int64_t group = num_heads / 2; group > 0; group /= 2) {
    int64_t jobs = num_sequences * (num_heads / group);
    int64_t work = ((jobs + core_count - 1) / core_count) * group;
    if (work < best_work) {
      best_group = group;
      best_work = work;
    }
  }
  return best_group;
}

}  // namespace

using xllm::triton_jit::JITKernel;

std::pair<torch::Tensor, torch::Tensor> fused_sigmoid_gating_delta_rule_update(
    const torch::Tensor& A_log,
    torch::Tensor& a,
    torch::Tensor& b,
    const torch::Tensor& dt_bias,
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& initial_state,
    torch::Tensor& ssm_state_indices,
    torch::Tensor& cu_seqlens,
    double scale,
    bool use_qk_l2norm_in_kernel,
    float softplus_beta,
    float softplus_threshold,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    bool inplace_final_state,
    bool is_kda,
    bool kda_use_safe_gate,
    float kda_gate_lower_bound) {
  CHECK_EQ(q.dim(), 4) << "q must be 4D [B, T, H, K]";
  CHECK_EQ(k.dim(), 4) << "k must be 4D [B, T, H, K]";
  CHECK_EQ(v.dim(), 4) << "v must be 4D [B, T, HV, V]";
  CHECK_EQ(A_log.dim(), 1) << "A_log must be 1D [HV]";
  CHECK_EQ(dt_bias.dim(), 1) << "dt_bias must be 1D [HV] or [HV * K]";
  CHECK_EQ(a.dim(), 2) << "a must be 2D [tokens, HV] or [tokens, HV * K]";
  CHECK_EQ(b.dim(), 2) << "b must be 2D [tokens, HV]";
  CHECK(!kda_use_safe_gate || is_kda) << "safe KDA gate requires is_kda=true";
  if (kda_use_safe_gate) {
    CHECK_LT(kda_gate_lower_bound, 0.0f)
        << "KDA gate lower bound must be negative";
  }

  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  a = a.contiguous();
  b = b.contiguous();
  initial_state = initial_state;
  int64_t batch_size = k.size(0);
  int64_t seq_len = k.size(1);
  int64_t num_k_heads = k.size(2);
  int64_t head_k_dim = k.size(3);
  int64_t num_v_heads = v.size(2);
  int64_t head_v_dim = v.size(3);
  torch::Tensor query_start_loc =
      cu_seqlens.defined()
          ? cu_seqlens.contiguous().to(torch::kInt32)
          : torch::arange(
                0,
                (batch_size + 1) * std::max<int64_t>(seq_len, 1),
                std::max<int64_t>(seq_len, 1),
                torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
  int64_t num_sequences = query_start_loc.size(0) - 1;
  if (cu_seqlens.defined()) {
    CHECK_EQ(batch_size, 1) << "cu_seqlens path expects flattened batch size 1";
  }
  CHECK_EQ(q.sizes(), k.sizes()) << "q/k shape mismatch";
  CHECK_EQ(v.size(0), batch_size) << "v batch dimension mismatch";
  CHECK_EQ(v.size(1), seq_len) << "v sequence dimension mismatch";
  CHECK_EQ(A_log.size(0), num_v_heads) << "A_log head dimension mismatch";
  CHECK_EQ(dt_bias.size(0), is_kda ? num_v_heads * head_k_dim : num_v_heads)
      << "dt_bias head dimension mismatch";
  CHECK_EQ(a.size(0), batch_size * seq_len) << "a token dimension mismatch";
  CHECK_EQ(b.size(0), batch_size * seq_len) << "b token dimension mismatch";
  CHECK_EQ(a.size(1), is_kda ? num_v_heads * head_k_dim : num_v_heads)
      << "a head dimension mismatch";
  CHECK_EQ(b.size(1), num_v_heads) << "b head dimension mismatch";
  CHECK_EQ(initial_state.dim(), 4)
      << "initial_state must be 4D [slots, HV, V, K]";
  CHECK_EQ(initial_state.size(1), num_v_heads)
      << "initial_state head dimension mismatch";
  CHECK_EQ(initial_state.size(2), head_v_dim)
      << "initial_state value dimension mismatch";
  CHECK_EQ(initial_state.size(3), head_k_dim)
      << "initial_state key dimension mismatch";

  // Continuous batching can skip invalid state rows, and variable-length
  // batches can leave padded positions unwritten. Only those paths need a
  // zeroed output; dense fixed-length batches write every output element.
  const std::vector<int64_t> output_shape = {
      1, batch_size, seq_len, num_v_heads, head_v_dim};
  const torch::TensorOptions output_options = v.options().dtype(v.dtype());
  torch::Tensor out_storage =
      (ssm_state_indices.defined() || cu_seqlens.defined())
          ? torch::zeros(output_shape, output_options)
          : torch::empty(output_shape, output_options);
  torch::Tensor out = out_storage.select(/*dim=*/0, /*index=*/0);
  torch::Tensor final_state =
      inplace_final_state
          ? initial_state
          : torch::empty(
                {batch_size * seq_len, num_v_heads, head_v_dim, head_k_dim},
                initial_state.options());
  if (seq_len == 0 || num_sequences == 0) {
    return std::make_pair(out, final_state);
  }

  torch::Tensor num_accepted_tokens =
      num_accepted_tokens_opt.has_value()
          ? num_accepted_tokens_opt.value().contiguous().to(torch::kInt32)
          : torch::Tensor();
  std::optional<torch::Tensor> num_accepted_tokens_arg =
      num_accepted_tokens_opt.has_value()
          ? std::make_optional(num_accepted_tokens)
          : std::nullopt;
  torch::Tensor state_indices;
  if (ssm_state_indices.defined()) {
    state_indices = ssm_state_indices.contiguous().to(torch::kInt32);
  } else if (inplace_final_state) {
    torch::Tensor token_offsets =
        torch::arange(seq_len, query_start_loc.options());
    torch::Tensor seq_offsets =
        query_start_loc.slice(/*dim=*/0, /*start=*/0, /*end=*/num_sequences)
            .unsqueeze(/*dim=*/1);
    state_indices = seq_offsets + token_offsets.unsqueeze(/*dim=*/0);
  } else {
    state_indices = torch::zeros({num_sequences, 1}, query_start_loc.options());
  }
  int64_t stride_indices_seq = 1;
  int64_t stride_indices_tok = 1;
  if (state_indices.dim() == 1) {
    stride_indices_seq = state_indices.stride(0);
  } else {
    stride_indices_seq = state_indices.stride(0);
    stride_indices_tok = state_indices.stride(1);
  }

  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  int64_t core_count = prop->cluster_count * prop->core_num_per_cluster;

  // Small batches need more independent tiles. Low checkpoint-slot occupancy
  // also benefits from scheduling heads separately. Occupancy only selects a
  // kernel; token boundaries always come from query_start_loc on the device.
  const bool sparse_checkpoints =
      num_accepted_tokens_opt.has_value() && state_indices.dim() == 2 &&
      state_indices.size(1) > 1 &&
      batch_size * seq_len * 4 < num_sequences * state_indices.size(1) * 3;
  const bool use_glm_kda =
      is_kda && kda_use_safe_gate && use_qk_l2norm_in_kernel &&
      ssm_state_indices.defined() && num_k_heads == 8 && num_v_heads == 8 &&
      head_k_dim == 128 && head_v_dim == 128 && initial_state.is_contiguous() &&
      A_log.is_contiguous() && dt_bias.is_contiguous();
  int64_t kda_head_group =
      use_glm_kda
          ? choose_kda_head_group(num_sequences, num_v_heads, core_count)
          : num_v_heads;
  // A single head cannot amortize the grouped kernel's token preloading.
  // Underfilled batches with few tokens similarly favor independent tiles.
  // Token counts select a layout only; the kernel still reads each CU boundary.
  const bool short_underfilled_batch = num_sequences < core_count &&
                                       batch_size * seq_len <= num_sequences &&
                                       kda_head_group < num_v_heads;
  const bool use_direct_kda =
      use_glm_kda && (num_sequences <= core_count / 2 || sparse_checkpoints ||
                      kda_head_group == 1 || short_underfilled_batch);
  if (use_direct_kda) {
    // Sparse verification keeps a full value tile even for one sequence.
    int64_t direct_block_v =
        num_sequences == 1 && !sparse_checkpoints ? 32 : 128;
    int64_t direct_tiles = num_sequences * num_v_heads *
                           ((head_v_dim + direct_block_v - 1) / direct_block_v);
    cnrtQueue_t queue = torch_mlu::getCurMLUStream();
    JITKernel& direct_kernel = JITKernel::get(
        /*py_path=*/"xllm.core.kernels.mlu.triton_kernel.fused_recurrent_kda",
        /*fn_name=*/"fused_recurrent_kda_kernel");
    direct_kernel.launch(
        static_cast<void*>(queue),
        /*grid=*/{static_cast<uint32_t>(direct_tiles), 1, 1},
        /*cfg=*/{/*num_warps=*/1, /*num_stages=*/3},
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        final_state,
        out,
        query_start_loc,
        state_indices,
        num_accepted_tokens_arg,
        /*N=*/static_cast<int32_t>(num_sequences),
        /*H=*/static_cast<int32_t>(num_k_heads),
        /*HV=*/static_cast<int32_t>(num_v_heads),
        /*DK=*/static_cast<int32_t>(head_k_dim),
        /*DV=*/static_cast<int32_t>(head_v_dim),
        /*STRIDE_INDICES_SEQ=*/static_cast<int32_t>(stride_indices_seq),
        /*STRIDE_INDICES_TOK=*/static_cast<int32_t>(stride_indices_tok),
        /*SCALE=*/static_cast<float>(scale),
        /*LOWER_BOUND=*/kda_gate_lower_bound,
        /*SPEC=*/num_accepted_tokens_opt.has_value() ? 1 : 0,
        /*INPLACE=*/inplace_final_state ? 1 : 0,
        /*BV=*/static_cast<int32_t>(direct_block_v),
        /*BK=*/static_cast<int32_t>(head_k_dim));
    return std::make_pair(out, final_state);
  }

  int64_t block_k = head_k_dim;
  int64_t block_v = std::min<int64_t>(head_v_dim, 128);
  int64_t block_n = 1;
  // KDA carries an additional gate tile per sequence. Keeping one sequence per
  // tile prevents the GLM5 graph warmup shape from exceeding MLU590 NRAM.
  if (!is_kda) {
    if (num_sequences > core_count * 2) {
      block_n = kMaxBlockN;
    } else if (num_sequences > core_count) {
      block_n = 2;
    }
  }
  int64_t max_block_hv = kMaxBlockHv / block_n;

  int64_t block_hv = choose_block_hv(num_k_heads, num_v_heads, max_block_hv);
  // A single-token KDA update has no recurrence across heads or value rows.
  // Split both dimensions so each program owns a disjoint state tile; keep K
  // whole because splitting the reduction would require synchronization.
  const bool split_single_token = is_kda && num_sequences == 1 && seq_len == 1;
  if (split_single_token) {
    block_hv = num_v_heads / num_k_heads;
    int64_t num_hv_blocks = num_v_heads / block_hv;
    int64_t num_v_blocks = (head_v_dim + block_v - 1) / block_v;
    if (num_hv_blocks * num_v_blocks < core_count) {
      block_v = std::min<int64_t>(head_v_dim, kSplitBlockV);
    }
  }
  const bool use_factored_kda_reduce =
      use_glm_kda && block_k == 128 && block_v == 128 && block_n == 1 &&
      q.scalar_type() == torch::kBFloat16 &&
      k.scalar_type() == torch::kBFloat16 &&
      v.scalar_type() == torch::kBFloat16 &&
      a.scalar_type() == torch::kBFloat16 &&
      b.scalar_type() == torch::kBFloat16 &&
      initial_state.scalar_type() == torch::kFloat32 &&
      A_log.scalar_type() == torch::kFloat32 &&
      dt_bias.scalar_type() == torch::kFloat32;
  if (use_glm_kda) {
    // Three-dimensional update broadcasting keeps all head groups within NRAM.
    block_hv = kda_head_group;
  }
  const bool split_hv =
      split_single_token || (use_glm_kda && block_hv < num_v_heads);
  int64_t num_hv_blocks =
      split_hv ? (num_v_heads + block_hv - 1) / block_hv : 1;
  int64_t total_blocks = ((head_k_dim + block_k - 1) / block_k) *
                         ((head_v_dim + block_v - 1) / block_v) *
                         num_sequences * num_hv_blocks;

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  JITKernel& f = JITKernel::get(
      /*py_path=*/
      "xllm.core.kernels.mlu.triton_kernel.fused_sigmoid_gating_delta_rule_"
      "update",
      /*fn_name=*/"tmo_fused_sigmoid_gating_delta_rule_update_kernel");

  f.launch(static_cast<void*>(queue),
           /*grid=*/
           {static_cast<uint32_t>(std::min(total_blocks, core_count)), 1, 1},
           /*cfg=*/{/*num_warps=*/1, /*num_stages=*/3},
           A_log,
           a,
           b,
           dt_bias,
           softplus_beta,
           softplus_threshold,
           q,
           k,
           v,
           out_storage,
           initial_state,
           final_state,
           query_start_loc,
           state_indices,
           num_accepted_tokens_arg,
           static_cast<float>(scale),
           static_cast<int64_t>(num_sequences),
           static_cast<int64_t>(seq_len),
           static_cast<int32_t>(batch_size),
           static_cast<int32_t>(num_k_heads),
           static_cast<int32_t>(num_v_heads),
           /*BLOCK_HV=*/static_cast<int32_t>(block_hv),
           static_cast<int32_t>(head_k_dim),
           static_cast<int32_t>(head_v_dim),
           /*BK=*/static_cast<int32_t>(block_k),
           /*BV=*/static_cast<int32_t>(block_v),
           static_cast<int64_t>(initial_state.stride(0)),
           static_cast<int64_t>(final_state.stride(0)),
           stride_indices_seq,
           stride_indices_tok,
           /*USE_INITIAL_STATE=*/1,
           /*INPLACE_FINAL_STATE=*/inplace_final_state ? 1 : 0,
           /*USE_QK_L2NORM_IN_KERNEL=*/use_qk_l2norm_in_kernel ? 1 : 0,
           /*IS_VARLEN=*/cu_seqlens.defined() ? 1 : 0,
           /*IS_CONTINUOUS_BATCHING=*/ssm_state_indices.defined() ? 1 : 0,
           /*IS_SPEC_DECODING=*/num_accepted_tokens_opt.has_value() ? 1 : 0,
           /*IS_KDA=*/is_kda ? 1 : 0,
           /*KDA_USE_SAFE_GATE=*/kda_use_safe_gate ? 1 : 0,
           kda_gate_lower_bound,
           /*SPLIT_HV=*/split_hv ? 1 : 0,
           /*BLOCK_N=*/static_cast<int32_t>(block_n),
           /*BLOCK_QUERY_LEN=*/static_cast<int32_t>(kBlockQueryLen),
           /*FACTORED_REDUCE=*/use_factored_kda_reduce ? 1 : 0);

  return std::make_pair(out, final_state);
}

}  // namespace xllm::kernel::mlu
