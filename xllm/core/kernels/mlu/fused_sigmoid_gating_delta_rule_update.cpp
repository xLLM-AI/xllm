/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

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
    bool is_kda) {
  CHECK_EQ(q.dim(), 4) << "q must be 4D [B, T, H, K]";
  CHECK_EQ(k.dim(), 4) << "k must be 4D [B, T, H, K]";
  CHECK_EQ(v.dim(), 4) << "v must be 4D [B, T, HV, V]";
  CHECK_EQ(A_log.dim(), 1) << "A_log must be 1D [HV]";
  CHECK_EQ(dt_bias.dim(), 1) << "dt_bias must be 1D [HV] or [HV * K]";
  CHECK_EQ(a.dim(), 2) << "a must be 2D [tokens, HV] or [tokens, HV * K]";
  CHECK_EQ(b.dim(), 2) << "b must be 2D [tokens, HV]";

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

  torch::Tensor out_storage =
      torch::empty({1, batch_size, seq_len, num_v_heads, head_v_dim},
                   v.options().dtype(v.dtype()));
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

  int64_t block_k = head_k_dim;
  int64_t block_v = std::min<int64_t>(head_v_dim, 128);
  int64_t block_hv = std::min<int64_t>(num_v_heads, 4);

  int32_t kBlockN = 1;
  int32_t kMaxN = 128;
  if (batch_size <= core_count) {
    kBlockN = 1;
  } else if (batch_size <= core_count * 2) {
    kBlockN = 2;
  } else if (batch_size > 128) {
    kMaxN = 1024;
    // to avoid nram out of resource
    kBlockN = num_v_heads > 6 ? 1 : 4;
  }

  int64_t block_query_len = seq_len;
  int64_t total_blocks = ((head_k_dim + block_k - 1) / block_k) *
                         ((head_v_dim + block_v - 1) / block_v) *
                         ((num_v_heads + block_hv - 1) / block_hv) *
                         num_sequences;

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  JITKernel& f = JITKernel::get(
      /*py_path=*/
      "xllm.core.kernels.mlu.triton_kernel.fused_sigmoid_gating_delta_rule_"
      "update",
      /*fn_name=*/"tmo_fused_sigmoid_gating_delta_rule_update_kernel");

  f.launch(static_cast<void*>(queue),
           /*grid=*/
           {static_cast<uint32_t>(std::min(total_blocks, core_count)), 1, 1},
           /*cfg=*/{/*num_warps=*/1, /*num_stages=*/2},
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
           /*MAX_N=*/static_cast<int32_t>(kMaxN),
           /*BLOCK_N=*/static_cast<int32_t>(kBlockN),
           /*BLOCK_QUERY_LEN=*/static_cast<int32_t>(block_query_len),
           /*BLOCK_HV=*/static_cast<int32_t>(block_hv));

  return std::make_pair(out, final_state);
}

}  // namespace xllm::kernel::mlu
