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
#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

using xllm::triton_jit::JITKernel;

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const std::optional<torch::Tensor>& beta_opt,
    const std::optional<torch::Tensor>& initial_state_opt,
    bool inplace_final_state,
    const std::optional<torch::Tensor>& cu_seqlens_opt,
    const std::optional<torch::Tensor>& ssm_state_indices_opt,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    bool use_qk_l2norm_in_kernel) {
  torch::Tensor initial_state = initial_state_opt.value_or(torch::Tensor());
  torch::Tensor cu_seqlens = cu_seqlens_opt.value_or(torch::Tensor());
  torch::Tensor ssm_state_indices =
      ssm_state_indices_opt.value_or(torch::Tensor());
  torch::Tensor num_accepted_tokens =
      num_accepted_tokens_opt.value_or(torch::Tensor());

  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();
  auto v_contig = v.contiguous();
  auto g_contig = g.contiguous().to(torch::kFloat32);
  torch::Tensor beta = beta_opt.has_value() ? beta_opt->contiguous()
                                            : torch::ones_like(g_contig);

  int32_t B = static_cast<int32_t>(k_contig.size(0));
  int32_t T = static_cast<int32_t>(k_contig.size(1));
  int32_t H = static_cast<int32_t>(k_contig.size(2));
  int32_t K = static_cast<int32_t>(k_contig.size(3));
  int32_t HV = static_cast<int32_t>(v_contig.size(2));
  int32_t V = static_cast<int32_t>(v_contig.size(3));
  int64_t beta_ndim = beta.ndimension();
  CHECK(beta_ndim == 3 || beta_ndim == 4)
      << "beta must have shape [B, T, HV] or [B, T, HV, V].";
  CHECK_EQ(beta.size(0), static_cast<int64_t>(B))
      << "beta batch size mismatch.";
  CHECK_EQ(beta.size(1), static_cast<int64_t>(T))
      << "beta sequence length mismatch.";
  CHECK_EQ(beta.size(2), static_cast<int64_t>(HV))
      << "beta value head size mismatch.";
  if (beta_ndim == 4) {
    CHECK_EQ(beta.size(3), static_cast<int64_t>(V))
        << "beta value dimension mismatch.";
  }
  int32_t N =
      cu_seqlens.numel() > 0 ? static_cast<int32_t>(cu_seqlens.size(0) - 1) : B;

  torch::Tensor o = torch::empty_like(v_contig);

  torch::Tensor ht;
  torch::Tensor h0;
  if (inplace_final_state && initial_state.numel() > 0) {
    CHECK(initial_state.scalar_type() == torch::kFloat32)
        << "In-place update requires initial_state to be float32.";
    ht = initial_state;
    h0 = initial_state;
  } else {
    auto dtype =
        initial_state.numel() > 0 ? initial_state.dtype() : v_contig.dtype();
    ht = torch::empty(
        {T, HV, V, K},
        torch::TensorOptions().dtype(dtype).device(v_contig.device()));
    if (initial_state.numel() > 0) {
      h0 = initial_state;
    }
    ht = ht.to(torch::kFloat32);
  }
  ht = ht.to(torch::kFloat32);
  if (h0.numel() > 0) {
    h0 = h0.to(torch::kFloat32);
  } else {
    h0 = torch::empty({0},
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(v_contig.device()));
  }

  int32_t stride_init_state_token =
      h0.numel() > 0 ? static_cast<int32_t>(h0.stride(0)) : 0;
  int32_t stride_final_state_token = static_cast<int32_t>(ht.stride(0));
  int32_t stride_indices_seq = 1;
  int32_t stride_indices_tok = 1;
  if (ssm_state_indices.numel() > 0) {
    if (ssm_state_indices.ndimension() == 1) {
      stride_indices_seq = static_cast<int32_t>(ssm_state_indices.stride(0));
    } else {
      stride_indices_seq = static_cast<int32_t>(ssm_state_indices.stride(0));
      stride_indices_tok = static_cast<int32_t>(ssm_state_indices.stride(1));
    }
  }

  int32_t use_initial_state = (initial_state.numel() > 0) ? 1 : 0;
  int32_t inplace_final = inplace_final_state ? 1 : 0;
  int32_t is_beta_headwise =
      (beta.ndimension() == v_contig.ndimension()) ? 1 : 0;
  int32_t use_qk_l2norm = use_qk_l2norm_in_kernel ? 1 : 0;
  int32_t is_varlen = cu_seqlens.numel() > 0 ? 1 : 0;
  int32_t is_continuous_batching = ssm_state_indices.numel() > 0 ? 1 : 0;
  int32_t is_spec_decoding = num_accepted_tokens.numel() > 0 ? 1 : 0;
  int32_t is_kda = 0;

  int32_t BV = 8;
  int32_t NV = (V + BV - 1) / BV;
  float scale = 1.0f / std::sqrt(static_cast<float>(K));
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();

  JITKernel& f = JITKernel::get(
      /*py_path=*/"torch_mlu_ops.triton.fla.fused_recurrent_fn",
      /*fn_name=*/"tmo_fused_recurrent_gated_delta_rule_fwd_kernel");

  f.launch(
      static_cast<void*>(queue),
      /*grid=*/{static_cast<uint32_t>(NV), static_cast<uint32_t>(N * HV), 1},
      /*cfg=*/{/*num_warps=*/1, /*num_stages=*/1},
      q_contig,
      k_contig,
      v_contig,
      g_contig,
      beta,
      o,
      h0,
      ht,
      cu_seqlens_opt,
      ssm_state_indices_opt,
      num_accepted_tokens_opt,
      scale,
      N,
      T,
      B,
      H,
      HV,
      K,
      V,
      /*BK=*/128,
      /*BV=*/BV,
      stride_init_state_token,
      stride_final_state_token,
      stride_indices_seq,
      stride_indices_tok,
      use_initial_state,
      inplace_final,
      is_beta_headwise,
      use_qk_l2norm,
      is_varlen,
      is_continuous_batching,
      is_spec_decoding,
      is_kda);

  return std::make_pair(o, ht);
}

}  // namespace xllm::kernel::mlu
