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
#include <tuple>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

using xllm::triton_jit::JITKernel;

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
fused_post_conv_prep(const torch::Tensor& conv_output,
                     const torch::Tensor& a,
                     const torch::Tensor& b,
                     const torch::Tensor& A_log,
                     const torch::Tensor& dt_bias,
                     int64_t num_k_heads,
                     int64_t head_k_dim,
                     int64_t head_v_dim,
                     bool apply_l2norm,
                     bool output_g_exp) {
  CHECK_EQ(conv_output.dim(), 2) << "conv_output must be 2D [L, qkv_dim]";
  CHECK_EQ(a.dim(), 2) << "a must be 2D [L, num_v_heads]";
  CHECK_EQ(b.dim(), 2) << "b must be 2D [L, num_v_heads]";
  CHECK_EQ(A_log.dim(), 1) << "A_log must be 1D [num_v_heads]";
  CHECK_EQ(dt_bias.dim(), 1) << "dt_bias must be 1D [num_v_heads]";
  CHECK(!output_g_exp) << "fused_post_conv_prep outputs g, not exp(g)";

  int64_t num_tokens = conv_output.size(0);
  int64_t num_v_heads = A_log.size(0);
  int64_t qkv_dim = conv_output.size(1);
  CHECK_EQ(a.size(0), num_tokens) << "a token dimension mismatch";
  CHECK_EQ(b.size(0), num_tokens) << "b token dimension mismatch";
  CHECK_EQ(a.size(1), num_v_heads) << "a head dimension mismatch";
  CHECK_EQ(b.size(1), num_v_heads) << "b head dimension mismatch";
  CHECK_EQ(dt_bias.size(0), num_v_heads) << "dt_bias head dimension mismatch";
  CHECK_EQ(qkv_dim, 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim)
      << "conv_output qkv_dim mismatch";

  torch::Tensor q = torch::empty({num_tokens, num_k_heads, head_k_dim},
                                 conv_output.options());
  torch::Tensor k = torch::empty({num_tokens, num_k_heads, head_k_dim},
                                 conv_output.options());
  torch::Tensor v = torch::empty({num_tokens, num_v_heads, head_v_dim},
                                 conv_output.options());
  torch::Tensor g = torch::empty({num_tokens, num_v_heads},
                                 conv_output.options().dtype(torch::kFloat32));
  torch::Tensor beta = torch::empty(
      {num_tokens, num_v_heads}, conv_output.options().dtype(torch::kFloat32));

  if (num_tokens == 0) {
    return std::make_tuple(q, k, v, g, beta);
  }

  constexpr int64_t kBlockTokens = 512;
  constexpr int32_t kNumWarps = 4;
  int64_t num_token_blocks = (num_tokens + kBlockTokens - 1) / kBlockTokens;
  int64_t total_blocks = num_token_blocks * (num_k_heads + num_v_heads);
  constexpr int32_t kNumStages = 1;
  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  int64_t core_count = prop->cluster_count * prop->core_num_per_cluster;

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  JITKernel& f = JITKernel::get(
      /*py_path=*/"xllm.core.kernels.mlu.triton_kernel.fused_post_conv_prep",
      /*fn_name=*/"tmo_fused_post_conv_prep_kernel");

  f.launch(
      static_cast<void*>(queue),
      /*grid=*/
      {static_cast<uint32_t>(std::min(total_blocks, core_count / kNumWarps)),
       1,
       1},
      /*cfg=*/{/*num_warps=*/kNumWarps, /*num_stages=*/kNumStages},
      conv_output,
      a,
      b,
      A_log,
      dt_bias,
      q,
      k,
      v,
      g,
      beta,
      static_cast<int32_t>(conv_output.stride(0)),
      static_cast<int32_t>(a.stride(0)),
      static_cast<int32_t>(b.stride(0)),
      static_cast<int32_t>(q.stride(0)),
      static_cast<int32_t>(k.stride(0)),
      static_cast<int32_t>(v.stride(0)),
      static_cast<int32_t>(num_tokens),
      static_cast<int32_t>(num_k_heads),
      static_cast<int32_t>(num_v_heads),
      static_cast<int32_t>(head_k_dim),
      static_cast<int32_t>(head_v_dim),
      /*APPLY_L2NORM=*/apply_l2norm ? 1 : 0,
      /*L2NORM_EPS=*/1.0e-6f,
      /*SOFTPLUS_THRESHOLD=*/20.0f,
      /*BLOCK_T=*/static_cast<int32_t>(kBlockTokens),
      /*BK=*/static_cast<int32_t>(head_k_dim),
      /*BV=*/static_cast<int32_t>(head_v_dim));

  return std::make_tuple(q, k, v, g, beta);
}

}  // namespace xllm::kernel::mlu
