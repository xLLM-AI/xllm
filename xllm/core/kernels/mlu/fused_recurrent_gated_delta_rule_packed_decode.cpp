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
#include <utility>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

using xllm::triton_jit::JITKernel;

std::pair<torch::Tensor, torch::Tensor>
fused_recurrent_gated_delta_rule_packed_decode(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    double scale,
    torch::Tensor& ssm_cache,
    const torch::Tensor& ssm_state_indices,
    bool use_qk_l2norm_in_kernel) {
  torch::Tensor mixed_qkv_contig = mixed_qkv.contiguous();
  torch::Tensor a_contig = a.contiguous();
  torch::Tensor b_contig = b.contiguous();

  int32_t B = static_cast<int32_t>(mixed_qkv_contig.size(0));
  int32_t qkv_dim = static_cast<int32_t>(mixed_qkv_contig.size(1));
  int32_t HV = static_cast<int32_t>(ssm_cache.size(1));
  int32_t V = static_cast<int32_t>(ssm_cache.size(2));
  int32_t K = static_cast<int32_t>(ssm_cache.size(3));
  int32_t qk_dim = qkv_dim - HV * V;
  int32_t H = qk_dim / (2 * K);

  torch::Tensor out =
      torch::empty({B, 1, HV, V},
                   mixed_qkv_contig.options().dtype(mixed_qkv_contig.dtype()));

  int32_t stride_mixed_qkv_tok =
      static_cast<int32_t>(mixed_qkv_contig.stride(0));
  int32_t stride_a_tok = static_cast<int32_t>(a_contig.stride(0));
  int32_t stride_b_tok = static_cast<int32_t>(b_contig.stride(0));
  int32_t stride_init_state_token = static_cast<int32_t>(ssm_cache.stride(0));
  int32_t stride_final_state_token = static_cast<int32_t>(ssm_cache.stride(0));
  int32_t stride_indices_seq =
      static_cast<int32_t>(ssm_state_indices.stride(0));

  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  int32_t core_count = prop->cluster_count * prop->core_num_per_cluster;

  int32_t kBlockN = 1;
  constexpr int32_t kBlockHv = 1;
  constexpr int32_t kBlockV = 128;
  int32_t kBlockB = 128;
  if (B <= core_count) {
    kBlockN = 1;
  } else if (B <= core_count * 2) {
    kBlockN = 2;
  } else if (B > 128) {
    kBlockB = 1024;
    // to avoid nram out of resource
    kBlockN = HV > 6 ? 1 : 4;
  }

  int32_t num_v_blocks = (V + kBlockV - 1) / kBlockV;
  int32_t total_blocks = B * num_v_blocks;

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();

  JITKernel& f = JITKernel::get(
      /*py_path=*/
      "xllm.core.kernels.mlu.triton_kernel.fused_recurrent_gated_delta_rule_"
      "packed_decode",
      /*fn_name=*/"tmo_fused_recurrent_gated_delta_rule_packed_decode_kernel");

  f.launch(static_cast<void*>(queue),
           /*grid=*/
           {static_cast<uint32_t>(std::min(total_blocks, core_count)), 1, 1},
           /*cfg=*/{/*num_warps=*/1, /*num_stages=*/2},
           mixed_qkv_contig,
           a_contig,
           b_contig,
           A_log,
           dt_bias,
           out,
           ssm_cache,
           ssm_cache,
           ssm_state_indices,
           static_cast<float>(scale),
           stride_mixed_qkv_tok,
           stride_a_tok,
           stride_b_tok,
           stride_init_state_token,
           stride_final_state_token,
           stride_indices_seq,
           B,
           H,
           HV,
           K,
           V,
           /*BLOCK_N=*/kBlockN,
           /*BLOCK_HV=*/kBlockHv,
           /*BLOCK_V=*/kBlockV,
           /*BLOCK_K=*/K,
           /*SOFTPLUS_THRESHOLD=*/20.0f,
           /*USE_QK_L2NORM_IN_KERNEL=*/use_qk_l2norm_in_kernel ? 1 : 0,
           /*BLOCK_B=*/kBlockB);

  return std::make_pair(out, ssm_cache);
}

}  // namespace xllm::kernel::mlu
