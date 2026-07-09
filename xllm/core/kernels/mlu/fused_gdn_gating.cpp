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

#include <cstdint>
#include <utility>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

using xllm::triton_jit::JITKernel;

std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float beta,
    float threshold) {
  int32_t batch = static_cast<int32_t>(a.size(0));
  int32_t num_heads = static_cast<int32_t>(a.size(1));
  int32_t seq_len = 1;

  // g: (1, batch, num_heads) fp32; beta_output: same shape, b's dtype (bf16).
  torch::Tensor g =
      torch::empty({1, batch, num_heads}, a.options().dtype(torch::kFloat32));
  torch::Tensor beta_output =
      torch::empty({1, batch, num_heads}, b.options().dtype(b.dtype()));

  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  int32_t core_count = prop->cluster_count * prop->core_num_per_cluster;

  int32_t blk_heads = 8;
  int32_t num_head_blocks = (num_heads + blk_heads - 1) / blk_heads;

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();

  JITKernel& f = JITKernel::get(
      /*py_path=*/"xllm.core.kernels.mlu.triton_kernel.fused_gdn_gating",
      /*fn_name=*/"tmo_fused_gdn_gating_kernel");

  f.launch(static_cast<void*>(queue),
           /*grid=*/
           {static_cast<uint32_t>(core_count),
            static_cast<uint32_t>(seq_len),
            static_cast<uint32_t>(num_head_blocks)},
           /*cfg=*/{/*num_warps=*/1, /*num_stages=*/1},
           g,
           beta_output,
           A_log,
           a,
           b,
           dt_bias,
           seq_len,
           num_heads,
           beta,
           threshold,
           blk_heads,
           core_count,
           batch);

  return std::make_pair(g, beta_output);
}

}  // namespace xllm::kernel::mlu
