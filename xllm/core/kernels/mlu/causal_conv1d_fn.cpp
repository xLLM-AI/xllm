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

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

using xllm::triton_jit::JITKernel;

torch::Tensor causal_conv1d_fn(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& batch,
    const torch::Tensor& token_block_offset,
    int32_t nt,
    const std::optional<torch::Tensor>& bias_opt,
    const std::optional<torch::Tensor>& cache_indices_opt,
    const std::optional<torch::Tensor>& has_initial_state_opt,
    const std::optional<torch::Tensor>& initial_state_idx_opt,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    bool /*inplace_final_state*/) {
  torch::Tensor out = torch::zeros_like(x);
  CHECK_GT(nt, 0);
  int32_t dim = static_cast<int32_t>(x.size(0));
  int32_t cu_seqlen = static_cast<int32_t>(x.size(1));
  int32_t width = static_cast<int32_t>(weight.size(1));
  int32_t state_len = width - 1;
  int32_t num_cache_lines = static_cast<int32_t>(conv_states.size(0));
  int32_t stride_x_dim = static_cast<int32_t>(x.stride(0));
  int32_t stride_x_token = static_cast<int32_t>(x.stride(1));
  int32_t stride_w_dim = static_cast<int32_t>(weight.stride(0));
  int32_t stride_w_width = static_cast<int32_t>(weight.stride(1));
  int32_t stride_istate_seq = static_cast<int32_t>(conv_states.stride(0));
  int32_t stride_istate_dim = static_cast<int32_t>(conv_states.stride(1));
  int32_t stride_istate_token = static_cast<int32_t>(conv_states.stride(2));
  int32_t stride_cache_indices =
      cache_indices_opt.has_value()
          ? static_cast<int32_t>(cache_indices_opt->stride(0))
          : 0;
  int32_t stride_o_dim = 0;
  int32_t stride_o_token = 0;
  if (out.dim() == 2) {
    stride_o_dim = static_cast<int32_t>(out.stride(0));
    stride_o_token = static_cast<int32_t>(out.stride(1));
  } else {
    stride_o_dim = static_cast<int32_t>(out.stride(1));
    stride_o_token = static_cast<int32_t>(out.stride(2));
  }

  constexpr int32_t kBlockM = 64;
  constexpr int32_t kBlockN = 2048;
  int32_t np2_statelen = 1;
  while (np2_statelen < state_len) {
    np2_statelen <<= 1;
  }

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  int32_t core_count = prop->cluster_count * prop->core_num_per_cluster;
  int32_t num_programs = std::min(nt, core_count);

  JITKernel& f = JITKernel::get(
      /*py_path=*/"torch_mlu_ops.triton.conv.kernels",
      /*fn_name=*/"tmo_causal_conv1d_fwd_vllm_kernel");

  f.launch(static_cast<void*>(queue),
           /*grid=*/{static_cast<uint32_t>(num_programs), 1, 1},
           /*cfg=*/{/*num_warps=*/1, /*num_stages=*/3},
           x,
           weight,
           bias_opt,
           conv_states,
           cache_indices_opt,
           has_initial_state_opt,
           query_start_loc,
           batch,
           token_block_offset,
           nullptr,
           nullptr,
           initial_state_idx_opt,
           num_accepted_tokens_opt,
           out,
           dim,
           nt,
           cu_seqlen,
           num_cache_lines,
           stride_x_dim,
           stride_x_token,
           stride_w_dim,
           stride_w_width,
           stride_istate_seq,
           stride_istate_dim,
           stride_istate_token,
           stride_cache_indices,
           stride_o_dim,
           stride_o_token,
           /*stride_block_m=*/0,
           /*pad_slot_id=*/-1,
           /*null_block_id=*/0,
           /*HAS_BIAS=*/bias_opt.has_value() ? 1 : 0,
           /*KERNEL_WIDTH=*/width,
           /*SILU_ACTIVATION=*/1,
           /*IS_APC_ENABLED=*/0,
           /*HAS_NULL_BLOCK=*/1,
           /*NP2_STATELEN=*/np2_statelen,
           /*BLOCK_M=*/kBlockM,
           /*BLOCK_N=*/kBlockN);

  return out;
}

}  // namespace xllm::kernel::mlu
