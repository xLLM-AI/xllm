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
#include <mutex>
#include <optional>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {

using xllm::triton_jit::JITKernel;

torch::Tensor causal_conv1d_update_decode(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias_opt,
    const torch::Tensor& conv_state_indices,
    int32_t pad_slot_id,
    const std::optional<torch::Tensor>& query_start_loc_opt,
    int32_t /*max_query_len*/,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token_opt,
    const std::optional<torch::Tensor>& initial_state_idx_opt) {
  bool unsqueeze = (x.dim() == 2);
  torch::Tensor x_input = x;
  if (unsqueeze) {
    x_input = x.unsqueeze(-1);
  }

  int32_t batch = static_cast<int32_t>(conv_state_indices.size(0));
  int32_t dim = static_cast<int32_t>(x_input.size(1));
  int32_t seqlen = static_cast<int32_t>(x_input.size(2));
  int32_t width = static_cast<int32_t>(weight.size(1));
  int32_t state_len = width - 1;
  int32_t num_cache_lines = static_cast<int32_t>(conv_state.size(0));

  torch::Tensor out = torch::empty_like(x_input);

  int32_t stride_x_seq = static_cast<int32_t>(x_input.stride(0));
  int32_t stride_x_dim = static_cast<int32_t>(x_input.stride(1));
  int32_t stride_x_token = static_cast<int32_t>(x_input.stride(2));
  int32_t stride_w_dim = static_cast<int32_t>(weight.stride(0));
  int32_t stride_w_width = static_cast<int32_t>(weight.stride(1));
  int32_t stride_istate_seq = static_cast<int32_t>(conv_state.stride(0));
  int32_t stride_istate_dim = static_cast<int32_t>(conv_state.stride(1));
  int32_t stride_istate_tok = static_cast<int32_t>(conv_state.stride(2));
  int32_t stride_state_indices =
      static_cast<int32_t>(conv_state_indices.stride(0));
  int32_t stride_o_seq = static_cast<int32_t>(out.stride(0));
  int32_t stride_o_dim = static_cast<int32_t>(out.stride(1));
  int32_t stride_o_token = static_cast<int32_t>(out.stride(2));

  int32_t bd = 8;
  int32_t num_feature_blocks = (dim + bd - 1) / bd;
  // NP2_STATELEN: next power of two >= state_len.
  int32_t np2_statelen = 1;
  while (np2_statelen < state_len) {
    np2_statelen <<= 1;
  }

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();

  JITKernel& f = JITKernel::get(
      /*py_path=*/
      "xllm.core.kernels.mlu.triton_kernel.causal_conv1d_update_decode",
      /*fn_name=*/"tmo_causal_conv1d_update_decode_kernel");

  f.launch(static_cast<void*>(queue),
           /*grid=*/
           {static_cast<uint32_t>(num_feature_blocks),
            static_cast<uint32_t>(batch),
            1},
           /*cfg=*/{/*num_warps=*/1, /*num_stages=*/1},
           x,
           weight,
           bias_opt,
           conv_state,
           conv_state_indices,
           num_accepted_tokens_opt,
           query_start_loc_opt,
           block_idx_last_scheduled_token_opt,
           initial_state_idx_opt,
           out,
           batch,
           num_cache_lines,
           dim,
           seqlen,
           state_len,
           stride_x_seq,
           stride_x_dim,
           stride_x_token,
           stride_w_dim,
           stride_w_width,
           stride_istate_seq,
           stride_istate_dim,
           stride_istate_tok,
           stride_state_indices,
           stride_o_seq,
           stride_o_dim,
           stride_o_token,
           pad_slot_id,
           /*HAS_BIAS=*/bias_opt.has_value() ? 1 : 0,
           /*KERNEL_WIDTH=*/width,
           /*SILU_ACTIVATION=*/1,
           /*IS_VARLEN=*/0,
           /*IS_APC_ENABLED=*/0,
           /*IS_SPEC_DECODING=*/0,
           /*NP2_STATELEN=*/np2_statelen,
           /*USE_PAD_SLOT=*/1,
           /*BD=*/bd,
           /*BW=*/4);

  if (unsqueeze) {
    out = out.squeeze(-1);
  }
  return out;
}

}  // namespace xllm::kernel::mlu
