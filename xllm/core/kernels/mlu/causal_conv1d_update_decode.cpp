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

torch::Tensor causal_conv1d_update_decode(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias_opt,
    const std::optional<torch::Tensor>& conv_state_indices_opt,
    bool activation,
    int32_t pad_slot_id,
    const std::optional<torch::Tensor>& query_start_loc_opt,
    int32_t max_query_len,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token_opt,
    const std::optional<torch::Tensor>& initial_state_idx_opt) {
  torch::Tensor x_input = x.to(conv_state.dtype());
  bool is_varlen = query_start_loc_opt.has_value();
  bool unsqueeze = (!is_varlen && x_input.dim() == 2);
  if (unsqueeze) {
    x_input = x_input.unsqueeze(-1);
  }

  int32_t batch = 0;
  int32_t dim = 0;
  int32_t seqlen = 0;
  if (is_varlen) {
    CHECK_EQ(x_input.dim(), 2) << "varlen x must be 2D [num_tokens, dim]";
    CHECK_GT(max_query_len, 0) << "varlen path requires max_query_len > 0";
    batch = static_cast<int32_t>(query_start_loc_opt.value().size(0) - 1);
    dim = static_cast<int32_t>(x_input.size(1));
    seqlen = max_query_len;
  } else {
    CHECK_EQ(x_input.dim(), 3) << "x must be 3D [batch, dim, seqlen]";
    batch = static_cast<int32_t>(x_input.size(0));
    dim = static_cast<int32_t>(x_input.size(1));
    seqlen = static_cast<int32_t>(x_input.size(2));
  }
  int32_t width = static_cast<int32_t>(weight.size(1));
  int32_t state_len =
      num_accepted_tokens_opt.has_value() ? width - 1 + seqlen - 1 : width - 1;
  int32_t num_cache_lines = static_cast<int32_t>(conv_state.size(0));

  torch::Tensor out = torch::empty_like(x_input);
  torch::Tensor conv_state_indices =
      conv_state_indices_opt.has_value()
          ? conv_state_indices_opt.value().contiguous().to(torch::kInt32)
          : torch::arange(batch,
                          torch::TensorOptions()
                              .dtype(torch::kInt32)
                              .device(x_input.device()));
  torch::Tensor query_start_loc =
      query_start_loc_opt.has_value()
          ? query_start_loc_opt.value().contiguous().to(torch::kInt32)
          : torch::Tensor();
  torch::Tensor num_accepted_tokens =
      num_accepted_tokens_opt.has_value()
          ? num_accepted_tokens_opt.value().contiguous().to(torch::kInt32)
          : torch::Tensor();
  bool is_apc = block_idx_last_scheduled_token_opt.has_value();
  torch::Tensor block_idx_last_scheduled_token =
      is_apc ? block_idx_last_scheduled_token_opt.value().contiguous().to(
                   torch::kInt32)
             : torch::Tensor();
  torch::Tensor initial_state_idx =
      initial_state_idx_opt.has_value()
          ? initial_state_idx_opt.value().contiguous().to(torch::kInt32)
          : torch::Tensor();
  std::optional<torch::Tensor> query_start_loc_arg =
      is_varlen ? std::make_optional(query_start_loc) : std::nullopt;
  std::optional<torch::Tensor> num_accepted_tokens_arg =
      num_accepted_tokens_opt.has_value()
          ? std::make_optional(num_accepted_tokens)
          : std::nullopt;
  std::optional<torch::Tensor> block_idx_last_scheduled_token_arg =
      is_apc ? std::make_optional(block_idx_last_scheduled_token)
             : std::nullopt;
  std::optional<torch::Tensor> initial_state_idx_arg =
      initial_state_idx_opt.has_value() ? std::make_optional(initial_state_idx)
                                        : std::nullopt;

  int32_t stride_x_seq =
      is_varlen ? 0 : static_cast<int32_t>(x_input.stride(0));
  int32_t stride_x_dim = static_cast<int32_t>(x_input.stride(1));
  int32_t stride_x_token = is_varlen ? static_cast<int32_t>(x_input.stride(0))
                                     : static_cast<int32_t>(x_input.stride(2));
  int32_t stride_w_dim = static_cast<int32_t>(weight.stride(0));
  int32_t stride_w_width = static_cast<int32_t>(weight.stride(1));
  int32_t stride_istate_seq = static_cast<int32_t>(conv_state.stride(0));
  int32_t stride_istate_dim = static_cast<int32_t>(conv_state.stride(1));
  int32_t stride_istate_tok = static_cast<int32_t>(conv_state.stride(2));
  int32_t stride_state_indices =
      static_cast<int32_t>(conv_state_indices.stride(0));
  int32_t stride_o_seq = is_varlen ? 0 : static_cast<int32_t>(out.stride(0));
  int32_t stride_o_dim = static_cast<int32_t>(out.stride(1));
  int32_t stride_o_token = is_varlen ? static_cast<int32_t>(out.stride(0))
                                     : static_cast<int32_t>(out.stride(2));

  int32_t block_n = dim;
  constexpr int32_t kBlockB = 1024;
  int32_t num_feature_blocks = (dim + block_n - 1) / block_n;
  // NP2_STATELEN: next power of two >= state_len.
  int32_t np2_statelen = 1;
  while (np2_statelen < state_len) {
    np2_statelen <<= 1;
  }

  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  torch_mlu::DeviceProp* prop =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(prop != nullptr);
  int32_t core_count = prop->cluster_count * prop->core_num_per_cluster;

  JITKernel& f = JITKernel::get(
      /*py_path=*/
      "xllm.core.kernels.mlu.triton_kernel.causal_conv1d_update_decode",
      /*fn_name=*/"tmo_causal_conv1d_update_decode_kernel");

  f.launch(
      static_cast<void*>(queue),
      /*grid=*/
      {static_cast<uint32_t>(std::min(num_feature_blocks * batch, core_count)),
       1,
       1},
      /*cfg=*/{/*num_warps=*/1, /*num_stages=*/4},
      x_input,
      weight,
      bias_opt,
      conv_state,
      conv_state_indices,
      num_accepted_tokens_arg,
      query_start_loc_arg,
      block_idx_last_scheduled_token_arg,
      initial_state_idx_arg,
      out,
      batch,
      dim,
      seqlen,
      state_len,
      num_cache_lines,
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
      /*SILU_ACTIVATION=*/activation ? 1 : 0,
      /*IS_VARLEN=*/is_varlen ? 1 : 0,
      /*IS_APC_ENABLED=*/is_apc ? 1 : 0,
      /*IS_SPEC_DECODING=*/num_accepted_tokens_opt.has_value() ? 1 : 0,
      /*NP2_STATELEN=*/np2_statelen,
      /*HAS_NULL_BLOCK=*/(pad_slot_id >= 0 && is_apc) ? 1 : 0,
      /*BLOCK_N=*/block_n,
      /*BLOCK_B=*/kBlockB);

  if (unsqueeze) {
    out = out.squeeze(-1);
  }
  return out.to(x.dtype());
}

}  // namespace xllm::kernel::mlu
