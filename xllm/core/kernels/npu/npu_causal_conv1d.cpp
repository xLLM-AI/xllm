/* Copyright 2025-2026 The xLLM Authors.

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

#include <optional>
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

void causal_conv1d_out(const torch::Tensor& output,
                       const torch::Tensor& x,
                       const torch::Tensor& weight,
                       const torch::Tensor& conv_state,
                       const std::optional<torch::Tensor>& bias_opt,
                       const torch::IntArrayRef query_start_loc_opt,
                       const torch::IntArrayRef cache_indices_opt,
                       const torch::IntArrayRef initial_state_mode_opt,
                       const torch::IntArrayRef num_accepted_tokens_opt,
                       int64_t activation_mode,
                       int64_t pad_slot_id,
                       int64_t run_mode) {
  check_tensor(output, "output", "causal_conv1d");
  check_tensor(x, "x", "causal_conv1d");
  check_tensor(weight, "weight", "causal_conv1d");
  check_tensor(conv_state, "conv_state", "causal_conv1d");

  std::optional<torch::Tensor> bias_tensor = std::nullopt;
  if (bias_opt.has_value() && bias_opt.value().defined()) {
    bias_tensor = bias_opt.value();
  }

  EXEC_NPU_CMD(aclnnCausalConv1d,
               x,
               weight,
               bias_tensor,
               conv_state,
               query_start_loc_opt,
               cache_indices_opt,
               initial_state_mode_opt,
               num_accepted_tokens_opt,
               activation_mode,
               pad_slot_id,
               run_mode,
               output);
}

torch::Tensor causal_conv1d(const torch::Tensor& x,
                            const torch::Tensor& weight,
                            const torch::Tensor& conv_state,
                            const std::optional<torch::Tensor>& bias_opt,
                            const torch::IntArrayRef query_start_loc_opt,
                            const torch::IntArrayRef cache_indices_opt,
                            const torch::IntArrayRef initial_state_mode_opt,
                            const torch::IntArrayRef num_accepted_tokens_opt,
                            int64_t activation_mode,
                            int64_t pad_slot_id,
                            int64_t run_mode) {
  torch::Tensor output = torch::empty(x.sizes(), x.options());
  causal_conv1d_out(output,
                    x,
                    weight,
                    conv_state,
                    bias_opt,
                    query_start_loc_opt,
                    cache_indices_opt,
                    initial_state_mode_opt,
                    num_accepted_tokens_opt,
                    activation_mode,
                    pad_slot_id,
                    run_mode);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> causal_conv1d_qkv(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_state,
    const torch::IntArrayRef query_start_loc_opt,
    const torch::IntArrayRef cache_indices_opt,
    const torch::IntArrayRef initial_state_mode_opt,
    int64_t num_qk_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  constexpr int64_t kPackedQkvActivationMode = 2;
  constexpr int64_t kPadSlotId = -1;
  constexpr int64_t kForwardRunMode = 0;

  check_tensor(x, "x", "causal_conv1d_qkv");
  check_tensor(weight, "weight", "causal_conv1d_qkv");
  check_tensor(conv_state, "conv_state", "causal_conv1d_qkv");
  CHECK_EQ(x.dim(), 2) << "causal_conv1d_qkv expects x with shape [T, D]";
  CHECK_GT(num_qk_heads, 0) << "num_qk_heads must be positive";
  CHECK_GT(num_v_heads, 0) << "num_v_heads must be positive";
  CHECK_EQ(head_k_dim, 128) << "packed QKV requires head_k_dim=128";
  CHECK_EQ(head_v_dim, head_k_dim)
      << "packed QKV requires equal Q/K and V head dimensions";
  const int64_t q_elements_per_token = num_qk_heads * head_k_dim;
  const int64_t k_elements_per_token = num_qk_heads * head_k_dim;
  const int64_t v_elements_per_token = num_v_heads * head_v_dim;
  CHECK_EQ(x.size(1),
           q_elements_per_token + k_elements_per_token + v_elements_per_token)
      << "causal_conv1d_qkv input width does not match the local Q/K/V layout";

  auto packed = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
  std::optional<torch::Tensor> bias_opt = std::nullopt;
  torch::IntArrayRef num_accepted_tokens_opt;
  EXEC_NPU_CMD(aclnnCausalConv1dQkv,
               x,
               weight,
               bias_opt,
               conv_state,
               query_start_loc_opt,
               cache_indices_opt,
               initial_state_mode_opt,
               num_accepted_tokens_opt,
               kPackedQkvActivationMode,
               kPadSlotId,
               kForwardRunMode,
               q_elements_per_token,
               k_elements_per_token,
               v_elements_per_token,
               head_k_dim,
               packed);

  const int64_t num_tokens = x.size(0);
  auto packed_flat = packed.view({-1});
  const int64_t q_elements = num_tokens * q_elements_per_token;
  const int64_t k_elements = num_tokens * k_elements_per_token;
  const int64_t v_elements = num_tokens * v_elements_per_token;
  auto q = packed_flat.narrow(0, 0, q_elements)
               .view({1, num_tokens, num_qk_heads, head_k_dim});
  auto k = packed_flat.narrow(0, q_elements, k_elements)
               .view({1, num_tokens, num_qk_heads, head_k_dim});
  auto v = packed_flat.narrow(0, q_elements + k_elements, v_elements)
               .view({1, num_tokens, num_v_heads, head_v_dim});
  return {q, k, v};
}

namespace {

torch::Tensor causal_conv1d_python(const torch::Tensor& input,
                                   const torch::Tensor& weight,
                                   const torch::Tensor& conv_state,
                                   torch::IntArrayRef query_start_loc,
                                   int64_t activation_mode,
                                   int64_t run_mode) {
  CHECK(input.dim() == 2 || input.dim() == 3);
  CHECK_EQ(weight.dim(), 2);
  CHECK_EQ(conv_state.dim(), 3);
  CHECK_GT(input.size(0), 0);
  CHECK_GT(input.size(1), 0);
  if (input.dim() == 3) {
    CHECK(query_start_loc.empty());
    CHECK_EQ(input.size(0), conv_state.size(0));
  } else {
    CHECK_EQ(query_start_loc.size(), conv_state.size(0) + 1);
    CHECK_EQ(query_start_loc.front(), 0);
    CHECK_EQ(query_start_loc.back(), input.size(0));
  }
  CHECK_EQ(input.size(-1), weight.size(1));
  CHECK_EQ(input.size(-1), conv_state.size(2));
  CHECK_EQ(conv_state.size(1), weight.size(0) - 1);
  CHECK(activation_mode == 0 || activation_mode == 1);
  CHECK(run_mode == 0 || run_mode == 1);
  CHECK(input.scalar_type() == weight.scalar_type());
  CHECK(input.scalar_type() == conv_state.scalar_type());
  CHECK(input.is_contiguous());
  CHECK(weight.is_contiguous());
  CHECK(conv_state.is_contiguous());
  constexpr int64_t kPadSlotId = -1;
  std::vector<int64_t> initial_state_mode;
  if (run_mode == 0) {
    initial_state_mode.assign(conv_state.size(0), 1);
  }
  return causal_conv1d(input,
                       weight,
                       conv_state,
                       std::nullopt,
                       query_start_loc,
                       torch::IntArrayRef{},
                       initial_state_mode,
                       torch::IntArrayRef{},
                       activation_mode,
                       kPadSlotId,
                       run_mode);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(xllm_ops, module) {
  module.def(
      "causal_conv1d(Tensor input, Tensor weight, Tensor(a!) conv_state, "
      "int[] query_start_loc, int activation_mode, int run_mode) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, module) {
  module.impl("causal_conv1d", TORCH_FN(causal_conv1d_python));
}

}  // namespace xllm::kernel::npu
