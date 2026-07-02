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

#include <framework/core/device.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::fused_recurrent_gated_delta_rule;

constexpr int32_t kBatch = 1;
constexpr int32_t kTokens = 8;
constexpr int32_t kQueryHeads = 8;
constexpr int32_t kKeyDim = 128;
constexpr int32_t kValueHeads = 8;
constexpr int32_t kValueDim = 128;

struct GdrTestInputs {
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor g;
  torch::Tensor initial_state;
  torch::Tensor cu_seqlens;
  torch::Tensor ssm_state_indices;
};

int64_t num_elements(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  for (int64_t dim : shape) {
    total *= dim;
  }
  return total;
}

torch::Tensor make_bf16_tensor(const std::vector<int64_t>& shape,
                               const torch::Device& device,
                               float scale,
                               float offset) {
  torch::Tensor values =
      torch::arange(/*end=*/num_elements(shape),
                    torch::TensorOptions().dtype(torch::kFloat32))
          .mul_(scale)
          .add_(offset)
          .reshape(shape);
  return values
      .to(torch::TensorOptions().dtype(torch::kBFloat16).device(device))
      .contiguous();
}

GdrTestInputs make_inputs(const torch::Device& device) {
  GdrTestInputs inputs;
  inputs.q = make_bf16_tensor({kBatch, kTokens, kQueryHeads, kKeyDim},
                              device,
                              /*scale=*/1e-4f,
                              /*offset=*/-0.2f);
  inputs.k = make_bf16_tensor({kBatch, kTokens, kQueryHeads, kKeyDim},
                              device,
                              /*scale=*/-8e-5f,
                              /*offset=*/0.3f);
  inputs.v = make_bf16_tensor({kBatch, kTokens, kValueHeads, kValueDim},
                              device,
                              /*scale=*/1e-4f,
                              /*offset=*/-0.1f);
  inputs.g = make_bf16_tensor({kBatch, kTokens, kValueHeads},
                              device,
                              /*scale=*/-1e-3f,
                              /*offset=*/-0.2f);
  inputs.initial_state = torch::zeros(
      {kTokens, kValueHeads, kValueDim, kKeyDim},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  inputs.cu_seqlens = torch::tensor(
      {0, kTokens}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  inputs.ssm_state_indices =
      torch::arange(
          /*start=*/0,
          /*end=*/kTokens,
          torch::TensorOptions().dtype(torch::kInt32).device(device))
          .reshape({kBatch, kTokens});
  return inputs;
}

bool tensors_allclose(const torch::Tensor& a,
                      const torch::Tensor& b,
                      double rtol,
                      double atol) {
  torch::Tensor ac = a.to(torch::kCPU).to(torch::kFloat32).contiguous();
  torch::Tensor bc = b.to(torch::kCPU).to(torch::kFloat32).contiguous();
  if (ac.sizes() != bc.sizes()) {
    return false;
  }
  torch::Tensor diff = (ac - bc).abs();
  torch::Tensor tol = atol + rtol * bc.abs();
  return (diff <= tol).all().item<bool>();
}

class FusedRecurrentGdrBetaRegressionTest : public ::testing::Test {
 protected:
  torch::Device device() const {
    return torch::Device(torch::kPrivateUse1, /*index=*/0);
  }
};

TEST_F(FusedRecurrentGdrBetaRegressionTest,
       MissingBetaMatchesExplicitOnesForMultipleValueHeads) {
  torch::DeviceGuard guard(device());
  GdrTestInputs inputs = make_inputs(device());
  torch::Tensor explicit_beta = torch::ones_like(inputs.g);

  torch::Tensor explicit_initial_state = inputs.initial_state.clone();
  auto [explicit_output, explicit_state] = fused_recurrent_gated_delta_rule(
      inputs.q,
      inputs.k,
      inputs.v,
      inputs.g,
      /*beta_opt=*/explicit_beta,
      /*initial_state_opt=*/explicit_initial_state,
      /*inplace_final_state=*/true,
      /*cu_seqlens_opt=*/inputs.cu_seqlens,
      /*ssm_state_indices_opt=*/inputs.ssm_state_indices,
      /*num_accepted_tokens_opt=*/std::nullopt,
      /*use_qk_l2norm_in_kernel=*/true);
  torch_mlu::synchronize();

  torch::Tensor missing_initial_state = inputs.initial_state.clone();
  auto [missing_output, missing_state] = fused_recurrent_gated_delta_rule(
      inputs.q,
      inputs.k,
      inputs.v,
      inputs.g,
      /*beta_opt=*/std::nullopt,
      /*initial_state_opt=*/missing_initial_state,
      /*inplace_final_state=*/true,
      /*cu_seqlens_opt=*/inputs.cu_seqlens,
      /*ssm_state_indices_opt=*/inputs.ssm_state_indices,
      /*num_accepted_tokens_opt=*/std::nullopt,
      /*use_qk_l2norm_in_kernel=*/true);
  torch_mlu::synchronize();

  EXPECT_TRUE(tensors_allclose(missing_output,
                               explicit_output,
                               /*rtol=*/1e-3,
                               /*atol=*/1e-3))
      << "missing beta output must match explicit all-ones beta";
  EXPECT_TRUE(tensors_allclose(missing_state,
                               explicit_state,
                               /*rtol=*/1e-3,
                               /*atol=*/1e-3))
      << "missing beta final state must match explicit all-ones beta";
}

}  // namespace
}  // namespace xllm
