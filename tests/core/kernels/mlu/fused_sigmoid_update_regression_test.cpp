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

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::fused_sigmoid_gating_delta_rule_update;

TEST(FusedSigmoidUpdateTest, MultipleKTilesUseFullKeyDimension) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSeqLen = 1;
  constexpr int64_t kNumHeads = 1;
  constexpr int64_t kHeadKDim = 256;
  constexpr int64_t kHeadVDim = 1;

  torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor a_log = torch::zeros({kNumHeads}, bf16_options);
  torch::Tensor a =
      torch::zeros({kBatchSize * kSeqLen, kNumHeads}, bf16_options);
  torch::Tensor b =
      torch::zeros({kBatchSize * kSeqLen, kNumHeads}, bf16_options);
  torch::Tensor dt_bias = torch::zeros({kNumHeads}, bf16_options);
  torch::Tensor q =
      torch::ones({kBatchSize, kSeqLen, kNumHeads, kHeadKDim}, bf16_options);
  torch::Tensor k =
      torch::ones({kBatchSize, kSeqLen, kNumHeads, kHeadKDim}, bf16_options);
  torch::Tensor v =
      torch::ones({kBatchSize, kSeqLen, kNumHeads, kHeadVDim}, bf16_options);
  torch::Tensor initial_state =
      torch::zeros({kBatchSize, kNumHeads, kHeadVDim, kHeadKDim}, fp32_options);
  torch::Tensor state_indices;
  torch::Tensor cu_seqlens;

  auto [out, final_state] = fused_sigmoid_gating_delta_rule_update(
      a_log,
      a,
      b,
      dt_bias,
      q,
      k,
      v,
      initial_state,
      state_indices,
      cu_seqlens,
      /*scale=*/1.0,
      /*use_qk_l2norm_in_kernel=*/false,
      /*softplus_beta=*/1.0f,
      /*softplus_threshold=*/20.0f,
      /*num_accepted_tokens_opt=*/std::nullopt,
      /*inplace_final_state=*/false,
      /*is_kda=*/false);
  torch_mlu::synchronize();

  ASSERT_EQ(out.numel(), 1);
  EXPECT_FLOAT_EQ(out.item<float>(), 128.0f)
      << "the output must reduce across all 256 key elements";
  EXPECT_TRUE(torch::allclose(final_state,
                              torch::full_like(final_state, /*fill_value=*/0.5),
                              /*rtol=*/0.0,
                              /*atol=*/0.0));
}

}  // namespace
}  // namespace xllm
