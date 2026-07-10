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
#include <vector>

#include "kernels/mlu/chunk_gated_delta_rule.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::ChunkGatedDeltaRule;

int64_t numel(const std::vector<int64_t>& shape) {
  int64_t numel = 1;
  for (int64_t dim : shape) {
    numel *= dim;
  }
  return numel;
}

torch::Tensor make_deterministic_tensor(const std::vector<int64_t>& shape,
                                        torch::ScalarType dtype,
                                        const torch::Device& device,
                                        float scale) {
  int64_t element_count = numel(shape);
  torch::Tensor values = torch::arange(
      /*end=*/element_count, torch::TensorOptions().dtype(torch::kFloat32));
  float denom = static_cast<float>(element_count > 0 ? element_count : 1);
  values = (values / denom - 0.5f) * scale;
  return values.reshape(shape).to(
      torch::TensorOptions().dtype(dtype).device(device));
}

torch::Tensor make_chunk_indices(const torch::Tensor& cu_seqlens,
                                 int64_t chunk_size) {
  torch::Tensor lengths = cu_seqlens.narrow(/*dim=*/0,
                                            /*start=*/1,
                                            /*length=*/cu_seqlens.size(0) - 1) -
                          cu_seqlens.narrow(/*dim=*/0,
                                            /*start=*/0,
                                            /*length=*/cu_seqlens.size(0) - 1);
  torch::Tensor num_chunks = (lengths + chunk_size - 1) / chunk_size;
  num_chunks = num_chunks.to(torch::kLong);
  torch::Tensor cumsum = torch::cumsum(num_chunks, /*dim=*/0);
  int64_t total_chunks = cumsum[-1].item<int64_t>();
  torch::Tensor arange_total =
      torch::arange(total_chunks, cu_seqlens.options());
  torch::Tensor zeros = torch::zeros({1}, cumsum.options());
  torch::Tensor prefix =
      torch::cat({zeros, cumsum.slice(/*dim=*/0, /*start=*/0, /*end=*/-1)});
  torch::Tensor repeats_prefix = torch::repeat_interleave(prefix, num_chunks);
  torch::Tensor indices = arange_total - repeats_prefix;
  torch::Tensor mask = indices == 0;
  torch::Tensor col0 = mask.cumsum(/*dim=*/0) - 1;
  return torch::stack({col0, indices}, /*dim=*/1)
      .to(cu_seqlens)
      .to(torch::kInt32);
}

TEST(ChunkGatedDeltaRuleRegressionTest,
     OutputFinalStateFalseDoesNotStoreFinalState) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kNumKHeads = 4;
  constexpr int64_t kNumVHeads = 8;
  constexpr int64_t kHeadKDim = 128;
  constexpr int64_t kHeadVDim = 128;
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSeqLen = 64;
  constexpr int64_t kChunkSize = 64;

  torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  torch::Tensor q =
      make_deterministic_tensor({kBatchSize, kSeqLen, kNumKHeads, kHeadKDim},
                                torch::kBFloat16,
                                device,
                                /*scale=*/0.02f);
  torch::Tensor k =
      make_deterministic_tensor({kBatchSize, kSeqLen, kNumKHeads, kHeadKDim},
                                torch::kBFloat16,
                                device,
                                /*scale=*/0.015f);
  torch::Tensor v =
      make_deterministic_tensor({kBatchSize, kSeqLen, kNumVHeads, kHeadVDim},
                                torch::kBFloat16,
                                device,
                                /*scale=*/0.01f);
  torch::Tensor g = make_deterministic_tensor({kBatchSize, kSeqLen, kNumVHeads},
                                              torch::kFloat32,
                                              device,
                                              /*scale=*/0.001f);
  torch::Tensor beta = torch::full({kBatchSize, kSeqLen, kNumVHeads},
                                   /*value=*/0.25f,
                                   bf16_options);
  torch::Tensor initial_state = torch::zeros(
      {kBatchSize, kNumVHeads, kHeadVDim, kHeadKDim}, fp32_options);
  torch::Tensor cu_seqlens = torch::arange(/*start=*/0,
                                           /*end=*/(kBatchSize + 1) * kSeqLen,
                                           /*step=*/kSeqLen,
                                           int_options);
  torch::Tensor chunk_indices = make_chunk_indices(cu_seqlens, kChunkSize);

  ChunkGatedDeltaRule chunk_gdr(kNumKHeads, kNumVHeads);
  chunk_gdr->to(device);

  auto [output, final_state] =
      chunk_gdr->forward(q,
                         k,
                         v,
                         g,
                         beta,
                         initial_state,
                         cu_seqlens,
                         chunk_indices,
                         /*output_final_state=*/false,
                         /*use_qk_l2norm_in_kernel=*/false);
  torch_mlu::synchronize();

  EXPECT_EQ(output.sizes(),
            torch::IntArrayRef({kBatchSize, kSeqLen, kNumVHeads, kHeadVDim}));
  EXPECT_EQ(output.scalar_type(), bf16_options.dtype());
  torch::Tensor output_cpu = output.flatten().to(torch::kFloat32).cpu();
  EXPECT_TRUE(torch::isfinite(output_cpu).all().item<bool>())
      << "chunk gated delta rule output must be finite";
  EXPECT_TRUE(!final_state.defined() || final_state.numel() == 0)
      << "final_state must stay undefined or empty when output_final_state is "
         "false";
}

}  // namespace
}  // namespace xllm
