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

#include "kernels/dcu/aiter_moe_adapter.h"

#include <glog/logging.h>

#include <algorithm>

namespace aiter {
namespace native {

void moe_sorting_fwd(
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weights,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& sorted_weights,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& tokens_positions_per_expert,
    torch::Tensor& num_valid_ids,
    std::optional<torch::Tensor> moe_buf = std::nullopt,
    int num_experts = 0,
    int unit_size = 0,
    std::optional<torch::Tensor> local_expert_mask = std::nullopt);

torch::Tensor moe_c_moe_gemm_marlin_w8a8_fp8(
    torch::Tensor input,
    torch::Tensor b_qweight,
    torch::Tensor output,
    torch::Tensor a_scale,
    torch::Tensor b_scale,
    std::optional<torch::Tensor> topk_weights,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad,
    int64_t top_k,
    int64_t mode,
    int64_t delta,
    int64_t size_m);

}  // namespace native
}  // namespace aiter

namespace xllm {
namespace kernel {
namespace dcu {
namespace aiter {
namespace {

constexpr int64_t kMarlinKTile = 64;
constexpr int64_t kMarlinNTile = 16;
constexpr int64_t kMoeGemm1SmallTokenMode = 60121;
constexpr int64_t kMoeGemm1LargeTokenMode = 60098;
constexpr int64_t kMoeGemm2TinyTokenMode = 60042;
constexpr int64_t kMoeGemm2SmallTokenMode = 60038;
constexpr int64_t kMoeGemm2MediumTokenMode = 60043;
constexpr int64_t kMoeGemm2LargeTokenMode = 60046;
constexpr int64_t kMoeSelectedMMax = 512;

void check_fp8_moe_weight(const torch::Tensor& weight) {
  CHECK(weight.defined()) << "AITER MoE weight must be defined.";
  CHECK(weight.is_contiguous()) << "AITER MoE weight must be contiguous.";
  CHECK_EQ(weight.dim(), 3)
      << "AITER MoE weight must be [E,N,K], got " << weight.sizes();
  CHECK(weight.scalar_type() == torch::kFloat8_e4m3fn ||
        weight.scalar_type() == torch::kFloat8_e5m2)
      << "AITER MoE weight must be FP8.";
  CHECK_EQ(weight.size(2) % kMarlinKTile, 0)
      << "AITER MoE weight K must be divisible by " << kMarlinKTile;
}

void check_scale(const torch::Tensor& scale, const torch::Tensor& weight) {
  CHECK(scale.defined()) << "AITER MoE weight scale must be defined.";
  CHECK(scale.is_cuda()) << "AITER MoE weight scale must be on DCU.";
  CHECK(scale.is_contiguous()) << "AITER MoE weight scale must be contiguous.";
  CHECK_EQ(scale.scalar_type(), torch::kFloat32)
      << "AITER MoE weight scale must be float32.";
  CHECK_EQ(scale.dim(), 3) << "AITER MoE weight scale must be [E,N,1], got "
                           << scale.sizes();
  CHECK_EQ(scale.size(0), weight.size(0))
      << "AITER MoE weight scale expert dim mismatch.";
  CHECK_EQ(scale.size(1), weight.size(1))
      << "AITER MoE weight scale output channel dim mismatch.";
  CHECK_EQ(scale.size(2), 1) << "AITER MoE weight scale last dim must be 1.";
}

}  // namespace

int64_t select_moe_gemm1_mode(int64_t num_tokens) {
  CHECK_GT(num_tokens, 0) << "num_tokens must be positive.";
  if (num_tokens <= 256) {
    return kMoeGemm1SmallTokenMode;
  }
  return kMoeGemm1LargeTokenMode;
}

int64_t select_moe_gemm2_mode(int64_t num_tokens) {
  CHECK_GT(num_tokens, 0) << "num_tokens must be positive.";
  if (num_tokens == 2 || num_tokens == 3) {
    return kMoeGemm2TinyTokenMode;
  }
  if (num_tokens <= 16) {
    return kMoeGemm2SmallTokenMode;
  }
  if (num_tokens <= 64) {
    return kMoeGemm2MediumTokenMode;
  }
  if (num_tokens <= 256) {
    return kMoeGemm2LargeTokenMode;
  }
  return kMoeGemm2MediumTokenMode;
}

int64_t select_moe_m_key(int64_t num_tokens) {
  CHECK_GT(num_tokens, 0) << "num_tokens must be positive.";
  return std::min<int64_t>(num_tokens, kMoeSelectedMMax);
}

torch::Tensor moe_layout_shuffle_gemm1(const torch::Tensor& weight) {
  // The exported AITER C kernel currently expects the same marlin packing for
  // the first and second MoE GEMMs.
  return moe_layout_shuffle_gemm2(weight);
}

torch::Tensor moe_layout_shuffle_gemm2(const torch::Tensor& weight) {
  check_fp8_moe_weight(weight);
  CHECK_EQ(weight.size(1) % kMarlinNTile, 0)
      << "AITER MoE weight N must be divisible by " << kMarlinNTile;
  torch::Tensor transposed = weight.permute({0, 2, 1}).contiguous();
  const int64_t experts = transposed.size(0);
  const int64_t size_k = transposed.size(1);
  const int64_t size_n = transposed.size(2);

  torch::Tensor shuffled = transposed.reshape({experts,
                                               size_k / kMarlinKTile,
                                               kMarlinKTile,
                                               size_n / kMarlinNTile,
                                               kMarlinNTile});
  shuffled = shuffled.permute({0, 1, 3, 4, 2}).contiguous();
  shuffled = shuffled.reshape(
      {experts, size_k / kMarlinKTile, size_n / kMarlinNTile, 1, 16, 4, 16});
  shuffled = shuffled.permute({0, 1, 2, 3, 5, 4, 6}).contiguous();
  return shuffled.view(weight.sizes());
}

MoeSortedTokens moe_align_block_size(const torch::Tensor& topk_ids,
                                     const torch::Tensor& topk_weights,
                                     int64_t num_experts,
                                     int64_t block_size) {
  CHECK(topk_ids.defined()) << "topk_ids must be defined.";
  CHECK(topk_weights.defined()) << "topk_weights must be defined.";
  CHECK(topk_ids.is_cuda() && topk_weights.is_cuda())
      << "topk tensors must be on DCU.";
  CHECK_EQ(topk_ids.sizes(), topk_weights.sizes())
      << "topk ids/weights shape mismatch.";
  CHECK_EQ(topk_ids.dim(), 2) << "topk_ids must be [M,topk].";
  CHECK_GT(num_experts, 0) << "num_experts must be positive.";
  CHECK_GT(block_size, 0) << "block_size must be positive.";

  torch::Tensor ids = topk_ids.scalar_type() == torch::kInt32
                          ? topk_ids.contiguous()
                          : topk_ids.to(torch::kInt32).contiguous();
  torch::Tensor weights = topk_weights.to(torch::kFloat32).contiguous();

  const int64_t topk = ids.size(1);
  const int64_t max_num_tokens_padded =
      ids.numel() + num_experts * block_size - topk;
  const int64_t max_num_m_blocks =
      (max_num_tokens_padded + block_size - 1) / block_size;

  torch::TensorOptions int_options =
      ids.options().dtype(torch::kInt32).device(ids.device());
  torch::TensorOptions fp_options =
      ids.options().dtype(torch::kFloat32).device(ids.device());
  MoeSortedTokens sorted_tokens{
      torch::empty({max_num_tokens_padded}, int_options),
      torch::empty({max_num_tokens_padded}, fp_options),
      torch::empty({max_num_m_blocks}, int_options),
      torch::empty({num_experts * 2}, int_options),
      torch::empty({1}, int_options)};

  ::aiter::native::moe_sorting_fwd(ids,
                                   weights,
                                   sorted_tokens.sorted_token_ids,
                                   sorted_tokens.sorted_weights,
                                   sorted_tokens.expert_ids,
                                   sorted_tokens.tokens_positions_per_expert,
                                   sorted_tokens.num_tokens_post_pad,
                                   std::nullopt,
                                   static_cast<int>(num_experts),
                                   static_cast<int>(block_size),
                                   std::nullopt);
  return sorted_tokens;
}

torch::Tensor moe_gemm_fp8_channelwise(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& output,
    const torch::Tensor& input_scale,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& topk_weights,
    const MoeSortedTokens& sorted_tokens,
    int64_t topk,
    int64_t mode,
    int64_t delta,
    int64_t selected_m) {
  CHECK(input.defined()) << "AITER MoE input must be defined.";
  CHECK(input.is_cuda()) << "AITER MoE input must be on DCU.";
  CHECK(input.is_contiguous()) << "AITER MoE input must be contiguous.";
  CHECK(input.scalar_type() == torch::kFloat8_e4m3fn ||
        input.scalar_type() == torch::kFloat8_e5m2)
      << "AITER MoE input must be FP8.";
  CHECK(output.defined()) << "AITER MoE output must be defined.";
  CHECK(output.is_cuda()) << "AITER MoE output must be on DCU.";
  CHECK(output.is_contiguous()) << "AITER MoE output must be contiguous.";
  CHECK(input_scale.defined()) << "AITER MoE input scale must be defined.";
  CHECK(input_scale.is_cuda()) << "AITER MoE input scale must be on DCU.";
  CHECK(input_scale.is_contiguous())
      << "AITER MoE input scale must be contiguous.";
  check_fp8_moe_weight(weight);
  CHECK(weight.is_cuda()) << "AITER MoE weight must be on DCU.";
  check_scale(weight_scale, weight);

  return ::aiter::native::moe_c_moe_gemm_marlin_w8a8_fp8(
      input,
      weight,
      output,
      input_scale,
      weight_scale,
      topk_weights,
      sorted_tokens.sorted_token_ids,
      sorted_tokens.expert_ids,
      sorted_tokens.num_tokens_post_pad,
      topk,
      mode,
      delta,
      selected_m);
}

}  // namespace aiter
}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
