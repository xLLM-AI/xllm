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

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kBlockSize = 256;

__device__ __forceinline__ float positive_probability(float value) {
  return (value > 0.0f && isfinite(value)) ? value : 0.0f;
}

__global__ void rejection_sample_target_only_k1_kernel(
    const int64_t* __restrict__ draft_token_ids,
    const float* __restrict__ draft_probs,
    const float* __restrict__ target_probs,
    const float* __restrict__ uniform_rand,
    const float* __restrict__ recovery_exponential,
    const int64_t* __restrict__ bonus_token_ids,
    int32_t vocab_size,
    int64_t* __restrict__ output) {
  const int32_t batch_index = static_cast<int32_t>(blockIdx.x);
  const int32_t thread_index = static_cast<int32_t>(threadIdx.x);
  const int64_t row_offset = static_cast<int64_t>(batch_index) * vocab_size;
  const int64_t draft_token = draft_token_ids[batch_index];
  const int64_t output_offset = static_cast<int64_t>(batch_index) * 2;

  __shared__ bool rejected;
  __shared__ float shared_scores[kBlockSize];
  __shared__ int64_t shared_tokens[kBlockSize];
  if (thread_index == 0) {
    const float draft_probability =
        positive_probability(draft_probs[batch_index]);
    const float target_probability =
        (draft_token >= 0 && draft_token < vocab_size)
            ? positive_probability(target_probs[row_offset + draft_token])
            : 0.0f;
    const float acceptance_probability =
        draft_probability > 0.0f ? target_probability / draft_probability
                                 : (target_probability > 0.0f ? 1.0f : 0.0f);
    rejected = uniform_rand[batch_index] >= acceptance_probability;
    if (!rejected) {
      output[output_offset] = draft_token;
      output[output_offset + 1] = bonus_token_ids[batch_index];
    }
  }
  __syncthreads();
  if (!rejected) {
    return;
  }

  float best_score = -1.0f;
  int64_t best_token = 0;
  for (int32_t token = thread_index; token < vocab_size;
       token += static_cast<int32_t>(blockDim.x)) {
    if (static_cast<int64_t>(token) == draft_token) {
      continue;
    }
    const float probability =
        positive_probability(target_probs[row_offset + token]);
    const float exponential =
        fmaxf(recovery_exponential[row_offset + token], FLT_MIN);
    const float score = probability / exponential;
    if (score > best_score ||
        (score == best_score && static_cast<int64_t>(token) < best_token)) {
      best_score = score;
      best_token = token;
    }
  }
  shared_scores[thread_index] = best_score;
  shared_tokens[thread_index] = best_token;
  __syncthreads();

  for (int32_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (thread_index < stride) {
      const float other_score = shared_scores[thread_index + stride];
      const int64_t other_token = shared_tokens[thread_index + stride];
      if (other_score > shared_scores[thread_index] ||
          (other_score == shared_scores[thread_index] &&
           other_token < shared_tokens[thread_index])) {
        shared_scores[thread_index] = other_score;
        shared_tokens[thread_index] = other_token;
      }
    }
    __syncthreads();
  }

  if (thread_index == 0) {
    output[output_offset] = shared_tokens[0];
    output[output_offset + 1] = -1;
  }
}

}  // namespace

torch::Tensor rejection_sample_target_only_k1(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& recovery_exponential,
    const torch::Tensor& bonus_token_ids) {
  CHECK_EQ(draft_token_ids.dim(), 2);
  CHECK_EQ(draft_token_ids.size(1), 1);
  CHECK_EQ(draft_probs.sizes(), draft_token_ids.sizes());
  CHECK_EQ(target_probs.dim(), 3);
  CHECK_EQ(target_probs.size(0), draft_token_ids.size(0));
  CHECK_EQ(target_probs.size(1), 1);
  CHECK_EQ(uniform_rand.sizes(), draft_token_ids.sizes());
  CHECK_EQ(recovery_exponential.sizes(), target_probs.sizes());
  CHECK_EQ(bonus_token_ids.numel(), draft_token_ids.size(0));
  CHECK_GT(target_probs.size(2), 0);

  const int64_t batch_size = draft_token_ids.size(0);
  auto ids = draft_token_ids.to(torch::kInt64).contiguous();
  auto draft_probability = draft_probs.to(torch::kFloat32).contiguous();
  auto target_probability = target_probs.to(torch::kFloat32).contiguous();
  auto acceptance_rand = uniform_rand.to(torch::kFloat32).contiguous();
  auto resample_exponential =
      recovery_exponential.to(torch::kFloat32).contiguous();
  auto bonus =
      bonus_token_ids.to(torch::kInt64).reshape({batch_size}).contiguous();
  auto output = torch::empty(
      {batch_size, 2},
      torch::TensorOptions().dtype(torch::kInt64).device(ids.device()));
  if (batch_size == 0) {
    return output;
  }

  const at::cuda::OptionalCUDAGuard device_guard(ids.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  rejection_sample_target_only_k1_kernel<<<batch_size, kBlockSize, 0, stream>>>(
      ids.data_ptr<int64_t>(),
      draft_probability.data_ptr<float>(),
      target_probability.data_ptr<float>(),
      acceptance_rand.data_ptr<float>(),
      resample_exponential.data_ptr<float>(),
      bonus.data_ptr<int64_t>(),
      static_cast<int32_t>(target_probability.size(2)),
      output.data_ptr<int64_t>());
  return output;
}

}  // namespace xllm::kernel::musa
