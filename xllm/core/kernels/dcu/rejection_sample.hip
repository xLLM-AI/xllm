/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <glog/logging.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cstdint>
#include <optional>

#include "core/kernels/dcu/dcu_ops_api.h"

namespace xllm::kernel::dcu {
namespace {

constexpr int32_t kBlockThreads = 256;

__device__ __forceinline__ float safe_prob(float p) {
  return p > 0.0f ? p : 0.0f;
}

__global__ void rejection_sample_kernel(
    const int32_t* __restrict__ draft_token_ids,
    const int32_t* __restrict__ num_draft_tokens,
    const int32_t* __restrict__ cu_num_draft_tokens,
    const float* __restrict__ draft_probs,
    const float* __restrict__ target_probs,
    const int32_t* __restrict__ bonus_token_ids,
    const float* __restrict__ uniform_rand,
    const float* __restrict__ uniform_probs,
    int32_t batch_size,
    int32_t vocab_size,
    int32_t* __restrict__ output) {
  const int32_t seq_id = static_cast<int32_t>(blockIdx.x);
  if (seq_id >= batch_size) {
    return;
  }

  const int32_t n_draft = num_draft_tokens[seq_id];
  const int32_t draft_end = cu_num_draft_tokens[seq_id];
  const int32_t draft_start = draft_end - n_draft;
  const int32_t out_start = draft_start + seq_id;

  if (threadIdx.x == 0) {
    for (int32_t i = 0; i < n_draft + 1; ++i) {
      output[out_start + i] = -1;
    }
  }
  __syncthreads();

  for (int32_t draft_idx = 0; draft_idx < n_draft; ++draft_idx) {
    const int32_t row = draft_start + draft_idx;
    const int32_t draft_token = draft_token_ids[row];
    if (draft_token < 0 || draft_token >= vocab_size) {
      return;
    }

    const int64_t row_offset = static_cast<int64_t>(row) * vocab_size;
    const float selected_draft_prob =
        safe_prob(draft_probs[row_offset + draft_token]);
    const float selected_target_prob =
        safe_prob(target_probs[row_offset + draft_token]);
    const float accept_prob =
        selected_draft_prob > 0.0f
            ? selected_target_prob / selected_draft_prob
            : (selected_target_prob > 0.0f ? 1.0f : 0.0f);

    if (uniform_rand[row] < accept_prob) {
      if (threadIdx.x == 0) {
        output[out_start + draft_idx] = draft_token;
      }
      __syncthreads();
      continue;
    }

    float best_score = -1.0f;
    int32_t best_token = 0;
    for (int32_t token = static_cast<int32_t>(threadIdx.x);
         token < vocab_size;
         token += static_cast<int32_t>(blockDim.x)) {
      const float recovered_prob =
          fmaxf(target_probs[row_offset + token] -
                    draft_probs[row_offset + token],
                0.0f);
      const float u = fmaxf(uniform_probs[row_offset + token], FLT_MIN);
      const float score = recovered_prob / u;
      if (score > best_score ||
          (score == best_score && token < best_token)) {
        best_score = score;
        best_token = token;
      }
    }

    __shared__ float shared_scores[kBlockThreads];
    __shared__ int32_t shared_tokens[kBlockThreads];
    shared_scores[threadIdx.x] = best_score;
    shared_tokens[threadIdx.x] = best_token;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other_score = shared_scores[threadIdx.x + stride];
        const int32_t other_token = shared_tokens[threadIdx.x + stride];
        if (other_score > shared_scores[threadIdx.x] ||
            (other_score == shared_scores[threadIdx.x] &&
             other_token < shared_tokens[threadIdx.x])) {
          shared_scores[threadIdx.x] = other_score;
          shared_tokens[threadIdx.x] = other_token;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      output[out_start + draft_idx] = shared_tokens[0];
    }
    return;
  }

  if (threadIdx.x == 0) {
    output[out_start + n_draft] = bonus_token_ids[seq_id];
  }
}

}  // namespace

torch::Tensor rejection_sample(const torch::Tensor& draft_token_ids,
                               const torch::Tensor& num_draft_tokens,
                               const torch::Tensor& cu_num_draft_tokens,
                               const std::optional<torch::Tensor>& draft_probs,
                               const torch::Tensor& target_probs,
                               const torch::Tensor& bonus_token_ids,
                               const torch::Tensor& uniform_rand,
                               const torch::Tensor& uniform_probs,
                               int64_t max_spec_len) {
  (void)max_spec_len;
  CHECK(draft_probs.has_value())
      << "DCU rejection_sample requires dense draft_probs";

  CHECK(draft_token_ids.is_cuda());
  CHECK(num_draft_tokens.is_cuda());
  CHECK(cu_num_draft_tokens.is_cuda());
  CHECK(draft_probs.value().is_cuda());
  CHECK(target_probs.is_cuda());
  CHECK(bonus_token_ids.is_cuda());
  CHECK(uniform_rand.is_cuda());
  CHECK(uniform_probs.is_cuda());
  CHECK_EQ(draft_token_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(num_draft_tokens.scalar_type(), torch::kInt32);
  CHECK_EQ(cu_num_draft_tokens.scalar_type(), torch::kInt32);
  CHECK_EQ(bonus_token_ids.scalar_type(), torch::kInt32);
  torch::Tensor draft_probs_tensor =
      draft_probs.value().to(torch::kFloat32).contiguous();
  torch::Tensor target_probs_tensor =
      target_probs.to(torch::kFloat32).contiguous();
  torch::Tensor uniform_rand_tensor =
      uniform_rand.to(torch::kFloat32).contiguous();
  torch::Tensor uniform_probs_tensor =
      uniform_probs.to(torch::kFloat32).contiguous();

  CHECK_EQ(draft_probs_tensor.dim(), 2);
  CHECK_EQ(target_probs_tensor.dim(), 2);
  CHECK_EQ(uniform_probs_tensor.sizes(), target_probs_tensor.sizes());
  CHECK_EQ(draft_probs_tensor.sizes(), target_probs_tensor.sizes());

  const int64_t batch_size = num_draft_tokens.size(0);
  CHECK_EQ(cu_num_draft_tokens.numel(), batch_size);
  CHECK_EQ(bonus_token_ids.numel(), batch_size);
  CHECK_EQ(uniform_rand_tensor.numel(), draft_token_ids.numel());
  CHECK_EQ(draft_probs_tensor.size(0), draft_token_ids.numel());
  CHECK_LE(batch_size, INT32_MAX);
  CHECK_LE(draft_probs_tensor.size(1), INT32_MAX);

  auto output = torch::empty(
      {batch_size + draft_token_ids.size(0)},
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .device(draft_token_ids.device()));
  if (batch_size == 0) {
    return output;
  }

  const c10::hip::HIPStream stream = c10::hip::getCurrentHIPStream(
      static_cast<c10::DeviceIndex>(draft_token_ids.device().index()));
  rejection_sample_kernel<<<batch_size, kBlockThreads, 0, stream>>>(
      draft_token_ids.data_ptr<int32_t>(),
      num_draft_tokens.data_ptr<int32_t>(),
      cu_num_draft_tokens.data_ptr<int32_t>(),
      draft_probs_tensor.data_ptr<float>(),
      target_probs_tensor.data_ptr<float>(),
      bonus_token_ids.data_ptr<int32_t>(),
      uniform_rand_tensor.data_ptr<float>(),
      uniform_probs_tensor.data_ptr<float>(),
      static_cast<int32_t>(batch_size),
      static_cast<int32_t>(draft_probs_tensor.size(1)),
      output.data_ptr<int32_t>());
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace xllm::kernel::dcu
