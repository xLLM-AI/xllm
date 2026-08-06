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

#include <cstdint>

#include "core/kernels/musa/gdn_ops.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kBlockSize = 256;

template <typename CopyType>
__global__ void scatter_selected_state_kernel(
    CopyType* __restrict__ cache,
    const CopyType* __restrict__ intermediate,
    const int64_t* __restrict__ logical_state_indices,
    const int64_t* __restrict__ accepted_tokens,
    int64_t accepted_token_width,
    int64_t verify_sequence_length,
    int64_t values_per_state,
    int64_t checkpoint_stride,
    bool add_step_to_destination) {
  const int64_t sequence_index = static_cast<int64_t>(blockIdx.y);
  int64_t accepted_count = 0;
  const int64_t accepted_offset = sequence_index * accepted_token_width;
  for (int64_t token_index = 0; token_index < accepted_token_width;
       ++token_index) {
    if (accepted_tokens[accepted_offset + token_index] < 0) {
      break;
    }
    ++accepted_count;
  }
  if (accepted_count == 0 || accepted_count > verify_sequence_length) {
    return;
  }
  const int64_t selected_step = accepted_count - 1;
  const int64_t source_state =
      sequence_index * verify_sequence_length + selected_step;
  const int64_t destination_state =
      logical_state_indices[sequence_index] * checkpoint_stride +
      (add_step_to_destination ? selected_step : 0);

  int64_t value_index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        static_cast<int64_t>(threadIdx.x);
  if (value_index >= values_per_state) {
    return;
  }
  cache[destination_state * values_per_state + value_index] =
      intermediate[source_state * values_per_state + value_index];
}

template <typename CopyType>
void launch_scatter_selected_state(torch::Tensor& cache,
                                   const torch::Tensor& intermediate,
                                   const torch::Tensor& logical_state_indices,
                                   const torch::Tensor& accepted_tokens,
                                   int64_t checkpoint_stride,
                                   bool add_step_to_destination,
                                   cudaStream_t stream) {
  const int64_t batch_size = intermediate.size(0);
  const int64_t verify_sequence_length = intermediate.size(1);
  const int64_t values_per_state =
      intermediate.numel() / batch_size / verify_sequence_length;
  const int64_t blocks = (values_per_state + kBlockSize - 1) / kBlockSize;
  const dim3 grid(static_cast<uint32_t>(blocks),
                  static_cast<uint32_t>(batch_size));
  scatter_selected_state_kernel<CopyType><<<grid, kBlockSize, 0, stream>>>(
      reinterpret_cast<CopyType*>(cache.data_ptr()),
      reinterpret_cast<const CopyType*>(intermediate.data_ptr()),
      logical_state_indices.data_ptr<int64_t>(),
      accepted_tokens.data_ptr<int64_t>(),
      accepted_tokens.size(1),
      verify_sequence_length,
      values_per_state,
      checkpoint_stride,
      add_step_to_destination);
}

void scatter_selected_state(torch::Tensor& cache,
                            const torch::Tensor& intermediate,
                            const torch::Tensor& logical_state_indices,
                            const torch::Tensor& accepted_tokens,
                            int64_t checkpoint_stride,
                            bool add_step_to_destination,
                            cudaStream_t stream) {
  CHECK(cache.defined() && cache.numel() > 0);
  CHECK(intermediate.defined() && intermediate.numel() > 0);
  CHECK(cache.is_contiguous());
  CHECK(intermediate.is_contiguous());
  CHECK(cache.scalar_type() == intermediate.scalar_type());
  CHECK(intermediate.size(0) == logical_state_indices.size(0));
  CHECK(intermediate.size(0) == accepted_tokens.size(0));

  if (cache.scalar_type() == torch::kFloat32) {
    launch_scatter_selected_state<float>(cache,
                                         intermediate,
                                         logical_state_indices,
                                         accepted_tokens,
                                         checkpoint_stride,
                                         add_step_to_destination,
                                         stream);
    return;
  }
  if (cache.scalar_type() == torch::kBFloat16 ||
      cache.scalar_type() == torch::kFloat16) {
    launch_scatter_selected_state<uint16_t>(cache,
                                            intermediate,
                                            logical_state_indices,
                                            accepted_tokens,
                                            checkpoint_stride,
                                            add_step_to_destination,
                                            stream);
    return;
  }
  CHECK(false) << "unsupported GDN MTP state dtype: " << cache.scalar_type();
}

}  // namespace

void scatter_gdn_mtp_verify_states(torch::Tensor& ssm_cache,
                                   const torch::Tensor& ssm_intermediate,
                                   torch::Tensor& conv_cache,
                                   const torch::Tensor& conv_intermediate,
                                   const torch::Tensor& logical_state_indices,
                                   const torch::Tensor& accepted_tokens,
                                   int64_t checkpoint_stride) {
  CHECK(logical_state_indices.scalar_type() == torch::kInt64);
  CHECK(accepted_tokens.scalar_type() == torch::kInt64);
  CHECK(logical_state_indices.is_contiguous());
  CHECK(accepted_tokens.is_contiguous());

  const at::cuda::OptionalCUDAGuard device_guard(accepted_tokens.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  if (ssm_cache.defined() && ssm_cache.numel() > 0) {
    scatter_selected_state(ssm_cache,
                           ssm_intermediate,
                           logical_state_indices,
                           accepted_tokens,
                           checkpoint_stride,
                           true,
                           stream);
  }
  if (conv_cache.defined() && conv_cache.numel() > 0 &&
      conv_intermediate.defined() && conv_intermediate.numel() > 0) {
    scatter_selected_state(conv_cache,
                           conv_intermediate,
                           logical_state_indices,
                           accepted_tokens,
                           1,
                           false,
                           stream);
  }
}

}  // namespace xllm::kernel::musa
