/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

// Fused MoE combine kernel — reorder + weighted sum in one pass.
// Replaces: torch::zeros + index_copy_ + view + multiply + sum
//
// Algorithm per token (each block handles one token):
//   1. For each of its topk experts, read gemm2 at flat_idx directly
//      (gemm2 is flat-index-ordered after scatter via index_copy_ with dst_src)
//   2. Multiply by router weight
//   3. Accumulate into output[token]
//
// Grid:  num_tokens (N) blocks
// Block: HIDDEN_DIM / HIDDEN_TILE threads

#include <c10/cuda/CUDAGuard.h>

#include "device_utils.cuh"
#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::cuda {

constexpr int32_t kCombineBlockSize = 256;

template <typename scalar_t>
__global__ void XLLM_KERNEL_ATTR(kCombineBlockSize) moe_combine_kernel(
    const scalar_t* __restrict__ gemm2,       // [N*topk, H] flat-index-ordered
    const float* __restrict__ reduce_weight,  // [N, topk]
    scalar_t* __restrict__ output,            // [N, H]
    int64_t N,
    int32_t topk,
    int64_t H) {
  int64_t token_id = blockIdx.x;  // 0 .. N-1
  if (token_id >= N) return;

  int32_t tid = threadIdx.x;
  int32_t stride = kCombineBlockSize;

  // Accumulate over topk experts for this token
  for (int64_t h = tid; h < H; h += stride) {
    float acc = 0.0f;
    for (int32_t k = 0; k < topk; ++k) {
      int64_t flat_idx = token_id * topk + k;
      float w = reduce_weight[flat_idx];
      acc += w * static_cast<float>(gemm2[flat_idx * H + h]);
    }
    output[token_id * H + h] = static_cast<scalar_t>(acc);
  }
}

// ---- Host-side orchestrator ----
torch::Tensor moe_combine_result(
    const torch::Tensor& gemm2,          // [N*topk, H] flat-index-ordered
    const torch::Tensor& reduce_weight,  // [N, topk] float or same as gemm2
    int64_t N,
    int32_t topk) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t H = gemm2.size(1);
  auto dtype = gemm2.scalar_type();

  auto output = torch::empty({N, H}, gemm2.options());
  auto rw = reduce_weight.to(gemm2.device(), torch::kFloat32).contiguous();

  if (dtype == torch::kFloat16) {
    moe_combine_kernel<c10::Half>
        <<<N, kCombineBlockSize, 0, stream>>>(gemm2.data_ptr<c10::Half>(),
                                              rw.data_ptr<float>(),
                                              output.data_ptr<c10::Half>(),
                                              N,
                                              topk,
                                              H);
  } else if (dtype == torch::kBFloat16) {
    moe_combine_kernel<c10::BFloat16>
        <<<N, kCombineBlockSize, 0, stream>>>(gemm2.data_ptr<c10::BFloat16>(),
                                              rw.data_ptr<float>(),
                                              output.data_ptr<c10::BFloat16>(),
                                              N,
                                              topk,
                                              H);
  } else {
    moe_combine_kernel<float>
        <<<N, kCombineBlockSize, 0, stream>>>(gemm2.data_ptr<float>(),
                                              rw.data_ptr<float>(),
                                              output.data_ptr<float>(),
                                              N,
                                              topk,
                                              H);
  }

  return output;
}

}  // namespace xllm::kernel::cuda
