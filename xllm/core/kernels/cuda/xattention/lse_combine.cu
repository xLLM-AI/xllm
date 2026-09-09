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
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include <cmath>

#include "kernels/cuda/utils.h"
#include "xattention_ops_api.h"

namespace {

// Fused log-sum-exp combine kernel.
//
// Layout and strategy (aligned with the TileLang version):
//   - Each block is responsible for one (batch_idx, head_idx) pair, i.e. one
//     row in the flattened [B * H, D] layout.
//   - Threads within a block parallelize along the head_dim (D) dimension to
//     ensure coalesced global memory access.
//
// Tensors:
//   shared_o    : [B, H, D] - shared attention output
//   shared_lse  : [B, H, 1] - shared log-sum-exp (FP32)
//   unshared_o  : [B, H, D] - unshared attention output
//   unshared_lse: [B, H, 1] - unshared log-sum-exp (FP32)
//   output      : [B, H, D] - combined output
template <typename scalar_t, typename out_scalar_t>
__global__ void lse_combine_kernel(
    out_scalar_t* __restrict__ output,        // [B, H, D]
    const scalar_t* __restrict__ shared_o,    // [B, H, D]
    const float* __restrict__ shared_lse,     // [B, H, 1], always FP32
    const scalar_t* __restrict__ unshared_o,  // [B, H, D]
    const float* __restrict__ unshared_lse,   // [B, H, 1], always FP32
    const int64_t B,                          // batch_size * beam_size
    const int64_t H,                          // num_heads
    const int64_t D) {                        // head_dim
  const int64_t total_elements = B * H;
  const int64_t idx = static_cast<int64_t>(blockIdx.y);

  if (idx >= total_elements) {
    return;
  }

  // Load LSE scalars for this (batch, head) pair.
  const float shared_lse_val = shared_lse[idx];
  const float unshared_lse_val = unshared_lse[idx];

  // 1. Compute element-wise max LSE.
  const float lse_max = fmaxf(shared_lse_val, unshared_lse_val);

  // 2. Compute base-2 exponentials relative to max.
  const float exp_shared = exp2f(shared_lse_val - lse_max);
  const float exp_unshared = exp2f(unshared_lse_val - lse_max);

  // 3. Compute merged LSE.
  const float lse_new = lse_max + log2f(exp_shared + exp_unshared);

  // 4. Compute normalized weights.
  const float w_shared = exp2f(shared_lse_val - lse_new);
  const float w_unshared = exp2f(unshared_lse_val - lse_new);

  // 5. Weighted combine along the head_dim.
  const int64_t base_idx = idx * D;
  // Threads in the block parallelize along D with stride blockDim.x for
  // coalesced global memory access.
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    const float shared_val = static_cast<float>(shared_o[base_idx + d]);
    const float unshared_val = static_cast<float>(unshared_o[base_idx + d]);
    const float combined = w_shared * shared_val + w_unshared * unshared_val;
    output[base_idx + d] = static_cast<out_scalar_t>(combined);
  }
}

}  // namespace

namespace xllm::kernel::cuda {

// Host wrapper for the fused LSE combine kernel.
//
// All inputs are expected to be on the same CUDA device:
//   shared_o    : [B, H, D], floating type (including Half/BFloat16)
//   shared_lse  : [B, H, 1], float32
//   unshared_o  : [B, H, D], same type/shape as shared_o
//   unshared_lse: [B, H, 1], float32
//   output      : [B, H, D], will be resized/allocated as needed.
void lse_combine(torch::Tensor output,
                 torch::Tensor shared_o,
                 torch::Tensor shared_lse,
                 torch::Tensor unshared_o,
                 torch::Tensor unshared_lse) {
  CHECK_EQ(shared_o.dim(), 3) << "shared_o must be 3D [B, H, D]";
  CHECK_EQ(unshared_o.dim(), 3) << "unshared_o must be 3D [B, H, D]";
  CHECK_EQ(shared_lse.dim(), 3) << "shared_lse must be 3D [B, H, 1]";
  CHECK_EQ(unshared_lse.dim(), 3) << "unshared_lse must be 3D [B, H, 1]";

  const int64_t B = shared_o.size(0);
  const int64_t H = shared_o.size(1);
  const int64_t D = shared_o.size(2);

  CHECK_EQ(shared_o.sizes(), unshared_o.sizes())
      << "shared_o and unshared_o must have same shape";
  CHECK_EQ(shared_lse.scalar_type(), torch::kFloat32)
      << "shared_lse must be float32";
  CHECK_EQ(unshared_lse.scalar_type(), torch::kFloat32)
      << "unshared_lse must be float32";
  CHECK_EQ(shared_lse.size(0), B)
      << "shared_lse shape mismatch, expected [B, H, 1]";
  CHECK_EQ(shared_lse.size(1), H)
      << "shared_lse shape mismatch, expected [B, H, 1]";
  CHECK_EQ(shared_lse.size(2), 1)
      << "shared_lse shape mismatch, expected [B, H, 1]";
  CHECK_EQ(unshared_lse.size(0), B)
      << "unshared_lse shape mismatch, expected [B, H, 1]";
  CHECK_EQ(unshared_lse.size(1), H)
      << "unshared_lse shape mismatch, expected [B, H, 1]";
  CHECK_EQ(unshared_lse.size(2), 1)
      << "unshared_lse shape mismatch, expected [B, H, 1]";

  // Ensure output has the correct shape and dtype.
  if (!output.defined() || output.sizes() != shared_o.sizes()) {
    output = torch::empty_like(shared_o);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(shared_o));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Launch kernel: one block per (batch, head) pair, threads along D.
  const int64_t total_elements = B * H;
  const int threads_per_block = 128;
  dim3 block_dim(threads_per_block, 1, 1);
  dim3 grid_dim(1, static_cast<unsigned int>(total_elements), 1);

  DISPATCH_FLOATING_TYPES(
      shared_o.scalar_type(), "lse_combine_kernel_input", [&] {
        using in_t = scalar_t;
        DISPATCH_FLOATING_TYPES(
            output.scalar_type(), "lse_combine_kernel_output", [&] {
              using out_t = scalar_t;
              lse_combine_kernel<in_t, out_t>
                  <<<grid_dim, block_dim, 0, stream>>>(
                      output.data_ptr<out_t>(),
                      shared_o.data_ptr<in_t>(),
                      shared_lse.data_ptr<float>(),
                      unshared_o.data_ptr<in_t>(),
                      unshared_lse.data_ptr<float>(),
                      B,
                      H,
                      D);
            });
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
