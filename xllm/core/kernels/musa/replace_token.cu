/* Copyright 2025-2026 The xLLM Authors.

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
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

namespace {

constexpr int32_t kBlockSize = 256;

__global__ void replace_token_kernel(int32_t* __restrict__ dst,
                                     const int64_t* __restrict__ src,
                                     int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  int32_t val = dst[idx];
  if (val < 0) {
    dst[idx] = static_cast<int32_t>(src[(-val) - 1]);
  }
}

}  // namespace

void replace_token(torch::Tensor& dst,
                   torch::Tensor& src,
                   bool synchronize_stream) {
  CHECK(dst.scalar_type() == torch::kInt)
      << "replace_token: dst must be int32, got " << dst.scalar_type();
  CHECK(src.scalar_type() == torch::kLong)
      << "replace_token: src must be int64, got " << src.scalar_type();

  const at::cuda::OptionalCUDAGuard device_guard(dst.device());
  int64_t n = dst.numel();
  if (n == 0) {
    return;
  }

  int32_t grid_size = static_cast<int32_t>((n + kBlockSize - 1) / kBlockSize);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  replace_token_kernel<<<grid_size, kBlockSize, 0, stream>>>(
      dst.data_ptr<int32_t>(), src.data_ptr<int64_t>(), n);

  if (synchronize_stream) {
    cudaStreamSynchronize(stream);
  }
}

}  // namespace xllm::kernel::musa
