/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <limits>

#include "cuda_ops_api.h"
#include "device_utils.cuh"
#include "utils.h"

namespace xllm::kernel::cuda {

namespace {

template <typename scalar_t, int kThreads>
__global__ void log_softmax_last_dim_kernel(const scalar_t* __restrict__ input,
                                            const float* __restrict__ temps,
                                            bool has_temps,
                                            float* __restrict__ output,
                                            int32_t k,
                                            int64_t stride) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * stride;

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  // NOTE: cub::BlockReduce::Reduce/Sum only returns valid result to thread 0.
  // Use shared variables to broadcast reduction results to all threads.
  __shared__ float s_row_max;
  __shared__ float s_log_denom;
  extern __shared__ float s_data[];

  float inv_temp = 1.0f;
  if (has_temps) {
    float t = temps[row];
    if (t == 0.0f) {
      t = 1.0f;
    }
    inv_temp = 1.0f / t;
  }

  // Load data into shared memory with temperature scaling.
  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    s_data[col] = static_cast<float>(input[base + col]) * inv_temp;
  }
  __syncthreads();

  float thread_max = -std::numeric_limits<float>::infinity();
  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    thread_max = s_data[col] > thread_max ? s_data[col] : thread_max;
  }
  float row_max_local =
      BlockReduce(reduce_storage).Reduce(thread_max, MaxReduceOp());
  if (threadIdx.x == 0) {
    s_row_max = row_max_local;
  }
  __syncthreads();

  float thread_sum = 0.0f;
  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    thread_sum += expf(s_data[col] - s_row_max);
  }
  float row_sum_local = BlockReduce(reduce_storage).Sum(thread_sum);
  if (threadIdx.x == 0) {
    s_log_denom = logf(row_sum_local) + s_row_max;
  }
  __syncthreads();

  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    output[base + col] = s_data[col] - s_log_denom;
  }
}

#if defined(USE_DCU)
template <typename scalar_t, int kThreads>
__global__ void log_softmax_last_dim_large_k_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ temps,
    bool has_temps,
    float* __restrict__ output,
    int32_t k,
    int64_t stride) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * stride;

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float s_row_max;
  __shared__ float s_log_denom;

  float inv_temp = 1.0f;
  if (has_temps) {
    float t = temps[row];
    if (t == 0.0f) {
      t = 1.0f;
    }
    inv_temp = 1.0f / t;
  }

  float thread_max = -std::numeric_limits<float>::infinity();
  float thread_sum = 0.0f;

  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    float val = static_cast<float>(input[base + col]) * inv_temp;
    float old_max = thread_max;
    thread_max = val > thread_max ? val : thread_max;
    thread_sum =
        thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
  }

  float row_max_local =
      BlockReduce(reduce_storage).Reduce(thread_max, MaxReduceOp());
  if (threadIdx.x == 0) {
    s_row_max = row_max_local;
  }
  __syncthreads();

  float rescaled_sum = thread_sum * expf(thread_max - s_row_max);
  float row_sum_local = BlockReduce(reduce_storage).Sum(rescaled_sum);
  if (threadIdx.x == 0) {
    s_log_denom = logf(row_sum_local) + s_row_max;
  }
  __syncthreads();

  for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
    float val = static_cast<float>(input[base + col]) * inv_temp;
    output[base + col] = val - s_log_denom;
  }
}
#endif

}  // namespace

torch::Tensor air_log_softmax_last_dim(const torch::Tensor& input,
                                       const torch::Tensor& temperatures) {
  CHECK(input.is_cuda()) << "air_log_softmax_last_dim: input must be CUDA";
  CHECK(input.dim() == 2)
      << "air_log_softmax_last_dim: input must be 2D [B, K]";

  const int64_t batch64 = input.size(0);
  const int64_t k64 = input.size(1);
  CHECK(batch64 >= 0 && batch64 <= INT32_MAX)
      << "air_log_softmax_last_dim: batch too large";
  CHECK(k64 > 0 && k64 <= INT32_MAX) << "air_log_softmax_last_dim: k too large";

  const int32_t batch = static_cast<int32_t>(batch64);
  const int32_t k = static_cast<int32_t>(k64);

  if (batch == 0) {
    return torch::empty(
        {batch64, k64},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
  }

  bool has_temps = temperatures.defined();
  torch::Tensor temps = temperatures;
  if (has_temps) {
    CHECK(temps.is_cuda())
        << "air_log_softmax_last_dim: temperatures must be CUDA";
    CHECK(temps.dim() == 1)
        << "air_log_softmax_last_dim: temperatures must be 1D [B]";
    CHECK(temps.size(0) == batch64)
        << "air_log_softmax_last_dim: temperatures size mismatch";
    CHECK(temps.scalar_type() == torch::kFloat32)
        << "air_log_softmax_last_dim: temperatures must be float32";
    temps = temps.contiguous();
  }

  c10::cuda::CUDAGuard device_guard(input.device());
  auto in = input.contiguous();
  auto out = torch::empty(
      {batch64, k64},
      torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

  const int64_t stride = k64;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int kThreads = 256;
  dim3 grid(batch);
  dim3 block(std::min<int32_t>(kThreads, 1024));
  const size_t shared_mem_bytes = k * sizeof(float);

  // Guard against user-controlled k exceeding device shared memory limit.
  // The kernel uses both static shared memory (BlockReduce::TempStorage for
  // cub reductions) and dynamic shared memory (extern __shared__ s_data[] for
  // input data). These occupy separate regions within the same per-block shared
  // memory pool, so the guard must account for both.
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  constexpr size_t kStaticSmem =
      sizeof(typename BlockReduce::TempStorage) + 2 * sizeof(float);
  const size_t total_shared_bytes = shared_mem_bytes + kStaticSmem;

  // Use ATen's per-device cached properties to avoid repeated driver queries
  // and correctly handle multi-GPU environments.
  const int max_smem =
      at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
  const bool use_smem_kernel =
      total_shared_bytes <= static_cast<size_t>(max_smem);

#if !defined(USE_DCU)
  CHECK(use_smem_kernel) << "air_log_softmax_last_dim: k (" << k
                         << ") requires " << total_shared_bytes
                         << " bytes shared memory "
                         << "(dynamic=" << shared_mem_bytes
                         << " + static=" << kStaticSmem
                         << "), exceeding device limit (" << max_smem
                         << " bytes)";
#endif

  DISPATCH_FLOATING_TYPES(in.scalar_type(), "air_log_softmax_last_dim", [&] {
    const scalar_t* in_ptr = in.data_ptr<scalar_t>();
    const float* t_ptr = has_temps ? temps.data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

#if defined(USE_DCU)
    if (use_smem_kernel) {
      log_softmax_last_dim_kernel<scalar_t, kThreads>
          <<<grid, block, shared_mem_bytes, stream>>>(
              in_ptr, t_ptr, has_temps, out_ptr, k, stride);
    } else {
      log_softmax_last_dim_large_k_kernel<scalar_t, kThreads>
          <<<grid, block, 0, stream>>>(
              in_ptr, t_ptr, has_temps, out_ptr, k, stride);
    }
#else
    log_softmax_last_dim_kernel<scalar_t, kThreads>
        <<<grid, block, shared_mem_bytes, stream>>>(
            in_ptr, t_ptr, has_temps, out_ptr, k, stride);
#endif
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace xllm::kernel::cuda
