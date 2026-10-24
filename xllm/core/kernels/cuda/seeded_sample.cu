/* Copyright 2026 The xLLM Authors.

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

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <limits>

#include "core/kernels/cuda/device_utils.cuh"
#include "core/kernels/cuda/seeded_sample.h"

namespace xllm::kernel::cuda {
namespace {

constexpr int32_t kThreads = 256;

__device__ double request_uniform(int64_t seed, int64_t offset) {
  uint64_t value = static_cast<uint64_t>(seed) +
                   static_cast<uint64_t>(offset) * 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<double>(value >> 11) * 0x1.0p-53;
}

__global__ void seeded_sample_kernel(const float* probs,
                                     int64_t row_stride,
                                     int32_t vocab,
                                     const int64_t* seeds,
                                     const int64_t* offsets,
                                     const bool* do_sample,
                                     const double* uniforms,
                                     int64_t* output) {
  using Scan = cub::BlockScan<double, kThreads>;
  using Pair = cub::KeyValuePair<int32_t, float>;
  using Reduce = cub::BlockReduce<Pair, kThreads>;
  using Select = cub::BlockReduce<int32_t, kThreads>;
  __shared__ union {
    Scan::TempStorage scan;
    Reduce::TempStorage reduce;
    Select::TempStorage select;
  } storage;
  __shared__ int32_t selected_lane;
  __shared__ int32_t selected_chunk;
  __shared__ double residual;
  const int32_t tid = threadIdx.x;
  const int32_t row = blockIdx.x;
  const float* weights = probs + row * row_stride;
  if (do_sample != nullptr && !do_sample[row]) {
    Pair maximum(vocab, -INFINITY);
    for (int32_t column = tid; column < vocab; column += kThreads) {
      maximum = cub::ArgMax()(maximum, Pair(column, weights[column]));
    }
    const Pair best = Reduce(storage.reduce).Reduce(maximum, cub::ArgMax());
    if (tid == 0) {
      output[row] = best.key;
    }
    return;
  }

  // First select a strided lane, then select within it. Both summation orders
  // depend only on vocabulary width, never the batch row or grid dimensions.
  double mass = 0.0;
  for (int32_t column = tid; column < vocab; column += kThreads) {
    const float weight = weights[column];
    CUDA_KERNEL_ASSERT(isfinite(weight) && weight >= 0.0F);
    mass += weight;
  }
  double prefix;
  double total;
  Scan(storage.scan).ExclusiveSum(mass, prefix, total);
  CUDA_KERNEL_ASSERT(total > 0.0 && isfinite(total));
  CUDA_KERNEL_ASSERT(seeds[row] != -1 || uniforms != nullptr);
  const double uniform = seeds[row] == -1
                             ? uniforms[row]
                             : request_uniform(seeds[row], offsets[row]);
  const double target = fmin(uniform * total, nextafter(total, 0.0));
  // Select by lower bounds only: independently rounded interval ends can leave
  // gaps or overlaps. The first positive lane has prefix zero, so a valid row
  // always has a candidate. A reduction gives ties exactly one owner.
  __syncthreads();
  const int32_t lane =
      Select(storage.select)
          .Reduce(mass > 0.0 && prefix <= target ? tid : -1, MaxReduceOp());
  if (tid == 0) {
    selected_lane = lane;
    CUDA_KERNEL_ASSERT(selected_lane >= 0);
  }
  __syncthreads();
  if (tid == selected_lane) {
    residual = fmin(fmax(target - prefix, 0.0), nextafter(mass, 0.0));
  }
  __syncthreads();

  // Each thread now owns consecutive elements of the selected strided lane.
  // At common vocabulary sizes this is only 1-2 elements per thread.
  const int32_t lane_size = (vocab - 1 - selected_lane) / kThreads + 1;
  const int32_t chunk = (lane_size + kThreads - 1) / kThreads;
  const int32_t begin = tid * chunk;
  const int32_t end = min(begin + chunk, lane_size);
  mass = 0.0;
  for (int32_t i = begin; i < end; ++i) {
    mass += weights[selected_lane + i * kThreads];
  }
  __syncthreads();
  Scan(storage.scan).ExclusiveSum(mass, prefix, total);
  const double local_target = fmin(residual, nextafter(total, 0.0));
  __syncthreads();
  const int32_t selected =
      Select(storage.select)
          .Reduce(mass > 0.0 && prefix <= local_target ? tid : -1,
                  MaxReduceOp());
  if (tid == 0) {
    selected_chunk = selected;
    CUDA_KERNEL_ASSERT(selected_chunk >= 0);
  }
  __syncthreads();
  if (tid == selected_chunk) {
    const double chunk_target =
        fmin(fmax(local_target - prefix, 0.0), nextafter(mass, 0.0));
    // Accumulate from zero in the same order as this chunk's mass. Its last
    // positive element must cross the clamped target, even after rounding.
    double cumulative = 0.0;
    int64_t token = -1;
    for (int32_t i = begin; i < end; ++i) {
      const int32_t column = selected_lane + i * kThreads;
      cumulative += weights[column];
      if (chunk_target < cumulative) {
        token = column;
        break;
      }
    }
    CUDA_KERNEL_ASSERT(token >= 0);
    output[row] = token;
  }
}

void check_metadata(const torch::Tensor& tensor,
                    const torch::Tensor& probs,
                    torch::ScalarType dtype) {
  CHECK(tensor.defined());
  CHECK(tensor.device() == probs.device());
  CHECK(tensor.scalar_type() == dtype);
  CHECK_EQ(tensor.dim(), 1);
  CHECK_EQ(tensor.size(0), probs.size(0));
  CHECK(tensor.is_contiguous());
}

}  // namespace

torch::Tensor seeded_sample(const torch::Tensor& probs,
                            const torch::Tensor& seeds,
                            const torch::Tensor& offsets,
                            const torch::Tensor& do_sample,
                            bool has_unseeded_sample) {
  CHECK(probs.defined());
  CHECK(probs.is_cuda());
  CHECK(probs.scalar_type() == torch::kFloat32);
  CHECK_EQ(probs.dim(), 2);
  CHECK_EQ(probs.stride(1), 1);
  CHECK_GT(probs.size(1), 0);
  CHECK_LT(probs.size(1), std::numeric_limits<int32_t>::max() - kThreads);
  CHECK_LE(probs.size(0), std::numeric_limits<int32_t>::max());
  check_metadata(seeds, probs, torch::kInt64);
  check_metadata(offsets, probs, torch::kInt64);
  if (do_sample.defined()) {
    check_metadata(do_sample, probs, torch::kBool);
  }
  const c10::cuda::CUDAGuard guard(probs.device());
  torch::Tensor output = torch::empty({probs.size(0)}, seeds.options());
  if (probs.size(0) == 0) {
    return output;
  }
  // Allocate randomness only for batches that contain unseeded random rows.
  // No device-to-host query is necessary, including during graph capture.
  const torch::Tensor uniforms =
      has_unseeded_sample
          ? torch::rand({probs.size(0)}, probs.options().dtype(torch::kFloat64))
          : torch::Tensor();
  seeded_sample_kernel<<<probs.size(0),
                         kThreads,
                         0,
                         c10::cuda::getCurrentCUDAStream(probs.get_device())>>>(
      probs.data_ptr<float>(),
      probs.stride(0),
      probs.size(1),
      seeds.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      do_sample.defined() ? do_sample.data_ptr<bool>() : nullptr,
      uniforms.defined() ? uniforms.data_ptr<double>() : nullptr,
      output.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace xllm::kernel::cuda
