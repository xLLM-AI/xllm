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

#include <ATen/hip/HIPGeneratorImpl.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hipcub/hipcub.hpp>
#include <hiprand/hiprand_kernel.h>

#include <c10/util/Optional.h>

#include <cstdint>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "kernels/dcu/dcu_ops_api.h"

namespace {

constexpr uint32_t kBlockThreads = 512;
constexpr uint32_t kVecSize = 4;

at::Generator get_default_generator(c10::DeviceIndex device_index) {
  static std::unordered_map<c10::DeviceIndex, at::Generator> cache;
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device_index);
  if (it != cache.end()) {
    return it->second;
  }
  at::globalContext().lazyInitCUDA();
  at::Generator gen = at::cuda::detail::getDefaultCUDAGenerator(device_index);
  cache.emplace(device_index, gen);
  return gen;
}

std::tuple<int64_t, int64_t> get_seed_and_offset(
    int64_t increment,
    const torch::Device& device,
    c10::optional<at::Generator> generator = c10::nullopt) {
  at::Generator gen = generator.has_value()
                          ? generator.value()
                          : get_default_generator(device.index());
  std::lock_guard<std::mutex> lock(gen.mutex());
  at::CUDAGeneratorImpl* cuda_gen =
      at::check_generator<at::CUDAGeneratorImpl>(gen);

  int64_t seed = static_cast<int64_t>(cuda_gen->current_seed());
  int64_t offset = static_cast<int64_t>(cuda_gen->get_offset());
  offset += (increment + 3) / 4 * 4;
  cuda_gen->set_offset(static_cast<uint64_t>(offset));

  return std::make_tuple(seed, offset);
}

// Shared memory layout for the sampling kernel.
// The union reuses space between hipcub primitives since they execute
// sequentially with __syncthreads() barriers between uses.
struct SamplingTempStorage {
  union {
    typename hipcub::BlockScan<float, kBlockThreads>::TempStorage scan;
    typename hipcub::BlockReduce<float, kBlockThreads>::TempStorage reduce;
    typename hipcub::BlockReduce<int32_t, kBlockThreads>::TempStorage
        reduce_int;
  } block_prim;
  int32_t sampled_id;
  int32_t last_valid_id;
  int32_t done;
  float aggregate_value;
};

// Samples one token per batch row from probability distributions.
// Algorithm: prefix-sum CDF + threshold search, matching FlashInfer's
// Original sampling kernel semantics.
//
// Each HIP block processes one row of the [batch, vocab] probs matrix.
// 1. Generate uniform random number u via Philox RNG
// 2. Iterate over vocab in chunks of (kBlockThreads * kVecSize)
// 3. Per chunk: BlockReduce sum of filtered probs (prob > 0)
// 4. If running aggregate exceeds u, compute inclusive CDF via BlockScan
//    and locate the first index where CDF > u using atomicMin
// 5. Fallback to last valid index when u approaches 1.0
template <uint32_t VecSize>
__global__ void __launch_bounds__(1024, 1)
    sampling_from_prob_kernel(const float* __restrict__ probs,
                              int32_t* __restrict__ output,
                              uint32_t d,
                              uint64_t philox_seed,
                              uint64_t philox_offset) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  hiprandStatePhilox4_32_10_t state;
  hiprand_init(philox_seed, bx, philox_offset, &state);
  float u = hiprand_uniform(&state);

  extern __shared__ __align__(alignof(SamplingTempStorage)) uint8_t smem[];
  SamplingTempStorage& ts = reinterpret_cast<SamplingTempStorage&>(smem);

  if (tx == 0) {
    ts.sampled_id = static_cast<int32_t>(d);
    ts.last_valid_id = -1;
    ts.done = 0;
  }
  __syncthreads();

  float aggregate = 0.0f;
  const uint32_t num_iters =
      (d + kBlockThreads * VecSize - 1) / (kBlockThreads * VecSize);

  for (uint32_t i = 0; i < num_iters; ++i) {
    if (ts.done != 0) {
      break;
    }

    // Load and filter probs for this chunk.
    float filtered[VecSize];
    bool valid[VecSize];

#pragma unroll
    for (uint32_t j = 0; j < VecSize; ++j) {
      const uint32_t idx = (i * kBlockThreads + tx) * VecSize + j;
      if (idx < d) {
        const float p = probs[bx * d + idx];
        valid[j] = (p > 0.0f);
        filtered[j] = valid[j] ? p : 0.0f;
      } else {
        valid[j] = false;
        filtered[j] = 0.0f;
      }
    }

    // Sum of filtered probs in this chunk.
    float aggregate_local =
        hipcub::BlockReduce<float, kBlockThreads>(ts.block_prim.reduce)
            .Sum(filtered);
    if (tx == 0) {
      ts.aggregate_value = aggregate_local;
    }
    __syncthreads();
    aggregate_local = ts.aggregate_value;

    if (aggregate + aggregate_local > u) {
      // Target lies in this chunk. Compute inclusive CDF.
      float inclusive_cdf[VecSize];
      hipcub::BlockScan<float, kBlockThreads>(ts.block_prim.scan)
          .InclusiveSum(filtered, inclusive_cdf);
      __syncthreads();

      // atomicMin on all positions where CDF + aggregate > u.
      // Because CDF is monotonically non-decreasing, atomicMin naturally
      // selects the first (smallest) qualifying index. No need for
      // BlockAdjacentDifference to detect the false->true edge.
#pragma unroll
      for (uint32_t j = 0; j < VecSize; ++j) {
        if ((inclusive_cdf[j] + aggregate > u) && valid[j]) {
          atomicMin(
              &(ts.sampled_id),
              static_cast<int32_t>((i * kBlockThreads + tx) * VecSize + j));
        }
      }
      __syncthreads();
    }

    // Track last valid index as fallback for u close to 1.0.
    int32_t valid_index[VecSize];
#pragma unroll
    for (uint32_t j = 0; j < VecSize; ++j) {
      valid_index[j] =
          valid[j]
              ? static_cast<int32_t>((i * kBlockThreads + tx) * VecSize + j)
              : -1;
    }
    int32_t max_valid_index =
        hipcub::BlockReduce<int32_t, kBlockThreads>(ts.block_prim.reduce_int)
            .Reduce(valid_index, hipcub::Max());
    if (tx == 0 && max_valid_index != -1 &&
        max_valid_index < static_cast<int32_t>(d)) {
      ts.last_valid_id = max_valid_index;
    }
    __syncthreads();

    aggregate += aggregate_local;
    if (tx == 0 && aggregate > u) {
      ts.done = 1;
    }
    __syncthreads();
  }

  int32_t sampled_id = ts.sampled_id;
  if (sampled_id == static_cast<int32_t>(d)) {
    // u is very close to 1.0 and sum(probs) < u
    sampled_id = (ts.last_valid_id != -1) ? ts.last_valid_id : 0;
  }
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

}  // namespace

namespace xllm::kernel::dcu {

torch::Tensor random_sample(const torch::Tensor& probs) {
  CHECK_EQ(probs.dim(), 2) << "probs must be a 2D tensor";
  const torch::Device device = probs.device();
  const int64_t batch_size = probs.size(0);
  CHECK(probs.is_cuda()) << "probs must be on a HIP/DCU device";
  CHECK_EQ(probs.scalar_type(), torch::kFloat32)
      << "probs must be float32";
  CHECK_GE(batch_size, 0);
  CHECK_LE(batch_size, INT32_MAX);
  CHECK_GE(probs.size(1), 1);
  CHECK_LE(probs.size(1), UINT32_MAX);

  const torch::ScalarType out_dtype = torch::kInt32;
  torch::Tensor samples = torch::empty(
      {batch_size}, torch::TensorOptions().dtype(out_dtype).device(device));
  if (batch_size == 0) {
    return samples;
  }

  const std::tuple<int64_t, int64_t> seed_and_offset =
      get_seed_and_offset(batch_size, device);
  const uint64_t seed = static_cast<uint64_t>(std::get<0>(seed_and_offset));
  const uint64_t offset = static_cast<uint64_t>(std::get<1>(seed_and_offset));

  const c10::hip::HIPStream stream = c10::hip::getCurrentHIPStream(
      static_cast<c10::DeviceIndex>(device.index()));
  const uint32_t d = static_cast<uint32_t>(probs.size(1));
  const uint32_t smem_size = sizeof(SamplingTempStorage);

  sampling_from_prob_kernel<kVecSize>
      <<<batch_size, kBlockThreads, smem_size, stream>>>(
          probs.data_ptr<float>(),
          samples.data_ptr<int32_t>(),
          d,
          seed,
          offset);
  C10_HIP_KERNEL_LAUNCH_CHECK();

  return samples;
}

}  // namespace xllm::kernel::dcu
