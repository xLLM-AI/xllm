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

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/Sort.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <torch/torch.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "core/kernels/cuda/device_utils.cuh"
#include "core/kernels/cuda/fused_logits_transform.h"
#include "core/kernels/cuda/utils.h"

namespace xllm::kernel::cuda {
namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kMaxBlocksPerRow = 32;
constexpr int32_t kRadixBins = 256;
constexpr int32_t kCandidates = 4096;
constexpr int32_t kSmallBatchRows = 32;

__device__ uint32_t ordered_key(float value) {
  if (value == 0.0F) {
    return 0x80000000U;
  }
  const uint32_t bits = __float_as_uint(value);
  return bits ^ ((bits & 0x80000000U) ? 0xffffffffU : 0x80000000U);
}

__device__ float key_value(uint32_t key) {
  return __uint_as_float(key ^
                         ((key & 0x80000000U) ? 0x80000000U : 0xffffffffU));
}

template <typename Scalar>
__global__ void select_top_k_kernel(const Scalar* logits,
                                    const int64_t* top_k,
                                    float* thresholds,
                                    int32_t width,
                                    int64_t stride) {
  using Reduce = cub::BlockReduce<uint32_t, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ int32_t histogram[kThreads / WARP_SIZE][kRadixBins];
  __shared__ uint32_t prefix;
  __shared__ uint32_t mask;
  __shared__ int32_t rank;
  __shared__ bool finished;
  const int32_t row = blockIdx.x;
  const int64_t k = top_k[row];
  if (k <= 0 || k >= width) {
    if (threadIdx.x == 0) {
      thresholds[row] = -INFINITY;
    }
    return;
  }
  if (threadIdx.x == 0) {
    prefix = 0;
    mask = 0;
    rank = static_cast<int32_t>(k);
    finished = false;
  }
  __syncthreads();
  const Scalar* values = logits + row * stride;
  for (int32_t shift = 24; shift >= 0; shift -= 8) {
    for (int32_t bin = threadIdx.x; bin < kRadixBins * kThreads / WARP_SIZE;
         bin += kThreads) {
      histogram[bin / kRadixBins][bin % kRadixBins] = 0;
    }
    __syncthreads();
    uint32_t local_min = UINT32_MAX;
    uint32_t local_max = 0;
    for (int64_t col = threadIdx.x; col < width; col += kThreads) {
      const uint32_t key = ordered_key(static_cast<float>(values[col]));
      const bool matches = (key & mask) == prefix;
      local_min = matches ? min(local_min, key) : local_min;
      local_max = matches ? max(local_max, key) : local_max;
      const uint32_t bin = (key >> shift) & 0xffU;
#if __CUDA_ARCH__ >= 700
      const uint32_t active = __activemask();
      const uint32_t group =
          __match_any_sync(active, matches ? bin : kRadixBins);
      if (matches && (threadIdx.x % WARP_SIZE) == __ffs(group) - 1) {
        atomicAdd(&histogram[threadIdx.x / WARP_SIZE][bin], __popc(group));
      }
#else
      if (matches) {
        atomicAdd(&histogram[threadIdx.x / WARP_SIZE][bin], 1);
      }
#endif
    }
    const uint32_t minimum = Reduce(storage).Reduce(local_min, MinReduceOp());
    __syncthreads();
    const uint32_t maximum = Reduce(storage).Reduce(local_max, MaxReduceOp());
    if (threadIdx.x == 0 && minimum == maximum) {
      thresholds[row] = key_value(minimum);
      finished = true;
    }
    __syncthreads();
    if (finished) {
      return;
    }
    if (threadIdx.x < kRadixBins) {
      int32_t total = 0;
      for (int32_t warp = 0; warp < kThreads / WARP_SIZE; ++warp) {
        total += histogram[warp][threadIdx.x];
      }
      histogram[0][threadIdx.x] = total;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int32_t bin = kRadixBins - 1; bin >= 0; --bin) {
        const int32_t count = histogram[0][bin];
        if (count >= rank) {
          prefix |= static_cast<uint32_t>(bin) << shift;
          mask |= 0xffU << shift;
          break;
        }
        rank -= count;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    thresholds[row] = key_value(prefix);
  }
}

template <typename Scalar>
__global__ void apply_threshold_kernel(Scalar* logits,
                                       const float* thresholds,
                                       int32_t width,
                                       int64_t stride) {
  const int32_t row = blockIdx.x;
  Scalar* values = logits + row * stride;
  const float threshold = thresholds[row];
  for (int64_t col = threadIdx.x + static_cast<int64_t>(blockIdx.y) * kThreads;
       col < width;
       col += static_cast<int64_t>(gridDim.y) * kThreads) {
    if (static_cast<float>(values[col]) < threshold) {
      values[col] = static_cast<Scalar>(-INFINITY);
    }
  }
}

template <typename Scalar>
__global__ void compact_sort_kernel(const Scalar* logits,
                                    const float* thresholds,
                                    Scalar* sorted,
                                    int64_t* segment_ends,
                                    int32_t width,
                                    int64_t stride) {
  using Sort = cub::BlockRadixSort<float, kThreads, kCandidates / kThreads>;
  __shared__ union {
    float candidates[kCandidates];
    typename Sort::TempStorage sort;
  } storage;
  __shared__ int32_t count;
  const int32_t row = blockIdx.x;
  const int64_t offset = static_cast<int64_t>(row) * width;
  const float threshold = thresholds[row];
  if (threadIdx.x == 0) {
    count = 0;
  }
  for (int32_t index = threadIdx.x; index < kCandidates; index += kThreads) {
    storage.candidates[index] = -INFINITY;
  }
  __syncthreads();
  const Scalar* values = logits + row * stride;
  for (int64_t chunk = 0; chunk < width; chunk += kThreads) {
    const int64_t col = chunk + threadIdx.x;
    const float value =
        col < width ? static_cast<float>(values[col]) : -INFINITY;
    const uint32_t group =
        __ballot_sync(0xffffffffU, col < width && value >= threshold);
    const int32_t lane = threadIdx.x % WARP_SIZE;
    int32_t base = 0;
    if (lane == 0) {
      base = atomicAdd(&count, __popc(group));
    }
    base = __shfl_sync(0xffffffffU, base, 0);
    const int32_t index = base + __popc(group & ((1U << lane) - 1));
    if ((group & (1U << lane)) && index < kCandidates) {
      storage.candidates[index] = value;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // Empty segments leave the block-sorted rows untouched in the following
    // device sort; only rows exceeding the capacity use a full radix sort.
    segment_ends[row] = offset + (count > kCandidates ? width : 0);
  }
  if (count > kCandidates) {
    return;
  }
  float keys[kCandidates / kThreads];
#pragma unroll
  for (int32_t index = 0; index < kCandidates / kThreads; ++index) {
    keys[index] =
        storage.candidates[threadIdx.x * (kCandidates / kThreads) + index];
  }
  __syncthreads();
  Sort(storage.sort).Sort(keys);
#pragma unroll
  for (int32_t index = 0; index < kCandidates / kThreads; ++index) {
    sorted[offset + width - kCandidates +
           threadIdx.x * (kCandidates / kThreads) + index] =
        static_cast<Scalar>(keys[index]);
  }
}

class RowOffset final {
 public:
  explicit RowOffset(int64_t width) : width_(width) {}

  __host__ __device__ int64_t operator()(int32_t row) const {
    return static_cast<int64_t>(row) * width_;
  }

 private:
  int64_t width_;
};

class RelativeOffset final {
 public:
  explicit RelativeOffset(int64_t base) : base_(base) {}

  __host__ __device__ int64_t operator()(int64_t offset) const {
    return offset - base_;
  }

 private:
  int64_t base_;
};

template <typename Scalar>
using PackedKey = std::conditional_t<sizeof(Scalar) == 4, uint64_t, uint32_t>;

template <typename Scalar>
__global__ void pack_keys_kernel(const Scalar* logits,
                                 PackedKey<Scalar>* keys,
                                 int64_t elements,
                                 int32_t width,
                                 int64_t stride) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / width;
    const Scalar value = logits[row * stride + index % width];
    uint32_t key;
    if constexpr (sizeof(Scalar) == 4) {
      key = ordered_key(value);
    } else {
      const uint32_t bits = value.x;
      key = bits ^ ((bits & 0x8000U) ? 0xffffU : 0x8000U);
    }
    keys[index] =
        (static_cast<PackedKey<Scalar>>(row) << (sizeof(Scalar) * 8)) | key;
  }
}

template <typename Scalar>
__global__ void unpack_keys_kernel(const PackedKey<Scalar>* keys,
                                   Scalar* sorted,
                                   int64_t elements) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if constexpr (sizeof(Scalar) == 4) {
      sorted[index] = key_value(static_cast<uint32_t>(keys[index]));
    } else {
      const uint32_t key = static_cast<uint32_t>(keys[index]) & 0xffffU;
      Scalar value;
      value.x =
          static_cast<uint16_t>(key ^ ((key & 0x8000U) ? 0x8000U : 0xffffU));
      sorted[index] = value;
    }
  }
}

template <typename Scalar>
torch::Tensor sort_small_batch_values(const torch::Tensor& logits,
                                      cudaStream_t stream) {
  const int32_t elements = static_cast<int32_t>(logits.numel());
  const int32_t blocks = static_cast<int32_t>(
      std::min<int64_t>((logits.numel() + kThreads - 1) / kThreads, 4096));
  const auto options = logits.options().dtype(torch::kUInt8);
  auto input = torch::empty(
      {logits.numel() * static_cast<int64_t>(sizeof(PackedKey<Scalar>))},
      options);
  auto output = torch::empty_like(input);
  auto* keys_in = reinterpret_cast<PackedKey<Scalar>*>(input.data_ptr());
  auto* keys_out = reinterpret_cast<PackedKey<Scalar>*>(output.data_ptr());
  pack_keys_kernel<Scalar>
      <<<blocks, kThreads, 0, stream>>>(logits.data_ptr<Scalar>(),
                                        keys_in,
                                        elements,
                                        static_cast<int32_t>(logits.size(1)),
                                        logits.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int32_t end_bit =
      sizeof(Scalar) * 8 +
      std::bit_width(static_cast<uint32_t>(logits.size(0) - 1));
  size_t bytes = 0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      nullptr, bytes, keys_in, keys_out, elements, 0, end_bit, stream));
  auto temporary = torch::empty({static_cast<int64_t>(bytes)}, options);
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(temporary.data_ptr(),
                                                bytes,
                                                keys_in,
                                                keys_out,
                                                elements,
                                                0,
                                                end_bit,
                                                stream));
  auto sorted = torch::empty(logits.sizes(), logits.options());
  unpack_keys_kernel<Scalar><<<blocks, kThreads, 0, stream>>>(
      keys_out, sorted.data_ptr<Scalar>(), elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return sorted;
}

template <typename Scalar>
torch::Tensor sort_values(const torch::Tensor& logits,
                          cudaStream_t stream,
                          torch::Tensor output = {},
                          const torch::Tensor& segment_ends = {}) {
  if (!segment_ends.defined() && logits.size(0) > 1 &&
      logits.size(0) <= kSmallBatchRows && logits.numel() <= INT32_MAX) {
    return sort_small_batch_values<Scalar>(logits, stream);
  }
  using Key = std::conditional_t<
      std::is_same_v<Scalar, c10::Half>,
      __half,
      std::conditional_t<std::is_same_v<Scalar, c10::BFloat16>,
                         __nv_bfloat16,
                         float>>;
  const auto input = logits.contiguous();
  if (!output.defined()) {
    output = torch::empty_like(input);
  }
  const auto* source = reinterpret_cast<const Key*>(input.data_ptr<Scalar>());
  auto* destination = reinterpret_cast<Key*>(output.data_ptr<Scalar>());
  const int64_t width = logits.size(1);
  const int64_t rows = logits.size(0);
  // Bound each invocation's element count to CUB's signed 32-bit limit.
  const int64_t rows_per_batch = INT32_MAX / width;
  for (int64_t first = 0; first < rows; first += rows_per_batch) {
    const int32_t count =
        static_cast<int32_t>(std::min(rows - first, rows_per_batch));
    const int32_t elements = static_cast<int32_t>(count * width);
    const auto offsets = thrust::make_transform_iterator(
        thrust::counting_iterator<int32_t>(0), RowOffset(width));
    size_t bytes = 0;
    const auto sort = [&](void* temporary) {
      if (count == 1 && !segment_ends.defined()) {
        return cub::DeviceRadixSort::SortKeys(temporary,
                                              bytes,
                                              source,
                                              destination,
                                              elements,
                                              0,
                                              sizeof(Key) * 8,
                                              stream);
      }
      if (segment_ends.defined()) {
        const auto ends = thrust::make_transform_iterator(
            segment_ends.data_ptr<int64_t>() + first,
            RelativeOffset(first * width));
        return cub::DeviceSegmentedRadixSort::SortKeys(temporary,
                                                       bytes,
                                                       source,
                                                       destination,
                                                       elements,
                                                       count,
                                                       offsets,
                                                       ends,
                                                       0,
                                                       sizeof(Key) * 8,
                                                       stream);
      }
      return cub::DeviceSegmentedRadixSort::SortKeys(temporary,
                                                     bytes,
                                                     source,
                                                     destination,
                                                     elements,
                                                     count,
                                                     offsets,
                                                     offsets + 1,
                                                     0,
                                                     sizeof(Key) * 8,
                                                     stream);
    };
    C10_CUDA_CHECK(sort(nullptr));
    auto temporary = torch::empty({static_cast<int64_t>(bytes)},
                                  logits.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(sort(temporary.data_ptr()));
    source += elements;
    destination += elements;
  }
  return output;
}

__device__ float parameter_value(const void* values,
                                 torch::ScalarType dtype,
                                 int32_t row) {
  if (dtype == torch::kFloat32) {
    return static_cast<const float*>(values)[row];
  }
  if (dtype == torch::kFloat16) {
    return static_cast<float>(static_cast<const c10::Half*>(values)[row]);
  }
  return static_cast<float>(static_cast<const c10::BFloat16*>(values)[row]);
}

__device__ float probability_cutoff(const void* top_p,
                                    torch::ScalarType dtype,
                                    int32_t row) {
  float cutoff = 1.0F - parameter_value(top_p, dtype, row);
  if (dtype == torch::kFloat16) {
    cutoff = static_cast<float>(static_cast<c10::Half>(cutoff));
  } else if (dtype == torch::kBFloat16) {
    cutoff = static_cast<float>(static_cast<c10::BFloat16>(cutoff));
  }
  return cutoff;
}

struct TopPBoundary {
  float value;
  int32_t discarded_ties;
};

template <typename Scalar>
__device__ TopPBoundary find_top_p_boundary(const Scalar* sorted,
                                            const float* cumulative,
                                            float cutoff,
                                            int32_t width) {
  int32_t first = 0;
  int32_t last = width - 1;
  while (first < last) {
    const int32_t middle = first + (last - first) / 2;
    if (cumulative[middle] <= cutoff) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  const int32_t boundary = first;
  const float value = static_cast<float>(sorted[boundary]);
  first = 0;
  while (first < last) {
    const int32_t middle = first + (last - first) / 2;
    if (static_cast<float>(sorted[middle]) < value) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return {value, boundary - first};
}

template <typename Scalar>
__global__ void filter_count_kernel(Scalar* logits,
                                    const Scalar* sorted,
                                    const float* cumulative,
                                    const void* top_p,
                                    torch::ScalarType top_p_dtype,
                                    int32_t* tie_counts,
                                    int32_t width,
                                    int64_t stride) {
  using Reduce = cub::BlockReduce<int32_t, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ TopPBoundary boundary;
  const int32_t row = blockIdx.x;
  const int64_t offset = static_cast<int64_t>(row) * width;
  if (threadIdx.x == 0) {
    boundary = find_top_p_boundary(sorted + offset,
                                   cumulative + offset,
                                   probability_cutoff(top_p, top_p_dtype, row),
                                   width);
  }
  __syncthreads();
  const int64_t tile =
      (static_cast<int64_t>(width) + gridDim.y - 1) / gridDim.y;
  const int64_t begin = blockIdx.y * tile;
  const int64_t end = min(begin + tile, static_cast<int64_t>(width));
  Scalar* values = logits + row * stride;
  int32_t count = 0;
  for (int64_t col = begin + threadIdx.x; col < end; col += kThreads) {
    const float value = static_cast<float>(values[col]);
    count += value == boundary.value;
    if (value < boundary.value) {
      values[col] = static_cast<Scalar>(-INFINITY);
    }
  }
  const int32_t total = Reduce(storage).Sum(count);
  if (threadIdx.x == 0) {
    tie_counts[static_cast<int64_t>(row) * gridDim.y + blockIdx.y] = total;
  }
}

template <typename Scalar>
__global__ void filter_ties_kernel(Scalar* logits,
                                   const Scalar* sorted,
                                   const float* cumulative,
                                   const void* top_p,
                                   torch::ScalarType top_p_dtype,
                                   const int32_t* tie_counts,
                                   int32_t width,
                                   int64_t stride) {
  using Scan = cub::BlockScan<int32_t, kThreads>;
  __shared__ typename Scan::TempStorage storage;
  __shared__ TopPBoundary boundary;
  __shared__ int32_t preceding;
  const int32_t row = blockIdx.x;
  const int64_t offset = static_cast<int64_t>(row) * width;
  if (threadIdx.x == 0) {
    boundary = find_top_p_boundary(sorted + offset,
                                   cumulative + offset,
                                   probability_cutoff(top_p, top_p_dtype, row),
                                   width);
    preceding = 0;
    for (int32_t tile = 0; tile < blockIdx.y; ++tile) {
      preceding += tie_counts[static_cast<int64_t>(row) * gridDim.y + tile];
    }
  }
  __syncthreads();
  if (boundary.discarded_ties == 0 || preceding >= boundary.discarded_ties) {
    return;
  }
  const int64_t tile =
      (static_cast<int64_t>(width) + gridDim.y - 1) / gridDim.y;
  const int64_t begin = blockIdx.y * tile;
  const int64_t end = min(begin + tile, static_cast<int64_t>(width));
  Scalar* values = logits + row * stride;
  for (int64_t chunk = begin; chunk < end; chunk += kThreads) {
    const int64_t col = chunk + threadIdx.x;
    const bool tied =
        col < end && static_cast<float>(values[col]) == boundary.value;
    int32_t rank;
    int32_t total;
    Scan(storage).InclusiveSum(static_cast<int32_t>(tied), rank, total);
    if (tied && preceding + rank <= boundary.discarded_ties) {
      values[col] = static_cast<Scalar>(-INFINITY);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      preceding += total;
    }
    __syncthreads();
  }
}

template <typename Scalar>
__global__ void scale_min_p_kernel(Scalar* logits,
                                   const void* temperatures,
                                   torch::ScalarType temperature_dtype,
                                   const float* log_min_p,
                                   float* partial_max,
                                   int32_t width,
                                   int64_t stride) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ float threshold;
  const int32_t row = blockIdx.x;
  Scalar* values = logits + row * stride;
  const float requested_temperature =
      temperatures ? parameter_value(temperatures, temperature_dtype, row)
                   : 1.0F;
  const float temperature =
      requested_temperature != 0.0F ? requested_temperature : 1.0F;
  float local_max = -INFINITY;
  for (int64_t col = threadIdx.x + static_cast<int64_t>(blockIdx.y) * kThreads;
       col < width;
       col += static_cast<int64_t>(gridDim.y) * kThreads) {
    // Match the in-place framework operation's rounding before min-p.
    const Scalar scaled = static_cast<Scalar>(
        __fdiv_rn(static_cast<float>(values[col]), temperature));
    values[col] = scaled;
    local_max = fmaxf(local_max, static_cast<float>(scaled));
  }
  if (!log_min_p) {
    return;
  }
  const float maximum = Reduce(storage).Reduce(local_max, MaxReduceOp());
  if (partial_max) {
    if (threadIdx.x == 0) {
      partial_max[static_cast<int64_t>(row) * gridDim.y + blockIdx.y] = maximum;
    }
    return;
  }
  if (threadIdx.x == 0) {
    threshold = maximum + log_min_p[row];
  }
  __syncthreads();
  for (int64_t col = threadIdx.x + static_cast<int64_t>(blockIdx.y) * kThreads;
       col < width;
       col += static_cast<int64_t>(gridDim.y) * kThreads) {
    if (static_cast<float>(values[col]) < threshold) {
      values[col] = static_cast<Scalar>(-INFINITY);
    }
  }
}

template <typename Scalar>
__global__ void apply_min_p_kernel(Scalar* logits,
                                   const float* partial_max,
                                   const float* log_min_p,
                                   int32_t width,
                                   int64_t stride) {
  __shared__ float threshold;
  const int32_t row = blockIdx.x;
  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (int32_t block = 0; block < gridDim.y; ++block) {
      maximum = fmaxf(
          maximum, partial_max[static_cast<int64_t>(row) * gridDim.y + block]);
    }
    threshold = maximum + log_min_p[row];
  }
  __syncthreads();
  Scalar* values = logits + row * stride;
  for (int64_t col = threadIdx.x + static_cast<int64_t>(blockIdx.y) * kThreads;
       col < width;
       col += static_cast<int64_t>(gridDim.y) * kThreads) {
    if (static_cast<float>(values[col]) < threshold) {
      values[col] = static_cast<Scalar>(-INFINITY);
    }
  }
}

template <typename Scalar>
__global__ void mask_top_k_kernel(Scalar* sorted,
                                  Scalar* logits,
                                  const int64_t* top_k,
                                  int32_t width,
                                  int64_t stride) {
  const int32_t row = blockIdx.x;
  Scalar* values = sorted + static_cast<int64_t>(row) * width;
  const int64_t requested_k = top_k[row];
  const int32_t k = requested_k <= 0 || requested_k > width
                        ? width
                        : static_cast<int32_t>(requested_k);
  const float threshold = static_cast<float>(values[width - k]);
  for (int64_t col = threadIdx.x + static_cast<int64_t>(blockIdx.y) * kThreads;
       col < width;
       col += static_cast<int64_t>(gridDim.y) * kThreads) {
    if (static_cast<float>(values[col]) < threshold) {
      values[col] = static_cast<Scalar>(-INFINITY);
    }
    if (static_cast<float>(logits[row * stride + col]) < threshold) {
      logits[row * stride + col] = static_cast<Scalar>(-INFINITY);
    }
  }
}

template <typename Scalar>
__global__ void filter_mask_kernel(uint8_t* mask,
                                   const Scalar* sorted,
                                   const int64_t* indices,
                                   const float* cumulative,
                                   const void* top_p,
                                   torch::ScalarType top_p_dtype,
                                   int32_t width) {
  const int32_t row = blockIdx.x;
  const int64_t offset = static_cast<int64_t>(row) * width;
  const float cutoff = probability_cutoff(top_p, top_p_dtype, row);
  const auto keep_token = [&](int64_t col) {
    return col == width - 1 || !(cumulative[offset + col] <= cutoff);
  };
  for (int64_t col = threadIdx.x + static_cast<int64_t>(blockIdx.y) * kThreads;
       col < width;
       col += static_cast<int64_t>(gridDim.y) * kThreads) {
    const bool keep = keep_token(col);
    const int64_t token = indices[offset + col];
    mask[offset + token] =
        keep && static_cast<float>(sorted[offset + col]) != -INFINITY;
  }
}

template <typename Scalar>
__global__ void apply_filter_mask_kernel(Scalar* logits,
                                         const uint8_t* mask,
                                         int32_t width,
                                         int64_t stride) {
  const int32_t row = blockIdx.x;
  const int64_t offset = static_cast<int64_t>(row) * width;
  Scalar* values = logits + row * stride;
  for (int64_t col = threadIdx.x + static_cast<int64_t>(blockIdx.y) * kThreads;
       col < width;
       col += static_cast<int64_t>(gridDim.y) * kThreads) {
    if (!mask[offset + col]) {
      values[col] = static_cast<Scalar>(-INFINITY);
    }
  }
}

void check_parameter(const torch::Tensor& parameter,
                     const torch::Tensor& logits,
                     torch::ScalarType dtype) {
  if (!parameter.defined()) {
    return;
  }
  CHECK(parameter.device() == logits.device());
  CHECK(parameter.scalar_type() == dtype);
  CHECK_EQ(parameter.dim(), 1);
  CHECK_EQ(parameter.size(0), logits.size(0));
  CHECK(parameter.is_contiguous());
}

}  // namespace

void fused_logits_transform(torch::Tensor& logits,
                            const torch::Tensor& temperatures,
                            const torch::Tensor& min_p,
                            const torch::Tensor& top_k,
                            const torch::Tensor& top_p) {
  CHECK(logits.is_cuda());
  CHECK_EQ(logits.dim(), 2);
  CHECK_EQ(logits.stride(1), 1);
  CHECK_GE(logits.stride(0), logits.size(1));
  CHECK_GT(logits.size(1), 0);
  CHECK_LE(logits.size(0), INT32_MAX);
  CHECK_LE(logits.size(1), INT32_MAX);
  CHECK(logits.scalar_type() == torch::kFloat32 ||
        logits.scalar_type() == torch::kFloat16 ||
        logits.scalar_type() == torch::kBFloat16);
  for (const auto& parameter : {temperatures, min_p, top_p}) {
    if (!parameter.defined()) {
      continue;
    }
    CHECK(parameter.scalar_type() == torch::kFloat32 ||
          parameter.scalar_type() == torch::kFloat16 ||
          parameter.scalar_type() == torch::kBFloat16);
    check_parameter(parameter, logits, parameter.scalar_type());
  }
  check_parameter(top_k, logits, torch::kInt64);
  const c10::cuda::CUDAGuard guard(logits.device());
  const int32_t rows = static_cast<int32_t>(logits.size(0));
  const int32_t width = static_cast<int32_t>(logits.size(1));
  if (rows == 0) {
    return;
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t blocks_per_row =
      (static_cast<int64_t>(width) + kThreads - 1) / kThreads;
  const dim3 grid(
      rows,
      blocks_per_row < kMaxBlocksPerRow ? blocks_per_row : kMaxBlocksPerRow);

  const auto log_min_p =
      min_p.defined() ? min_p.to(torch::kFloat32).log() : torch::Tensor();
  const auto partial_max =
      min_p.defined() && grid.y > 1
          ? torch::empty({rows, grid.y},
                         logits.options().dtype(torch::kFloat32))
          : torch::Tensor();
  DISPATCH_FLOATING_TYPES(logits.scalar_type(), "scale_min_p", [&] {
    if (temperatures.defined() || min_p.defined()) {
      scale_min_p_kernel<scalar_t><<<grid, kThreads, 0, stream>>>(
          logits.data_ptr<scalar_t>(),
          temperatures.defined() ? temperatures.data_ptr() : nullptr,
          temperatures.defined() ? temperatures.scalar_type() : torch::kFloat32,
          log_min_p.defined() ? log_min_p.data_ptr<float>() : nullptr,
          partial_max.defined() ? partial_max.data_ptr<float>() : nullptr,
          width,
          logits.stride(0));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      if (partial_max.defined()) {
        apply_min_p_kernel<scalar_t>
            <<<grid, kThreads, 0, stream>>>(logits.data_ptr<scalar_t>(),
                                            partial_max.data_ptr<float>(),
                                            log_min_p.data_ptr<float>(),
                                            width,
                                            logits.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
  });
  if (!top_k.defined() && !top_p.defined()) {
    return;
  }
  // Small framework sorts have a distinct tie order. Large sorts are stable,
  // so their boundary ties can be reconstructed without sorting token indices.
  const bool small_sort = at::native::should_use_small_sort(logits, 1);
  const bool select_k =
      !small_sort && top_k.defined() && rows > kSmallBatchRows;
  torch::Tensor sorted;
  torch::Tensor indices;
  torch::Tensor thresholds;
  if (select_k) {
    thresholds = torch::empty({rows}, logits.options().dtype(torch::kFloat32));
    DISPATCH_FLOATING_TYPES(logits.scalar_type(), "select_top_k", [&] {
      select_top_k_kernel<scalar_t>
          <<<rows, kThreads, 0, stream>>>(logits.data_ptr<scalar_t>(),
                                          top_k.data_ptr<int64_t>(),
                                          thresholds.data_ptr<float>(),
                                          width,
                                          logits.stride(0));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      apply_threshold_kernel<scalar_t>
          <<<grid, kThreads, 0, stream>>>(logits.data_ptr<scalar_t>(),
                                          thresholds.data_ptr<float>(),
                                          width,
                                          logits.stride(0));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!top_p.defined()) {
      return;
    }
  }
  if (small_sort) {
    std::tie(sorted, indices) = logits.sort(/*dim=*/-1, /*descending=*/false);
  } else {
    DISPATCH_FLOATING_TYPES(logits.scalar_type(), "sort_values", [&] {
      if (select_k) {
        sorted = torch::full(logits.sizes(), -INFINITY, logits.options());
        const auto ends =
            torch::empty({rows}, logits.options().dtype(torch::kInt64));
        compact_sort_kernel<scalar_t>
            <<<rows, kThreads, 0, stream>>>(logits.data_ptr<scalar_t>(),
                                            thresholds.data_ptr<float>(),
                                            sorted.data_ptr<scalar_t>(),
                                            ends.data_ptr<int64_t>(),
                                            width,
                                            logits.stride(0));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        sorted = sort_values<scalar_t>(logits, stream, sorted, ends);
      } else {
        sorted = sort_values<scalar_t>(logits, stream);
      }
    });
  }
  DISPATCH_FLOATING_TYPES(logits.scalar_type(), "mask_top_k", [&] {
    if (top_k.defined() && !select_k) {
      mask_top_k_kernel<scalar_t>
          <<<grid, kThreads, 0, stream>>>(sorted.data_ptr<scalar_t>(),
                                          logits.data_ptr<scalar_t>(),
                                          top_k.data_ptr<int64_t>(),
                                          width,
                                          logits.stride(0));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });
  if (!top_p.defined()) {
    return;
  }
  // Preserve the full-width framework reductions: compact softmax/cumsum
  // would round differently and can change the token at the top-p boundary.
  auto cumulative = sorted.softmax(-1, torch::kFloat32);
  torch::cumsum_out(cumulative, cumulative, -1);
  if (!small_sort) {
    const auto tie_counts =
        torch::empty({rows, grid.y}, logits.options().dtype(torch::kInt32));
    DISPATCH_FLOATING_TYPES(logits.scalar_type(), "filter_threshold", [&] {
      filter_count_kernel<scalar_t>
          <<<grid, kThreads, 0, stream>>>(logits.data_ptr<scalar_t>(),
                                          sorted.data_ptr<scalar_t>(),
                                          cumulative.data_ptr<float>(),
                                          top_p.data_ptr(),
                                          top_p.scalar_type(),
                                          tie_counts.data_ptr<int32_t>(),
                                          width,
                                          logits.stride(0));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      filter_ties_kernel<scalar_t>
          <<<grid, kThreads, 0, stream>>>(logits.data_ptr<scalar_t>(),
                                          sorted.data_ptr<scalar_t>(),
                                          cumulative.data_ptr<float>(),
                                          top_p.data_ptr(),
                                          top_p.scalar_type(),
                                          tie_counts.data_ptr<int32_t>(),
                                          width,
                                          logits.stride(0));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  // Scatter byte masks so the working set stays small; write the wider
  // logits in token order to avoid costly random stores on large batches.
  const auto mask =
      torch::empty(logits.sizes(), logits.options().dtype(torch::kUInt8));
  DISPATCH_FLOATING_TYPES(logits.scalar_type(), "filter_mask", [&] {
    filter_mask_kernel<scalar_t>
        <<<grid, kThreads, 0, stream>>>(mask.data_ptr<uint8_t>(),
                                        sorted.data_ptr<scalar_t>(),
                                        indices.data_ptr<int64_t>(),
                                        cumulative.data_ptr<float>(),
                                        top_p.data_ptr(),
                                        top_p.scalar_type(),
                                        width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    apply_filter_mask_kernel<scalar_t>
        <<<grid, kThreads, 0, stream>>>(logits.data_ptr<scalar_t>(),
                                        mask.data_ptr<uint8_t>(),
                                        width,
                                        logits.stride(0));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
