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

// c10/cuda headers are remapped by mcc_wrapper on MUSA builds; keep the
// shared CUDA exception macros while launching with an explicit musaStream_t.
#include <c10/cuda/CUDAException.h>
#include <musa.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/kernels/cuda/device_utils.cuh"
#include "core/kernels/musa/musa_ops_api.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm::kernel::musa {
namespace {

constexpr int32_t kCombineBlockSize = 256;
constexpr std::size_t kSharedMemoryBytes = 0;
constexpr int32_t kValuesPerChunk = 8;

template <typename ScalarType>
__global__ void XLLM_KERNEL_ATTR(kCombineBlockSize)
    moe_combine_indexed_kernel(const ScalarType* __restrict__ gemm2_sorted,
                               const int32_t* __restrict__ sorted_positions,
                               const float* __restrict__ reduce_weight,
                               ScalarType* __restrict__ output,
                               int64_t num_tokens,
                               int32_t top_k,
                               int64_t gemm2_rows,
                               int64_t hidden_size) {
  const int64_t token_id = static_cast<int64_t>(blockIdx.x);
  if (token_id >= num_tokens) {
    return;
  }

  const int64_t thread_id = static_cast<int64_t>(threadIdx.x);
  for (int64_t hidden_idx = thread_id; hidden_idx < hidden_size;
       hidden_idx += static_cast<int64_t>(kCombineBlockSize)) {
    float accumulator = 0.0f;
    for (int32_t top_k_idx = 0; top_k_idx < top_k; ++top_k_idx) {
      const int64_t flat_idx = token_id * static_cast<int64_t>(top_k) +
                               static_cast<int64_t>(top_k_idx);
      const int32_t sorted_idx = sorted_positions[flat_idx];
      if (sorted_idx >= 0 && static_cast<int64_t>(sorted_idx) < gemm2_rows) {
        accumulator +=
            reduce_weight[flat_idx] *
            static_cast<float>(
                gemm2_sorted[static_cast<int64_t>(sorted_idx) * hidden_size +
                             hidden_idx]);
      }
    }
    output[token_id * hidden_size + hidden_idx] =
        static_cast<ScalarType>(accumulator);
  }
}

union Bf16Pack8 {
  int4 vector;
  c10::BFloat16 values[kValuesPerChunk];
};

__global__ void XLLM_KERNEL_ATTR(kCombineBlockSize)
    moe_combine_indexed_bf16_vec8_kernel(
        const c10::BFloat16* __restrict__ gemm2_sorted,
        const int32_t* __restrict__ sorted_positions,
        const float* __restrict__ reduce_weight,
        c10::BFloat16* __restrict__ output,
        int64_t num_tokens,
        int32_t top_k,
        int64_t gemm2_rows,
        int64_t hidden_size) {
  const int64_t token_id = static_cast<int64_t>(blockIdx.x);
  if (token_id >= num_tokens) {
    return;
  }

  const int64_t chunks_per_row =
      hidden_size / static_cast<int64_t>(kValuesPerChunk);
  for (int64_t chunk_idx = static_cast<int64_t>(threadIdx.x);
       chunk_idx < chunks_per_row;
       chunk_idx += static_cast<int64_t>(blockDim.x)) {
    float accumulators[kValuesPerChunk] = {};
    const int64_t column = chunk_idx * static_cast<int64_t>(kValuesPerChunk);
    for (int32_t top_k_idx = 0; top_k_idx < top_k; ++top_k_idx) {
      const int64_t flat_idx = token_id * static_cast<int64_t>(top_k) +
                               static_cast<int64_t>(top_k_idx);
      const int32_t sorted_idx = sorted_positions[flat_idx];
      if (sorted_idx < 0 || static_cast<int64_t>(sorted_idx) >= gemm2_rows) {
        continue;
      }

      const float weight = reduce_weight[flat_idx];
      Bf16Pack8 input_pack;
      input_pack.vector = *reinterpret_cast<const int4*>(
          gemm2_sorted + static_cast<int64_t>(sorted_idx) * hidden_size +
          column);
#pragma unroll
      for (int32_t value_idx = 0; value_idx < kValuesPerChunk; ++value_idx) {
        accumulators[value_idx] +=
            weight * static_cast<float>(input_pack.values[value_idx]);
      }
    }

    Bf16Pack8 output_pack;
#pragma unroll
    for (int32_t value_idx = 0; value_idx < kValuesPerChunk; ++value_idx) {
      output_pack.values[value_idx] =
          static_cast<c10::BFloat16>(accumulators[value_idx]);
    }
    *reinterpret_cast<int4*>(output + token_id * hidden_size + column) =
        output_pack.vector;
  }
}

}  // namespace

torch::Tensor moe_combine_result_indexed(const torch::Tensor& gemm2_sorted,
                                         const torch::Tensor& sorted_positions,
                                         const torch::Tensor& reduce_weight,
                                         int64_t num_tokens,
                                         int32_t top_k) {
  CHECK_GE(num_tokens, 0);
  CHECK_LE(num_tokens,
           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));
  CHECK_GT(top_k, 0);
  CHECK_EQ(gemm2_sorted.dim(), 2);
  CHECK_EQ(sorted_positions.dim(), 1);
  CHECK_EQ(reduce_weight.dim(), 2);
  CHECK_LE(num_tokens,
           std::numeric_limits<int64_t>::max() / static_cast<int64_t>(top_k));
  const int64_t num_routed_tokens = num_tokens * static_cast<int64_t>(top_k);
  CHECK_GE(gemm2_sorted.size(0), num_routed_tokens);
  CHECK_EQ(sorted_positions.numel(), num_routed_tokens);
  CHECK_EQ(reduce_weight.size(0), num_tokens);
  CHECK_EQ(reduce_weight.size(1), top_k);
  CHECK_EQ(sorted_positions.scalar_type(), torch::kInt32);
  CHECK(reduce_weight.is_floating_point());
  CHECK(gemm2_sorted.device().is_cuda() ||
        gemm2_sorted.device().is_privateuseone());
  CHECK(gemm2_sorted.is_contiguous());
  CHECK(sorted_positions.is_contiguous());
  CHECK(sorted_positions.device() == gemm2_sorted.device());

  const int64_t hidden_size = gemm2_sorted.size(1);
  CHECK_GT(hidden_size, 0);
  const torch::ScalarType scalar_type = gemm2_sorted.scalar_type();
  CHECK(scalar_type == torch::kFloat16 || scalar_type == torch::kBFloat16 ||
        scalar_type == torch::kFloat32);
  torch::Tensor output =
      torch::empty({num_tokens, hidden_size}, gemm2_sorted.options());
  if (num_tokens == 0) {
    return output;
  }

  torch::Tensor reduce_weight_fp32 =
      reduce_weight.to(gemm2_sorted.device(), torch::kFloat32).contiguous();
  const musaStream_t stream =
      static_cast<musaStream_t>(c10::musa::getCurrentMUSAStream().stream());

  if (scalar_type == torch::kFloat16) {
    moe_combine_indexed_kernel<c10::Half>
        <<<num_tokens, kCombineBlockSize, kSharedMemoryBytes, stream>>>(
            gemm2_sorted.data_ptr<c10::Half>(),
            sorted_positions.data_ptr<int32_t>(),
            reduce_weight_fp32.data_ptr<float>(),
            output.data_ptr<c10::Half>(),
            num_tokens,
            top_k,
            gemm2_sorted.size(0),
            hidden_size);
  } else if (scalar_type == torch::kBFloat16) {
    const c10::BFloat16* input_ptr = gemm2_sorted.data_ptr<c10::BFloat16>();
    c10::BFloat16* output_ptr = output.data_ptr<c10::BFloat16>();
    const bool is_vec8_aligned =
        reinterpret_cast<uintptr_t>(input_ptr) % alignof(int4) == 0 &&
        reinterpret_cast<uintptr_t>(output_ptr) % alignof(int4) == 0;
    if (hidden_size % static_cast<int64_t>(kValuesPerChunk) == 0 &&
        is_vec8_aligned) {
      moe_combine_indexed_bf16_vec8_kernel<<<num_tokens,
                                             kCombineBlockSize,
                                             kSharedMemoryBytes,
                                             stream>>>(
          input_ptr,
          sorted_positions.data_ptr<int32_t>(),
          reduce_weight_fp32.data_ptr<float>(),
          output_ptr,
          num_tokens,
          top_k,
          gemm2_sorted.size(0),
          hidden_size);
    } else {
      moe_combine_indexed_kernel<c10::BFloat16>
          <<<num_tokens, kCombineBlockSize, kSharedMemoryBytes, stream>>>(
              input_ptr,
              sorted_positions.data_ptr<int32_t>(),
              reduce_weight_fp32.data_ptr<float>(),
              output_ptr,
              num_tokens,
              top_k,
              gemm2_sorted.size(0),
              hidden_size);
    }
  } else {
    moe_combine_indexed_kernel<float>
        <<<num_tokens, kCombineBlockSize, kSharedMemoryBytes, stream>>>(
            gemm2_sorted.data_ptr<float>(),
            sorted_positions.data_ptr<int32_t>(),
            reduce_weight_fp32.data_ptr<float>(),
            output.data_ptr<float>(),
            num_tokens,
            top_k,
            gemm2_sorted.size(0),
            hidden_size);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace xllm::kernel::musa
