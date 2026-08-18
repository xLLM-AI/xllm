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
#include <musa_bf16.h>
#include <musa_fp8.h>

#include <cstdint>
#include <limits>
#include <tuple>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm::kernel::musa {

namespace {

// DeepSeek block-FP8 activation quantization, specialized for the Qwen3.5
// serving path: bf16 input, e4m3 output, group size 128 along K, row-major
// scale grid [M, K/128]. Uses 16 lanes per group (8 bf16 elements per
// lane), and maps the two logical dimensions to a
// 2-D grid: blockIdx.x selects a tile of groups and blockIdx.y selects a row.
constexpr int32_t kGroupSize = 128;
constexpr int32_t kThreadsPerGroup = 16;
constexpr int32_t kElemsPerThread = kGroupSize / kThreadsPerGroup;  // 8
constexpr int32_t kActThreadsPerGroup = 32;
constexpr int32_t kActElemsPerThread = kGroupSize / kActThreadsPerGroup;  // 4
constexpr int32_t kHardwareWarpThreads = 32;
constexpr int32_t kMaxMoeExperts = 256;
constexpr int32_t kMaxMoeTopk = 16;
constexpr int32_t kRaggedAlignment = 128;
constexpr int32_t kRaggedCombineThreads = 256;
constexpr float kFp8E4M3Max = 448.0f;
constexpr float kEps = 1e-10f;

template <int32_t ThreadsPerGroup>
__device__ __forceinline__ float group_reduce_max(float value, int32_t lane) {
  static_assert(ThreadsPerGroup <= kHardwareWarpThreads,
                "FP8 quantization subwarp must fit in one hardware warp");
  constexpr uint32_t kSubwarpMaskBits = ThreadsPerGroup == kHardwareWarpThreads
                                            ? 0xffffffffu
                                            : (1u << ThreadsPerGroup) - 1u;
  // threadIdx.x is a block-wide index. Restrict it to the physical warp lane
  // before constructing the mask; using threadIdx.x directly would shift by
  // 32 for the second warp and is undefined for a 32-bit mask.
  const int32_t warp_lane =
      static_cast<int32_t>(threadIdx.x) & (kHardwareWarpThreads - 1);
  const uint32_t mask = kSubwarpMaskBits << (warp_lane - lane);

  if constexpr (ThreadsPerGroup >= 32) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, 16));
  }
  if constexpr (ThreadsPerGroup >= 16) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, 8));
  }
  if constexpr (ThreadsPerGroup >= 8) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, 4));
  }
  if constexpr (ThreadsPerGroup >= 4) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, 2));
  }
  if constexpr (ThreadsPerGroup >= 2) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, 1));
  }
  return value;
}

// The SLC store is available on MP31. Keep a normal global store for other
// MUSA targets so the kernel remains usable in development and correctness
// builds on older devices.
__device__ __forceinline__ void store_b64_slc_new(uint64_t* ptr,
                                                  uint64_t value) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  asm volatile(
      "LSU.ST.B64 %1, %0, _, 8, 1, 1, inner_persist=4, outer_persist=2, "
      "chrnt=l2_l3, slc=new, persist=0, stride_add_first=0"
      :
      : "R"(ptr), "R"(value));
#else
  *ptr = value;
#endif
}

__global__ void per_token_group_quant_fp8_bf16_g128_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    int64_t hidden_size,
    int64_t hidden_dim_num_groups,
    int64_t num_tokens) {
  const int32_t subwarp_id =
      static_cast<int32_t>(threadIdx.x) / kThreadsPerGroup;
  const int32_t subwarps_per_block =
      static_cast<int32_t>(blockDim.x) / kThreadsPerGroup;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % kThreadsPerGroup;
  const int64_t token_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t group_idx =
      static_cast<int64_t>(blockIdx.x) * subwarps_per_block + subwarp_id;
  if (token_idx >= num_tokens || group_idx >= hidden_dim_num_groups) {
    return;
  }

  const int64_t group_offset =
      token_idx * hidden_size + group_idx * static_cast<int64_t>(kGroupSize);
  const int64_t in_offset =
      group_offset + static_cast<int64_t>(lane) * kElemsPerThread;

  // Every lane consumes 16 contiguous bytes (8 bf16 values). Group starts
  // and lane offsets are 16-byte aligned for the contiguous tensors produced
  // by the caller.
  const int4 input_vec = *reinterpret_cast<const int4*>(input + in_offset);
  const __mt_bfloat16* input_values =
      reinterpret_cast<const __mt_bfloat16*>(&input_vec);

  float vals[kElemsPerThread];
  float local_absmax = 0.0f;
#pragma unroll
  for (int32_t j = 0; j < kElemsPerThread; ++j) {
    vals[j] = __bfloat162float(input_values[j]);
    local_absmax = fmaxf(local_absmax, fabsf(vals[j]));
  }

  // Every lane ends with the group absmax, so all lanes can quantize locally.
  local_absmax = group_reduce_max<kThreadsPerGroup>(local_absmax, lane);

  const float scale_inv = fmaxf(local_absmax / kFp8E4M3Max, kEps);
  if (lane == 0) {
    output_s[token_idx * hidden_dim_num_groups + group_idx] = scale_inv;
  }

  const float scale = 1.0f / scale_inv;
  const int64_t output_offset =
      group_offset + static_cast<int64_t>(lane) * kElemsPerThread;
  const float4 scaled_0 = make_float4(
      vals[0] * scale, vals[1] * scale, vals[2] * scale, vals[3] * scale);
  const float4 scaled_1 = make_float4(
      vals[4] * scale, vals[5] * scale, vals[6] * scale, vals[7] * scale);
  const uint32_t packed_0 = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(scaled_0, __MT_SATFINITE, __MT_E4M3));
  const uint32_t packed_1 = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(scaled_1, __MT_SATFINITE, __MT_E4M3));
  const uint64_t packed =
      static_cast<uint64_t>(packed_0) | (static_cast<uint64_t>(packed_1) << 32);
  store_b64_slc_new(reinterpret_cast<uint64_t*>(output_q + output_offset),
                    packed);
}

__global__ void moe_preprocess_histogram_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ expert_counts,
    int64_t assignment_count,
    int32_t num_experts) {
  const int64_t assignment_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (assignment_idx >= assignment_count) {
    return;
  }
  const int32_t expert_id = topk_ids[assignment_idx];
  if (expert_id >= 0 && expert_id < num_experts) {
    atomicAdd(expert_counts + expert_id, 1);
  }
}

__global__ void moe_preprocess_prefix_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_cursor,
    int32_t num_experts) {
  if (threadIdx.x != 0) {
    return;
  }
  int32_t offset = 0;
  for (int32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    expert_cursor[expert_id] = offset;
    offset += expert_counts[expert_id];
  }
}

__global__ void moe_preprocess_prefix_aligned_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_cursor,
    int32_t* __restrict__ group_m_counts,
    int32_t num_experts,
    int32_t alignment,
    int32_t padded_rows) {
  if (threadIdx.x != 0) {
    return;
  }
  int32_t offset = 0;
  for (int32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    expert_cursor[expert_id] = offset;
    const int32_t count = expert_counts[expert_id];
    const int32_t aligned_count =
        ((count + alignment - 1) / alignment) * alignment;
    if (group_m_counts != nullptr) {
      group_m_counts[expert_id] = aligned_count;
    }
    offset += aligned_count;
  }
  // The allocation is an upper bound on the sum of per-expert aligned
  // counts. Keep the grouped-GEMM counts summing to the allocated M by
  // assigning the harmless padding tail to the final expert.
  if (group_m_counts != nullptr && num_experts > 0 && offset < padded_rows) {
    group_m_counts[num_experts - 1] += padded_rows - offset;
  }
}

// Token-major BF16 gather/reorder for the contiguous Ragged GEMM path.  The
// histogram/prefix kernels provide each expert's aligned start; this kernel
// performs one hidden-state read per input token and fans it out to top-k
// expert rows.  `original_to_padded` is consumed after the down GEMM to avoid
// a second host-side sort/index pipeline.
__global__ void moe_preprocess_assign_bf16_kernel(
    const __mt_bfloat16* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ expert_cursor,
    int32_t* __restrict__ original_to_padded,
    int32_t* __restrict__ row_expert_ids,
    __mt_bfloat16* __restrict__ output,
    int64_t num_tokens,
    int64_t hidden_size,
    int32_t topk,
    int32_t num_experts) {
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
  if (token_idx >= num_tokens) {
    return;
  }

  __shared__ int32_t destinations[kMaxMoeTopk];
  const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
  if (thread_idx < topk) {
    const int64_t assignment_idx = token_idx * topk + thread_idx;
    const int32_t expert_id = topk_ids[assignment_idx];
    if (expert_id >= 0 && expert_id < num_experts) {
      const int32_t destination = atomicAdd(expert_cursor + expert_id, 1);
      destinations[thread_idx] = destination;
      original_to_padded[assignment_idx] = destination;
      row_expert_ids[destination] = expert_id;
    } else {
      destinations[thread_idx] = -1;
      original_to_padded[assignment_idx] = -1;
    }
  }
  __syncthreads();

  constexpr int32_t kBf16PerVector = sizeof(int4) / sizeof(__mt_bfloat16);
  if (hidden_size % kBf16PerVector == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(int4) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(int4) == 0) {
    const int64_t vectors_per_row = hidden_size / kBf16PerVector;
    const auto* input_vec = reinterpret_cast<const int4*>(input);
    auto* output_vec = reinterpret_cast<int4*>(output);
    for (int64_t vector_idx = thread_idx; vector_idx < vectors_per_row;
         vector_idx += static_cast<int64_t>(blockDim.x)) {
      const int4 value = input_vec[token_idx * vectors_per_row + vector_idx];
      for (int32_t topk_idx = 0; topk_idx < topk; ++topk_idx) {
        const int32_t destination = destinations[topk_idx];
        if (destination >= 0) {
          output_vec[static_cast<int64_t>(destination) * vectors_per_row +
                     vector_idx] = value;
        }
      }
    }
    return;
  }

  for (int64_t hidden_idx = thread_idx; hidden_idx < hidden_size;
       hidden_idx += static_cast<int64_t>(blockDim.x)) {
    const __mt_bfloat16 value = input[token_idx * hidden_size + hidden_idx];
    for (int32_t topk_idx = 0; topk_idx < topk; ++topk_idx) {
      const int32_t destination = destinations[topk_idx];
      if (destination >= 0) {
        output[static_cast<int64_t>(destination) * hidden_size + hidden_idx] =
            value;
      }
    }
  }
}

// Token-major fused routing/gather/quant
// (deep_gemm_contig_preprocess_fp8_assign_compact). Each block
// reads one hidden row once per g128 group, quantizes it once, and writes the
// same packed values into all selected expert-major destinations.
__global__ void moe_preprocess_assign_quant_kernel(
    const __mt_bfloat16* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ expert_cursor,
    int32_t* __restrict__ src_to_dst,
    __mt_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    int64_t num_tokens,
    int64_t hidden_size,
    int32_t hidden_dim_num_groups,
    int32_t topk,
    int32_t num_experts) {
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
  if (token_idx >= num_tokens) {
    return;
  }

  __shared__ int32_t destinations[kMaxMoeTopk];
  const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
  if (thread_idx < topk) {
    const int64_t assignment_idx = token_idx * topk + thread_idx;
    const int32_t expert_id = topk_ids[assignment_idx];
    if (expert_id >= 0 && expert_id < num_experts) {
      const int32_t destination = atomicAdd(expert_cursor + expert_id, 1);
      destinations[thread_idx] = destination;
      src_to_dst[assignment_idx] = destination;
    } else {
      destinations[thread_idx] = -1;
      src_to_dst[assignment_idx] = -1;
    }
  }
  __syncthreads();

  const int32_t group_idx = thread_idx / kActThreadsPerGroup;
  const int32_t lane = thread_idx % kActThreadsPerGroup;
  if (group_idx >= hidden_dim_num_groups) {
    return;
  }

  const int64_t elem_offset = static_cast<int64_t>(lane) * kActElemsPerThread;
  const int64_t input_base = token_idx * hidden_size +
                             static_cast<int64_t>(group_idx) * kGroupSize +
                             elem_offset;
  const uint64_t input_u64 =
      *reinterpret_cast<const uint64_t*>(input + input_base);
  const __mt_bfloat16* input_values =
      reinterpret_cast<const __mt_bfloat16*>(&input_u64);

  float values[kActElemsPerThread];
  float local_absmax = kEps;
#pragma unroll
  for (int32_t j = 0; j < kActElemsPerThread; ++j) {
    const float value = __bfloat162float(input_values[j]);
    values[j] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }
  local_absmax = group_reduce_max<kActThreadsPerGroup>(local_absmax, lane);
  const float scale_inv = local_absmax / kFp8E4M3Max;
  const float scale = kFp8E4M3Max / local_absmax;
  const float4 scaled = make_float4(values[0] * scale,
                                    values[1] * scale,
                                    values[2] * scale,
                                    values[3] * scale);
  const uint32_t packed = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(scaled, __MT_SATFINITE, __MT_E4M3));

  for (int32_t topk_idx = 0; topk_idx < topk; ++topk_idx) {
    const int32_t destination = destinations[topk_idx];
    if (destination < 0) {
      continue;
    }
    const int64_t output_base =
        static_cast<int64_t>(destination) * hidden_size +
        static_cast<int64_t>(group_idx) * kGroupSize + elem_offset;
    *reinterpret_cast<uint32_t*>(output_q + output_base) = packed;
    if (lane == 0) {
      output_s[static_cast<int64_t>(destination) * hidden_dim_num_groups +
               group_idx] = scale_inv;
    }
  }
}

// Decode specialization: each top-k assignment owns one fixed aligned block.
// Only the first row is valid; Mate's Ragged kernel skips the -1 padding rows.
// A token is read and quantized once, then copied to its top-k block starts.
__global__ void moe_ragged_preprocess_assign_quant_kernel(
    const __mt_bfloat16* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    __mt_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    int32_t* __restrict__ row_expert_ids,
    int64_t num_tokens,
    int64_t hidden_size,
    int32_t hidden_dim_num_groups,
    int32_t topk) {
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
  if (token_idx >= num_tokens) {
    return;
  }

  const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
  const int64_t token_block_start =
      token_idx * static_cast<int64_t>(topk) * kRaggedAlignment;
  const int32_t rows_per_token = topk * kRaggedAlignment;
  for (int32_t row = thread_idx; row < rows_per_token;
       row += static_cast<int32_t>(blockDim.x)) {
    row_expert_ids[token_block_start + row] = -1;
  }
  __syncthreads();
  if (thread_idx < topk) {
    const int64_t assignment_idx = token_idx * topk + thread_idx;
    row_expert_ids[token_block_start +
                   static_cast<int64_t>(thread_idx) * kRaggedAlignment] =
        topk_ids[assignment_idx];
  }

  const int32_t group_idx = thread_idx / kActThreadsPerGroup;
  const int32_t lane = thread_idx % kActThreadsPerGroup;
  if (group_idx >= hidden_dim_num_groups) {
    return;
  }

  const int64_t elem_offset = static_cast<int64_t>(lane) * kActElemsPerThread;
  const int64_t input_base = token_idx * hidden_size +
                             static_cast<int64_t>(group_idx) * kGroupSize +
                             elem_offset;
  const uint64_t input_u64 =
      *reinterpret_cast<const uint64_t*>(input + input_base);
  const __mt_bfloat16* input_values =
      reinterpret_cast<const __mt_bfloat16*>(&input_u64);

  float values[kActElemsPerThread];
  float local_absmax = kEps;
#pragma unroll
  for (int32_t j = 0; j < kActElemsPerThread; ++j) {
    const float value = __bfloat162float(input_values[j]);
    values[j] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }
  local_absmax = group_reduce_max<kActThreadsPerGroup>(local_absmax, lane);
  const float scale_inv = local_absmax / kFp8E4M3Max;
  const float scale = kFp8E4M3Max / local_absmax;
  const float4 scaled = make_float4(values[0] * scale,
                                    values[1] * scale,
                                    values[2] * scale,
                                    values[3] * scale);
  const uint32_t packed = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(scaled, __MT_SATFINITE, __MT_E4M3));

  for (int32_t topk_idx = 0; topk_idx < topk; ++topk_idx) {
    const int64_t row =
        token_block_start + static_cast<int64_t>(topk_idx) * kRaggedAlignment;
    const int64_t output_base = row * hidden_size +
                                static_cast<int64_t>(group_idx) * kGroupSize +
                                elem_offset;
    *reinterpret_cast<uint32_t*>(output_q + output_base) = packed;
    if (lane == 0) {
      output_s[row * hidden_dim_num_groups + group_idx] = scale_inv;
    }
  }
}

// BF16 decode specialization matching the fixed-block Ragged FP8 layout.
// Each top-k assignment owns one aligned block, while only its first row is
// valid. Mate's Ragged GEMM uses row_expert_ids to skip every padding row.
__global__ void moe_ragged_preprocess_assign_bf16_kernel(
    const __mt_bfloat16* __restrict__ input,
    const int32_t* __restrict__ topk_ids,
    __mt_bfloat16* __restrict__ output,
    int32_t* __restrict__ row_expert_ids,
    int64_t num_tokens,
    int64_t hidden_size,
    int32_t topk) {
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
  if (token_idx >= num_tokens) {
    return;
  }

  const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
  const int64_t token_block_start =
      token_idx * static_cast<int64_t>(topk) * kRaggedAlignment;
  const int32_t rows_per_token = topk * kRaggedAlignment;
  for (int32_t row = thread_idx; row < rows_per_token;
       row += static_cast<int32_t>(blockDim.x)) {
    row_expert_ids[token_block_start + row] = -1;
  }
  __syncthreads();

  if (thread_idx < topk) {
    const int64_t assignment_idx = token_idx * topk + thread_idx;
    row_expert_ids[token_block_start +
                   static_cast<int64_t>(thread_idx) * kRaggedAlignment] =
        topk_ids[assignment_idx];
  }

  for (int64_t hidden_idx = thread_idx; hidden_idx < hidden_size;
       hidden_idx += static_cast<int64_t>(blockDim.x)) {
    const __mt_bfloat16 value = input[token_idx * hidden_size + hidden_idx];
    for (int32_t topk_idx = 0; topk_idx < topk; ++topk_idx) {
      const int64_t row =
          token_block_start + static_cast<int64_t>(topk_idx) * kRaggedAlignment;
      output[row * hidden_size + hidden_idx] = value;
    }
  }
}

// Small-batch BF16 decode router. Keeping the complete histogram and prefix
// operation in one thread block avoids the global cursor atomic behavior that
// is not reliable for MUSA overlap batches. The resulting rows remain grouped
// by expert, as required by Mate's 16-bit Ragged GEMM.
__global__ void moe_decode_route_bf16_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ original_to_padded,
    int32_t* __restrict__ row_expert_ids,
    int32_t* __restrict__ group_m_counts,
    int64_t assignment_count,
    int32_t num_experts,
    int32_t alignment,
    int32_t padded_rows) {
  __shared__ int32_t expert_counts[kMaxMoeExperts];
  __shared__ int32_t expert_cursor[kMaxMoeExperts];
  const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
  if (thread_idx == 0) {
    for (int32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      expert_counts[expert_id] = 0;
    }
    for (int64_t assignment_idx = 0; assignment_idx < assignment_count;
         ++assignment_idx) {
      const int32_t expert_id = topk_ids[assignment_idx];
      if (expert_id >= 0 && expert_id < num_experts) {
        ++expert_counts[expert_id];
      }
    }
    int32_t offset = 0;
    for (int32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      expert_cursor[expert_id] = offset;
      const int32_t count = expert_counts[expert_id];
      const int32_t aligned_count =
          ((count + alignment - 1) / alignment) * alignment;
      if (group_m_counts != nullptr) {
        group_m_counts[expert_id] = aligned_count;
      }
      offset += aligned_count;
    }
    // The allocation formula is an upper bound on the sum of per-expert
    // aligned counts. Keep grouped-GEMM counts summing to the allocated M by
    // assigning any final tail to the last expert. Those rows are padding and
    // are ignored by the indexed combine.
    if (group_m_counts != nullptr && num_experts > 0 && offset < padded_rows) {
      group_m_counts[num_experts - 1] += padded_rows - offset;
    }
    for (int64_t assignment_idx = 0; assignment_idx < assignment_count;
         ++assignment_idx) {
      const int32_t expert_id = topk_ids[assignment_idx];
      if (expert_id >= 0 && expert_id < num_experts) {
        const int32_t destination = expert_cursor[expert_id]++;
        original_to_padded[assignment_idx] = destination;
        row_expert_ids[destination] = expert_id;
      } else {
        original_to_padded[assignment_idx] = -1;
      }
    }
  }
}

__global__ void moe_decode_assign_bf16_kernel(
    const __mt_bfloat16* __restrict__ input,
    const int32_t* __restrict__ original_to_padded,
    __mt_bfloat16* __restrict__ output,
    int64_t num_tokens,
    int64_t hidden_size,
    int32_t topk) {
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
  if (token_idx >= num_tokens) {
    return;
  }

  const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
  for (int64_t hidden_idx = thread_idx; hidden_idx < hidden_size;
       hidden_idx += static_cast<int64_t>(blockDim.x)) {
    const __mt_bfloat16 value = input[token_idx * hidden_size + hidden_idx];
    for (int32_t topk_idx = 0; topk_idx < topk; ++topk_idx) {
      const int64_t assignment_idx = token_idx * topk + topk_idx;
      const int32_t destination = original_to_padded[assignment_idx];
      if (destination >= 0) {
        output[static_cast<int64_t>(destination) * hidden_size + hidden_idx] =
            value;
      }
    }
  }
}

__global__ void moe_ragged_swiglu_bf16_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_bfloat16* __restrict__ output,
    int64_t assignment_count,
    int64_t intermediate_size) {
  const int64_t assignment_idx = static_cast<int64_t>(blockIdx.x);
  if (assignment_idx >= assignment_count) {
    return;
  }

  const int64_t row = assignment_idx * kRaggedAlignment;
  const int64_t gate_base = row * intermediate_size * 2;
  const int64_t up_base = gate_base + intermediate_size;
  const int64_t output_base = row * intermediate_size;
  for (int64_t intermediate_idx = static_cast<int64_t>(threadIdx.x);
       intermediate_idx < intermediate_size;
       intermediate_idx += static_cast<int64_t>(blockDim.x)) {
    const float gate = __bfloat162float(input[gate_base + intermediate_idx]);
    const __mt_bfloat16 activated =
        __float2bfloat16_rn(gate / (1.0f + expf(-gate)));
    output[output_base + intermediate_idx] =
        activated * input[up_base + intermediate_idx];
  }
}

__global__ void moe_indexed_swiglu_bf16_kernel(
    const __mt_bfloat16* __restrict__ input,
    const int32_t* __restrict__ valid_rows,
    __mt_bfloat16* __restrict__ output,
    int64_t assignment_count,
    int64_t intermediate_size) {
  const int64_t assignment_idx = static_cast<int64_t>(blockIdx.x);
  if (assignment_idx >= assignment_count) {
    return;
  }

  const int32_t row_idx = valid_rows[assignment_idx];
  if (row_idx < 0) {
    return;
  }
  const int64_t row = static_cast<int64_t>(row_idx);
  const int64_t gate_base = row * intermediate_size * 2;
  const int64_t up_base = gate_base + intermediate_size;
  const int64_t output_base = row * intermediate_size;
  for (int64_t intermediate_idx = static_cast<int64_t>(threadIdx.x);
       intermediate_idx < intermediate_size;
       intermediate_idx += static_cast<int64_t>(blockDim.x)) {
    const float gate = __bfloat162float(input[gate_base + intermediate_idx]);
    const __mt_bfloat16 activated =
        __float2bfloat16_rn(gate / (1.0f + expf(-gate)));
    output[output_base + intermediate_idx] =
        activated * input[up_base + intermediate_idx];
  }
}

union Bf16Pack8 {
  int4 vector;
  __mt_bfloat16 values[8];
};

// Flatten rows and 16-byte chunks into one grid (act_and_mul_flat_vec8).
// This avoids launching one mostly idle block for
// every routed assignment when the expert intermediate size is only 512.
__global__ void moe_indexed_swiglu_bf16_vec8_kernel(
    const __mt_bfloat16* __restrict__ input,
    const int32_t* __restrict__ valid_rows,
    __mt_bfloat16* __restrict__ output,
    int32_t assignment_count,
    int64_t intermediate_size,
    int32_t chunks_per_row,
    int32_t total_chunks) {
  const int32_t chunk_idx =
      static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunk_idx >= total_chunks) {
    return;
  }

  const int32_t assignment_idx = chunk_idx / chunks_per_row;
  if (assignment_idx >= assignment_count) {
    return;
  }
  const int32_t row_idx = valid_rows[assignment_idx];
  if (row_idx < 0) {
    return;
  }

  constexpr int32_t kValuesPerChunk = 8;
  const int32_t chunk_in_row = chunk_idx - assignment_idx * chunks_per_row;
  const int32_t column = chunk_in_row * kValuesPerChunk;
  const int64_t row = static_cast<int64_t>(row_idx);
  const int64_t gate_base = row * intermediate_size * 2 + column;
  const int64_t up_base = gate_base + intermediate_size;
  const int64_t output_base = row * intermediate_size + column;

  Bf16Pack8 gate;
  Bf16Pack8 up;
  Bf16Pack8 result;
  gate.vector = *reinterpret_cast<const int4*>(input + gate_base);
  up.vector = *reinterpret_cast<const int4*>(input + up_base);
#pragma unroll
  for (int32_t i = 0; i < kValuesPerChunk; ++i) {
    const float gate_value = __bfloat162float(gate.values[i]);
    const float up_value = __bfloat162float(up.values[i]);
    const float activated =
        0.5f * gate_value * (1.0f + tanhf(0.5f * gate_value));
    result.values[i] = __float2bfloat16_rn(activated * up_value);
  }
  *reinterpret_cast<int4*>(output + output_base) = result.vector;
}

__global__ void moe_ragged_swiglu_quant_kernel(
    const __mt_bfloat16* __restrict__ input,
    __mt_fp8_e4m3* __restrict__ output_q,
    float* __restrict__ output_s,
    int64_t assignment_count,
    int64_t intermediate_size,
    int32_t hidden_dim_num_groups) {
  const int64_t assignment_idx = static_cast<int64_t>(blockIdx.x);
  if (assignment_idx >= assignment_count) {
    return;
  }

  const int32_t thread_idx = static_cast<int32_t>(threadIdx.x);
  const int32_t group_idx = thread_idx / kActThreadsPerGroup;
  const int32_t lane = thread_idx % kActThreadsPerGroup;
  if (group_idx >= hidden_dim_num_groups) {
    return;
  }

  const int64_t row = assignment_idx * kRaggedAlignment;
  const int64_t elem_offset = static_cast<int64_t>(lane) * kActElemsPerThread;
  const int64_t gate_base = row * intermediate_size * 2 +
                            static_cast<int64_t>(group_idx) * kGroupSize +
                            elem_offset;
  const int64_t up_base = gate_base + intermediate_size;
  const uint64_t gate_u64 =
      *reinterpret_cast<const uint64_t*>(input + gate_base);
  const uint64_t up_u64 = *reinterpret_cast<const uint64_t*>(input + up_base);
  const __mt_bfloat16* gate_values =
      reinterpret_cast<const __mt_bfloat16*>(&gate_u64);
  const __mt_bfloat16* up_values =
      reinterpret_cast<const __mt_bfloat16*>(&up_u64);

  float values[kActElemsPerThread];
  float local_absmax = kEps;
#pragma unroll
  for (int32_t j = 0; j < kActElemsPerThread; ++j) {
    const float gate = __bfloat162float(gate_values[j]);
    const float half_gate = 0.5f * gate;
    const __mt_bfloat16 activated =
        __float2bfloat16_rn(half_gate * (1.0f + tanhf(half_gate)));
    const __mt_bfloat16 product = activated * up_values[j];
    const float value = __bfloat162float(product);
    values[j] = value;
    local_absmax = fmaxf(local_absmax, fabsf(value));
  }
  local_absmax = group_reduce_max<kActThreadsPerGroup>(local_absmax, lane);
  const float scale_inv = local_absmax / kFp8E4M3Max;
  const float scale = kFp8E4M3Max / local_absmax;
  const float4 scaled = make_float4(values[0] * scale,
                                    values[1] * scale,
                                    values[2] * scale,
                                    values[3] * scale);
  const uint32_t packed = static_cast<uint32_t>(
      __musa_cvt_float4_to_fp8x4(scaled, __MT_SATFINITE, __MT_E4M3));
  const int64_t output_base = row * intermediate_size +
                              static_cast<int64_t>(group_idx) * kGroupSize +
                              elem_offset;
  *reinterpret_cast<uint32_t*>(output_q + output_base) = packed;
  if (lane == 0) {
    output_s[row * hidden_dim_num_groups + group_idx] = scale_inv;
  }
}

__global__ void moe_ragged_combine_bf16_kernel(
    const __mt_bfloat16* __restrict__ down,
    const float* __restrict__ topk_weights,
    __mt_bfloat16* __restrict__ output,
    int64_t num_tokens,
    int32_t topk,
    int64_t hidden_size) {
  const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
  if (token_idx >= num_tokens) {
    return;
  }

  for (int64_t hidden_idx = static_cast<int64_t>(threadIdx.x);
       hidden_idx < hidden_size;
       hidden_idx += kRaggedCombineThreads) {
    float value = 0.0f;
#pragma unroll
    for (int32_t topk_idx = 0; topk_idx < topk; ++topk_idx) {
      const int64_t assignment_idx = token_idx * topk + topk_idx;
      const int64_t row = assignment_idx * kRaggedAlignment;
      value += topk_weights[assignment_idx] *
               __bfloat162float(down[row * hidden_size + hidden_idx]);
    }
    output[token_idx * hidden_size + hidden_idx] = __float2bfloat16_rn(value);
  }
}

int32_t choose_subwarps_per_block(int64_t hidden_dim_num_groups) {
  if (hidden_dim_num_groups % 16 == 0) {
    return 16;
  }
  if (hidden_dim_num_groups % 8 == 0) {
    return 8;
  }
  if (hidden_dim_num_groups % 4 == 0) {
    return 4;
  }
  if (hidden_dim_num_groups % 2 == 0) {
    return 2;
  }
  return 1;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> per_token_group_quant_fp8(
    const torch::Tensor& input,
    int64_t group_size) {
  CHECK_EQ(input.scalar_type(), torch::kBFloat16)
      << "per_token_group_quant_fp8 supports BF16 input only.";
  CHECK_EQ(group_size, kGroupSize);
  CHECK_GT(input.dim(), 0);
  CHECK(input.is_contiguous())
      << "per_token_group_quant_fp8 requires contiguous input.";
  CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(int4) == 0)
      << "per_token_group_quant_fp8 requires 16-byte-aligned input.";

  const int64_t k = input.size(-1);
  CHECK_GT(k, 0);
  CHECK_EQ(k % kGroupSize, 0);
  const int64_t k_groups = k / kGroupSize;
  const int64_t m = input.numel() / k;

  auto out_q =
      torch::empty(input.sizes(), input.options().dtype(torch::kFloat8_e4m3fn));
  auto scale_sizes = input.sizes().vec();
  scale_sizes.back() = k_groups;
  auto out_scale =
      torch::empty(scale_sizes, input.options().dtype(torch::kFloat32));

  if (m == 0 || k_groups == 0) {
    return std::make_tuple(out_q, out_scale);
  }

  const int32_t subwarps_per_block = choose_subwarps_per_block(k_groups);
  const int64_t grid_x =
      (k_groups + subwarps_per_block - 1) / subwarps_per_block;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  per_token_group_quant_fp8_bf16_g128_kernel<<<
      dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(m)),
      static_cast<unsigned int>(subwarps_per_block * kThreadsPerGroup),
      0,
      stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__mt_fp8_e4m3*>(out_q.data_ptr<c10::Float8_e4m3fn>()),
      out_scale.data_ptr<float>(),
      k,
      k_groups,
      m);
  C10_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(out_q, out_scale);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_preprocess_fp8(const torch::Tensor& input,
                         const torch::Tensor& topk_ids,
                         int64_t num_experts,
                         int64_t group_size) {
  CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
        input.is_contiguous())
      << "fused_moe_preprocess_fp8 requires contiguous BF16 [M, K].";
  CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_ids.dim() == 2 &&
        topk_ids.is_contiguous())
      << "fused_moe_preprocess_fp8 requires contiguous int32 top-k ids.";
  CHECK_EQ(input.device(), topk_ids.device());
  CHECK_EQ(topk_ids.size(0), input.size(0));
  CHECK_EQ(group_size, kGroupSize);
  CHECK_EQ(input.size(1) % kGroupSize, 0);
  CHECK_GT(num_experts, 0);
  CHECK_LE(num_experts, kMaxMoeExperts);
  CHECK_GT(topk_ids.size(1), 0);
  CHECK_LE(topk_ids.size(1), kMaxMoeTopk);

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = input.size(1);
  const int32_t hidden_dim_num_groups =
      static_cast<int32_t>(hidden_size / kGroupSize);
  const int32_t topk = static_cast<int32_t>(topk_ids.size(1));
  const int64_t assignment_count = num_tokens * topk;
  CHECK_GT(hidden_dim_num_groups, 0);
  CHECK_LE(hidden_dim_num_groups * kActThreadsPerGroup, 1024);
  CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(uint64_t) == 0)
      << "fused_moe_preprocess_fp8 requires 8-byte-aligned input.";

  auto output_q = torch::empty({assignment_count, hidden_size},
                               input.options().dtype(torch::kFloat8_e4m3fn));
  auto output_s = torch::empty({assignment_count, hidden_dim_num_groups},
                               input.options().dtype(torch::kFloat32));
  auto src_to_dst = torch::empty({assignment_count}, topk_ids.options());
  auto expert_counts = torch::zeros({num_experts}, topk_ids.options());
  auto expert_cursor = torch::empty({num_experts}, topk_ids.options());
  if (num_tokens == 0) {
    return std::make_tuple(output_q, output_s, src_to_dst, expert_counts);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int32_t kHistogramThreads = 256;
  const int64_t histogram_blocks =
      (assignment_count + kHistogramThreads - 1) / kHistogramThreads;
  moe_preprocess_histogram_kernel<<<histogram_blocks,
                                    kHistogramThreads,
                                    0,
                                    stream>>>(
      topk_ids.data_ptr<int32_t>(),
      expert_counts.data_ptr<int32_t>(),
      assignment_count,
      static_cast<int32_t>(num_experts));
  moe_preprocess_prefix_kernel<<<1, 1, 0, stream>>>(
      expert_counts.data_ptr<int32_t>(),
      expert_cursor.data_ptr<int32_t>(),
      static_cast<int32_t>(num_experts));
  moe_preprocess_assign_quant_kernel<<<num_tokens,
                                       hidden_dim_num_groups *
                                           kActThreadsPerGroup,
                                       0,
                                       stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      topk_ids.data_ptr<int32_t>(),
      expert_cursor.data_ptr<int32_t>(),
      src_to_dst.data_ptr<int32_t>(),
      reinterpret_cast<__mt_fp8_e4m3*>(output_q.data_ptr<c10::Float8_e4m3fn>()),
      output_s.data_ptr<float>(),
      num_tokens,
      hidden_size,
      hidden_dim_num_groups,
      topk,
      static_cast<int32_t>(num_experts));
  C10_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output_q, output_s, src_to_dst, expert_counts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_preprocess_bf16(const torch::Tensor& input,
                          const torch::Tensor& topk_ids,
                          int64_t num_experts,
                          int64_t alignment) {
  CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
        input.is_contiguous())
      << "fused_moe_preprocess_bf16 requires contiguous BF16 [M, K].";
  CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_ids.dim() == 2 &&
        topk_ids.is_contiguous())
      << "fused_moe_preprocess_bf16 requires contiguous int32 top-k ids.";
  CHECK_EQ(input.device(), topk_ids.device());
  CHECK_EQ(topk_ids.size(0), input.size(0));
  CHECK_GT(num_experts, 0);
  CHECK_LE(num_experts, kMaxMoeExperts);
  CHECK_GT(topk_ids.size(1), 0);
  CHECK_LE(topk_ids.size(1), kMaxMoeTopk);
  CHECK(alignment == 128 || alignment == 256)
      << "fused_moe_preprocess_bf16 alignment must be 128 or 256.";

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = input.size(1);
  const int32_t topk = static_cast<int32_t>(topk_ids.size(1));
  const int64_t assignment_count = num_tokens * topk;
  const int64_t padded_rows =
      ((assignment_count + num_experts * (alignment - 1) + alignment - 1) /
       alignment) *
      alignment;

  torch::Tensor output =
      torch::empty({padded_rows, hidden_size}, input.options());
  torch::Tensor row_expert_ids =
      torch::full({padded_rows}, -1, topk_ids.options());
  torch::Tensor original_to_padded =
      torch::empty({assignment_count}, topk_ids.options());
  torch::Tensor expert_counts = torch::zeros({num_experts}, topk_ids.options());
  torch::Tensor group_m_counts =
      torch::zeros({num_experts}, topk_ids.options());
  if (num_tokens == 0) {
    return std::make_tuple(
        output, row_expert_ids, original_to_padded, group_m_counts);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int32_t kHistogramThreads = 256;
  const int64_t histogram_blocks =
      (assignment_count + kHistogramThreads - 1) / kHistogramThreads;
  moe_preprocess_histogram_kernel<<<histogram_blocks,
                                    kHistogramThreads,
                                    0,
                                    stream>>>(
      topk_ids.data_ptr<int32_t>(),
      expert_counts.data_ptr<int32_t>(),
      assignment_count,
      static_cast<int32_t>(num_experts));
  torch::Tensor expert_cursor = torch::empty_like(expert_counts);
  moe_preprocess_prefix_aligned_kernel<<<1, 1, 0, stream>>>(
      expert_counts.data_ptr<int32_t>(),
      expert_cursor.data_ptr<int32_t>(),
      group_m_counts.data_ptr<int32_t>(),
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(alignment),
      static_cast<int32_t>(padded_rows));
  moe_preprocess_assign_bf16_kernel<<<static_cast<unsigned int>(num_tokens),
                                      256,
                                      0,
                                      stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      topk_ids.data_ptr<int32_t>(),
      expert_cursor.data_ptr<int32_t>(),
      original_to_padded.data_ptr<int32_t>(),
      row_expert_ids.data_ptr<int32_t>(),
      reinterpret_cast<__mt_bfloat16*>(output.data_ptr<at::BFloat16>()),
      num_tokens,
      hidden_size,
      topk,
      static_cast<int32_t>(num_experts));
  C10_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(
      output, row_expert_ids, original_to_padded, group_m_counts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_ragged_preprocess_fp8(const torch::Tensor& input,
                                const torch::Tensor& topk_ids,
                                int64_t group_size,
                                int64_t alignment) {
  CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
        input.is_contiguous())
      << "Ragged MoE preprocess requires contiguous BF16 [M, K].";
  CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_ids.dim() == 2 &&
        topk_ids.is_contiguous())
      << "Ragged MoE preprocess requires contiguous int32 top-k ids.";
  CHECK_EQ(input.device(), topk_ids.device());
  CHECK_EQ(topk_ids.size(0), input.size(0));
  CHECK_EQ(group_size, kGroupSize);
  CHECK_EQ(alignment, kRaggedAlignment);
  CHECK_EQ(input.size(1) % kGroupSize, 0);
  CHECK_GT(topk_ids.size(1), 0);
  CHECK_LE(topk_ids.size(1), kMaxMoeTopk);

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = input.size(1);
  const int32_t hidden_dim_num_groups =
      static_cast<int32_t>(hidden_size / kGroupSize);
  const int32_t topk = static_cast<int32_t>(topk_ids.size(1));
  const int64_t assignment_count = num_tokens * topk;
  const int64_t padded_rows = assignment_count * alignment;
  CHECK_GT(hidden_dim_num_groups, 0);
  CHECK_LE(hidden_dim_num_groups * kActThreadsPerGroup, 1024);
  CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(uint64_t) == 0)
      << "Ragged MoE preprocess requires 8-byte-aligned input.";

  torch::Tensor output_q = torch::empty(
      {padded_rows, hidden_size}, input.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor output_s = torch::empty({padded_rows, hidden_dim_num_groups},
                                        input.options().dtype(torch::kFloat32));
  torch::Tensor row_expert_ids =
      torch::empty({padded_rows}, topk_ids.options());
  if (num_tokens == 0) {
    return std::make_tuple(output_q, output_s, row_expert_ids);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  moe_ragged_preprocess_assign_quant_kernel<<<
      static_cast<unsigned int>(num_tokens),
      static_cast<unsigned int>(hidden_dim_num_groups * kActThreadsPerGroup),
      0,
      stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      topk_ids.data_ptr<int32_t>(),
      reinterpret_cast<__mt_fp8_e4m3*>(output_q.data_ptr<c10::Float8_e4m3fn>()),
      output_s.data_ptr<float>(),
      row_expert_ids.data_ptr<int32_t>(),
      num_tokens,
      hidden_size,
      hidden_dim_num_groups,
      topk);
  C10_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output_q, output_s, row_expert_ids);
}

std::tuple<torch::Tensor, torch::Tensor> fused_moe_ragged_preprocess_bf16(
    const torch::Tensor& input,
    const torch::Tensor& topk_ids,
    int64_t alignment) {
  CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
        input.is_contiguous())
      << "Ragged BF16 MoE preprocess requires contiguous BF16 [M, K].";
  CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_ids.dim() == 2 &&
        topk_ids.is_contiguous())
      << "Ragged BF16 MoE preprocess requires contiguous int32 top-k ids.";
  CHECK_EQ(input.device(), topk_ids.device());
  CHECK_EQ(topk_ids.size(0), input.size(0));
  CHECK_EQ(alignment, kRaggedAlignment);
  CHECK_GT(topk_ids.size(1), 0);
  CHECK_LE(topk_ids.size(1), kMaxMoeTopk);

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = input.size(1);
  const int32_t topk = static_cast<int32_t>(topk_ids.size(1));
  const int64_t padded_rows = num_tokens * topk * alignment;
  torch::Tensor output =
      torch::empty({padded_rows, hidden_size}, input.options());
  torch::Tensor row_expert_ids =
      torch::empty({padded_rows}, topk_ids.options());
  if (num_tokens == 0) {
    return std::make_tuple(output, row_expert_ids);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  moe_ragged_preprocess_assign_bf16_kernel<<<static_cast<unsigned int>(
                                                 num_tokens),
                                             256,
                                             0,
                                             stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      topk_ids.data_ptr<int32_t>(),
      reinterpret_cast<__mt_bfloat16*>(output.data_ptr<at::BFloat16>()),
      row_expert_ids.data_ptr<int32_t>(),
      num_tokens,
      hidden_size,
      topk);
  C10_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output, row_expert_ids);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_moe_decode_preprocess_bf16(const torch::Tensor& input,
                                 const torch::Tensor& topk_ids,
                                 int64_t num_experts,
                                 int64_t alignment) {
  CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
        input.is_contiguous())
      << "Decode BF16 MoE preprocess requires contiguous BF16 [M, K].";
  CHECK(topk_ids.scalar_type() == torch::kInt32 && topk_ids.dim() == 2 &&
        topk_ids.is_contiguous())
      << "Decode BF16 MoE preprocess requires contiguous int32 top-k ids.";
  CHECK_EQ(input.device(), topk_ids.device());
  CHECK_EQ(topk_ids.size(0), input.size(0));
  CHECK_GT(num_experts, 0);
  CHECK_LE(num_experts, kMaxMoeExperts);
  CHECK_EQ(alignment, kRaggedAlignment);
  CHECK_GT(topk_ids.size(1), 0);
  CHECK_LE(topk_ids.size(1), kMaxMoeTopk);

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_size = input.size(1);
  const int32_t topk = static_cast<int32_t>(topk_ids.size(1));
  const int64_t assignment_count = num_tokens * topk;
  const int64_t padded_rows = assignment_count * alignment;
  torch::Tensor output =
      torch::empty({padded_rows, hidden_size}, input.options());
  torch::Tensor row_expert_ids =
      torch::full({padded_rows}, -1, topk_ids.options());
  torch::Tensor original_to_padded =
      torch::empty({assignment_count}, topk_ids.options());
  if (num_tokens == 0) {
    return std::make_tuple(output, row_expert_ids, original_to_padded);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int32_t kRouteThreads = 256;
  moe_decode_route_bf16_kernel<<<1, kRouteThreads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(),
      original_to_padded.data_ptr<int32_t>(),
      row_expert_ids.data_ptr<int32_t>(),
      nullptr,
      assignment_count,
      static_cast<int32_t>(num_experts),
      static_cast<int32_t>(alignment),
      static_cast<int32_t>(padded_rows));
  moe_decode_assign_bf16_kernel<<<static_cast<unsigned int>(num_tokens),
                                  256,
                                  0,
                                  stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      original_to_padded.data_ptr<int32_t>(),
      reinterpret_cast<__mt_bfloat16*>(output.data_ptr<at::BFloat16>()),
      num_tokens,
      hidden_size,
      topk);
  C10_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output, row_expert_ids, original_to_padded);
}

torch::Tensor fused_moe_ragged_swiglu_bf16(const torch::Tensor& input,
                                           int64_t alignment) {
  CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
        input.is_contiguous())
      << "Ragged BF16 SwiGLU requires contiguous BF16 [M, 2N].";
  CHECK_EQ(alignment, kRaggedAlignment);
  CHECK_EQ(input.size(0) % alignment, 0);
  CHECK_EQ(input.size(1) % 2, 0);

  const int64_t padded_rows = input.size(0);
  const int64_t assignment_count = padded_rows / alignment;
  const int64_t intermediate_size = input.size(1) / 2;
  torch::Tensor output =
      torch::empty({padded_rows, intermediate_size}, input.options());
  if (assignment_count == 0) {
    return output;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int32_t kSwiGluThreads = 256;
  moe_ragged_swiglu_bf16_kernel<<<static_cast<unsigned int>(assignment_count),
                                  kSwiGluThreads,
                                  0,
                                  stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__mt_bfloat16*>(output.data_ptr<at::BFloat16>()),
      assignment_count,
      intermediate_size);
  C10_CUDA_CHECK(cudaGetLastError());
  return output;
}

torch::Tensor fused_moe_indexed_swiglu_bf16(const torch::Tensor& input,
                                            const torch::Tensor& valid_rows) {
  CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
        input.is_contiguous())
      << "Indexed BF16 SwiGLU requires contiguous BF16 [M, 2N].";
  CHECK(valid_rows.scalar_type() == torch::kInt32 && valid_rows.dim() == 1 &&
        valid_rows.is_contiguous())
      << "Indexed BF16 SwiGLU requires contiguous int32 row indices.";
  CHECK_EQ(input.device(), valid_rows.device());
  CHECK_EQ(input.size(1) % 2, 0);

  const int64_t assignment_count = valid_rows.size(0);
  const int64_t intermediate_size = input.size(1) / 2;
  CHECK_GT(intermediate_size, 0);
  torch::Tensor output =
      torch::empty({input.size(0), intermediate_size}, input.options());
  if (assignment_count == 0) {
    return output;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int32_t kValuesPerChunk = 8;
  const int64_t chunks_per_row = intermediate_size / kValuesPerChunk;
  const int64_t total_chunks = assignment_count * chunks_per_row;
  if (intermediate_size % kValuesPerChunk == 0 &&
      reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(int4) == 0 &&
      reinterpret_cast<uintptr_t>(output.data_ptr()) % alignof(int4) == 0 &&
      total_chunks <= std::numeric_limits<int32_t>::max() &&
      assignment_count <= std::numeric_limits<int32_t>::max()) {
    constexpr int32_t kFlatSwiGluThreads = 512;
    const int64_t blocks =
        (total_chunks + kFlatSwiGluThreads - 1) / kFlatSwiGluThreads;
    moe_indexed_swiglu_bf16_vec8_kernel<<<static_cast<unsigned int>(blocks),
                                          kFlatSwiGluThreads,
                                          0,
                                          stream>>>(
        reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
        valid_rows.data_ptr<int32_t>(),
        reinterpret_cast<__mt_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<int32_t>(assignment_count),
        intermediate_size,
        static_cast<int32_t>(chunks_per_row),
        static_cast<int32_t>(total_chunks));
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  constexpr int32_t kSwiGluThreads = 256;
  moe_indexed_swiglu_bf16_kernel<<<static_cast<unsigned int>(assignment_count),
                                   kSwiGluThreads,
                                   0,
                                   stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      valid_rows.data_ptr<int32_t>(),
      reinterpret_cast<__mt_bfloat16*>(output.data_ptr<at::BFloat16>()),
      assignment_count,
      intermediate_size);
  C10_CUDA_CHECK(cudaGetLastError());
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> fused_moe_ragged_swiglu_quant_fp8(
    const torch::Tensor& input,
    int64_t group_size,
    int64_t alignment) {
  CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
        input.is_contiguous())
      << "Ragged SwiGLU quant requires contiguous BF16 [M, 2N].";
  CHECK_EQ(group_size, kGroupSize);
  CHECK_EQ(alignment, kRaggedAlignment);
  CHECK_EQ(input.size(0) % alignment, 0);
  CHECK_EQ(input.size(1) % (2 * kGroupSize), 0);

  const int64_t padded_rows = input.size(0);
  const int64_t assignment_count = padded_rows / alignment;
  const int64_t intermediate_size = input.size(1) / 2;
  const int32_t hidden_dim_num_groups =
      static_cast<int32_t>(intermediate_size / kGroupSize);
  CHECK_GT(hidden_dim_num_groups, 0);
  CHECK_LE(hidden_dim_num_groups * kActThreadsPerGroup, 1024);
  CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(uint64_t) == 0)
      << "Ragged SwiGLU quant requires 8-byte-aligned input.";

  torch::Tensor output_q =
      torch::empty({padded_rows, intermediate_size},
                   input.options().dtype(torch::kFloat8_e4m3fn));
  torch::Tensor output_s = torch::empty({padded_rows, hidden_dim_num_groups},
                                        input.options().dtype(torch::kFloat32));
  if (assignment_count == 0) {
    return std::make_tuple(output_q, output_s);
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  moe_ragged_swiglu_quant_kernel<<<
      static_cast<unsigned int>(assignment_count),
      static_cast<unsigned int>(hidden_dim_num_groups * kActThreadsPerGroup),
      0,
      stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__mt_fp8_e4m3*>(output_q.data_ptr<c10::Float8_e4m3fn>()),
      output_s.data_ptr<float>(),
      assignment_count,
      intermediate_size,
      hidden_dim_num_groups);
  C10_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output_q, output_s);
}

torch::Tensor fused_moe_ragged_combine(const torch::Tensor& down,
                                       const torch::Tensor& topk_weights,
                                       int64_t num_tokens,
                                       int64_t alignment) {
  CHECK(down.scalar_type() == torch::kBFloat16 && down.dim() == 2 &&
        down.is_contiguous())
      << "Ragged MoE combine requires contiguous BF16 [M, H].";
  CHECK(topk_weights.scalar_type() == torch::kFloat32 &&
        topk_weights.dim() == 2 && topk_weights.is_contiguous())
      << "Ragged MoE combine requires contiguous FP32 top-k weights.";
  CHECK_EQ(down.device(), topk_weights.device());
  CHECK_GE(num_tokens, 0);
  CHECK_EQ(topk_weights.size(0), num_tokens);
  CHECK_EQ(alignment, kRaggedAlignment);
  const int64_t topk = topk_weights.size(1);
  CHECK_GT(topk, 0);
  CHECK_EQ(down.size(0), num_tokens * topk * alignment);
  CHECK_LE(topk, kMaxMoeTopk);

  const int64_t hidden_size = down.size(1);
  torch::Tensor output =
      torch::empty({num_tokens, hidden_size}, down.options());
  if (num_tokens == 0) {
    return output;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(down));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  moe_ragged_combine_bf16_kernel<<<static_cast<unsigned int>(num_tokens),
                                   kRaggedCombineThreads,
                                   0,
                                   stream>>>(
      reinterpret_cast<const __mt_bfloat16*>(down.data_ptr<at::BFloat16>()),
      topk_weights.data_ptr<float>(),
      reinterpret_cast<__mt_bfloat16*>(output.data_ptr<at::BFloat16>()),
      num_tokens,
      static_cast<int32_t>(topk),
      hidden_size);
  C10_CUDA_CHECK(cudaGetLastError());
  return output;
}

}  // namespace xllm::kernel::musa
