/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include <cmath>
#include <type_traits>

#include "core/kernels/cuda/device_utils.cuh"
#include "core/kernels/musa/musa_ops_api.h"

using at::device_of;
using ::xllm::kernel::cuda::xllm_ldg;

namespace {

static constexpr unsigned int kFinalMask = 0xffffffffU;

template <typename T, int num>
struct packed_as;
template <>
struct packed_as<uint, 1> {
  using type = uint;
};
template <>
struct packed_as<uint, 2> {
  using type = uint2;
};
template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(kFinalMask, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

template <typename scalar_t, int head_dim, bool interleave>
__global__ void fused_qknorm_rope_kernel(void* qkv_void,
                                         int const num_heads_q,
                                         int const num_heads_k,
                                         int const total_heads_per_token,
                                         int const k_head_offset,
                                         float const eps,
                                         float const weight_offset,
                                         void const* q_weight_void,
                                         void const* k_weight_void,
                                         void const* cos_sin_cache_void,
                                         int32_t const* position_ids,
                                         int const num_tokens,
                                         int const rotary_dim) {
  using T = scalar_t;
  T* qkv = reinterpret_cast<T*>(qkv_void);
  T const* q_weight = reinterpret_cast<T const*>(q_weight_void);
  T const* k_weight = reinterpret_cast<T const*>(k_weight_void);
  T const* cos_sin_cache = reinterpret_cast<T const*>(cos_sin_cache_void);

  int const warps_per_block = blockDim.x / 32;
  int const warp_id = threadIdx.x / 32;
  int const lane_id = threadIdx.x % 32;

  int const global_warp_idx = blockIdx.x * warps_per_block + warp_id;

  int const total_qk_heads = num_heads_q + num_heads_k;

  int const token_idx = global_warp_idx / total_qk_heads;
  int const local_head_idx = global_warp_idx % total_qk_heads;

  if (token_idx >= num_tokens) {
    return;
  }

  bool const is_q = local_head_idx < num_heads_q;
  int const head_idx = is_q ? local_head_idx : local_head_idx - num_heads_q;

  static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
  constexpr int kNumElemsPerThread = head_dim / 32;
  float elements[kNumElemsPerThread];
  constexpr int kElemSizeBytes = kNumElemsPerThread * sizeof(scalar_t);
  static_assert(kElemSizeBytes % 4 == 0,
                "numSizeBytes must be a multiple of 4");
  constexpr int kVecSize = kElemSizeBytes / 4;
  using vec_T = typename packed_as<uint, kVecSize>::type;

  int offset_warp;
  if (is_q) {
    offset_warp =
        token_idx * total_heads_per_token * head_dim + head_idx * head_dim;
  } else {
    offset_warp = token_idx * total_heads_per_token * head_dim +
                  k_head_offset * head_dim + head_idx * head_dim;
  }
  int offset_thread = offset_warp + lane_id * kNumElemsPerThread;

  float sum_of_squares = 0.0f;

  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offset_thread]);
    constexpr int kNumPackedElems = kElemSizeBytes / sizeof(scalar_t);
#pragma unroll
    for (int i = 0; i < kNumPackedElems; i++) {
      scalar_t val = *(reinterpret_cast<scalar_t*>(&vec) + i);
      float fval = static_cast<float>(val);
      sum_of_squares += fval * fval;
      elements[i] = fval;
    }
  }

  sum_of_squares = warp_reduce_sum(sum_of_squares);

  float rms_rcp = rsqrtf(sum_of_squares / static_cast<float>(head_dim) + eps);

#pragma unroll
  for (int i = 0; i < kNumElemsPerThread; i++) {
    int dim = lane_id * kNumElemsPerThread + i;
    float weight = is_q ? static_cast<float>(q_weight[dim])
                        : static_cast<float>(k_weight[dim]);
    weight += weight_offset;
    elements[i] *= rms_rcp * weight;
  }

  float elements2[kNumElemsPerThread];

  int64_t pos_id = static_cast<int64_t>(position_ids[token_idx]);

  T const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
  int const embed_dim = rotary_dim / 2;
  T const* cos_ptr = cache_ptr;
  T const* sin_ptr = cache_ptr + embed_dim;
  int const rotary_lanes = rotary_dim / kNumElemsPerThread;
  if (lane_id < rotary_lanes) {
    if constexpr (interleave) {
#pragma unroll
      for (int i = 0; i < kNumElemsPerThread / 2; ++i) {
        int const idx0 = 2 * i;
        int const idx1 = 2 * i + 1;
        int const dim_idx = lane_id * kNumElemsPerThread + idx0;

        float const val0 = elements[idx0];
        float const val1 = elements[idx1];

        int const half_dim = dim_idx / 2;
        float const cos_val = static_cast<float>(xllm_ldg(cos_ptr + half_dim));
        float const sin_val = static_cast<float>(xllm_ldg(sin_ptr + half_dim));

        elements[idx0] = val0 * cos_val - val1 * sin_val;
        elements[idx1] = val0 * sin_val + val1 * cos_val;
      }
    } else {
      __syncwarp();
      int pair_offset = (rotary_dim / 2) / kNumElemsPerThread;
#pragma unroll
      for (int i = 0; i < kNumElemsPerThread; i++) {
        elements2[i] = __shfl_xor_sync(kFinalMask, elements[i], pair_offset);

        if (lane_id < pair_offset) {
          elements2[i] = -elements2[i];
        }
        int dim_idx = lane_id * kNumElemsPerThread + i;

        dim_idx = (dim_idx * 2) % rotary_dim;
        int half_dim = dim_idx / 2;
        float cos_val = static_cast<float>(xllm_ldg(cos_ptr + half_dim));
        float sin_val = static_cast<float>(xllm_ldg(sin_ptr + half_dim));

        elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
      }
      __syncwarp();
    }
  }
  {
    vec_T vec;
    constexpr int kNumPackedElems = kElemSizeBytes / sizeof(scalar_t);
#pragma unroll
    for (int i = 0; i < kNumPackedElems; i++) {
      *(reinterpret_cast<scalar_t*>(&vec) + i) =
          static_cast<scalar_t>(elements[i]);
    }
    *reinterpret_cast<vec_T*>(&qkv[offset_thread]) = vec;
  }
}

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

template <typename scalar_t>
void launch_fused_qknorm_rope(void* qkv,
                              int const num_tokens,
                              int const num_heads_q,
                              int const num_heads_k,
                              int const total_heads_per_token,
                              int const k_head_offset,
                              int const head_dim,
                              int const rotary_dim,
                              float const eps,
                              float const weight_offset,
                              void const* q_weight,
                              void const* k_weight,
                              void const* cos_sin_cache,
                              bool const interleave,
                              int32_t const* position_ids,
                              cudaStream_t stream) {
  constexpr int kBlockSize = 256;

  int const warps_per_block = kBlockSize / 32;
  int const total_qk_heads = num_heads_q + num_heads_k;
  int const total_warps = num_tokens * total_qk_heads;

  int const grid_size = div_up(total_warps, warps_per_block);
  dim3 gridDim(grid_size);
  dim3 blockDim(kBlockSize);

  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fused_qknorm_rope_kernel<scalar_t, 64, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(qkv,
                                               num_heads_q,
                                               num_heads_k,
                                               total_heads_per_token,
                                               k_head_offset,
                                               eps,
                                               weight_offset,
                                               q_weight,
                                               k_weight,
                                               cos_sin_cache,
                                               position_ids,
                                               num_tokens,
                                               rotary_dim);
      });
      break;
    case 128:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fused_qknorm_rope_kernel<scalar_t, 128, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(qkv,
                                               num_heads_q,
                                               num_heads_k,
                                               total_heads_per_token,
                                               k_head_offset,
                                               eps,
                                               weight_offset,
                                               q_weight,
                                               k_weight,
                                               cos_sin_cache,
                                               position_ids,
                                               num_tokens,
                                               rotary_dim);
      });
      break;
    case 256:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fused_qknorm_rope_kernel<scalar_t, 256, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(qkv,
                                               num_heads_q,
                                               num_heads_k,
                                               total_heads_per_token,
                                               k_head_offset,
                                               eps,
                                               weight_offset,
                                               q_weight,
                                               k_weight,
                                               cos_sin_cache,
                                               position_ids,
                                               num_tokens,
                                               rotary_dim);
      });
      break;
    default:
      CHECK(false) << "Unsupported head dimension for fusedQKNormRope: "
                   << head_dim;
  }
}

}  // namespace

namespace xllm::kernel::musa {

void fused_qk_norm_rope(torch::Tensor& qkv,
                        int64_t num_heads_q,
                        int64_t num_heads_k,
                        int64_t num_heads_v,
                        int64_t head_dim,
                        double eps,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin_cache,
                        bool interleaved,
                        const torch::Tensor& position_ids,
                        int64_t k_head_offset) {
  CHECK(qkv.is_contiguous()) << "qkv must be contiguous";
  CHECK(position_ids.is_contiguous()) << "position_ids must be contiguous";
  CHECK(position_ids.scalar_type() == torch::kInt32)
      << "position_ids dtype is " << position_ids.scalar_type()
      << ", while Int32 is expected";
  CHECK(q_weight.is_contiguous()) << "q_weight must be contiguous";
  CHECK(k_weight.is_contiguous()) << "k_weight must be contiguous";
  CHECK(cos_sin_cache.is_contiguous()) << "cos_sin_cache must be contiguous";

  CHECK(qkv.dim() == 2) << "QKV tensor must be 2D";
  CHECK(position_ids.dim() == 1) << "Position IDs must be 1D";
  CHECK(q_weight.dim() == 1) << "Query weights must be 1D";
  CHECK(k_weight.dim() == 1) << "Key weights must be 1D";
  CHECK(cos_sin_cache.dim() == 2) << "Cos/sin cache must be 2D";

  int64_t num_tokens = qkv.size(0);
  int64_t total_heads_per_token =
      k_head_offset > 0 ? k_head_offset + num_heads_k + num_heads_v
                        : num_heads_q + num_heads_k + num_heads_v;

  CHECK(qkv.size(1) == total_heads_per_token * head_dim)
      << "QKV tensor size mismatch";

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // Qwen3.5's [Q|G|K|V] layout passes a nonzero K-head offset and uses
  // Gemma RMSNorm semantics, whose effective scale is (1 + weight).
  // Standard Qwen2 attention passes offset zero and keeps regular RMSNorm.
  const float weight_offset = k_head_offset > 0 ? 1.0f : 0.0f;

  if (qkv.scalar_type() == at::ScalarType::BFloat16) {
    launch_fused_qknorm_rope<__nv_bfloat16>(
        qkv.data_ptr(),
        static_cast<int>(num_tokens),
        static_cast<int>(num_heads_q),
        static_cast<int>(num_heads_k),
        static_cast<int>(total_heads_per_token),
        static_cast<int>(k_head_offset > 0 ? k_head_offset : num_heads_q),
        static_cast<int>(head_dim),
        static_cast<int>(cos_sin_cache.size(1)),
        static_cast<float>(eps),
        weight_offset,
        q_weight.data_ptr(),
        k_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        interleaved,
        reinterpret_cast<int32_t const*>(position_ids.data_ptr()),
        stream);
  } else if (qkv.scalar_type() == at::ScalarType::Half) {
    launch_fused_qknorm_rope<__half>(
        qkv.data_ptr(),
        static_cast<int>(num_tokens),
        static_cast<int>(num_heads_q),
        static_cast<int>(num_heads_k),
        static_cast<int>(total_heads_per_token),
        static_cast<int>(k_head_offset > 0 ? k_head_offset : num_heads_q),
        static_cast<int>(head_dim),
        static_cast<int>(cos_sin_cache.size(1)),
        static_cast<float>(eps),
        weight_offset,
        q_weight.data_ptr(),
        k_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        interleaved,
        reinterpret_cast<int32_t const*>(position_ids.data_ptr()),
        stream);
  } else {
    CHECK(false) << "Unsupported dtype for fused_qk_norm_rope: "
                 << qkv.scalar_type();
  }
}

}  // namespace xllm::kernel::musa
