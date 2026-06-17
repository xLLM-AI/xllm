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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include <cmath>
#include <type_traits>

#if !defined(USE_DCU)
#include "cuda_ops_api.h"
#else
#include "dcu_ops_api.h"
#endif
#include "device_utils.cuh"
#include "type_convert.cuh"
#include "utils.h"

using at::device_of;

// Borrowed from:
// https://github.com/vllm-project/vllm/blob/022f3cea5327cc720a325c50931e1edcfdf2d32b/csrc/fused_qknorm_rope_kernel.cu

#if defined(USE_DCU)
static constexpr unsigned long long kFinalMask = 0xffffffffffffffffULL;
#else
static constexpr unsigned int kFinalMask = 0xffffffffU;
#endif

namespace {

using namespace xllm::kernel::cuda;

template <typename T, int num>
struct packed_as;
// Specialization for packed_as used in this kernel.
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

// Perform per-head QK Norm and RoPE in a single kernel.
// scalar_t_in: data type of QKV and RMSNorm weights
// scalar_t_cache: data type of cos/sin cache
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
template <typename scalar_t_in,
          typename scalar_t_cache,
          int head_dim,
          bool interleave>
__global__ void fused_qknorm_rope_kernel(
    void* qkv_void,                  // Combined QKV tensor
    int const num_heads_q,           // Number of query heads
    int const num_heads_k,           // Number of key heads
    int const num_heads_v,           // Number of value heads
    float const eps,                 // Epsilon for RMS normalization
    void const* q_weight_void,       // RMSNorm weights for query
    void const* k_weight_void,       // RMSNorm weights for key
    void const* cos_sin_cache_void,  // Pre-computed cos/sin cache
    int64_t const* position_ids,     // Position IDs for RoPE
    int const num_tokens,            // Number of tokens
    int const rotary_dim             // Dimension for RoPE
) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_DCU)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif

    using Converter = _typeConvert<scalar_t_in>;
    static_assert(Converter::exists,
                  "Input QKV data type is not supported for this CUDA "
                  "architecture or toolkit version.");
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;

    using CacheConverter = _typeConvert<scalar_t_cache>;
    static_assert(CacheConverter::exists,
                  "Cache data type is not supported for this CUDA architecture "
                  "or toolkit version.");
    using T_cache = typename CacheConverter::hip_type;

    T_in* qkv = reinterpret_cast<T_in*>(qkv_void);
    T_in const* q_weight = reinterpret_cast<T_in const*>(q_weight_void);
    T_in const* k_weight = reinterpret_cast<T_in const*>(k_weight_void);
    T_cache const* cos_sin_cache =
        reinterpret_cast<T_cache const*>(cos_sin_cache_void);

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    // Calculate global warp index to determine which head/token this warp
    // processes
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    // Total number of attention heads (Q and K)
    int const total_qk_heads = num_heads_q + num_heads_k;

    // Determine which token and head type (Q or K) this warp processes
    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;

    // Skip if this warp is assigned beyond the number of tokens
    if (tokenIdx >= num_tokens) return;

    bool const isQ = localHeadIdx < num_heads_q;
    int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

    int const num_heads = num_heads_q + num_heads_k + num_heads_v;

    static_assert(head_dim % (32 * 2) == 0,
                  "head_dim must be divisible by 64 (each warp processes one "
                  "head, and each thread gets even number of "
                  "elements)");
    constexpr int kNumElemsPerThread = head_dim / 32;
    float elements[kNumElemsPerThread];
    constexpr int kElemSizeBytes = kNumElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(kElemSizeBytes % 4 == 0,
                  "numSizeBytes must be a multiple of 4");
    constexpr int kVecSize =
        kElemSizeBytes /
        4;  // Use packed_as<uint, kVecSize> to perform loading/saving.
    using vec_T = typename packed_as<uint, kVecSize>::type;

    int offsetWarp;  // Offset for the warp
    if (isQ) {
      // Q segment: token offset + head offset within Q segment
      offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    } else {
      // K segment: token offset + entire Q segment + head offset within K
      // segment
      offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim +
                   headIdx * head_dim;
    }
    int offsetThread = offsetWarp + laneId * kNumElemsPerThread;

    // Sum of squares for RMSNorm
    float sumOfSquares = 0.0f;

    // Load.
    {
      vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
      constexpr int kNumPackedElems = kElemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < kNumPackedElems; i++) {
        // Interpret the generic vector chunk as the specific packed type
        T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
        // Convert to float2 for computation
        float2 vals = Converter::convert(packed_val);
        sumOfSquares += vals.x * vals.x;
        sumOfSquares += vals.y * vals.y;

        elements[2 * i] = vals.x;
        elements[2 * i + 1] = vals.y;
      }
    }

    // Reduce sum across warp using the utility function
    sumOfSquares = warp_reduce_sum(sumOfSquares);

    // Compute RMS normalization factor
    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

    // Normalize elements
#pragma unroll
    for (int i = 0; i < kNumElemsPerThread; i++) {
      int dim = laneId * kNumElemsPerThread + i;
      float weight = isQ ? Converter::convert(q_weight[dim])
                         : Converter::convert(k_weight[dim]);
      elements[i] *= rms_rcp * weight;
    }

    // Apply RoPE to normalized elements
    float
        elements2[kNumElemsPerThread];  // Additional buffer required for RoPE.

    int64_t pos_id = position_ids[tokenIdx];

    // Calculate cache pointer for this position - similar to
    // pos_encoding_kernels.cu
    T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    int const embed_dim = rotary_dim / 2;
    T_cache const* cos_ptr = cache_ptr;
    T_cache const* sin_ptr = cache_ptr + embed_dim;
    int const rotary_lanes = rotary_dim / kNumElemsPerThread;  // rotary range
    if (laneId < rotary_lanes) {
      if constexpr (interleave) {
        // Perform interleaving. Use pre-computed cos/sin values.
#pragma unroll
        for (int i = 0; i < kNumElemsPerThread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          // Global dimension index in the head
          int const dim_idx = laneId * kNumElemsPerThread + idx0;

          float const val0 = elements[idx0];
          float const val1 = elements[idx1];

          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(xllm_ldg(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(xllm_ldg(sin_ptr + half_dim));

          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        // Before data exchange with in warp, we need to sync.
        __syncwarp();
        int pairOffset = (rotary_dim / 2) / kNumElemsPerThread;
        // Get the data from the other half of the warp. Use pre-computed
        // cos/sin values.
#pragma unroll
        for (int i = 0; i < kNumElemsPerThread; i++) {
          elements2[i] = __shfl_xor_sync(kFinalMask, elements[i], pairOffset);

          if (laneId < pairOffset) {
            elements2[i] = -elements2[i];
          }
          int dim_idx = laneId * kNumElemsPerThread + i;

          dim_idx = (dim_idx * 2) % rotary_dim;
          int half_dim = dim_idx / 2;
          float cos_val = CacheConverter::convert(xllm_ldg(cos_ptr + half_dim));
          float sin_val = CacheConverter::convert(xllm_ldg(sin_ptr + half_dim));

          elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
        }
        // __shfl_xor_sync does not provide memfence. Need to sync again.
        __syncwarp();
      }
    }
    // Store.
    {
      vec_T vec;
      constexpr int kNumPackedElems = kElemSizeBytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < kNumPackedElems; i++) {
        // Convert from float2 back to the specific packed type
        float2 vals = {elements[2 * i], elements[2 * i + 1]};
        T2_in packed_val = Converter::convert(vals);
        // Place it into the generic vector
        *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
      }
      *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
    }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_DCU)
  }
#endif
}

// Borrowed from
// https://github.com/flashinfer-ai/flashinfer/blob/8125d079a43e9a0ba463a4ed1b639cefd084cec9/include/flashinfer/pos_enc.cuh#L568
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

template <typename scalar_t_in, typename scalar_t_cache>
void launch_fused_qknorm_rope(void* qkv,
                              int const num_tokens,
                              int const num_heads_q,
                              int const num_heads_k,
                              int const num_heads_v,
                              int const head_dim,
                              int const rotary_dim,
                              float const eps,
                              void const* q_weight,
                              void const* k_weight,
                              void const* cos_sin_cache,
                              bool const interleave,
                              int64_t const* position_ids,
                              cudaStream_t stream) {
  constexpr int kBlockSize = 256;

  int const warpsPerBlock = kBlockSize / 32;
  int const totalQKHeads = num_heads_q + num_heads_k;
  int const totalWarps = num_tokens * totalQKHeads;

  int const gridSize = div_up(totalWarps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(kBlockSize);

  switch (head_dim) {
    case 64:
      DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        fused_qknorm_rope_kernel<scalar_t_in, scalar_t_cache, 64, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(qkv,
                                               num_heads_q,
                                               num_heads_k,
                                               num_heads_v,
                                               eps,
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
        fused_qknorm_rope_kernel<scalar_t_in, scalar_t_cache, 128, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(qkv,
                                               num_heads_q,
                                               num_heads_k,
                                               num_heads_v,
                                               eps,
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
        fused_qknorm_rope_kernel<scalar_t_in, scalar_t_cache, 256, INTERLEAVE>
            <<<gridDim, blockDim, 0, stream>>>(qkv,
                                               num_heads_q,
                                               num_heads_k,
                                               num_heads_v,
                                               eps,
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

namespace xllm::kernel::cuda {

void fused_qk_norm_rope(
    torch::Tensor& qkv,   // Combined QKV tensor [num_tokens,
                          // (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,  // Number of query heads
    int64_t num_heads_k,  // Number of key heads
    int64_t num_heads_v,  // Number of value heads
    int64_t head_dim,     // Dimension per head
    double eps,           // Epsilon for RMS normalization
    const torch::Tensor& q_weight,  // RMSNorm weights for query [head_dim]
    const torch::Tensor& k_weight,  // RMSNorm weights for key [head_dim]
    const torch::Tensor&
        cos_sin_cache,  // Cos/sin cache [max_position, rotary_dim]
    bool interleaved,   // Whether RoPE is applied in interleaved style
    const torch::Tensor& position_ids  // Position IDs for RoPE [num_tokens]
) {
  // Input validation
  CHECK(qkv.is_cuda()) << "qkv must be a CUDA tensor";
  CHECK(qkv.is_contiguous()) << "qkv must be contiguous";
  CHECK(position_ids.is_cuda()) << "position_ids must be a CUDA tensor";
  CHECK(position_ids.is_contiguous()) << "position_ids must be contiguous";
  CHECK(q_weight.is_cuda()) << "q_weight must be a CUDA tensor";
  CHECK(q_weight.is_contiguous()) << "q_weight must be contiguous";
  CHECK(k_weight.is_cuda()) << "k_weight must be a CUDA tensor";
  CHECK(k_weight.is_contiguous()) << "k_weight must be contiguous";
  CHECK(cos_sin_cache.is_cuda()) << "cos_sin_cache must be a CUDA tensor";
  CHECK(cos_sin_cache.is_contiguous()) << "cos_sin_cache must be contiguous";
  CHECK(position_ids.scalar_type() == torch::kInt64)
      << "position_ids dtype is " << position_ids.scalar_type()
      << ", while Int64 is expected";

  CHECK(qkv.dim() == 2) << "QKV tensor must be 2D: [num_tokens, "
                        << "(num_heads_q+num_heads_k+num_heads_v)*head_dim]";
  CHECK(position_ids.dim() == 1) << "Position IDs must be 1D: [num_tokens]";
  CHECK(q_weight.dim() == 1) << "Query weights must be 1D: [head_dim]";
  CHECK(k_weight.dim() == 1) << "Key weights must be 1D: [head_dim]";
  CHECK(cos_sin_cache.dim() == 2)
      << "Cos/sin cache must be 2D: [max_position, rotary_dim]";
  CHECK(q_weight.size(0) == head_dim)
      << "Query weights size must match head dimension";
  CHECK(k_weight.size(0) == head_dim)
      << "Key weights size must match head dimension";

  CHECK(cos_sin_cache.size(1) % 2 == 0) << "rotary_dim must be even";
  CHECK(cos_sin_cache.size(1) <= head_dim)
      << "rotary_dim must be less than or equal to head_dim";

  CHECK(qkv.scalar_type() == q_weight.scalar_type() &&
        qkv.scalar_type() == k_weight.scalar_type())
      << "qkv, q_weight and k_weight must have the same dtype";

  int64_t num_tokens = qkv.size(0);
  CHECK(position_ids.size(0) == num_tokens)
      << "Number of tokens in position_ids must match QKV";

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  CHECK(qkv.size(1) == total_heads * head_dim)
      << "QKV tensor size must match total number of heads and head dimension";

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_HALF_TYPES(qkv.scalar_type(), "fused_qk_norm_rope_kernel", [&] {
    using qkv_scalar_t = scalar_t;
    DISPATCH_FLOATING_TYPES(
        cos_sin_cache.scalar_type(), "fused_qk_norm_rope_kernel", [&] {
          using cache_scalar_t = scalar_t;
          launch_fused_qknorm_rope<qkv_scalar_t, cache_scalar_t>(
              qkv.data_ptr(),
              static_cast<int>(num_tokens),
              static_cast<int>(num_heads_q),
              static_cast<int>(num_heads_k),
              static_cast<int>(num_heads_v),
              static_cast<int>(head_dim),
              static_cast<int>(cos_sin_cache.size(1)),
              static_cast<float>(eps),
              q_weight.data_ptr(),
              k_weight.data_ptr(),
              cos_sin_cache.data_ptr(),
              interleaved,
              reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
              stream);
        });
  });
}

}  // namespace xllm::kernel::cuda
