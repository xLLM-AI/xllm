
/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// refers to
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/moeTopKFuncs.cuh

#pragma once

#include <cooperative_groups.h>
#if !defined(USE_DCU)
#include <cooperative_groups/reduce.h>
#endif

#if defined(USE_MACA)
#include <cuda_bf16.h>
#endif

#if !defined(USE_DCU)
#include <cub/cub.cuh>
#else
#include <hipcub/hipcub.hpp>
#endif

#include "core/kernels/cuda/arch_condition.h"

#if defined(USE_DCU)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#endif

#include "core/kernels/cuda/device_utils.cuh"

namespace xllm::kernel::cuda {
namespace reduce_topk {
namespace cg = cooperative_groups;
static constexpr int kWarpSize = 32;
#if !defined(USE_DCU)
static constexpr bool kTllmGenHasFastRedux = arch::is_major_v<10>;
#else
static constexpr bool kTllmGenHasFastRedux = false;
#endif

template <typename T_>
struct TopKRedType {
  using T = T_;
  static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, half> ||
          std::is_same_v<T, BFloat16Type> || std::is_same_v<T, int>,
      "Top K reduction only implemented for int, float, float16 and bfloat16");

  using TypeCmp = std::conditional_t<sizeof(T) == 4, uint64_t, uint32_t>;
  using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int16_t>;
#if defined(USE_DCU)
  using UnsignedBits = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
#endif

  static constexpr int kMoveBits = (sizeof(T) == 4) ? 32 : 16;
  static constexpr int kMaxIdx = 65535;
  TypeCmp compValIdx;

  static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0) {
#if !defined(USE_DCU)
    auto valueBits = cub::Traits<T>::TwiddleIn(
        reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
#else
    UnsignedBits valueBits = reinterpret_cast<UnsignedBits&>(val);
    constexpr UnsignedBits kSignMask =
        static_cast<UnsignedBits>(UnsignedBits{1} << (sizeof(T) * 8 - 1));
    if constexpr (std::is_same_v<T, int>) {
      valueBits = static_cast<UnsignedBits>(valueBits ^ kSignMask);
    } else {
      valueBits = (valueBits & kSignMask)
                      ? static_cast<UnsignedBits>(~valueBits)
                      : static_cast<UnsignedBits>(valueBits ^ kSignMask);
    }
#endif
    TypeCmp compactTmp = valueBits;
    compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
    // Use 65535 minus idx to give higher priority to elements with smaller
    // indices.
    return compactTmp;
  }

  static __host__ __device__ void unpack(T& value,
                                         int32_t& index,
                                         TypeCmp cmp) {
    // Since "65535-idx" is always smaller than 65536 and positive, we can
    // directly use it as the lower 16 bits
    index = kMaxIdx - static_cast<int32_t>((cmp & 0xFFFF));

    auto compactTmp = cmp >> kMoveBits;
#if !defined(USE_DCU)
    auto valueBits = cub::Traits<T>::TwiddleOut(
        reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(compactTmp));
#else
    UnsignedBits valueBits = static_cast<UnsignedBits>(compactTmp);
    constexpr UnsignedBits kSignMask =
        static_cast<UnsignedBits>(UnsignedBits{1} << (sizeof(T) * 8 - 1));
    if constexpr (std::is_same_v<T, int>) {
      valueBits = static_cast<UnsignedBits>(valueBits ^ kSignMask);
    } else {
      valueBits = (valueBits & kSignMask)
                      ? static_cast<UnsignedBits>(valueBits ^ kSignMask)
                      : static_cast<UnsignedBits>(~valueBits);
    }
#endif
    value = reinterpret_cast<T&>(valueBits);
  }

  __host__ __device__ TopKRedType() = default;

  __host__ __device__ TopKRedType(T val, int32_t idx)
      : compValIdx(makeCmpVal(val, idx)) {}

  __host__ __device__ operator TypeCmp() const noexcept { return compValIdx; }

  __device__ inline TypeCmp reduce(
      cg::thread_block_tile<kWarpSize> const& warp) {
#if defined(USE_DCU)
    TypeCmp result = compValIdx;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      TypeCmp other = warp.shfl_down(result, offset);
      result = other > result ? other : result;
    }
    return warp.shfl(result, 0);
#else
    if constexpr (!kTllmGenHasFastRedux || sizeof(TypeCmp) == 8) {
      return cg::reduce(warp, compValIdx, cg::greater<TypeCmp>{});
    } else {
      TypeCmp result;
      asm("redux.sync.max.u32 %0, %1, 0xffffffff;\n"
          : "=r"(result)
          : "r"(compValIdx));
      return result;
    }
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K_, bool Enable_>
struct TopKIdx {
  // by default, empty
};

template <int K_>
struct TopKIdx<K_, true> {
  static constexpr int K = K_;
  int32_t val[K];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TOPK_SWAP(I, J)                                         \
  {                                                             \
    auto pairMin = min(topK[I].compValIdx, topK[J].compValIdx); \
    auto pairMax = max(topK[I].compValIdx, topK[J].compValIdx); \
    topK[I].compValIdx = pairMax;                               \
    topK[J].compValIdx = pairMin;                               \
  }

template <int N, typename RedType>
struct Sort;

template <typename RedType>
struct Sort<1, RedType> {
  static __device__ void run(RedType* topK) {}
};

template <typename RedType>
struct Sort<2, RedType> {
  static __device__ void run(RedType* topK) { TOPK_SWAP(0, 1); }
};

template <typename RedType>
struct Sort<3, RedType> {
  static __device__ void run(RedType* topK) {
    TOPK_SWAP(0, 1);
    TOPK_SWAP(1, 2);
    TOPK_SWAP(0, 1);
  }
};

template <typename RedType>
struct Sort<4, RedType> {
  static __device__ void run(RedType* topK) {
    TOPK_SWAP(0, 2);
    TOPK_SWAP(1, 3);
    TOPK_SWAP(0, 1);
    TOPK_SWAP(2, 3);
    TOPK_SWAP(1, 2);
  }
};

template <int K, typename Type>
__forceinline__ __device__ void reduceTopK(
    cg::thread_block_tile<kWarpSize> const& warp,
    Type (&out)[K],
    int32_t (&outIdx)[K],
    Type value,
    int32_t idx,
    Type const minValue,
    int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWarpSize, "Top K must have K < kWarpSize");
  using RedType = TopKRedType<Type>;
  RedType topK{value, idx};
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  for (int kk = 0; kk < actualK; ++kk)  //@todo: check if actualK is correct
  {
    topK =
        kk > 0 && packedMax == topK.compValIdx ? RedType{minValue, idx} : topK;
    // get the next largest value
    packedMax = topK.reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopKFunc(cg::thread_block_tile<kWarpSize> const& warp,
                               Type (&out)[K],
                               int32_t (&outIdx)[K],
                               Type (&value)[N],
                               int32_t (&idx)[N],
                               Type minValue,
                               int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWarpSize, "Top K must have K < kWarpSize");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(N < 5,
                "Only support candidates number less than or equal to 128");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};
  }

  if constexpr (!IsSorted) {
    Sort<N, RedType>::run(topK);
  }
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  for (int kk = 0; kk < actualK; ++kk) {
    bool update = kk > 0 && packedMax == topK[0].compValIdx;
#pragma unroll
    for (int nn = 0; nn < N; ++nn) {
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]}
                 : update              ? topK[nn + 1]
                                       : topK[nn];
    }
    // get the next largest value
    packedMax = topK[0].reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopK(
    cg::thread_block_tile<kWarpSize> const& warp,
    Type (&out)[K],
    int32_t (&outIdx)[K],
    Type (&value)[N],
    int32_t (&idx)[N],
    Type const minValue,
    int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < kWarpSize, "Top K must have K < kWarpSize");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(
      N <= 16,
      "Only support candidates number less than or equal to 16*32=512");
  static_assert(N <= 4 || N % 4 == 0,
                "Only support candidates number is a multiple of 4*32=128 or "
                "less than or equal to 4");
  using RedType = TopKRedType<Type>;

  if constexpr (N <= 4) {
    reduceTopKFunc<K, Type, N>(
        warp, out, outIdx, value, idx, minValue, actualK);
  } else {
    constexpr int kNumLoops = N / 4;
    constexpr int kNumResults = (kNumLoops * K - 1) / kWarpSize + 1;

    Type topKBufferValue[kNumResults];
    int32_t topKBufferIdx[kNumResults];
    int32_t laneIdx = threadIdx.x % kWarpSize;

    // Sentinel index must be in [0, kMaxIdx] to survive makeCmpVal pack/unpack
    // (kMaxIdx - idx is stored in 16 bits; -1 would become 0 and unpack to
    // 65535). Use kMaxIdx so sentinel slots have smallest compValIdx for
    // minValue and lose to any real candidate.
    for (int ii = 0; ii < kNumResults; ++ii) {
      topKBufferValue[ii] = minValue;
      topKBufferIdx[ii] = RedType::kMaxIdx;
    }
    for (int loop = 0; loop < kNumLoops; ++loop) {
      int start = loop * 4;
      Type topKValue[K];
      int32_t topKIdx[K];
      Type inValue[4];
      int32_t inIdx[4];
      for (int i = 0; i < 4; ++i) {
        inValue[i] = value[start + i];
        inIdx[i] = idx[start + i];
      }
      reduceTopKFunc<K, Type, 4>(
          warp, topKValue, topKIdx, inValue, inIdx, minValue, actualK);
      int inOffset = laneIdx % K;
      if (laneIdx >= loop * K && laneIdx < (loop + 1) * K) {
        topKBufferValue[0] = topKValue[inOffset];
        topKBufferIdx[0] = topKIdx[inOffset];
      }
      if (loop == kNumLoops - 1 && (laneIdx < (kNumLoops * K - kWarpSize))) {
        topKBufferValue[1] = topKValue[inOffset];
        topKBufferIdx[1] = topKIdx[inOffset];
      }
    }

    reduceTopKFunc<K, Type, kNumResults>(
        warp, out, outIdx, topKBufferValue, topKBufferIdx, minValue, actualK);
  }
};

#undef TOPK_SWAP

}  // namespace reduce_topk
}  // namespace xllm::kernel::cuda
