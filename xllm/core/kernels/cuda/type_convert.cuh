/* Copyright 2025 The vLLM Authors and The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/type_convert.cuh

/* Converter helpers for the conversion from torch types to HIP/CUDA types,
   and the associated type conversions within HIP/CUDA. These helpers need
   to be implemented for now because the relevant type conversion
   operators/constructors are not consistently implemented by HIP/CUDA, so
   a generic conversion via type casts cannot be implemented.

   Each helper should have the member static constexpr bool `exists`:
   If false, the optimized kernel is not used for the corresponding torch type.
   If true, the helper should be fully defined as shown in the examples below.
 */
namespace xllm::kernel::cuda {
template <typename torch_type>
class _typeConvert {
 public:
  static constexpr bool exists = false;
};

template <>
class _typeConvert<float> {
 public:
  static constexpr bool exists = true;
  using hip_type = float;
  using packed_hip_type = float2;
  using packed_hip_type4 = float4;  // For 128-bit vectorization

  __device__ static __forceinline__ float convert(hip_type x) { return x; }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return x;
  }
  __device__ static __forceinline__ float4 convert(packed_hip_type4 x) {
    return x;
  }
};

#if defined(USE_DCU) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)) || \
    defined(USE_MACA)
// CUDA < 12.0 runs into issues with packed type conversion
template <>
class _typeConvert<c10::Half> {
 public:
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;

  __device__ static __forceinline__ float convert(hip_type x) {
    return __half2float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return __half22float2(x);
  }
  __device__ static __forceinline__ hip_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};
#endif  // defined(USE_DCU) || CUDA_VERSION >= 12000

#if defined(USE_DCU)
template <>
class _typeConvert<c10::BFloat16> {
 public:
  static constexpr bool exists = true;
  using hip_type = __hip_bfloat16;
  using packed_hip_type = __hip_bfloat162;

  __device__ static __forceinline__ float convert(hip_type x) {
    return __bfloat162float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static __forceinline__ hip_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};
#elif defined(CUDA_VERSION) && (CUDA_VERSION >= 12000) &&   \
        defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) || \
    defined(USE_MACA)

// CUDA_ARCH < 800 does not have BF16 support.
template <>
class _typeConvert<c10::BFloat16> {
 public:
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;

  __device__ static __forceinline__ float convert(hip_type x) {
    return __bfloat162float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static __forceinline__ hip_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};
#endif

/* Vector helper to generate vectorized and packed FP16/BF16 ops
   for appropriate specializations of fused_add_rms_norm_kernel.
   Only functions that are necessary in that kernel are implemented.
   Alignment to 16 bytes is required to use 128-bit global memory ops.
 */

template <typename scalar_t, int width>
class alignas(16) _f16Vec {
 public:
  /* Not theoretically necessary that width is a power of 2 but should
     almost always be the case for optimization purposes */
  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        if constexpr (std::is_same_v<T2, float2>) {
          data[i] += other.data[i];
          data[i + 1] += other.data[i + 1];
        } else {
          T2 temp{data[i], data[i + 1]};
          temp += T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] += other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        if constexpr (std::is_same_v<T2, float2>) {
          data[i] *= other.data[i];
          data[i + 1] *= other.data[i + 1];
        } else {
          T2 temp{data[i], data[i + 1]};
          temp *= T2{other.data[i], other.data[i + 1]};
          data[i] = temp.x;
          data[i + 1] = temp.y;
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] *= other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
        temp_f.x *= scale;
        temp_f.y *= scale;
        T2 temp = Converter::convert(temp_f);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float temp = Converter::convert(data[i]) * scale;
        data[i] = Converter::convert(temp);
      }
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        result += z.x * z.x + z.y * z.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        result += x * x;
      }
    }
    return result;
  }
};
}  // namespace xllm::kernel::cuda
