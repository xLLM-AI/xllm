/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

#pragma once
// clang-format off
#include <c10/util/Float8_e4m3fn.h>
#include <cmath>
#include <torch/types.h>
// clang-format on
namespace xllm {
namespace kernel {
namespace cuda {

// FP8 type max value definitions
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, int8_t>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

template <typename T>
__host__ __device__ static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

// Minimum scaling factor for quantization types
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  __device__ __host__ static inline float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

template <>
struct min_scaling_factor<int8_t> {
  __device__ __host__ static inline float val() {
    return std::numeric_limits<float>::epsilon();
  }
};

// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

template <typename quant_type_t, size_t vec_size>
struct __align__(vec_size * sizeof(quant_type_t)) q8_n_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fn>);
  quant_type_t val[vec_size];
};

// Atomic max for float
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

// FP8 conversion functions
namespace fp8 {

#ifdef ENABLE_FP8

#include <cuda_fp8.h>

// float -> c10::Float8_e4m3fn conversion
template <typename Tout, typename Tin>
__inline__ __device__ Tout
vec_conversion(const Tin& x,
               const __nv_fp8_interpretation_t fp8_type = __NV_E4M3) {
  return x;
}

template <>
__inline__ __device__ c10::Float8_e4m3fn
vec_conversion<c10::Float8_e4m3fn, float>(
    const float& a,
    const __nv_fp8_interpretation_t fp8_type) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return static_cast<c10::Float8_e4m3fn>(a);
#else
  return c10::Float8_e4m3fn(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type),
                            c10::Float8_e4m3fn::from_bits());
#endif
}

#endif  // ENABLE_FP8

}  // namespace fp8

// Scaled FP8 conversion with saturation
template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r =
      fmaxf(-quant_type_max_v<fp8_type>, fminf(x, quant_type_max_v<fp8_type>));

#ifdef ENABLE_FP8
  // Use hardware cvt instruction for fp8 on nvidia
  return fp8::vec_conversion<fp8_type, float>(r);
#else
  return static_cast<fp8_type>(r);
#endif
}

// Vectorization utilities
template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
  ScaOp scalar_op;

  __device__ __forceinline__ void operator()(
      vec_n_t<OutT, VEC_SIZE>& dst,
      const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(dst.val[i], src.val[i]);
    }
  }
};

template <int VEC_SIZE,
          typename InT,
          typename OutT,
          typename VecOp,
          typename ScaOp>
__device__ inline void vectorize_with_alignment(
    const InT* in,
    OutT* out,
    int len,
    int tid,
    int stride,
    VecOp&& vec_op,       // vec_n_t<InT,16> -> vec_n_t<OutT,16>
    ScaOp&& scalar_op) {  // InT -> OutT
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  // Fast path when the whole region is already aligned
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);

    for (int i = tid; i < num_vec; i += stride) {
      vout_t tmp;
      vin_t src = v_in[i];
      vec_op(tmp, src);
      v_out[i] = tmp;
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);
  int alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = alignment_bytes & (WIDTH - 1);
  prefix_elems /= sizeof(InT);
  prefix_elems = min(prefix_elems, len);

  // Prefix handling
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(out[i], in[i]);
  }

  in += prefix_elems;
  out += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  using vout_t = vec_n_t<OutT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);

  // Vectorized main part
  for (int i = tid; i < num_vec; i += stride) {
    vout_t tmp;
    vin_t src = v_in[i];
    vec_op(tmp, src);
    v_out[i] = tmp;
  }

  // Tail handling
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(out[i], in[i]);
  }
}

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
__device__ __forceinline__ void vectorize_with_alignment(const InT* in,
                                                         OutT* out,
                                                         int len,
                                                         int tid,
                                                         int stride,
                                                         ScaOp&& scalar_op) {
  using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>>;
  vectorize_with_alignment<VEC_SIZE>(in,
                                     out,
                                     len,
                                     tid,
                                     stride,
                                     Vec{scalar_op},
                                     std::forward<ScaOp>(scalar_op));
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
