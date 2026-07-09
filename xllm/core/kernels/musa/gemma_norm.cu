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

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/cuda.h>

#include <cstdint>

#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::cuda {

namespace {

template <typename T>
struct __align__(16) Vec8Storage {
  T elem[8];
};

struct __align__(32) Float8Storage {
  float elem[8];
};

template <typename T>
class __align__(16) Vec8 final {
 public:
  union {
    Vec8Storage<T> storage;
    T elem[8];
  } val;

  __device__ __forceinline__ Vec8() {}

  template <typename Offset>
  static __device__ __forceinline__ Vec8 load(const T* ptr, Offset idx) {
    return *reinterpret_cast<const Vec8*>(ptr + idx);
  }

  template <typename Offset>
  static __device__ __forceinline__ Vec8 load_byp_slc(const T* ptr,
                                                      Offset idx) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
    uint4 raw;
    const T* addr = ptr + idx;
    asm volatile(
        "LSU.LD.B128 %0, %1, _, 16, 1, 1, inner_persist=0, outer_persist=2, "
        "chrnt=l2_l3, slc=byp, persist=0, stride_add_first=0"
        : "=R"(raw)
        : "R"(addr));
    Vec8 dst;
    *reinterpret_cast<uint4*>(&dst) = raw;
    return dst;
#else
    return *reinterpret_cast<const Vec8*>(ptr + idx);
#endif
  }
};

class __align__(32) Float8 final {
 public:
  union {
    Float8Storage storage;
    float elem[8];
  } val;

  __device__ __forceinline__ Float8() {}
};

template <typename T>
__device__ __forceinline__ float gemma_to_float(T value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else {
    return static_cast<float>(value);
  }
}

template <typename T>
__device__ __forceinline__ T gemma_from_float(float value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16_rn(value);
  } else {
    return static_cast<T>(value);
  }
}

__device__ __forceinline__ float fast_rsqrt(float value) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  const float half_value = 0.5f * value;
  float y = __frsqrt_rn(value);
  y = y * (1.5f - half_value * y * y);
  return y;
#else
  return rsqrtf(value);
#endif
}

__device__ __forceinline__ void block_sync() {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  __syncthreads_lm();
#else
  __syncthreads();
#endif
}

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  block_sync();

  value = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  block_sync();
  return warp_sums[0];
}

__device__ __forceinline__ float block_sum_8warps(float value,
                                                  float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  block_sync();

  value = lane < 8 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  block_sync();
  return warp_sums[0];
}

__device__ __forceinline__ float block_sum_4warps(float value,
                                                  float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  block_sync();

  value = lane < 4 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  block_sync();
  return warp_sums[0];
}

template <typename T, bool GEMMA, bool CACHE>
__global__ void __launch_bounds__(1024, 1)
    rmsnorm_vec8_kernel(const T* __restrict__ input,
                        const T* __restrict__ weight,
                        T* __restrict__ out,
                        int rows,
                        int hidden,
                        int input_outer_dim,
                        int64_t input_outer_stride,
                        int64_t input_row_stride,
                        int64_t out_row_stride,
                        float inv_hidden,
                        float eps) {
  constexpr int kVec = 8;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + (CACHE ? hidden : 0);

  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / kVec;
  const int64_t input_base =
      static_cast<int64_t>(row / input_outer_dim) * input_outer_stride +
      static_cast<int64_t>(row % input_outer_dim) * input_row_stride;
  const int64_t out_base = static_cast<int64_t>(row) * out_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count;
       vec_idx += static_cast<int>(blockDim.x)) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load(input + input_base, col);
    Float8 x_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value = gemma_to_float<T>(x.val.elem[i]);
      sum += value * value;
      x_float.val.elem[i] = value;
    }
    if constexpr (CACHE) {
      *reinterpret_cast<Float8*>(cached + col) = x_float;
    }
  }

  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count;
       vec_idx += static_cast<int>(blockDim.x)) {
    const int col = vec_idx * kVec;
    Float8 x_float;
    if constexpr (CACHE) {
      x_float = *reinterpret_cast<Float8*>(cached + col);
    } else {
      Vec8<T> x = Vec8<T>::load(input + input_base, col);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        x_float.val.elem[i] = gemma_to_float<T>(x.val.elem[i]);
      }
    }
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value =
          gemma_to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] =
          gemma_from_float<T>(x_float.val.elem[i] * scale * weight_value);
    }
    *reinterpret_cast<Vec8<T>*>(out + out_base + col) = dst;
  }
}

template <typename T, bool GEMMA, int H, int WARPS>
__global__ void __launch_bounds__(256, 1)
    rmsnorm_small_h_one_vec_register_kernel(const T* __restrict__ input,
                                            const T* __restrict__ weight,
                                            T* __restrict__ out,
                                            int rows,
                                            int hidden,
                                            int input_outer_dim,
                                            int64_t input_outer_stride,
                                            int64_t input_row_stride,
                                            int64_t out_row_stride,
                                            float inv_hidden,
                                            float eps) {
  constexpr int kVec = 8;
  extern __shared__ float warp_sums[];

  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int col = tid * kVec;
  const int64_t input_base =
      static_cast<int64_t>(row / input_outer_dim) * input_outer_stride +
      static_cast<int64_t>(row % input_outer_dim) * input_row_stride;
  const int64_t out_base = static_cast<int64_t>(row) * out_row_stride;
  float sum = 0.0f;
  Float8 x_float;

  Vec8<T> x = Vec8<T>::load(input + input_base, col);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float value = gemma_to_float<T>(x.val.elem[i]);
    sum += value * value;
    x_float.val.elem[i] = value;
  }

  if constexpr (WARPS == 4) {
    sum = block_sum_4warps(sum, warp_sums);
  } else {
    sum = block_sum_8warps(sum, warp_sums);
  }

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  Vec8<T> w = Vec8<T>::load(weight, col);
  Vec8<T> dst;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float weight_value =
        gemma_to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
    dst.val.elem[i] =
        gemma_from_float<T>(x_float.val.elem[i] * scale * weight_value);
  }
  *reinterpret_cast<Vec8<T>*>(out + out_base + col) = dst;
}

template <typename T, bool GEMMA>
__global__ void rmsnorm_scalar_kernel(const T* __restrict__ input,
                                      const T* __restrict__ weight,
                                      T* __restrict__ out,
                                      int rows,
                                      int hidden,
                                      int input_outer_dim,
                                      int64_t input_outer_stride,
                                      int64_t input_row_stride,
                                      int64_t out_row_stride,
                                      float inv_hidden,
                                      float eps) {
  extern __shared__ float warp_sums[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t input_base =
      static_cast<int64_t>(row / input_outer_dim) * input_outer_stride +
      static_cast<int64_t>(row % input_outer_dim) * input_row_stride;
  const int64_t out_base = static_cast<int64_t>(row) * out_row_stride;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += static_cast<int>(blockDim.x)) {
    const float value = gemma_to_float<T>(input[input_base + col]);
    sum += value * value;
  }
  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int col = tid; col < hidden; col += static_cast<int>(blockDim.x)) {
    const float weight_value =
        gemma_to_float<T>(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    out[out_base + col] = gemma_from_float<T>(
        gemma_to_float<T>(input[input_base + col]) * scale * weight_value);
  }
}

template <typename T, bool GEMMA, bool CACHE>
__global__ void __launch_bounds__(1024, 1)
    fused_add_rmsnorm_vec8_kernel(T* __restrict__ input,
                                  T* __restrict__ residual,
                                  const T* __restrict__ weight,
                                  int rows,
                                  int hidden,
                                  int64_t input_row_stride,
                                  int64_t residual_row_stride,
                                  float inv_hidden,
                                  float eps) {
  constexpr int kVec = 8;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + (CACHE ? hidden : 0);

  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / kVec;
  const int64_t input_base = static_cast<int64_t>(row) * input_row_stride;
  const int64_t residual_base = static_cast<int64_t>(row) * residual_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count;
       vec_idx += static_cast<int>(blockDim.x)) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load_byp_slc(input + input_base, col);
    Vec8<T> r = Vec8<T>::load_byp_slc(residual + residual_base, col);
    Vec8<T> residual_out;
    Float8 sum_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value =
          gemma_to_float<T>(x.val.elem[i]) + gemma_to_float<T>(r.val.elem[i]);
      sum += value * value;
      residual_out.val.elem[i] = gemma_from_float<T>(value);
      sum_float.val.elem[i] = value;
    }
    *reinterpret_cast<Vec8<T>*>(residual + residual_base + col) = residual_out;
    if constexpr (CACHE) {
      *reinterpret_cast<Float8*>(cached + col) = sum_float;
    }
  }

  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count;
       vec_idx += static_cast<int>(blockDim.x)) {
    const int col = vec_idx * kVec;
    Float8 sum_float;
    if constexpr (CACHE) {
      sum_float = *reinterpret_cast<Float8*>(cached + col);
    } else {
      Vec8<T> r = Vec8<T>::load(residual + residual_base, col);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        sum_float.val.elem[i] = gemma_to_float<T>(r.val.elem[i]);
      }
    }
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value =
          gemma_to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] =
          gemma_from_float<T>(sum_float.val.elem[i] * scale * weight_value);
    }
    *reinterpret_cast<Vec8<T>*>(input + input_base + col) = dst;
  }
}

template <typename T, bool GEMMA>
__global__ void fused_add_rmsnorm_scalar_kernel(T* __restrict__ input,
                                                T* __restrict__ residual,
                                                const T* __restrict__ weight,
                                                int rows,
                                                int hidden,
                                                int64_t input_row_stride,
                                                int64_t residual_row_stride,
                                                float inv_hidden,
                                                float eps) {
  extern __shared__ float warp_sums[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t input_base = static_cast<int64_t>(row) * input_row_stride;
  const int64_t residual_base = static_cast<int64_t>(row) * residual_row_stride;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += static_cast<int>(blockDim.x)) {
    const float value = gemma_to_float<T>(input[input_base + col]) +
                        gemma_to_float<T>(residual[residual_base + col]);
    residual[residual_base + col] = gemma_from_float<T>(value);
    sum += value * value;
  }
  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int col = tid; col < hidden; col += static_cast<int>(blockDim.x)) {
    const float weight_value =
        gemma_to_float<T>(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    input[input_base + col] =
        gemma_from_float<T>(gemma_to_float<T>(residual[residual_base + col]) *
                            scale * weight_value);
  }
}

inline int vec8_block_threads(int hidden) {
  const int vec_count = hidden / 8;
  const int rounded = ((vec_count + 31) / 32) * 32;
  return rounded < 1024 ? rounded : 1024;
}

inline int rmsnorm_block_threads(int rows, int hidden) {
  if (hidden <= 512) {
    return 64;
  }
  if (hidden <= 4096) {
    if (rows <= 16) {
      const int threads = vec8_block_threads(hidden);
      return threads < 512 ? threads : 512;
    }
    if (rows <= 256) {
      const int threads = vec8_block_threads(hidden);
      return threads < 256 ? threads : 256;
    }
    return 128;
  }
  if (hidden <= 8192) {
    const int threads = vec8_block_threads(hidden);
    return threads < 512 ? threads : 512;
  }
  const int threads = vec8_block_threads(hidden);
  return threads < 896 ? threads : 896;
}

inline int fused_block_threads(int hidden) {
  return vec8_block_threads(hidden);
}

inline int cached_vec8_shared_bytes(int hidden,
                                    int block_threads,
                                    int cache_hidden_limit) {
  const int reduce_floats = (block_threads + 31) / 32;
  const int cached_floats = hidden <= cache_hidden_limit ? hidden : 0;
  return (cached_floats + reduce_floats) * static_cast<int>(sizeof(float));
}

template <typename T>
void launch_rmsnorm_gemma(const T* input_ptr,
                          const T* weight_ptr,
                          T* out_ptr,
                          int rows,
                          int hidden,
                          int input_outer_dim,
                          int64_t input_outer_stride,
                          int64_t input_row_stride,
                          int64_t out_row_stride,
                          float inv_hidden,
                          float eps,
                          cudaStream_t stream) {
  if ((hidden % 8) == 0 && hidden <= 32768) {
    if (rows <= 16 && hidden == 1024) {
      constexpr int threads = 128;
      constexpr int smem_bytes = 4 * static_cast<int>(sizeof(float));
      rmsnorm_small_h_one_vec_register_kernel<T,
                                              /*GEMMA=*/true,
                                              /*H=*/1024,
                                              /*WARPS=*/4>
          <<<rows, threads, smem_bytes, stream>>>(input_ptr,
                                                  weight_ptr,
                                                  out_ptr,
                                                  rows,
                                                  hidden,
                                                  input_outer_dim,
                                                  input_outer_stride,
                                                  input_row_stride,
                                                  out_row_stride,
                                                  inv_hidden,
                                                  eps);
      return;
    }
    if (rows <= 16 && hidden == 2048) {
      constexpr int threads = 256;
      constexpr int smem_bytes = 8 * static_cast<int>(sizeof(float));
      rmsnorm_small_h_one_vec_register_kernel<T,
                                              /*GEMMA=*/true,
                                              /*H=*/2048,
                                              /*WARPS=*/8>
          <<<rows, threads, smem_bytes, stream>>>(input_ptr,
                                                  weight_ptr,
                                                  out_ptr,
                                                  rows,
                                                  hidden,
                                                  input_outer_dim,
                                                  input_outer_stride,
                                                  input_row_stride,
                                                  out_row_stride,
                                                  inv_hidden,
                                                  eps);
      return;
    }
    const int threads = rmsnorm_block_threads(rows, hidden);
    if (hidden <= 8192) {
      const int smem = cached_vec8_shared_bytes(hidden, threads, 8192);
      rmsnorm_vec8_kernel<T, /*GEMMA=*/true, /*CACHE=*/true>
          <<<rows, threads, smem, stream>>>(input_ptr,
                                            weight_ptr,
                                            out_ptr,
                                            rows,
                                            hidden,
                                            input_outer_dim,
                                            input_outer_stride,
                                            input_row_stride,
                                            out_row_stride,
                                            inv_hidden,
                                            eps);
    } else {
      const int smem = cached_vec8_shared_bytes(hidden, threads, 8192);
      rmsnorm_vec8_kernel<T, /*GEMMA=*/true, /*CACHE=*/false>
          <<<rows, threads, smem, stream>>>(input_ptr,
                                            weight_ptr,
                                            out_ptr,
                                            rows,
                                            hidden,
                                            input_outer_dim,
                                            input_outer_stride,
                                            input_row_stride,
                                            out_row_stride,
                                            inv_hidden,
                                            eps);
    }
  } else {
    constexpr int threads = 256;
    constexpr int smem =
        ((threads + 31) / 32) * static_cast<int>(sizeof(float));
    rmsnorm_scalar_kernel<T, /*GEMMA=*/true>
        <<<rows, threads, smem, stream>>>(input_ptr,
                                          weight_ptr,
                                          out_ptr,
                                          rows,
                                          hidden,
                                          input_outer_dim,
                                          input_outer_stride,
                                          input_row_stride,
                                          out_row_stride,
                                          inv_hidden,
                                          eps);
  }
}

template <typename T>
void launch_fused_add_rmsnorm_gemma(T* input_ptr,
                                    T* residual_ptr,
                                    const T* weight_ptr,
                                    int rows,
                                    int hidden,
                                    int64_t input_row_stride,
                                    int64_t residual_row_stride,
                                    float inv_hidden,
                                    float eps,
                                    cudaStream_t stream) {
  if ((hidden % 8) == 0 && hidden <= 32768) {
    const int threads = fused_block_threads(hidden);
    const int smem = cached_vec8_shared_bytes(hidden, threads, 32768);
    fused_add_rmsnorm_vec8_kernel<T, /*GEMMA=*/true, /*CACHE=*/true>
        <<<rows, threads, smem, stream>>>(input_ptr,
                                          residual_ptr,
                                          weight_ptr,
                                          rows,
                                          hidden,
                                          input_row_stride,
                                          residual_row_stride,
                                          inv_hidden,
                                          eps);
  } else {
    constexpr int threads = 256;
    constexpr int smem =
        ((threads + 31) / 32) * static_cast<int>(sizeof(float));
    fused_add_rmsnorm_scalar_kernel<T, /*GEMMA=*/true>
        <<<rows, threads, smem, stream>>>(input_ptr,
                                          residual_ptr,
                                          weight_ptr,
                                          rows,
                                          hidden,
                                          input_row_stride,
                                          residual_row_stride,
                                          inv_hidden,
                                          eps);
  }
}

}  // namespace

void gemma_rms_norm(torch::Tensor output,
                    torch::Tensor input,
                    torch::Tensor weight,
                    double eps) {
  CHECK(input.scalar_type() == output.scalar_type());
  CHECK(input.scalar_type() == weight.scalar_type());
  CHECK(output.is_contiguous());
  CHECK(weight.is_contiguous());
  CHECK(input.stride(-1) == 1)
      << "gemma_rms_norm requires the last dim to be contiguous "
         "(stride(-1)==1). Got strides="
      << input.strides() << ", sizes=" << input.sizes();
  CHECK(input.dim() >= 2 && input.dim() <= 3)
      << "gemma_rms_norm supports 2D [rows, hidden] or 3D [outer, mid, "
         "hidden] inputs only. Got sizes="
      << input.sizes();

  const int hidden = input.size(-1);
  const int rows = input.numel() / hidden;
  const int64_t input_row_stride = input.stride(-2);
  const int input_outer_dim =
      input.dim() == 3 ? static_cast<int>(input.size(-2)) : rows;
  const int64_t input_outer_stride = input.dim() == 3 ? input.stride(-3) : 0;
  const int64_t out_row_stride = output.stride(-2);
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "gemma_rms_norm",
      AT_DISPATCH_CASE(torch::ScalarType::Half, [&] {
        launch_rmsnorm_gemma<__half>(
            reinterpret_cast<const __half*>(input.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<c10::Half>()),
            rows,
            hidden,
            input_outer_dim,
            input_outer_stride,
            input_row_stride,
            out_row_stride,
            inv_hidden,
            static_cast<float>(eps),
            stream);
      }) AT_DISPATCH_CASE(torch::ScalarType::BFloat16, [&] {
        launch_rmsnorm_gemma<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(
                input.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<c10::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<c10::BFloat16>()),
            rows,
            hidden,
            input_outer_dim,
            input_outer_stride,
            input_row_stride,
            out_row_stride,
            inv_hidden,
            static_cast<float>(eps),
            stream);
      }) AT_DISPATCH_CASE(torch::ScalarType::Float, [&] {
        launch_rmsnorm_gemma<float>(input.data_ptr<float>(),
                                    weight.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    rows,
                                    hidden,
                                    input_outer_dim,
                                    input_outer_stride,
                                    input_row_stride,
                                    out_row_stride,
                                    inv_hidden,
                                    static_cast<float>(eps),
                                    stream);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_add_gemma_rms_norm(torch::Tensor& input,
                              torch::Tensor& residual,
                              torch::Tensor& weight,
                              double epsilon) {
  CHECK(input.scalar_type() == residual.scalar_type());
  CHECK(input.scalar_type() == weight.scalar_type());
  CHECK(residual.is_contiguous());
  CHECK(weight.is_contiguous());
  CHECK(input.is_contiguous())
      << "fused_add_gemma_rms_norm requires a contiguous `input` (in-place "
         "write back). Got strides="
      << input.strides() << ", sizes=" << input.sizes();

  const int hidden = input.size(-1);
  const int64_t input_row_stride = input.stride(-2);
  const int64_t residual_row_stride = residual.stride(-2);
  const int rows = input.numel() / hidden;
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "fused_add_gemma_rms_norm",
      AT_DISPATCH_CASE(torch::ScalarType::Half, [&] {
        launch_fused_add_rmsnorm_gemma<__half>(
            reinterpret_cast<__half*>(input.data_ptr<c10::Half>()),
            reinterpret_cast<__half*>(residual.data_ptr<c10::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<c10::Half>()),
            rows,
            hidden,
            input_row_stride,
            residual_row_stride,
            inv_hidden,
            static_cast<float>(epsilon),
            stream);
      }) AT_DISPATCH_CASE(torch::ScalarType::BFloat16, [&] {
        launch_fused_add_rmsnorm_gemma<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr<c10::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                residual.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(
                weight.data_ptr<c10::BFloat16>()),
            rows,
            hidden,
            input_row_stride,
            residual_row_stride,
            inv_hidden,
            static_cast<float>(epsilon),
            stream);
      }) AT_DISPATCH_CASE(torch::ScalarType::Float, [&] {
        launch_fused_add_rmsnorm_gemma<float>(input.data_ptr<float>(),
                                              residual.data_ptr<float>(),
                                              weight.data_ptr<float>(),
                                              rows,
                                              hidden,
                                              input_row_stride,
                                              residual_row_stride,
                                              inv_hidden,
                                              static_cast<float>(epsilon),
                                              stream);
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
