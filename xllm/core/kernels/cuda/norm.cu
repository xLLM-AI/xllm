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

#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#include <cstdint>
#include <cub/cub.cuh>

#include "cuda_ops_api.h"
#include "device_utils.cuh"
#include "fp8_quant_utils.cuh"
#include "type_convert.cuh"

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu

#if CUB_VERSION >= 200800
#include <cuda/std/functional>
using CubAddOp = ::cuda::std::plus<>;
using CubMaxOp = ::cuda::maximum<>;
#else   // if CUB_VERSION < 200800
using CubAddOp = cub::Sum;
using CubMaxOp = cub::Max;
#endif  // CUB_VERSION

namespace {

using namespace xllm::kernel::cuda;

template <typename scalar_t>
__global__ void XLLM_KERNEL_ATTR(1024)
    rms_norm_kernel(scalar_t* __restrict__ out,          // [..., hidden_size]
                    const scalar_t* __restrict__ input,  // [..., hidden_size]
                    const int64_t input_stride,
                    const scalar_t* __restrict__ weight,  // [hidden_size]
                    const float epsilon,
                    const int num_tokens,
                    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = static_cast<float>(input[blockIdx.x * input_stride + idx]);
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input[blockIdx.x * input_stride + idx]);
    out[blockIdx.x * hidden_size + idx] =
        (static_cast<scalar_t>(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
XLLM_KERNEL_ATTR(1024) fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
XLLM_KERNEL_ATTR(1024) fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = static_cast<float>(z);
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(residual[blockIdx.x * hidden_size + idx]);
    input[blockIdx.x * input_stride + idx] =
        (static_cast<scalar_t>(x * s_variance)) * weight[idx];
  }
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                \
  DISPATCH_FLOATING_TYPES(                                              \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {           \
        fused_add_rms_norm_kernel<scalar_t, width>                      \
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),    \
                                         input_stride,                  \
                                         residual.data_ptr<scalar_t>(), \
                                         weight.data_ptr<scalar_t>(),   \
                                         epsilon,                       \
                                         num_tokens,                    \
                                         hidden_size);                  \
      });

// ============================================================================
// Fused RMSNorm + Static FP8 Quantization Kernels
// ============================================================================
// These kernels combine RMSNorm and FP8 quantization to reduce memory
// bandwidth by avoiding the intermediate write-back to global memory.

// Dispatch macro for FP8 types
#define DISPATCH_FP8_TYPES(TYPE, NAME, ...)         \
  [&] {                                             \
    const auto& the_type = TYPE;                    \
    switch (the_type) {                             \
      case at::ScalarType::Float8_e4m3fn: {         \
        using fp8_t = c10::Float8_e4m3fn;           \
        return __VA_ARGS__();                       \
      }                                             \
      default:                                      \
        AT_ERROR(#NAME,                             \
                 " not implemented for FP8 type '", \
                 toString(the_type),                \
                 "'");                              \
    }                                               \
  }()

/**
 * Fused RMSNorm + Static FP8 Quantization kernel (without residual)
 * Combines RMSNorm and FP8 quantization in a single kernel to reduce
 * memory bandwidth by avoiding intermediate write-back.
 *
 * @tparam scalar_t Input data type (float, half, bfloat16)
 * @tparam fp8_type Output FP8 type (c10::Float8_e4m3fn)
 * @param out Output FP8 tensor [num_tokens, hidden_size]
 * @param input Input tensor [num_tokens, hidden_size]
 * @param input_stride Stride of input tensor in the token dimension
 * @param weight RMSNorm weight tensor [hidden_size]
 * @param scale FP8 quantization scale (scalar)
 * @param epsilon RMSNorm epsilon
 * @param num_tokens Number of tokens
 * @param hidden_size Hidden dimension size
 */
template <typename scalar_t, typename fp8_type>
__global__ void rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,          // [num_tokens, hidden_size]
    const scalar_t* __restrict__ input,  // [num_tokens, hidden_size]
    const int64_t input_stride,
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float* __restrict__ scale,      // [1]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const scalar_t* input_row = input + blockIdx.x * input_stride;

  // Step 1: Compute variance for RMSNorm
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = static_cast<float>(input_row[idx]);
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Step 2: Precompute scale inverse to avoid division
  const float scale_inv = 1.0f / (*scale);

  // Step 3: Fused RMSNorm + FP8 quantization
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(input_row[idx]);
    float out_norm = (static_cast<scalar_t>(x * s_variance)) *
                     static_cast<float>(weight[idx]);
    out[blockIdx.x * hidden_size + idx] =
        xllm::kernel::cuda::scaled_fp8_conversion<true, fp8_type>(out_norm,
                                                                  scale_inv);
  }
}

/**
 * Fused Add + RMSNorm + Static FP8 Quantization kernel (with residual)
 * Optimized version with packed + vectorized operations for FP16/BF16.
 *
 * @tparam scalar_t Input data type (float, half, bfloat16)
 * @tparam width Vector width for optimization (0, 8)
 * @tparam fp8_type Output FP8 type (c10::Float8_e4m3fn)
 */
template <typename scalar_t, int width, typename fp8_type>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,    // [num_tokens, hidden_size]
    scalar_t* __restrict__ input,  // [num_tokens, hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [num_tokens, hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float* __restrict__ scale,      // [1]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;

  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  // Step 1: Fused add and compute variance
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;  // Store updated residual
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Step 2: Precompute scale inverse
  const float scale_inv = 1.0f / (*scale);

  // Step 3: Fused RMSNorm + FP8 quantization
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];

    // Convert each element to FP8
#pragma unroll
    for (int i = 0; i < width; ++i) {
      float val = _typeConvert<scalar_t>::convert(temp.data[i]);
      out[id * width + i] =
          xllm::kernel::cuda::scaled_fp8_conversion<true, fp8_type>(val,
                                                                    scale_inv);
    }
  }
}

/**
 * Generic fused add + RMSNorm + FP8 quant kernel (fallback for unaligned data)
 */
template <typename scalar_t, int width, typename fp8_type>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,    // [num_tokens, hidden_size]
    scalar_t* __restrict__ input,  // [num_tokens, hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [num_tokens, hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float* __restrict__ scale,      // [1]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  // Step 1: Fused add and compute variance
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = static_cast<float>(z);
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;  // Store updated residual
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Step 2: Precompute scale inverse
  const float scale_inv = 1.0f / (*scale);

  // Step 3: Fused RMSNorm + FP8 quantization
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = static_cast<float>(residual[blockIdx.x * hidden_size + idx]);
    float out_norm = (static_cast<scalar_t>(x * s_variance)) *
                     static_cast<float>(weight[idx]);
    out[blockIdx.x * hidden_size + idx] =
        xllm::kernel::cuda::scaled_fp8_conversion<true, fp8_type>(out_norm,
                                                                  scale_inv);
  }
}

#define LAUNCH_FUSED_ADD_RMS_NORM_STATIC_FP8_QUANT(width)                     \
  DISPATCH_FLOATING_TYPES(                                                    \
      input.scalar_type(), "fused_add_rms_norm_static_fp8_quant", [&] {       \
        DISPATCH_FP8_TYPES(                                                   \
            out.scalar_type(), "fused_add_rms_norm_static_fp8_quant", [&] {   \
              fused_add_rms_norm_static_fp8_quant_kernel<scalar_t,            \
                                                         width,               \
                                                         fp8_t>               \
                  <<<grid, block, 0, stream>>>(out.data_ptr<fp8_t>(),         \
                                               input.data_ptr<scalar_t>(),    \
                                               input_stride,                  \
                                               residual.data_ptr<scalar_t>(), \
                                               weight.data_ptr<scalar_t>(),   \
                                               scale.data_ptr<float>(),       \
                                               epsilon,                       \
                                               num_tokens,                    \
                                               hidden_size);                  \
            });                                                               \
      });

}  // namespace

namespace xllm::kernel::cuda {

// flashinfer rmsnorm ops
// void rmsnorm(torch::Tensor output,
//              torch::Tensor input,
//              torch::Tensor weight,
//              double eps) {
//   FunctionFactory::get_instance().rmsnorm_func("norm").call(
//       output, input, weight, eps, support_pdl());
// }

void rms_norm(torch::Tensor output,  // [..., hidden_size]
              torch::Tensor input,   // [..., hidden_size]
              torch::Tensor weight,  // [hidden_size]
              double eps) {
  CHECK(output.is_contiguous());
  CHECK(input.stride(-1) == 1);
  CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int64_t input_stride = input.stride(-2);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    rms_norm_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(),
                                     input.data_ptr<scalar_t>(),
                                     input_stride,
                                     weight.data_ptr<scalar_t>(),
                                     eps,
                                     num_tokens,
                                     hidden_size);
  });
}

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  CHECK(weight.scalar_type() == input.scalar_type());
  CHECK(input.scalar_type() == residual.scalar_type());
  CHECK(residual.is_contiguous());
  CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  constexpr int kVectorWidth = 8;
  constexpr int kReqAlignmentBytes =
      kVectorWidth * 2;  // kVectorWidth * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % kReqAlignmentBytes == 0 &&
                          res_ptr % kReqAlignmentBytes == 0 &&
                          wt_ptr % kReqAlignmentBytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % kVectorWidth == 0 && input_stride % kVectorWidth == 0;
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

// ============================================================================
// Fused RMSNorm + Static FP8 Quantization Host Functions
// ============================================================================

void rms_norm_static_fp8_quant(torch::Tensor& out,    // [..., hidden_size], FP8
                               torch::Tensor& input,  // [..., hidden_size]
                               torch::Tensor& weight,  // [hidden_size]
                               torch::Tensor& scale,   // [1]
                               double epsilon) {
  CHECK(out.is_contiguous());
  CHECK(input.stride(-1) == 1);
  CHECK(weight.is_contiguous());
  CHECK(scale.is_contiguous());

  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  // For large num_tokens, use smaller blocks to increase SM concurrency
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, max_block_size));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_static_fp8_quant", [&] {
        DISPATCH_FP8_TYPES(out.scalar_type(), "rms_norm_static_fp8_quant", [&] {
          rms_norm_static_fp8_quant_kernel<scalar_t, fp8_t>
              <<<grid, block, 0, stream>>>(out.data_ptr<fp8_t>(),
                                           input.data_ptr<scalar_t>(),
                                           input_stride,
                                           weight.data_ptr<scalar_t>(),
                                           scale.data_ptr<float>(),
                                           epsilon,
                                           num_tokens,
                                           hidden_size);
        });
      });
}

void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,       // [..., hidden_size], FP8
    torch::Tensor& input,     // [..., hidden_size]
    torch::Tensor& residual,  // [..., hidden_size]
    torch::Tensor& weight,    // [hidden_size]
    torch::Tensor& scale,     // [1]
    double epsilon) {
  CHECK(out.is_contiguous());
  CHECK(residual.is_contiguous());
  CHECK(weight.is_contiguous());
  CHECK(scale.is_contiguous());
  CHECK(residual.scalar_type() == input.scalar_type());
  CHECK(weight.scalar_type() == input.scalar_type());

  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Check alignment for vectorized kernel
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  constexpr int kVectorWidth = 8;
  constexpr int kReqAlignmentBytes = kVectorWidth * 2;

  bool ptrs_are_aligned = inp_ptr % kReqAlignmentBytes == 0 &&
                          res_ptr % kReqAlignmentBytes == 0 &&
                          wt_ptr % kReqAlignmentBytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % kVectorWidth == 0 && input_stride % kVectorWidth == 0;

  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width) {
    LAUNCH_FUSED_ADD_RMS_NORM_STATIC_FP8_QUANT(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM_STATIC_FP8_QUANT(0);
  }
}

}  // namespace xllm::kernel::cuda
