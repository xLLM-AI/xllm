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

// clang-format off
#include "fp8_quant_utils.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>
// clang-format on

namespace xllm {
namespace kernel {
namespace cuda {

// Dispatch macro for floating types
#define XLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                       \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    switch (the_type) {                                                     \
      case at::ScalarType::Float: {                                         \
        using scalar_t = float;                                             \
        return __VA_ARGS__();                                               \
      }                                                                     \
      case at::ScalarType::Half: {                                          \
        using scalar_t = at::Half;                                          \
        return __VA_ARGS__();                                               \
      }                                                                     \
      case at::ScalarType::BFloat16: {                                      \
        using scalar_t = at::BFloat16;                                      \
        return __VA_ARGS__();                                               \
      }                                                                     \
      default:                                                              \
        AT_ERROR(                                                           \
            #NAME, " not implemented for type '", toString(the_type), "'"); \
    }                                                                       \
  }()

// Dispatch macro for FP8 types
#define XLLM_DISPATCH_FP8_TYPES(TYPE, NAME, ...)    \
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
 * Static scaled FP8 quantization kernel
 * Each block handles one token, quantizing all hidden dimensions
 *
 * @param out Output FP8 tensor [num_tokens, hidden_size]
 * @param input Input tensor [num_tokens, hidden_size]
 * @param scale Quantization scale (scalar, inverted)
 * @param hidden_size Hidden dimension size
 * @param in_row_stride Input row stride
 * @param out_row_stride Output row stride
 */
template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel_strided(
    fp8_type* __restrict__ out,
    const scalar_t* __restrict__ input,
    const float* __restrict__ scale,
    int hidden_size,
    int64_t in_row_stride,
    int64_t out_row_stride) {
  const int64_t token_idx = blockIdx.x;  // one token per block
  const int tid = threadIdx.x;

  const scalar_t* token_in = input + token_idx * in_row_stride;
  fp8_type* token_out = out + token_idx * out_row_stride;

  const float inv_scale = 1.0f / (*scale);

  // Vectorized conversion with 16-byte alignment
  vectorize_with_alignment<16>(
      token_in,
      token_out,
      hidden_size,
      tid,
      blockDim.x,
      [=] __device__(fp8_type & dst, const scalar_t& src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                    inv_scale);
      });
}

/**
 * Static scaled FP8 quantization (entry function)
 * Quantizes input tensor to FP8 using a pre-computed scale factor
 *
 * @param out Output FP8 tensor [..., hidden_size]
 * @param input Input tensor [..., hidden_size]
 * @param scale Scale tensor [1] - pre-computed scaling factor
 */
void static_scaled_fp8_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale)  // [1]
{
  TORCH_CHECK(input.stride(-1) == 1,
              "last dimension of input must be contiguous");
  TORCH_CHECK(out.stride(-1) == 1,
              "last dimension of output must be contiguous");

  const int hidden_size = input.size(-1);
  const int num_tokens = input.numel() / hidden_size;
  const int block_size = 256;
  dim3 grid(num_tokens);
  dim3 block(block_size);

  const int64_t in_row_stride = input.stride(-2);
  const int64_t out_row_stride = out.stride(-2);

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  XLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        XLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
              scaled_fp8_quant_kernel_strided<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(out.data_ptr<fp8_t>(),
                                               input.data_ptr<scalar_t>(),
                                               scale.data_ptr<float>(),
                                               hidden_size,
                                               in_row_stride,
                                               out_row_stride);
            });
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm
