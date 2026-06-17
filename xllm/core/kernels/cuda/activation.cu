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

#include "cuda_ops_api.h"
#include "device_utils.cuh"

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu

namespace {

using ::xllm::kernel::cuda::xllm_ldg;

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

// Check if pointer is 16-byte aligned for int4 vectorized access
__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

// Activation and gating kernel template with 128-bit vectorized access
// optimization.
template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void XLLM_KERNEL_ATTR(1024)
    act_and_mul_kernel(scalar_t* __restrict__ out,          // [..., d]
                       const scalar_t* __restrict__ input,  // [..., 2, d]
                       const int d) {
  constexpr int kVecSize = 16 / sizeof(scalar_t);
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  // Check alignment for 128-bit vectorized access.
  // All three pointers must be 16-byte aligned for safe int4 operations.
  const bool aligned = is_16byte_aligned(x_ptr) && is_16byte_aligned(y_ptr) &&
                       is_16byte_aligned(out_ptr);

  if (aligned && d >= kVecSize) {
    // Fast path: 128-bit vectorized loop
    const int4* x_vec = reinterpret_cast<const int4*>(x_ptr);
    const int4* y_vec = reinterpret_cast<const int4*>(y_ptr);
    int4* out_vec = reinterpret_cast<int4*>(out_ptr);
    const int num_vecs = d / kVecSize;
    const int vec_end = num_vecs * kVecSize;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 x = xllm_ldg(&x_vec[i]), y = xllm_ldg(&y_vec[i]), r;
      auto* xp = reinterpret_cast<scalar_t*>(&x);
      auto* yp = reinterpret_cast<scalar_t*>(&y);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int j = 0; j < kVecSize; j++) {
        rp[j] = compute<scalar_t, ACT_FN, act_first>(xp[j], yp[j]);
      }
      out_vec[i] = r;
    }
    // Scalar cleanup for remaining elements
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = compute<scalar_t, ACT_FN, act_first>(xllm_ldg(&x_ptr[i]),
                                                        xllm_ldg(&y_ptr[i]));
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = xllm_ldg(&x_ptr[idx]);
      const scalar_t y = xllm_ldg(&y_ptr[idx]);
      out_ptr[idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  const float f = static_cast<float>(x);
  return static_cast<T>(f / (1.0f + expf(-f)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = static_cast<float>(x);
  constexpr float kAlpha = M_SQRT1_2;
  return static_cast<T>(f * 0.5f * (1.0f + ::erf(f * kAlpha)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = static_cast<float>(x);
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float kKappa = 0.044715;
  float x_cube = f * f * f;
  float inner = kBeta * (f + kKappa * x_cube);
  return static_cast<T>(0.5f * f * (1.0f + ::tanhf(inner)));
}

#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                   \
  int d = input.size(-1) / 2;                                              \
  int64_t num_tokens = input.numel() / input.size(-1);                     \
  dim3 grid(num_tokens);                                                   \
  dim3 block(std::min(d, 1024));                                           \
  if (num_tokens == 0) {                                                   \
    return;                                                                \
  }                                                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));        \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();            \
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "act_and_mul_kernel", [&] { \
    act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>              \
        <<<grid, block, 0, stream>>>(                                      \
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);      \
  });

void silu_and_mul(torch::Tensor out,    // [..., d]
                  torch::Tensor input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(silu_kernel, true);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(gelu_tanh_kernel, true);
}
}  // namespace

namespace xllm::kernel::cuda {

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode) {
  if (act_mode != "silu" && act_mode != "gelu" && act_mode != "gelu_tanh") {
    LOG(FATAL) << "Unsupported act mode: " << act_mode
               << ", only support silu, gelu, gelu_tanh";
  }

  // flashinfer act_and_mul ops
  // std::string uri = act_mode + "_and_mul";
  // FunctionFactory::get_instance().act_and_mul(uri).call(
  //     out, input, support_pdl());

  if (act_mode == "silu") {
    silu_and_mul(out, input);
  } else if (act_mode == "gelu") {
    gelu_and_mul(out, input);
  } else if (act_mode == "gelu_tanh") {
    gelu_tanh_and_mul(out, input);
  }
}

}  // namespace xllm::kernel::cuda
