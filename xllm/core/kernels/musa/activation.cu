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
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#include <cstdint>

#include "core/kernels/cuda/device_utils.cuh"
#include "core/kernels/musa/musa_ops_api.h"

namespace {

using ::xllm::kernel::cuda::xllm_ldg;

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void XLLM_KERNEL_ATTR(1024)
    act_and_mul_kernel(scalar_t* __restrict__ out,
                       const scalar_t* __restrict__ input,
                       const int32_t d) {
  constexpr int32_t kVecSize = 16 / sizeof(scalar_t);
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  const bool aligned = is_16byte_aligned(x_ptr) && is_16byte_aligned(y_ptr) &&
                       is_16byte_aligned(out_ptr);

  if (aligned && d >= kVecSize) {
    const int4* x_vec = reinterpret_cast<const int4*>(x_ptr);
    const int4* y_vec = reinterpret_cast<const int4*>(y_ptr);
    int4* out_vec = reinterpret_cast<int4*>(out_ptr);
    const int32_t num_vecs = d / kVecSize;
    const int32_t vec_end = num_vecs * kVecSize;

    for (int32_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 x = xllm_ldg(&x_vec[i]), y = xllm_ldg(&y_vec[i]), r;
      auto* xp = reinterpret_cast<scalar_t*>(&x);
      auto* yp = reinterpret_cast<scalar_t*>(&y);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int32_t j = 0; j < kVecSize; j++) {
        rp[j] = compute<scalar_t, ACT_FN, act_first>(xp[j], yp[j]);
      }
      out_vec[i] = r;
    }
    for (int32_t i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = compute<scalar_t, ACT_FN, act_first>(xllm_ldg(&x_ptr[i]),
                                                        xllm_ldg(&y_ptr[i]));
    }
  } else {
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = xllm_ldg(&x_ptr[idx]);
      const scalar_t y = xllm_ldg(&y_ptr[idx]);
      out_ptr[idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  const float f = static_cast<float>(x);
  return static_cast<T>(f / (1.0f + expf(-f)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  const float f = static_cast<float>(x);
  constexpr float kAlpha = M_SQRT1_2;
  return static_cast<T>(f * 0.5f * (1.0f + ::erf(f * kAlpha)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  const float f = static_cast<float>(x);
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float kKappa = 0.044715;
  float x_cube = f * f * f;
  float inner = kBeta * (f + kKappa * x_cube);
  return static_cast<T>(0.5f * f * (1.0f + ::tanhf(inner)));
}

#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                   \
  int32_t d = static_cast<int32_t>(input.size(-1) / 2);                    \
  int64_t num_tokens = input.numel() / input.size(-1);                     \
  dim3 grid(num_tokens);                                                   \
  dim3 block(std::min<int32_t>(d, 1024));                                  \
  if (num_tokens == 0) {                                                   \
    return;                                                                \
  }                                                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));        \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();            \
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "act_and_mul_kernel", [&] { \
    act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>              \
        <<<grid, block, 0, stream>>>(                                      \
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);      \
  });                                                                      \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

void silu_and_mul(torch::Tensor out, torch::Tensor input) {
  LAUNCH_ACTIVATION_GATE_KERNEL(silu_kernel, true);
}

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input) {
  LAUNCH_ACTIVATION_GATE_KERNEL(gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input) {
  LAUNCH_ACTIVATION_GATE_KERNEL(gelu_tanh_kernel, true);
}

template <typename scalar_t>
__global__ void XLLM_KERNEL_ATTR(1024)
    mul_sigmoid_gate_strided_2d_kernel(scalar_t* __restrict__ out,
                                       const scalar_t* __restrict__ gate,
                                       const int64_t n,
                                       const int64_t out_row_stride,
                                       const int64_t gate_row_stride) {
  const int64_t row = blockIdx.x;
  scalar_t* out_row = out + row * out_row_stride;
  const scalar_t* gate_row = gate + row * gate_row_stride;
  for (int64_t col = threadIdx.x; col < n; col += blockDim.x) {
    const float g = static_cast<float>(xllm_ldg(&gate_row[col]));
    const float s = 1.0f / (1.0f + expf(-g));
    out_row[col] = static_cast<scalar_t>(static_cast<float>(out_row[col]) * s);
  }
}

void launch_mul_sigmoid_gate_inplace(torch::Tensor& out,
                                     const torch::Tensor& gate) {
  const int64_t n = out.numel();
  if (n == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t last_dim = out.size(-1);
  const int64_t M = n / last_dim;
  const int64_t out_row_stride = (out.dim() <= 1) ? last_dim : out.stride(-2);
  const int64_t gate_row_stride =
      (gate.dim() <= 1) ? last_dim : gate.stride(-2);

  const int32_t threads =
      static_cast<int32_t>(std::min<int64_t>(last_dim, 1024));
  dim3 grid(static_cast<unsigned int>(M));
  dim3 block(static_cast<unsigned int>(threads));
  DISPATCH_FLOATING_TYPES(out.scalar_type(), "mul_sigmoid_gate_inplace", [&] {
    mul_sigmoid_gate_strided_2d_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),
                                     gate.data_ptr<scalar_t>(),
                                     last_dim,
                                     out_row_stride,
                                     gate_row_stride);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

namespace xllm::kernel::cuda {

void mul_sigmoid_gate_inplace(torch::Tensor& out, const torch::Tensor& gate) {
  CHECK(out.defined() && gate.defined()) << "out and gate must be defined";
  CHECK(out.sizes() == gate.sizes()) << "out and gate must have same shape";
  CHECK(out.scalar_type() == gate.scalar_type()) << "dtype mismatch";
  CHECK(out.device() == gate.device()) << "device mismatch";
  CHECK(out.dim() >= 1) << "out must be at least 1D";
  CHECK(out.stride(-1) == 1 && gate.stride(-1) == 1)
      << "out and gate must have last-dim stride == 1";
  if (out.dim() > 2) {
    const int64_t numel_per_row = out.size(-1);
    CHECK(out.numel() % numel_per_row == 0)
        << "out shape not collapsible to 2D";
    CHECK(gate.numel() % numel_per_row == 0)
        << "gate shape not collapsible to 2D";
  }
  launch_mul_sigmoid_gate_inplace(out, gate);
}

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode) {
  if (act_mode != "silu" && act_mode != "gelu" && act_mode != "gelu_tanh" &&
      act_mode != "gelu_pytorch_tanh") {
    LOG(FATAL) << "Unsupported act mode: " << act_mode
               << ", only support silu, gelu, gelu_tanh, gelu_pytorch_tanh";
  }

  if (act_mode == "silu") {
    silu_and_mul(out, input);
  } else if (act_mode == "gelu") {
    gelu_and_mul(out, input);
  } else if (act_mode == "gelu_tanh" || act_mode == "gelu_pytorch_tanh") {
    gelu_tanh_and_mul(out, input);
  }
}

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias,
                     std::optional<torch::Tensor> output_buf) {
  if (output_buf.has_value() && output_buf->defined() && a.dim() == 2 &&
      b.dim() == 2) {
    auto& out = *output_buf;
    auto bt = b.t();
    if (bias.has_value() && bias->defined()) {
      torch::addmm_out(out, *bias, a, bt);
    } else {
      torch::mm_out(out, a, bt);
    }
    return out;
  }
  namespace F = torch::nn::functional;
  return F::linear(a, b, bias.value_or(torch::Tensor()));
}

}  // namespace xllm::kernel::cuda
