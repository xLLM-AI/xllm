/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include <cstdint>
#include <type_traits>

#include "kernels/cuda/utils.h"
#include "xattention_ops_api.h"

namespace {

template <typename scalar_t>
struct VecType;

template <>
struct VecType<c10::Half> {
  using type = uint4;  // 8 elements * 2 bytes = 16 bytes
  static constexpr int32_t vec_width = 8;
};

template <>
struct VecType<c10::BFloat16> {
  using type = uint4;  // 8 elements * 2 bytes = 16 bytes
  static constexpr int32_t vec_width = 8;
};

template <>
struct VecType<float> {
  using type = float4;  // 4 elements * 4 bytes = 16 bytes
  static constexpr int32_t vec_width = 4;
};

// decoder reshape and cache kernel.
// Copies proj_k and proj_v into unshared_k_cache / unshared_v_cache.
// Inputs:
//   proj_k           : [batch_size, beam_size, kv_heads, head_dim]
//   proj_v           : [batch_size, beam_size, kv_heads, head_dim]
//   step             : [1] - current decode step
//   batch_size       : batch size
//   beam_size        : beam size
//   kv_heads         : number of kv heads
//   head_dim         : head dimension
//   k_stride0        : proj_k.stride(0)
//   k_stride1        : proj_k.stride(1)
//   v_stride0        : proj_v.stride(0)
//   v_stride1        : proj_v.stride(1)
//   cache_stride0    : unshared_k_cache.stride(0)
//   cache_stride1    : unshared_k_cache.stride(1)
//   cache_stride2    : unshared_k_cache.stride(2)
//   cache_stride3    : unshared_k_cache.stride(3)
// Outputs:
//   unshared_k_cache : [max_batch_size, beam_size, max_step, kv_heads,
//   head_dim]
//   unshared_v_cache : [max_batch_size, beam_size, max_step, kv_heads,
//   head_dim]

template <typename scalar_t>
__global__ void decoder_reshape_and_cache_kernel(
    const scalar_t* __restrict__ proj_k,
    const scalar_t* __restrict__ proj_v,
    scalar_t* __restrict__ unshared_k_cache,
    scalar_t* __restrict__ unshared_v_cache,
    const int32_t* __restrict__ step,
    const int64_t batch_size,
    const int64_t beam_size,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int64_t k_stride0,
    const int64_t k_stride1,
    const int64_t v_stride0,
    const int64_t v_stride1,
    const int64_t cache_stride0,
    const int64_t cache_stride1,
    const int64_t cache_stride2,
    const int64_t cache_stride3) {
  using VecTypeT = typename VecType<scalar_t>::type;
  constexpr int32_t VEC_WIDTH = VecType<scalar_t>::vec_width;

  const int64_t token_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t total_tokens = batch_size * beam_size;
  if (token_idx >= total_tokens) {
    return;
  }

  const int64_t batch_idx = token_idx / beam_size;
  const int64_t beam_idx = token_idx - batch_idx * beam_size;

  __shared__ int32_t current_step_s;
  if (threadIdx.x == 0) {
    current_step_s = __ldg(step);
  }
  __syncthreads();
  const int64_t current_step = static_cast<int64_t>(current_step_s);

  const int64_t vecs_per_head = head_dim / VEC_WIDTH;
  const int64_t total_vecs = kv_heads * vecs_per_head;

  const int64_t k_token_base = batch_idx * k_stride0 + beam_idx * k_stride1;
  const int64_t v_token_base = batch_idx * v_stride0 + beam_idx * v_stride1;
  const int64_t dst_token_base = batch_idx * cache_stride0 +
                                 beam_idx * cache_stride1 +
                                 current_step * cache_stride2;

  for (int64_t linear_idx = static_cast<int64_t>(threadIdx.x);
       linear_idx < total_vecs;
       linear_idx += static_cast<int64_t>(blockDim.x)) {
    const int64_t head_idx = linear_idx / vecs_per_head;
    const int64_t vec_idx = linear_idx - head_idx * vecs_per_head;
    const int64_t vec_offset = vec_idx * VEC_WIDTH;

    const auto* k_src_vec = reinterpret_cast<const VecTypeT*>(
        proj_k + k_token_base + head_idx * head_dim + vec_offset);
    const auto* v_src_vec = reinterpret_cast<const VecTypeT*>(
        proj_v + v_token_base + head_idx * head_dim + vec_offset);
    auto* k_dst_vec =
        reinterpret_cast<VecTypeT*>(unshared_k_cache + dst_token_base +
                                    head_idx * cache_stride3 + vec_offset);
    auto* v_dst_vec =
        reinterpret_cast<VecTypeT*>(unshared_v_cache + dst_token_base +
                                    head_idx * cache_stride3 + vec_offset);

    *k_dst_vec = *k_src_vec;
    *v_dst_vec = *v_src_vec;
  }
}

}  // namespace

namespace xllm::kernel::cuda {

void decoder_reshape_and_cache(torch::Tensor proj_k,
                               torch::Tensor proj_v,
                               torch::Tensor unshared_k_cache,
                               torch::Tensor unshared_v_cache,
                               torch::Tensor step) {
  CHECK_EQ(proj_k.dim(), 4) << "proj_k must be 4-dimensional";
  CHECK_EQ(proj_v.dim(), 4) << "proj_v must be 4-dimensional";
  CHECK_EQ(unshared_k_cache.dim(), 5)
      << "unshared_k_cache must be 5-dimensional";
  CHECK_EQ(unshared_v_cache.dim(), 5)
      << "unshared_v_cache must be 5-dimensional";
  CHECK(proj_k.is_cuda() && proj_v.is_cuda() && unshared_k_cache.is_cuda() &&
        unshared_v_cache.is_cuda() && step.is_cuda())
      << "all tensors must be CUDA tensors";
  CHECK_EQ(step.dim(), 1) << "step must be 1-dimensional";
  CHECK_EQ(step.size(0), 1) << "step must have shape [1]";
  CHECK_EQ(step.scalar_type(), at::ScalarType::Int)
      << "step must be int32 (torch::kInt32)";

  const int64_t batch_size = proj_k.size(0);
  const int64_t beam_size = proj_k.size(1);
  const int64_t kv_heads = proj_k.size(2);
  const int64_t head_dim = proj_k.size(3);

  CHECK_EQ(proj_v.sizes(), proj_k.sizes())
      << "proj_v and proj_k must have same shape";
  CHECK_EQ(unshared_k_cache.size(3), kv_heads)
      << "unshared_k_cache kv_heads mismatch";
  CHECK_EQ(unshared_k_cache.size(4), head_dim)
      << "unshared_k_cache head_dim mismatch";
  CHECK(unshared_v_cache.sizes() == unshared_k_cache.sizes())
      << "unshared_v_cache and unshared_k_cache must have same shape";

  // This kernel is specialized for qkv-slice layouts:
  // last dim contiguous and kv head stride tightly packed by head_dim.
  CHECK_EQ(proj_k.stride(3), 1) << "proj_k must satisfy stride(3)=1";
  CHECK_EQ(proj_v.stride(3), 1) << "proj_v must satisfy stride(3)=1";
  CHECK_EQ(proj_k.stride(2), head_dim)
      << "proj_k must satisfy stride(2)=head_dim";
  CHECK_EQ(proj_v.stride(2), head_dim)
      << "proj_v must satisfy stride(2)=head_dim";
  CHECK_EQ(unshared_k_cache.stride(4), 1)
      << "unshared_k_cache must satisfy stride(4)=1";
  CHECK_EQ(unshared_v_cache.stride(4), 1)
      << "unshared_v_cache must satisfy stride(4)=1";
  CHECK_EQ(unshared_k_cache.stride(3), head_dim)
      << "unshared_k_cache must satisfy stride(3)=head_dim";
  CHECK_EQ(unshared_v_cache.stride(3), head_dim)
      << "unshared_v_cache must satisfy stride(3)=head_dim";

  const at::cuda::OptionalCUDAGuard device_guard(device_of(proj_k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t k_stride0 = proj_k.stride(0);
  const int64_t k_stride1 = proj_k.stride(1);
  const int64_t v_stride0 = proj_v.stride(0);
  const int64_t v_stride1 = proj_v.stride(1);
  const int64_t cache_stride0 = unshared_k_cache.stride(0);
  const int64_t cache_stride1 = unshared_k_cache.stride(1);
  const int64_t cache_stride2 = unshared_k_cache.stride(2);
  const int64_t cache_stride3 = unshared_k_cache.stride(3);

  // Launch kernel: one block per (batch, beam), threads cover
  // kv_heads*head_dim.
  const int64_t total_tokens = batch_size * beam_size;
  dim3 grid_dim(1, static_cast<unsigned int>(total_tokens), 1);

  DISPATCH_FLOATING_TYPES(
      proj_k.scalar_type(), "decoder_reshape_and_cache_kernel", [&] {
        constexpr int32_t VEC_WIDTH = (std::is_same_v<scalar_t, c10::Half> ||
                                       std::is_same_v<scalar_t, c10::BFloat16>)
                                          ? 8
                                          : 4;  // FP16/BF16: 8, Float: 4
        constexpr int32_t kWarpSize = 32;
        constexpr int32_t kMaxThreadsPerBlock = 256;
        constexpr int32_t kAlignmentBytes = 16;  // 128-bit alignment

        CHECK(head_dim % VEC_WIDTH == 0)
            << "head_dim must be divisible by vector width: " << VEC_WIDTH;
        const int64_t vecs_per_head = head_dim / VEC_WIDTH;
        const int64_t total_vecs = kv_heads * vecs_per_head;
        CHECK(total_vecs > 0) << "total_vecs must be > 0";

        int32_t threads_per_block = static_cast<int32_t>(
            total_vecs > kMaxThreadsPerBlock ? kMaxThreadsPerBlock
                                             : total_vecs);
        threads_per_block =
            ((threads_per_block + kWarpSize - 1) / kWarpSize) * kWarpSize;
        if (threads_per_block < kWarpSize) {
          threads_per_block = kWarpSize;
        }
        dim3 block_dim(threads_per_block, 1, 1);

        const auto proj_k_ptr =
            reinterpret_cast<std::uintptr_t>(proj_k.data_ptr<scalar_t>());
        const auto proj_v_ptr =
            reinterpret_cast<std::uintptr_t>(proj_v.data_ptr<scalar_t>());
        const auto k_cache_ptr = reinterpret_cast<std::uintptr_t>(
            unshared_k_cache.data_ptr<scalar_t>());
        const auto v_cache_ptr = reinterpret_cast<std::uintptr_t>(
            unshared_v_cache.data_ptr<scalar_t>());
        CHECK(proj_k_ptr % kAlignmentBytes == 0)
            << "proj_k data_ptr must be 16-byte aligned";
        CHECK(proj_v_ptr % kAlignmentBytes == 0)
            << "proj_v data_ptr must be 16-byte aligned";
        CHECK(k_cache_ptr % kAlignmentBytes == 0)
            << "unshared_k_cache data_ptr must be 16-byte aligned";
        CHECK(v_cache_ptr % kAlignmentBytes == 0)
            << "unshared_v_cache data_ptr must be 16-byte aligned";

        const int64_t scalar_bytes = static_cast<int64_t>(sizeof(scalar_t));
        CHECK((k_stride0 * scalar_bytes) % kAlignmentBytes == 0)
            << "proj_k stride(0) bytes must be 16-byte aligned";
        CHECK((k_stride1 * scalar_bytes) % kAlignmentBytes == 0)
            << "proj_k stride(1) bytes must be 16-byte aligned";
        CHECK((v_stride0 * scalar_bytes) % kAlignmentBytes == 0)
            << "proj_v stride(0) bytes must be 16-byte aligned";
        CHECK((v_stride1 * scalar_bytes) % kAlignmentBytes == 0)
            << "proj_v stride(1) bytes must be 16-byte aligned";
        CHECK((cache_stride0 * scalar_bytes) % kAlignmentBytes == 0)
            << "cache stride(0) bytes must be 16-byte aligned";
        CHECK((cache_stride1 * scalar_bytes) % kAlignmentBytes == 0)
            << "cache stride(1) bytes must be 16-byte aligned";
        CHECK((cache_stride2 * scalar_bytes) % kAlignmentBytes == 0)
            << "cache stride(2) bytes must be 16-byte aligned";
        CHECK((cache_stride3 * scalar_bytes) % kAlignmentBytes == 0)
            << "cache stride(3) bytes must be 16-byte aligned";

        decoder_reshape_and_cache_kernel<scalar_t>
            <<<grid_dim, block_dim, 0, stream>>>(
                proj_k.data_ptr<scalar_t>(),
                proj_v.data_ptr<scalar_t>(),
                unshared_k_cache.data_ptr<scalar_t>(),
                unshared_v_cache.data_ptr<scalar_t>(),
                step.data_ptr<int32_t>(),
                batch_size,
                beam_size,
                kv_heads,
                head_dim,
                k_stride0,
                k_stride1,
                v_stride0,
                v_stride1,
                cache_stride0,
                cache_stride1,
                cache_stride2,
                cache_stride3);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
