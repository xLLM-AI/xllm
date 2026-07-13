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

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <glog/logging.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace xllm {
namespace kernel {
namespace dcu {
namespace {

template <typename StorageT>
__global__ void gather_mla_latent_cache_kernel(
    const StorageT* __restrict__ k_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ kv_cu_seq_lens,
    int64_t block_table_stride,
    int64_t chunks_per_sequence,
    int64_t block_size,
    int64_t latent_dim,
    StorageT* __restrict__ output) {
  const int64_t global_chunk = static_cast<int64_t>(blockIdx.x);
  const int64_t seq_idx = global_chunk / chunks_per_sequence;
  const int64_t chunk_idx = global_chunk - seq_idx * chunks_per_sequence;
  const int64_t seq_start = static_cast<int64_t>(kv_cu_seq_lens[seq_idx]);
  const int64_t seq_end = static_cast<int64_t>(kv_cu_seq_lens[seq_idx + 1]);
  const int64_t seq_len = seq_end - seq_start;
  if (seq_len <= 0) {
    return;
  }

  const int64_t linear_idx =
      chunk_idx * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  const int64_t seq_numel = seq_len * latent_dim;
  if (linear_idx >= seq_numel) {
    return;
  }

  const int64_t token_idx = linear_idx / latent_dim;
  const int64_t dim_idx = linear_idx - token_idx * latent_dim;
  const int64_t logical_block_idx = token_idx / block_size;
  const int64_t token_offset = token_idx - logical_block_idx * block_size;
  const int32_t physical_block =
      block_table[seq_idx * block_table_stride + logical_block_idx];
  if (physical_block < 0) {
    return;
  }

  const int64_t src_row =
      static_cast<int64_t>(physical_block) * block_size + token_offset;
  const int64_t dst_row = seq_start + token_idx;
  output[dst_row * latent_dim + dim_idx] =
      k_cache[src_row * latent_dim + dim_idx];
}

template <typename StorageT>
void launch_gather_mla_latent_cache_kernel(
    const torch::Tensor& k_cache,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_cu_seq_lens,
    torch::Tensor& output,
    int64_t max_seq_len) {
  const int64_t batch_size = block_table.size(0);
  const int64_t block_size = k_cache.size(1);
  const int64_t latent_dim = k_cache.size(-1);
  const int64_t block_table_stride = block_table.size(1);
  constexpr int32_t kThreads = 256;
  const int64_t max_seq_numel = std::max<int64_t>(max_seq_len * latent_dim, 1);
  const int64_t chunks_per_sequence =
      (max_seq_numel + kThreads - 1) / kThreads;
  CHECK_GT(chunks_per_sequence, 0)
      << "gather_mla_latent_cache chunks_per_sequence must be positive.";
  CHECK_LE(batch_size * chunks_per_sequence, INT32_MAX)
      << "gather_mla_latent_cache launch grid is too large.";

  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
      k_cache.device());
  const hipStream_t stream =
      c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(k_cache.get_device());
  hipLaunchKernelGGL(
      (gather_mla_latent_cache_kernel<StorageT>),
      dim3(static_cast<uint32_t>(batch_size * chunks_per_sequence)),
      dim3(kThreads),
      0,
      stream,
      reinterpret_cast<const StorageT*>(k_cache.data_ptr()),
      block_table.data_ptr<int32_t>(),
      kv_cu_seq_lens.data_ptr<int32_t>(),
      block_table_stride,
      chunks_per_sequence,
      block_size,
      latent_dim,
      reinterpret_cast<StorageT*>(output.data_ptr()));
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor gather_mla_latent_cache(const torch::Tensor& k_cache,
                                      const torch::Tensor& block_table,
                                      const torch::Tensor& kv_cu_seq_lens,
                                      int64_t total_kv_len,
                                      int64_t max_seq_len) {
  CHECK(k_cache.defined()) << "gather_mla_latent_cache requires k_cache.";
  CHECK(block_table.defined())
      << "gather_mla_latent_cache requires block_table.";
  CHECK(kv_cu_seq_lens.defined())
      << "gather_mla_latent_cache requires kv_cu_seq_lens.";
  CHECK(k_cache.is_cuda()) << "k_cache must be on a HIP/DCU device.";
  CHECK(block_table.is_cuda()) << "block_table must be on a HIP/DCU device.";
  CHECK(kv_cu_seq_lens.is_cuda())
      << "kv_cu_seq_lens must be on a HIP/DCU device.";
  CHECK_EQ(k_cache.dim(), 4)
      << "k_cache must be [blocks, block_size, 1, dim], got "
      << k_cache.sizes();
  CHECK_EQ(k_cache.size(2), 1) << "MLA latent cache must have one KV head.";
  CHECK_EQ(block_table.dim(), 2)
      << "block_table must be [batch, max_blocks], got "
      << block_table.sizes();
  CHECK_EQ(kv_cu_seq_lens.dim(), 1)
      << "kv_cu_seq_lens must be [batch + 1], got "
      << kv_cu_seq_lens.sizes();
  CHECK_EQ(kv_cu_seq_lens.size(0), block_table.size(0) + 1)
      << "kv_cu_seq_lens size must match block_table batch.";
  CHECK_GT(total_kv_len, 0)
      << "gather_mla_latent_cache total_kv_len must be positive.";
  CHECK_GT(max_seq_len, 0)
      << "gather_mla_latent_cache max_seq_len must be positive.";
  CHECK(k_cache.is_contiguous()) << "k_cache must be contiguous.";

  torch::Tensor block_table_i32 =
      block_table.scalar_type() == torch::kInt32 ? block_table
                                                 : block_table.to(torch::kInt32);
  block_table_i32 = block_table_i32.contiguous();
  torch::Tensor kv_cu_seq_lens_i32 =
      kv_cu_seq_lens.scalar_type() == torch::kInt32
          ? kv_cu_seq_lens
          : kv_cu_seq_lens.to(torch::kInt32);
  kv_cu_seq_lens_i32 = kv_cu_seq_lens_i32.contiguous();

  torch::Tensor output =
      torch::empty({total_kv_len, k_cache.size(-1)}, k_cache.options());
  switch (k_cache.element_size()) {
    case 1:
      launch_gather_mla_latent_cache_kernel<uint8_t>(
          k_cache, block_table_i32, kv_cu_seq_lens_i32, output, max_seq_len);
      break;
    case 2:
      launch_gather_mla_latent_cache_kernel<uint16_t>(
          k_cache, block_table_i32, kv_cu_seq_lens_i32, output, max_seq_len);
      break;
    case 4:
      launch_gather_mla_latent_cache_kernel<uint32_t>(
          k_cache, block_table_i32, kv_cu_seq_lens_i32, output, max_seq_len);
      break;
    case 8:
      launch_gather_mla_latent_cache_kernel<uint64_t>(
          k_cache, block_table_i32, kv_cu_seq_lens_i32, output, max_seq_len);
      break;
    default:
      LOG(FATAL) << "Unsupported k_cache element size: "
                 << k_cache.element_size();
  }
  return output;
}

}  // namespace dcu
}  // namespace kernel
}  // namespace xllm
