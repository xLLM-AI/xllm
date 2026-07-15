/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>
#include <vector>

#include "core/kernels/musa/llm_decode_metadata_update.h"

namespace xllm::kernel::cuda {
namespace {

constexpr int32_t kThreadsPerBlock = 256;
constexpr int64_t kMaxBlocksPerLaunch = 4096;

__global__ void llm_decode_metadata_update_kernel(
    LlmDecodeMetadataUpdateParams params,
    int64_t max_work_size) {
  const int64_t thread_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  const int64_t dyn_indices_size =
      (params.src_paged_kv_indptr != nullptr && params.actual_batch_size > 0)
          ? static_cast<int64_t>(
                params.src_paged_kv_indptr[params.actual_batch_size])
          : params.actual_indices_size;

  for (int64_t idx = thread_idx; idx < max_work_size; idx += step) {
    if (idx < params.actual_num_tokens) {
      params.dst_tokens[idx] = params.src_tokens[idx];
      params.dst_positions[idx] = params.src_positions[idx];
      params.dst_new_cache_slots[idx] = params.src_new_cache_slots[idx];
    }
    if (idx >= params.actual_num_tokens && idx < params.padded_num_tokens) {
      params.dst_tokens[idx] = 0;
      params.dst_new_cache_slots[idx] = 0;
    }
    if (idx < params.actual_batch_size + 1) {
      params.dst_kv_seq_lens[idx] = params.src_kv_seq_lens[idx];
      params.dst_paged_kv_indptr[idx] = params.src_paged_kv_indptr[idx];
    }
    if (idx < params.actual_batch_size) {
      params.dst_kv_seq_lens_delta[idx] =
          params.src_kv_seq_lens[idx + 1] - params.src_kv_seq_lens[idx];
      params.dst_paged_kv_last_page_len[idx] =
          params.src_paged_kv_last_page_len[idx];
    }
    if (idx < dyn_indices_size) {
      params.dst_paged_kv_indices[idx] = params.src_paged_kv_indices[idx];
    }
  }
}

}  // namespace

void memcpy_async(void* dst,
                  const void* src,
                  size_t bytes,
                  cudaMemcpyKind kind,
                  LlmDecodeMetadataUpdateStream stream) {
  if (bytes == 0 || dst == nullptr || src == nullptr) {
    return;
  }
  const cudaError_t error = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  CHECK_EQ(error, cudaSuccess)
      << "llm_decode_metadata memcpy failed: " << cudaGetErrorString(error);
}

void update_llm_decode_metadata_from_host(
    const LlmDecodeMetadataHostUpdateParams& params,
    LlmDecodeMetadataUpdateStream stream) {
  const int64_t actual_num_tokens = params.actual_num_tokens;
  const int64_t padded_num_tokens = params.padded_num_tokens;
  const int64_t actual_batch_size = params.actual_batch_size;
  const int64_t actual_indices_size = params.actual_indices_size;
  if (actual_num_tokens <= 0 && actual_batch_size <= 0) {
    return;
  }

  const size_t token_bytes =
      static_cast<size_t>(actual_num_tokens) * sizeof(int32_t);
  memcpy_async(params.dst_tokens,
               params.src_tokens,
               token_bytes,
               cudaMemcpyDeviceToDevice,
               stream);
  memcpy_async(params.dst_positions,
               params.src_positions,
               token_bytes,
               cudaMemcpyDeviceToDevice,
               stream);
  memcpy_async(params.dst_new_cache_slots,
               params.src_new_cache_slots,
               token_bytes,
               cudaMemcpyDeviceToDevice,
               stream);

  if (padded_num_tokens > actual_num_tokens) {
    const size_t pad_bytes =
        static_cast<size_t>(padded_num_tokens - actual_num_tokens) *
        sizeof(int32_t);
    const cudaError_t zero_tokens = cudaMemsetAsync(
        params.dst_tokens + actual_num_tokens, 0, pad_bytes, stream);
    CHECK_EQ(zero_tokens, cudaSuccess)
        << "llm_decode_metadata token padding memset failed: "
        << cudaGetErrorString(zero_tokens);
    const cudaError_t zero_slots = cudaMemsetAsync(
        params.dst_new_cache_slots + actual_num_tokens, 0, pad_bytes, stream);
    CHECK_EQ(zero_slots, cudaSuccess)
        << "llm_decode_metadata slot padding memset failed: "
        << cudaGetErrorString(zero_slots);
  }

  if (actual_batch_size >= 0 && params.host_kv_seq_lens != nullptr) {
    const size_t kv_cu_bytes =
        static_cast<size_t>(actual_batch_size + 1) * sizeof(int32_t);
    memcpy_async(params.dst_kv_seq_lens,
                 params.host_kv_seq_lens,
                 kv_cu_bytes,
                 cudaMemcpyHostToDevice,
                 stream);
    if (actual_batch_size > 0 && params.dst_kv_seq_lens_delta != nullptr) {
      std::vector<int32_t> kv_delta(static_cast<size_t>(actual_batch_size));
      for (int64_t i = 0; i < actual_batch_size; ++i) {
        kv_delta[static_cast<size_t>(i)] =
            params.host_kv_seq_lens[i + 1] - params.host_kv_seq_lens[i];
      }
      memcpy_async(params.dst_kv_seq_lens_delta,
                   kv_delta.data(),
                   static_cast<size_t>(actual_batch_size) * sizeof(int32_t),
                   cudaMemcpyHostToDevice,
                   stream);
    }
  }

  if (params.host_paged_kv_indptr != nullptr && actual_batch_size >= 0) {
    memcpy_async(params.dst_paged_kv_indptr,
                 params.host_paged_kv_indptr,
                 static_cast<size_t>(actual_batch_size + 1) * sizeof(int32_t),
                 cudaMemcpyHostToDevice,
                 stream);
  }
  if (params.host_paged_kv_indices != nullptr && actual_indices_size > 0) {
    memcpy_async(params.dst_paged_kv_indices,
                 params.host_paged_kv_indices,
                 static_cast<size_t>(actual_indices_size) * sizeof(int32_t),
                 cudaMemcpyHostToDevice,
                 stream);
  }
  if (params.host_paged_kv_last_page_len != nullptr && actual_batch_size > 0) {
    memcpy_async(params.dst_paged_kv_last_page_len,
                 params.host_paged_kv_last_page_len,
                 static_cast<size_t>(actual_batch_size) * sizeof(int32_t),
                 cudaMemcpyHostToDevice,
                 stream);
  }
}

void update_llm_decode_metadata(const LlmDecodeMetadataUpdateParams& params,
                                LlmDecodeMetadataUpdateStream stream) {
  int64_t max_work_size = params.actual_num_tokens;
  if (params.padded_num_tokens > max_work_size) {
    max_work_size = params.padded_num_tokens;
  }
  if (params.actual_batch_size + 1 > max_work_size) {
    max_work_size = params.actual_batch_size + 1;
  }
  if (params.actual_indices_size > max_work_size) {
    max_work_size = params.actual_indices_size;
  }
  if (params.max_indices_size_for_graph_capacity > max_work_size) {
    max_work_size = params.max_indices_size_for_graph_capacity;
  }
  if (max_work_size <= 0) {
    return;
  }
  const int64_t num_blocks = std::min<int64_t>(
      (max_work_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
      kMaxBlocksPerLaunch);
  llm_decode_metadata_update_kernel<<<static_cast<uint32_t>(num_blocks),
                                      kThreadsPerBlock,
                                      /*shared_mem_bytes=*/0,
                                      stream>>>(params, max_work_size);
  const cudaError_t error = cudaGetLastError();
  CHECK_EQ(error, cudaSuccess)
      << "llm_decode_metadata_update kernel launch failed: "
      << cudaGetErrorString(error);
}

}  // namespace xllm::kernel::cuda
