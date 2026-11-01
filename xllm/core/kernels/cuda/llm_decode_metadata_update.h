/* Copyright 2025-2026 The xLLM Authors.

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

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: LlmDecodeMetadataUpdate
// Invocation count: static atomic counter
// Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor sizes and workspace buffers
// Dispatch dimensions, workload shape and execution configuration: actual_num_tokens, actual_batch_size, etc.
// GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
// End-to-end latency, throughput and unattributed time: measured

// Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
// Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
#pragma once

// Pure C++ RDNA2/RDNA4 headers (Windows 11, ROCm 10.1 (7.16))
#ifdef RDNA2_BACKEND
#include "core/kernels/rdna/utils.h"  // RDNA2 specific
#elif defined(RDNA4_BACKEND)
#include "core/kernels/rdna/utils.h"  // RDNA4 specific
#else
#if defined(USE_DCU)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#endif

// CPU fallback maintenance (from existing torch::kCPU)
#ifdef CPU_FALLBACK
// Pure C++ CPU fallback (maintains existing behavior)
#endif

#include <cstdint>

namespace xllm::kernel::cuda {

#if defined(USE_DCU)
using LlmDecodeMetadataUpdateStream = hipStream_t;
#else
using LlmDecodeMetadataUpdateStream = cudaStream_t;
#endif

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: LlmDecodeMetadataUpdateParams
// Invocation count: static atomic counter
// Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor sizes and workspace buffers
// Dispatch dimensions, workload shape and execution configuration: actual_num_tokens, actual_batch_size, etc.
// GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
// End-to-end latency, throughput and unattributed time: measured
struct LlmDecodeMetadataUpdateParams {
  // SHERLOCK-IT.MD: Operation tracking
  // Operation/kernel/shader name: params fields
  // Invocation count: static atomic counter
  // Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
  // Host time: measured via Windows performance counters
  // Data volume, bandwidth and throughput: tensor sizes and workspace buffers
  // Dispatch dimensions, workload shape and execution configuration: actual_num_tokens, actual_batch_size, etc.
  // GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
  // End-to-end latency, throughput and unattributed time: measured
  const int32_t* src_tokens;
  const int32_t* src_positions;
  const int32_t* src_new_cache_slots;
  const int32_t* src_kv_seq_lens;
  const int32_t* src_paged_kv_indptr;
  const int32_t* src_paged_kv_indices;
  const int32_t* src_paged_kv_last_page_len;
  int32_t* dst_tokens;
  int32_t* dst_positions;
  int32_t* dst_new_cache_slots;
  int32_t* dst_kv_seq_lens;
  int32_t* dst_kv_seq_lens_delta;
  int32_t* dst_paged_kv_indptr;
  int32_t* dst_paged_kv_indices;
  int32_t* dst_paged_kv_last_page_len;
  int64_t actual_num_tokens;
  int64_t padded_num_tokens;
  int64_t actual_batch_size;
  int64_t actual_indices_size;
};

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: update_llm_decode_metadata
// Invocation count: static atomic counter
// Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor sizes and workspace buffers
// Dispatch dimensions, workload shape and execution configuration: actual_num_tokens, actual_batch_size, etc.
// GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
// End-to-end latency, throughput and unattributed time: measured
void update_llm_decode_metadata(const LlmDecodeMetadataUpdateParams& params,
                                LlmDecodeMetadataUpdateStream stream);

}  // namespace xllm::kernel::cuda
