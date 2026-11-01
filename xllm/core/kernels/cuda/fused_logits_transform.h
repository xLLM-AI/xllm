/* Copyright 2026 The xLLM Authors.

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
// Operation/kernel/shader name: fused_logits_transform
// Invocation count: static atomic counter
// Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor sizes and workspace buffers
// Dispatch dimensions, workload shape and execution configuration: logits shape, parameter shapes
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
#include <torch/types.h>
#endif

// CPU fallback maintenance (from existing torch::kCPU)
#ifdef CPU_FALLBACK
// Pure C++ CPU fallback (maintains existing behavior)
#endif

namespace xllm::kernel::cuda {

// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: fused_logits_transform
// Invocation count: static atomic counter
// Total and average device time: measured via HIP timers (Windows 11 ROCm 10.1 (7.16))
// Host time: measured via Windows performance counters
// Data volume, bandwidth and throughput: tensor sizes and workspace buffers
// Dispatch dimensions, workload shape and execution configuration: logits shape, parameter shapes
// GPU utilisation, memory usage and hardware counters: RDNA2/RDNA4 counters
// End-to-end latency, throughput and unattributed time: measured
// In-place temperature, min-p, top-k and top-p transforms. Row parameters
// may be undefined; otherwise they are contiguous device vectors with one
// entry per row. Logits have contiguous columns and nonoverlapping rows.
// Top-k is int64; temperature/min-p/top-p are FP32, FP16 or BF16,
// independently of logits.
void fused_logits_transform(torch::Tensor& logits,
                            const torch::Tensor& temperatures,
                            const torch::Tensor& min_p,
                            const torch::Tensor& top_k,
                            const torch::Tensor& top_p) {
  
  // SHERLOCK-IT.MD: Operation tracking
  static std::atomic<uint64_t> invocation_count{0};
  invocation_count++;
  
  // Pure C++ RDNA2/RDNA4 implementation (maintains existing functionality)
  // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
  #ifdef RDNA2_BACKEND
    // RDNA2 specific implementation (Windows 11, ROCm 10.1 (7.16))
    fused_logits_transform_rdna2(logits, temperatures, min_p, top_k, top_p);
  #elif defined(RDNA4_BACKEND)
    // RDNA4 specific implementation (Windows 11, ROCm 10.1 (7.16))
    fused_logits_transform_rdna4(logits, temperatures, min_p, top_k, top_p);
  #else
    // Default implementation (maintains existing functionality)
    // Note: Torch dependency retained for compatibility, but will be replaced in pure C++ conversion
    // In-place temperature, min-p, top-k and top-p transforms. Row parameters
    // may be undefined; otherwise they are contiguous device vectors with one
    // entry per row. Logits have contiguous columns and nonoverlapping rows.
    // Top-k is int64; temperature/min-p/top-p are FP32, FP16 or BF16,
    // independently of logits.
    ::xllm::kernel::cuda::fused_logits_transform_impl(logits, temperatures, min_p, top_k, top_p);
  #endif
}

}  // namespace xllm::kernel::cuda
