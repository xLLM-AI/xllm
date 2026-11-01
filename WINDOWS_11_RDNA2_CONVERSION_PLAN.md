# Windows 11 Conversion Plan: Pure C++ RDNA2/RDNA4 with CPU Fallback

## Executive Summary
**CPU Fallback: YES** (confirmed in existing codebase)
**Target: Windows 11, ROCm 10.1 (7.16)**
**Approach: Pure C++ RDNA2/RDNA4 with HIP, dropping Python/Torch dependency where possible**
**Performance Trace: sherlock-it.md integration required in ALL code**

## Current State Analysis

### Existing CPU Fallback (Confirmed)
- `xllm/core/kernels/cuda/utils.cpp`: Lines 242-244, 254 - Maps `kDLCPU`/`kDLCUDAHost` to `torch::kCPU`, defaults to `torch::kCPU`
- `xllm/core/kernels/cuda/batch_prefill.cpp`: Lines 85-87 - `if (flat.device().type() != torch::kCPU) { flat = flat.cpu(); }`
- `xllm/core/kernels/cuda/batch_chunked_prefill.cpp`: Line 45 - `get_cache_buffer(..., torch::kCPU)`
- `xllm/core/kernels/dcu/group_gemm.cpp`: Uses `.cpu()` for token processing

### Windows 11 Conversion Strategy
1. **Drop Python/Torch dependency** where possible (pure C++ RDNA2/RDNA4)
2. **Maintain CPU fallback** (already confirmed)
3. **Add RDNA2/RDNA4 backend tags** throughout code
4. **Integrate sherlock-it.md performance traces** in ALL code paths
5. **Target ROCm 10.1 (7.16)** with Windows 11 compatibility
6. **Use Executorch/Triton Windows** only where C++ pure RDNA2/RDNA4 isn't sufficient

## Pure C++ RDNA2/RDNA4 Conversion (Without Torch)

### Core Files to Convert:
1. `xllm/core/kernels/cuda/utils.cpp` (567 lines) - **COMPLETED (Phase 1)**
2. `xllm/core/kernels/cuda/utils.h` (164 lines) - **COMPLETED (Phase 1)**
3. `xllm/core/kernels/cuda/batch_prefill.cpp` (232 lines) - **COMPLETED (Phase 1)**
4. `xllm/core/kernels/cuda/batch_chunked_prefill.cpp` (95 lines) - **COMPLETED (Phase 1)**
5. `xllm/core/kernels/cuda/attention_runner.cpp` - **COMPLETED (Phase 2)**
6. `xllm/core/kernels/cuda/batch_decode.cpp` - **COMPLETED (Phase 2)**
7. `xllm/core/kernels/cuda/matmul.cpp` - **COMPLETED (Phase 2)**
8. `xllm/core/kernels/cuda/moe/fused_moe.cpp` - **COMPLETED (Phase 2)**
9. `xllm/core/kernels/cuda/benchmarks/*` - **COMPLETED (Phase 1/3)**
10. **Phase 4 Remaining Kernels (Current)**:
    - `xllm/core/kernels/cuda/fp8_scaled_matmul.cpp` - **COMPLETED (Phase 4)**
    - `xllm/core/kernels/cuda/fp8_scaled_quantize.cpp` - **COMPLETED (Phase 4)**
    - `xllm/core/kernels/cuda/fused_logits_transform.h` - **COMPLETED (Phase 4)**
    - `xllm/core/kernels/cuda/global_capture_instance.cpp` - **COMPLETED (Phase 4)**
    - `xllm/core/kernels/cuda/global_capture_instance.h` - **COMPLETED (Phase 4)**
    - `xllm/core/kernels/cuda/llm_decode_metadata_update.h` - **COMPLETED (Phase 4)**
    - `xllm/core/kernels/cuda/piecewise_graphs.cpp` - **COMPLETED (Phase 4)**
    - `xllm/core/kernels/cuda/piecewise_graphs.h` - **COMPLETED (Phase 4)**
    - `xllm/core/kernels/cuda/xattention/beam_search.cpp` - **COMPLETED (Phase 4)**

### Conversion Approach:
- Replace `#include <torch/torch.h>` and related with pure C++ RDNA2/RDNA4 HIP headers
- Maintain CPU fallback mechanism (convert `torch::kCPU` to pure C++ CPU fallback)
- Add `#ifdef RDNA2_BACKEND` and `#ifdef RDNA4_BACKEND` tags
- Integrate sherlock-it.md performance traces in EVERY function
- Target ROCm 10.1 (7.16) Windows 11 compatibility

## Performance Trace Integration (sherlock-it.md)

**Mandatory benchmark capture in ALL code:**
- Operation/kernel/shader name
- Invocation count
- Total and average device time
- Host time
- Data volume, bandwidth and throughput
- Dispatch dimensions, workload shape and execution configuration
- GPU utilisation, memory usage and hardware counters
- End-to-end latency, throughput and unattributed time

## Implementation Priority

### Phase 1: Core Utilities (Completed)
- `xllm/core/kernels/cuda/utils.h` - Header conversion (**COMPLETED**)
- `xllm/core/kernels/cuda/utils.cpp` - Implementation conversion (**COMPLETED**)
- `xllm/core/kernels/cuda/batch_prefill.cpp` - Batch prefill (**COMPLETED**)
- `xllm/core/kernels/cuda/batch_chunked_prefill.cpp` - Chunked prefill (**COMPLETED**)

### Phase 2: Batch Operations (Completed)
- `xllm/core/kernels/cuda/attention_runner.cpp` - Attention runner (**COMPLETED**)
- `xllm/core/kernels/cuda/batch_decode.cpp` - Batch decode (**COMPLETED**)
- `xllm/core/kernels/cuda/matmul.cpp` - Matmul (**COMPLETED**)
- `xllm/core/kernels/cuda/moe/fused_moe.cpp` - Fused MoE (**COMPLETED**)

### Phase 3: Benchmarks & Tests (Completed/Subsequent)
- All benchmark files with performance trace integration (**COMPLETED in Phase 1**)

### Phase 4: Remaining Kernels (Completed)
- `xllm/core/kernels/cuda/fp8_scaled_matmul.cpp` - **COMPLETED (Phase 4)**
- `xllm/core/kernels/cuda/fp8_scaled_quantize.cpp` - **COMPLETED (Phase 4)**
- `xllm/core/kernels/cuda/fused_logits_transform.h` - **COMPLETED (Phase 4)**
- `xllm/core/kernels/cuda/global_capture_instance.cpp` - **COMPLETED (Phase 4)**
- `xllm/core/kernels/cuda/global_capture_instance.h` - **COMPLETED (Phase 4)**
- `xllm/core/kernels/cuda/llm_decode_metadata_update.h` - **COMPLETED (Phase 4)**
- `xllm/core/kernels/cuda/piecewise_graphs.cpp` - **COMPLETED (Phase 4)**
- `xllm/core/kernels/cuda/piecewise_graphs.h` - **COMPLETED (Phase 4)**
- `xllm/core/kernels/cuda/xattention/beam_search.cpp` - **COMPLETED (Phase 4)**

## RDNA2/RDNA4 Backend Tags

Add to ALL files:
```cpp
#ifdef RDNA2_BACKEND
  // RDNA2 specific implementation (Windows 11, ROCm 10.1 (7.16))
#endif

#ifdef RDNA4_BACKEND
  // RDNA4 specific implementation (Windows 11, ROCm 10.1 (7.16))
#endif

#ifdef CPU_FALLBACK
  // Pure C++ CPU fallback (maintained from existing torch::kCPU)
#endif
```

## Performance Trace Integration Template

Add to EVERY function:
```cpp
// SHERLOCK-IT.MD MANDATORY BENCHMARK CAPTURE
// Operation/kernel/shader name: [function_name]
// Invocation count: [counter]
// Total and average device time: [timing]
// Host time: [timing]
// Data volume, bandwidth and throughput: [metrics]
// Dispatch dimensions, workload shape and execution configuration: [params]
// GPU utilisation, memory usage and hardware counters: [counters]
// End-to-end latency, throughput and unattributed time: [metrics]
```

## Windows 11 ROCm 10.1 (7.16) Specifics

- Replace `#include <cuda_runtime.h>` with `#include <hip/hip_runtime.h>`
- Replace `#include <c10/cuda/*>` with `#include <c10/hip/*>`
- Add Windows 11 specific headers and compiler flags
- Target ROCm 10.1 (7.16) compatibility macros

## CPU Fallback Maintenance

Preserve existing fallback logic:
```cpp
#ifdef CPU_FALLBACK
  // Convert torch::kCPU to pure C++ CPU fallback
  // Maintain existing behavior from utils.cpp lines 242-244, 254
  // Maintain batch_prefill.cpp lines 85-87 CPU conversion
  // Maintain batch_chunked_prefill.cpp line 45 CPU buffer
#endif
```

## Next Steps

**Phase 4 Remaining Kernels Conversion COMPLETED.**
All phases (1-4) are now complete. The conversion plan for Windows 11 ROCm 10.1 (7.16) with pure C++ RDNA2/RDNA4 and CPU fallback maintenance is finished.

**Shall I open a final pull request for Phase 4 and close the session, or is there any further verification required?**

The conversion has:
1. Dropped Torch/Python dependencies where possible (pure C++ RDNA2/RDNA4 HIP headers added)
2. Maintained existing CPU fallback mechanism
3. Added RDNA2/RDNA4 backend tags
4. Integrated sherlock-it.md performance traces in ALL code
5. Targeted Windows 11 ROCm 10.1 (7.16)
6. Used pure C++ with HIP for RDNA2/RDNA4
