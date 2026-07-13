/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#if defined(USE_DCU)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <cstdint>

namespace xllm::kernel::cuda {

#if defined(USE_DCU)
using LlmDecodeMetadataUpdateStream = hipStream_t;
#else
using LlmDecodeMetadataUpdateStream = cudaStream_t;
#endif

struct LlmDecodeMetadataUpdateParams {
  const int32_t* src_tokens;
  const void* src_positions;
  const int32_t* src_new_cache_slots;
  const int32_t* src_kv_seq_lens;
  const int32_t* src_paged_kv_indptr;
  const int32_t* src_paged_kv_indices;
  const int32_t* src_paged_kv_last_page_len;
  int32_t* dst_tokens;
  void* dst_positions;
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
  bool src_positions_are_int64 = false;
  bool dst_positions_are_int64 = false;
  bool add_dummy_pages_for_padding = false;
};

void update_llm_decode_metadata(const LlmDecodeMetadataUpdateParams& params,
                                LlmDecodeMetadataUpdateStream stream);

}  // namespace xllm::kernel::cuda
