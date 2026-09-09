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

#include "base_attention_impl.h"

#include "kernels/cuda/utils.h"

namespace xllm {
namespace layer {

BaseAttentionImpl::BaseAttentionImpl(int64_t num_heads,
                                     int64_t head_size,
                                     float scale,
                                     int64_t num_kv_heads,
                                     int64_t sliding_window)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      sliding_window_(sliding_window) {
  // we only support bf16 kvcache for now
  decode_use_tensor_core_ = xllm::kernel::cuda::should_use_tensor_core(
      /*kv_cache_dtype=*/torch::ScalarType::BFloat16, num_heads, num_kv_heads);
}

}  // namespace layer
}  // namespace xllm
