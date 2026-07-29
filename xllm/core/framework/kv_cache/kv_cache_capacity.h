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

#pragma once

#include <cstdint>

#include "common/macros.h"

namespace xllm {

class KVCacheCapacity final {
 public:
  PROPERTY(int64_t, n_blocks) = 0;
  PROPERTY(int64_t, cache_size_in_bytes) = 0;
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, slot_size) = 0;

  // for index cache
  PROPERTY(int64_t, index_slot_size) = 0;
  PROPERTY(int64_t, num_indexer_layers) = 0;
  PROPERTY(bool, enable_indexer_cache_quant) = false;

  // for kv cache quantization scale cache
  PROPERTY(int64_t, scale_slot_size) = 0;

  // for linear attention
  PROPERTY(int64_t, linear_slot_size) = 0;
  PROPERTY(int64_t, linear_cache_size_in_bytes) = 0;
  PROPERTY(int64_t, linear_conv_state_len) = 0;
  PROPERTY(int64_t, linear_ssm_checkpoint_stride) = 1;
  PROPERTY(int64_t, n_layers) = 0;
  PROPERTY(int64_t, num_linear_state_blocks) = 0;
  PROPERTY(int64_t, num_full_attention_layers) = 0;
  PROPERTY(int64_t, num_linear_attention_layers) = 0;

  // DeepSeek V4 uses separate block pools for sliding-window and compressed
  // caches. These fields are only meaningful for deepseek_v4.
  PROPERTY(int64_t, swa_count) = 0;
  PROPERTY(int64_t, c4_count) = 0;
  PROPERTY(int64_t, c128_count) = 0;
};

}  // namespace xllm
