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

#include "framework/block/block_utils.h"

#include <glog/logging.h>

namespace xllm {

int64_t get_swa_blocks_per_seq(int64_t window_size, int64_t block_size) {
  CHECK_GT(window_size, 0) << "sliding_window_size must be positive";
  CHECK_GT(block_size, 0) << "block_size must be positive";
  // Align with vLLM/vllm-ascend sliding-window semantics: keep enough
  // contiguous KV blocks to cover `sliding_window - 1` history tokens plus
  // the current block being written.
  return (window_size - 1) / block_size + 1;
}

}  // namespace xllm
