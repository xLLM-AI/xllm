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

#include "block_manager_impl.h"

namespace xllm {

// Sliding-window sub-manager used by CompositeBlockManager.
class SlidingWindowBlockManager : public BlockManagerImpl {
 public:
  explicit SlidingWindowBlockManager(const Options& options);
  ~SlidingWindowBlockManager() override = default;

  std::vector<Block> allocate(size_t num_blocks) override;

  uint32_t swa_blocks_per_seq() const { return options_.swa_blocks_per_seq(); }
};

}  // namespace xllm
