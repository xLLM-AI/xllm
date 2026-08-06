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

#include <torch/torch.h>

#include <cmath>
#include <vector>

#include "dit_cache_impl.h"

namespace xllm {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

class ResidualCache : public DitCacheImpl {
 public:
  ResidualCache() = default;
  ~ResidualCache() = default;

  ResidualCache(const ResidualCache&) = delete;
  ResidualCache& operator=(const ResidualCache&) = delete;
  ResidualCache(ResidualCache&&) = default;
  ResidualCache& operator=(ResidualCache&&) = default;

  void init(const DiTCacheConfig& cfg,
            const ParallelArgs& parallel_args) override;
  // check whether to use cache
  bool cache_validation();

  // Reset all cached derivatives and internal state
  void reset_cache();

  // Mark the beginning of a new inference step
  void mark_step_begin();

  // calculate residual
  torch::Tensor get_residual(const torch::Tensor& hidden_states,
                             const std::string& key);

  // add residaul to hidden states
  torch::Tensor add_residual(const torch::Tensor& hidden_states,
                             const std::string& key);

  // Update internal caches with the new residual
  void update(const torch::Tensor& residual, const std::string& key);

  bool on_before_block(const CacheBlockIn& blockin) override;
  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override;

  bool on_before_step(const CacheStepIn& stepin) override;
  CacheStepOut on_after_step(const CacheStepIn& stepin) override;

 private:
  bool use_cache_ = false;
  bool update_cache_ = false;
  int64_t skip_interval_steps_;
  int64_t dit_cache_start_steps_;
  int64_t dit_cache_end_steps_;
  int64_t dit_cache_start_blocks_;
  int64_t dit_cache_end_blocks_;
};

}  // namespace xllm
