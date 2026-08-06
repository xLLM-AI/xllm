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

#include "dit_cache_impl.h"

namespace xllm {

class FBCache : public DitCacheImpl {
 public:
  FBCache() = default;
  ~FBCache() override = default;

  FBCache(const FBCache&) = delete;
  FBCache& operator=(const FBCache&) = delete;
  FBCache(FBCache&&) = default;
  FBCache& operator=(FBCache&&) = default;

  void init(const DiTCacheConfig& cfg,
            const ParallelArgs& parallel_args) override;

  bool on_before_block(const CacheBlockIn& blockin) override;
  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override;

  bool on_before_step(const CacheStepIn& stepin) override;
  CacheStepOut on_after_step(const CacheStepIn& stepin) override;

 private:
  std::pair<torch::Tensor, torch::Tensor> apply_prev_hidden_states_residual(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states);

  bool can_use_cache(const torch::Tensor& first_hidden_states_residual);

 private:
  float residual_diff_threshold_;
  bool use_cache_;
};

}  // namespace xllm
