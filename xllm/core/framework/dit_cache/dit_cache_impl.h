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
#include <string>
#include <unordered_map>

#include "dit_cache_config.h"
#include "dit_cache_type.h"

namespace xllm {

struct ParallelArgs;

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

class DitCacheImpl {
 public:
  DitCacheImpl() = default;
  virtual ~DitCacheImpl() = default;

  // Startup-time init. parallel_args carries the fixed parallel topology
  // (e.g. the SP group used to all-reduce the similarity metric).
  virtual void init(const DiTCacheConfig& cfg,
                    const ParallelArgs& parallel_args) = 0;

  virtual bool on_before_block(const CacheBlockIn& blockin) = 0;
  virtual CacheBlockOut on_after_block(const CacheBlockIn& blockin) = 0;

  virtual bool on_before_step(const CacheStepIn& stepin) = 0;
  virtual CacheStepOut on_after_step(const CacheStepIn& stepin) = 0;

  // Per-generation params, refreshed on every forward().
  virtual void set_context(const CacheContext& context) {
    infer_steps_ = context.infer_steps;
    num_blocks_ = context.num_blocks;
  }

 protected:
  int64_t warmup_steps_;
  int64_t current_step_;
  int64_t infer_steps_;
  int64_t num_blocks_;
  TensorMap buffers;
  // Non-owning: the worker's ParallelArgs outlives this cache singleton.
  const ParallelArgs* parallel_args_ = nullptr;

  static torch::Tensor get_tensor_or_empty(const TensorMap& m,
                                           const std::string& k);
  bool is_similar(const torch::Tensor& lhs,
                  const torch::Tensor& rhs,
                  float threshold) const;
};

std::unique_ptr<DitCacheImpl> create_dit_cache(const DiTCacheConfig& cfg);

}  // namespace xllm
