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

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "dit_cache_impl.h"

namespace xllm {

class DiTCache {
 public:
  DiTCache() = default;
  ~DiTCache() = default;

  DiTCache(const DiTCache&) = delete;
  DiTCache& operator=(const DiTCache&) = delete;
  DiTCache(DiTCache&&) = delete;
  DiTCache& operator=(DiTCache&&) = delete;

  static DiTCache& get_instance() {
    static DiTCache ditcache;
    return ditcache;
  }

  bool init(const DiTCacheConfig& cfg, const ParallelArgs& parallel_args);

  bool on_before_block(const CacheBlockIn& blockin, bool use_cfg = false);

  CacheBlockOut on_after_block(const CacheBlockIn& blockin,
                               bool use_cfg = false);

  bool on_before_step(const CacheStepIn& stepin, bool use_cfg = false);

  CacheStepOut on_after_step(const CacheStepIn& stepin, bool use_cfg = false);

  void set_context(const CacheContext& context);
  void reset_scope(int64_t scope_id);

 private:
  struct CachePair {
    std::unique_ptr<DitCacheImpl> cache;
    std::unique_ptr<DitCacheImpl> cond_cache;
  };

  bool create_scope(int64_t scope_id);
  CachePair& active_scope();

  DiTCacheConfig config_;
  const ParallelArgs* parallel_args_ = nullptr;
  int64_t active_scope_id_ = 0;
  std::unordered_map<int64_t, CachePair> scopes_;
};

}  // namespace xllm
