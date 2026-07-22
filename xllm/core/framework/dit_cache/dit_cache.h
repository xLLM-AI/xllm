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

  bool init(const DiTCacheConfig& cfg);

  DiTCache(const DiTCacheConfig& cfg) {
    active_cache_ = create_dit_cache(cfg);
    active_cond_cache_ = create_dit_cache(cfg);
    if (!active_cache_ || !active_cond_cache_) {
      LOG(ERROR) << "failed to initialized dit cache, "
                    "please check your config";
    }
    active_cache_->init(cfg);
    active_cond_cache_->init(cfg);
  }

  bool on_before_block(const CacheBlockIn& blockin, bool use_cfg = false);

  CacheBlockOut on_after_block(const CacheBlockIn& blockin,
                               bool use_cfg = false);

  bool on_before_step(const CacheStepIn& stepin, bool use_cfg = false);

  CacheStepOut on_after_step(const CacheStepIn& stepin, bool use_cfg = false);

  virtual void set_infer_steps(const int64_t& infer_steps) {
    active_cache_->set_infer_steps(infer_steps);
    active_cond_cache_->set_infer_steps(infer_steps);
  }

  virtual void set_num_blocks(const int64_t& num_blocks) {
    active_cache_->set_num_blocks(num_blocks);
    active_cond_cache_->set_num_blocks(num_blocks);
  }

  void set_runtime_context(const DiTCacheRuntimeContext& ctx);

 private:
  torch::Tensor get_tensor_or_empty(const TensorMap& m, const std::string& k);

  std::unique_ptr<DitCacheImpl> active_cache_;
  std::unique_ptr<DitCacheImpl> active_cond_cache_;
};

}  // namespace xllm
