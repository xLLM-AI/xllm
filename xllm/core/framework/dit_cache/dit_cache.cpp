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

#include "dit_cache.h"

namespace xllm {

bool DiTCache::init(const DiTCacheConfig& cfg) {
  active_cache_ = create_dit_cache(cfg);
  active_cond_cache_ = create_dit_cache(cfg);
  if (!active_cache_ || !active_cond_cache_) {
    return false;
  }
  active_cache_->init(cfg);
  active_cond_cache_->init(cfg);
  return true;
}

torch::Tensor DiTCache::get_tensor_or_empty(const TensorMap& m,
                                            const std::string& k) {
  auto it = m.find(k);
  if (it != m.end()) return it->second;
  return torch::Tensor();
}

bool DiTCache::on_before_block(const CacheBlockIn& blockin, bool use_cfg) {
  if (use_cfg) {
    return active_cond_cache_->on_before_block(blockin);
  }
  return active_cache_->on_before_block(blockin);
}

CacheBlockOut DiTCache::on_after_block(const CacheBlockIn& blockin,
                                       bool use_cfg) {
  if (use_cfg) {
    return active_cond_cache_->on_after_block(blockin);
  }
  return active_cache_->on_after_block(blockin);
}

bool DiTCache::on_before_step(const CacheStepIn& stepin, bool use_cfg) {
  if (use_cfg) {
    return active_cond_cache_->on_before_step(stepin);
  }
  return active_cache_->on_before_step(stepin);
}

CacheStepOut DiTCache::on_after_step(const CacheStepIn& stepin, bool use_cfg) {
  if (use_cfg) {
    return active_cond_cache_->on_after_step(stepin);
  }
  return active_cache_->on_after_step(stepin);
}

void DiTCache::set_runtime_context(const DiTCacheRuntimeContext& ctx) {
  if (active_cache_) active_cache_->set_runtime_context(ctx);
  if (active_cond_cache_) active_cond_cache_->set_runtime_context(ctx);
}

}  // namespace xllm
