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

#include <glog/logging.h>

namespace xllm {

bool DiTCache::init(const DiTCacheConfig& cfg,
                    const ParallelArgs& parallel_args) {
  config_ = cfg;
  parallel_args_ = &parallel_args;
  active_scope_id_ = 0;
  scopes_.clear();
  return create_scope(active_scope_id_);
}

void DiTCache::set_context(const CacheContext& context) {
  active_scope_id_ = context.scope_id;
  CHECK(create_scope(active_scope_id_))
      << "Failed to create DiT cache scope " << active_scope_id_;

  CachePair& scope = active_scope();
  scope.cache->set_context(context);
  scope.cond_cache->set_context(context);
}

void DiTCache::reset_scope(int64_t scope_id) {
  scopes_.erase(scope_id);
  CHECK(create_scope(scope_id))
      << "Failed to reset DiT cache scope " << scope_id;
}

bool DiTCache::create_scope(int64_t scope_id) {
  auto [it, inserted] = scopes_.try_emplace(scope_id);
  if (!inserted) {
    return true;
  }
  CHECK(parallel_args_ != nullptr) << "DiT cache must be initialized first";

  CachePair& scope = it->second;
  scope.cache = create_dit_cache(config_);
  scope.cond_cache = create_dit_cache(config_);
  if (!scope.cache || !scope.cond_cache) {
    scopes_.erase(it);
    return false;
  }
  scope.cache->init(config_, *parallel_args_);
  scope.cond_cache->init(config_, *parallel_args_);
  return true;
}

DiTCache::CachePair& DiTCache::active_scope() {
  auto it = scopes_.find(active_scope_id_);
  CHECK(it != scopes_.end()) << "Active DiT cache scope is not initialized";
  return it->second;
}

bool DiTCache::on_before_block(const CacheBlockIn& blockin, bool use_cfg) {
  CachePair& scope = active_scope();
  if (use_cfg) {
    return scope.cond_cache->on_before_block(blockin);
  }
  return scope.cache->on_before_block(blockin);
}

CacheBlockOut DiTCache::on_after_block(const CacheBlockIn& blockin,
                                       bool use_cfg) {
  CachePair& scope = active_scope();
  if (use_cfg) {
    return scope.cond_cache->on_after_block(blockin);
  }
  return scope.cache->on_after_block(blockin);
}

bool DiTCache::on_before_step(const CacheStepIn& stepin, bool use_cfg) {
  CachePair& scope = active_scope();
  if (use_cfg) {
    return scope.cond_cache->on_before_step(stepin);
  }
  return scope.cache->on_before_step(stepin);
}

CacheStepOut DiTCache::on_after_step(const CacheStepIn& stepin, bool use_cfg) {
  CachePair& scope = active_scope();
  if (use_cfg) {
    return scope.cond_cache->on_after_step(stepin);
  }
  return scope.cache->on_after_step(stepin);
}

}  // namespace xllm
