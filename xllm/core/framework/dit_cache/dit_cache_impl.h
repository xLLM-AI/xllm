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
#include <string>
#include <unordered_map>

#include "dit_cache_config.h"
#include "dit_cache_type.h"

namespace xllm {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

struct DiTCacheRuntimeContext {
  std::string model_name;
  // cfg parallel runtime info
  void* cfg_group = nullptr;
  int64_t cfg_rank = 0;
  int64_t cfg_world_size = 1;
  bool cfg_enabled = false;
  float true_cfg_scale = 1.0f;
  // sequence parallel runtime info
  void* sp_group = nullptr;
  int64_t sp_rank = 0;
  int64_t sp_world_size = 1;
  bool sp_enabled = false;
  // optional runtime metadata
  int64_t infer_steps = 0;
  int64_t num_blocks = 0;
};

class DitCacheImpl {
 public:
  DitCacheImpl() = default;
  virtual ~DitCacheImpl() = default;

  virtual void init(const DiTCacheConfig& cfg) = 0;

  virtual bool on_before_block(const CacheBlockIn& blockin) = 0;
  virtual CacheBlockOut on_after_block(const CacheBlockIn& blockin) = 0;

  virtual bool on_before_step(const CacheStepIn& stepin) = 0;
  virtual CacheStepOut on_after_step(const CacheStepIn& stepin) = 0;

  virtual void set_infer_steps(const int64_t& infer_steps) {
    infer_steps_ = infer_steps;
  }

  virtual void set_num_blocks(const int64_t& num_blocks) {
    num_blocks_ = num_blocks;
  }

  virtual void set_runtime_context(const DiTCacheRuntimeContext& ctx) {
    runtime_ctx_ = ctx;
  }

  const DiTCacheRuntimeContext& get_runtime_context() const {
    return runtime_ctx_;
  }

 protected:
  int64_t num_inference_steps_;
  int64_t warmup_steps_;
  int64_t current_step_;
  int64_t infer_steps_;
  int64_t num_blocks_;
  TensorMap buffers;
  DiTCacheRuntimeContext runtime_ctx_;

  static torch::Tensor get_tensor_or_empty(const TensorMap& m,
                                           const std::string& k);
  bool is_similar(const torch::Tensor& lhs,
                  const torch::Tensor& rhs,
                  float threshold) const;
};

std::unique_ptr<DitCacheImpl> create_dit_cache(const DiTCacheConfig& cfg);

}  // namespace xllm
