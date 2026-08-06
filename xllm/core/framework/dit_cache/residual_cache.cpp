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

#include "residual_cache.h"

namespace xllm {

void ResidualCache::init(const DiTCacheConfig& cfg,
                         const ParallelArgs& parallel_args) {
  parallel_args_ = &parallel_args;
  CHECK_GT(cfg.residual_cache.skip_interval_steps, 0)
      << "skip_interval_steps must be > 0";
  CHECK_GT(cfg.residual_cache.dit_cache_start_steps, 0)
      << "dit_cache_start_steps must be > 0";
  CHECK_GT(cfg.residual_cache.dit_cache_end_steps, 0)
      << "dit_cache_end_steps must be > 0";
  CHECK_GT(cfg.residual_cache.dit_cache_start_blocks, 0)
      << "dit_cache_start_blocks must be > 0";
  CHECK_GT(cfg.residual_cache.dit_cache_end_blocks, 0)
      << "dit_cache_end_blocks must be > 0";
  skip_interval_steps_ = cfg.residual_cache.skip_interval_steps;
  dit_cache_start_steps_ = cfg.residual_cache.dit_cache_start_steps;
  dit_cache_end_steps_ = cfg.residual_cache.dit_cache_end_steps;
  dit_cache_start_blocks_ = cfg.residual_cache.dit_cache_start_blocks;
  dit_cache_end_blocks_ = cfg.residual_cache.dit_cache_end_blocks;
  reset_cache();
}

void ResidualCache::reset_cache() {
  use_cache_ = false;
  update_cache_ = false;
}

void ResidualCache::mark_step_begin() { ++current_step_; }

torch::Tensor ResidualCache::get_residual(const torch::Tensor& hidden_states,
                                          const std::string& key) {
  return hidden_states - buffers[key];
}

torch::Tensor ResidualCache::add_residual(const torch::Tensor& hidden_states,
                                          const std::string& key) {
  return hidden_states + buffers[key];
}

void ResidualCache::update(const torch::Tensor& residual,
                           const std::string& key) {
  buffers[key] = residual;
}

bool ResidualCache::cache_validation() {
  bool step_valid =
      infer_steps_ > dit_cache_start_steps_ + dit_cache_end_steps_ &&
      infer_steps_ > dit_cache_start_steps_ &&
      infer_steps_ > dit_cache_end_steps_;
  bool block_valid =
      num_blocks_ > dit_cache_start_blocks_ + dit_cache_end_blocks_ &&
      num_blocks_ > dit_cache_start_blocks_ &&
      num_blocks_ > dit_cache_end_blocks_;
  return step_valid & block_valid;
}

bool ResidualCache::on_before_block(const CacheBlockIn& blockin) {
  // when infer_steps is less than skipped_skips, won't use cache
  if (!cache_validation() || !use_cache_ ||
      blockin.block_id < dit_cache_start_blocks_ ||
      blockin.block_id >= num_blocks_ - dit_cache_end_blocks_ - 1) {
    return false;
  }

  return true;
}

CacheBlockOut ResidualCache::on_after_block(const CacheBlockIn& blockin) {
  TensorMap out_map;
  auto hidden_states = get_tensor_or_empty(blockin.tensors, "hidden_states");
  auto encoder_hidden_states =
      get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
  if (cache_validation()) {
    if (use_cache_) {
      if (blockin.block_id == num_blocks_ - dit_cache_end_blocks_ - 1) {
        out_map["hidden_states"] = add_residual(hidden_states, "hidden_states");
        out_map["encoder_hidden_states"] =
            add_residual(encoder_hidden_states, "encoder_hidden_states");
        return CacheBlockOut(out_map);
      }
    } else if (update_cache_) {
      if (blockin.block_id == dit_cache_start_blocks_ - 1) {
        // cache
        update(hidden_states.clone(), "hidden_states");
        update(encoder_hidden_states.clone(), "encoder_hidden_states");
      } else if (blockin.block_id == num_blocks_ - dit_cache_end_blocks_ - 1) {
        // calculate residual and update cache
        update(get_residual(hidden_states, "hidden_states"), "hidden_states");
        update(get_residual(encoder_hidden_states, "encoder_hidden_states"),
               "encoder_hidden_states");
      }
    }
  }
  out_map["hidden_states"] = hidden_states;
  out_map["encoder_hidden_states"] = encoder_hidden_states;
  return CacheBlockOut(out_map);
}

bool ResidualCache::on_before_step(const CacheStepIn& stepin) {
  current_step_ = stepin.step_id;
  // if outside the target steps, do nothing
  if (!cache_validation() || current_step_ < dit_cache_start_steps_ - 1 ||
      current_step_ >= infer_steps_ - dit_cache_end_steps_) {
    reset_cache();
    return false;
  }
  // if inside target steps, use_cache when inside the interval
  // update cache when interval ends
  use_cache_ =
      ((current_step_ - (dit_cache_start_steps_ - 1)) % skip_interval_steps_ !=
       0);
  update_cache_ = !use_cache_;
  return false;
}

CacheStepOut ResidualCache::on_after_step(const CacheStepIn& stepin) {
  return CacheStepOut(stepin.tensors);
}

}  // namespace xllm
