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

#include "fbcache.h"

namespace xllm {

void FBCache::init(const DiTCacheConfig& cfg,
                   const ParallelArgs& parallel_args) {
  parallel_args_ = &parallel_args;
  CHECK_GE(cfg.fbcache.residual_diff_threshold, 0.0)
      << "residual_diff_threshold must be >= 0";
  CHECK_GE(cfg.fbcache.warmup_steps, 0) << "warmup_steps must be >= 0";

  residual_diff_threshold_ = cfg.fbcache.residual_diff_threshold;
  warmup_steps_ = cfg.fbcache.warmup_steps;
  current_step_ = 0;
  use_cache_ = false;
  buffers.clear();
}

bool FBCache::on_before_block(const CacheBlockIn& blockin) {
  return use_cache_;
}

CacheBlockOut FBCache::on_after_block(const CacheBlockIn& blockin) {
  // If this is not the first transform block
  // just return the original tensors without caching.
  if (blockin.block_id != 0 ||
      blockin.tensors.find("encoder_hidden_states") == blockin.tensors.end()) {
    TensorMap out_map;
    out_map["hidden_states"] =
        get_tensor_or_empty(blockin.tensors, "hidden_states");
    out_map["encoder_hidden_states"] =
        get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
    return CacheBlockOut(out_map);
  }

  auto hidden_states = get_tensor_or_empty(blockin.tensors, "hidden_states");
  auto original_hidden_states =
      get_tensor_or_empty(blockin.tensors, "original_hidden_states");
  auto encoder_hidden_states =
      get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");

  torch::Tensor first_hidden_states_residual =
      hidden_states - original_hidden_states;
  torch::Tensor output_hidden_states, output_encoder_hidden_states;

  if (can_use_cache(first_hidden_states_residual)) {
    use_cache_ = true;
    auto [new_hidden, new_encoder] = apply_prev_hidden_states_residual(
        original_hidden_states, encoder_hidden_states);
    output_hidden_states = std::move(new_hidden);
    output_encoder_hidden_states = std::move(new_encoder);
  } else {
    use_cache_ = false;
    buffers["first_hidden_states_residual"] =
        std::move(first_hidden_states_residual);
    output_hidden_states = hidden_states;
    output_encoder_hidden_states = encoder_hidden_states;
  }

  TensorMap out_map;
  out_map["hidden_states"] = std::move(output_hidden_states);
  out_map["encoder_hidden_states"] = std::move(output_encoder_hidden_states);

  return CacheBlockOut(out_map);
}

bool FBCache::on_before_step(const CacheStepIn& stepin) {
  current_step_ = stepin.step_id;
  use_cache_ = false;
  if (current_step_ == 1) buffers.clear();
  return false;
}

CacheStepOut FBCache::on_after_step(const CacheStepIn& stepin) {
  if (!use_cache_) {
    auto hidden_states = get_tensor_or_empty(stepin.tensors, "hidden_states");
    auto original_hidden_states =
        get_tensor_or_empty(stepin.tensors, "original_hidden_states");

    auto hidden_states_residual = hidden_states - original_hidden_states;
    buffers["hidden_states_residual"] = std::move(hidden_states_residual);

    auto encoder_hidden_states =
        get_tensor_or_empty(stepin.tensors, "encoder_hidden_states");
    auto original_encoder_hidden_states =
        get_tensor_or_empty(stepin.tensors, "original_encoder_hidden_states");

    if (encoder_hidden_states.defined() &&
        original_encoder_hidden_states.defined()) {
      auto encoder_hidden_states_residual =
          encoder_hidden_states - original_encoder_hidden_states;
      buffers["encoder_hidden_states_residual"] =
          std::move(encoder_hidden_states_residual);
    }
  }

  TensorMap out_map;
  auto hidden_states = get_tensor_or_empty(stepin.tensors, "hidden_states");
  if (hidden_states.defined())
    out_map["hidden_states"] = std::move(hidden_states);

  return CacheStepOut(out_map);
}

std::pair<torch::Tensor, torch::Tensor>
FBCache::apply_prev_hidden_states_residual(
    const torch::Tensor& hidden_states,
    const torch::Tensor& encoder_hidden_states) {
  auto hidden_states_residual =
      get_tensor_or_empty(buffers, "hidden_states_residual");
  if (hidden_states_residual.defined()) {
    auto new_hidden = hidden_states + hidden_states_residual;
    return {new_hidden.contiguous(), encoder_hidden_states};
  } else {
    return {hidden_states, encoder_hidden_states};
  }
}

bool FBCache::can_use_cache(const torch::Tensor& first_hidden_states_residual) {
  if (current_step_ <= warmup_steps_ || residual_diff_threshold_ <= 0.0f)
    return false;

  auto prev_first_hidden_states_residual =
      get_tensor_or_empty(buffers, "first_hidden_states_residual");
  if (!prev_first_hidden_states_residual.defined()) return false;

  return is_similar(prev_first_hidden_states_residual,
                    first_hidden_states_residual,
                    residual_diff_threshold_);
}

}  // namespace xllm
