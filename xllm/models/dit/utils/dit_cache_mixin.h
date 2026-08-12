/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <tuple>

#include "core/framework/dit_cache/dit_cache.h"

namespace xllm {
namespace dit {

// Mixin that wraps the DiTCache step/block callback boilerplate.
//
// Every DiT transformer that supports DiTCache repeats the same plumbing in
// its forward(): build a CacheStepIn, call on_before_step, loop over blocks
// calling on_before_block / on_after_block, then call on_after_step and read
// tensors back by string key. This mixin hoists that plumbing behind a few
// higher-order methods so the model only supplies the block computation.
//
// Usage:
//   class MyTransformerImpl : public torch::nn::Module,
//                             public dit::DiTCacheMixin { ... };
//
//   torch::Tensor forward(...) {
//     torch::Tensor original_hidden_states = hidden_states;
//     torch::Tensor original_encoder_hidden_states = encoder_hidden_states;
//     exec_cache_step(
//         step_idx, hidden_states, original_hidden_states,
//         [&]() {
//           // Double-stream blocks: block_id 0..num_double-1.
//           exec_cached_blocks(
//               num_double, hidden_states, encoder_hidden_states,
//               original_hidden_states, original_encoder_hidden_states,
//               [&](int64_t i,
//                   const torch::Tensor& h,
//                   const torch::Tensor& eh) {
//                 return blocks_[i]->forward(h, eh, temb, image_rotary_emb);
//               });
//           hidden_states = torch::cat({encoder_hidden_states, hidden_states},
//                                      1);
//           // Single-stream blocks: block_id continues at num_double.
//           exec_cached_single_blocks(
//               num_single, hidden_states, original_hidden_states,
//               [&](int64_t i, const torch::Tensor& h) {
//                 return single_blocks_[i]->forward(h, temb, image_rotary_emb);
//               },
//               /*block_id_offset=*/num_double);
//         });
//     ...
//   }
//
// The mixin holds no state; DiTCache is a process-wide singleton. All methods
// are const and route through DiTCache::get_instance().
class DiTCacheMixin {
 protected:
  // Wraps one inference step. `original_hidden_states` is the pre-step snapshot
  // of `hidden_states` (used by residual-based policies). When on_before_step
  // reports a cache hit, `step_body` is skipped entirely; otherwise it runs the
  // block loops (and any model-specific merge / narrow) while mutating
  // `hidden_states` through its capture list. On return `hidden_states` holds
  // the value produced by on_after_step.
  template <typename StepBody>
  void exec_cache_step(int64_t step_idx,
                       torch::Tensor& hidden_states,
                       const torch::Tensor& original_hidden_states,
                       const StepBody& step_body,
                       bool use_cfg = false) const {
    DiTCache& cache = DiTCache::get_instance();

    TensorMap step_in_map = {
        {"hidden_states", hidden_states},
        {"original_hidden_states", original_hidden_states}};
    CacheStepIn stepin_before(step_idx, step_in_map);
    bool use_step_cache = cache.on_before_step(stepin_before, use_cfg);

    if (!use_step_cache) {
      step_body();
    }

    TensorMap step_after_map = {
        {"hidden_states", hidden_states},
        {"original_hidden_states", original_hidden_states}};
    CacheStepIn stepin_after(step_idx, step_after_map);
    CacheStepOut stepout_after = cache.on_after_step(stepin_after, use_cfg);
    hidden_states = stepout_after.tensors.at("hidden_states");
  }

  // Runs a dual-stream block loop under cache control. `block_fn(i, h, eh)`
  // computes block `i` and returns {hidden_states, encoder_hidden_states} in
  // that order; it is called only when on_before_block reports no cache hit.
  // Both `hidden_states` and `encoder_hidden_states` are updated in place from
  // the on_after_block result. The `original_*` snapshots feed residual-based
  // policies. `block_id_offset` is added to the loop index so callers can lay
  // out block_id as a contiguous range across several loops (e.g. a
  // single-stream loop that continues numbering after the double-stream loop).
  template <typename BlockFn>
  void exec_cached_blocks(int64_t num_blocks,
                          torch::Tensor& hidden_states,
                          torch::Tensor& encoder_hidden_states,
                          const torch::Tensor& original_hidden_states,
                          const torch::Tensor& original_encoder_hidden_states,
                          const BlockFn& block_fn,
                          int64_t block_id_offset = 0,
                          bool use_cfg = false) const {
    DiTCache& cache = DiTCache::get_instance();

    for (int64_t i = 0; i < num_blocks; ++i) {
      int64_t block_id = i + block_id_offset;
      CacheBlockIn blockin_before(block_id);
      bool use_block_cache = cache.on_before_block(blockin_before, use_cfg);

      if (!use_block_cache) {
        std::tie(hidden_states, encoder_hidden_states) =
            block_fn(i, hidden_states, encoder_hidden_states);
      }

      TensorMap block_after_map = {
          {"hidden_states", hidden_states},
          {"encoder_hidden_states", encoder_hidden_states},
          {"original_hidden_states", original_hidden_states},
          {"original_encoder_hidden_states", original_encoder_hidden_states}};
      CacheBlockIn blockin_after(block_id, block_after_map);
      CacheBlockOut blockout_after =
          cache.on_after_block(blockin_after, use_cfg);

      hidden_states = blockout_after.tensors.at("hidden_states");
      encoder_hidden_states =
          blockout_after.tensors.at("encoder_hidden_states");
    }
  }

  // Runs a single-stream block loop under cache control. `block_fn(i, h)`
  // computes block `i` and returns the new hidden_states; it is called only
  // when on_before_block reports no cache hit. Only `hidden_states` is updated
  // from the on_after_block result. `block_id_offset` is added to the loop
  // index, so a single-stream loop can continue block_id numbering after a
  // preceding double-stream loop (keeping ResidualCache's block boundaries
  // accurate across both loops).
  //
  // NOTE: the after-block map intentionally omits `encoder_hidden_states`. For
  // FBCache-family policies the absence of that key forces a pass-through on
  // single-stream blocks (see FBCache::on_after_block). Models that instead
  // want the cache to engage on the single stream must drive DiTCache directly
  // rather than use this helper.
  template <typename BlockFn>
  void exec_cached_single_blocks(int64_t num_blocks,
                                 torch::Tensor& hidden_states,
                                 const torch::Tensor& original_hidden_states,
                                 const BlockFn& block_fn,
                                 int64_t block_id_offset = 0,
                                 bool use_cfg = false) const {
    DiTCache& cache = DiTCache::get_instance();

    for (int64_t i = 0; i < num_blocks; ++i) {
      int64_t block_id = i + block_id_offset;
      CacheBlockIn blockin_before(block_id);
      bool use_block_cache = cache.on_before_block(blockin_before, use_cfg);

      if (!use_block_cache) {
        hidden_states = block_fn(i, hidden_states);
      }

      TensorMap block_after_map = {
          {"hidden_states", hidden_states},
          {"original_hidden_states", original_hidden_states}};
      CacheBlockIn blockin_after(block_id, block_after_map);
      CacheBlockOut blockout_after =
          cache.on_after_block(blockin_after, use_cfg);

      hidden_states = blockout_after.tensors.at("hidden_states");
    }
  }
};

}  // namespace dit
}  // namespace xllm
