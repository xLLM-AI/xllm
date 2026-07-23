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

#include "speculative_engine.h"

#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include <algorithm>
#include <memory>

#include "common/metrics.h"
#include "llm_engine.h"
#include "runtime/forward_params.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

SpeculativeEngine::SpeculativeEngine(const runtime::Options& options)
    : SpeculativeEngine(options, /*use_draft_engine=*/true) {}

SpeculativeEngine::SpeculativeEngine(const runtime::Options& options,
                                     bool use_draft_engine)
    : options_(options), use_draft_engine_(use_draft_engine) {
  CHECK_GT(options.num_speculative_tokens(), 0)
      << "speculative tokens should not be zero";

  runtime::Options dist_options = options;
  dist_options.num_decoding_tokens(options.num_speculative_tokens() + 1);
  dist_options.enable_speculative_decode(true);
  dist_manager_ = std::make_shared<DistManager>(dist_options);

  runtime::Options target_engine_options = options_;
  target_engine_options.num_decoding_tokens(options.num_speculative_tokens() +
                                            1);
  engine_ = std::make_unique<LLMEngine>(target_engine_options, dist_manager_);

  if (use_draft_engine_) {
    // draft engine
    runtime::Options draft_engine_options = options_;
    draft_engine_options.model_path(options_.draft_model_path().value_or(""))
        .devices(options.draft_devices())
        .num_decoding_tokens(1)
        .enable_speculative_decode(/*enable_speculative_decode=*/false)
        .enable_graph(/*enable_graph=*/false)
        .is_draft_engine(true);
    draft_engine_ =
        std::make_unique<LLMEngine>(draft_engine_options, dist_manager_);

    // Currently target and draft engines must use the same device list.
    if (options.devices() != options.draft_devices()) {
      LOG(FATAL) << "Current only support target and draft engine using the "
                    "same devices";
    }
    share_device_ = true;
  }
}

SuffixSpeculativeEngine::SuffixSpeculativeEngine(
    const runtime::Options& options)
    : SpeculativeEngine(options, /*use_draft_engine=*/false) {}

bool SpeculativeEngine::init(MasterStatus master_status) {
  if (!init_model()) {
    return false;
  }

  if (!allocate_kv_cache()) {
    return false;
  }

  return true;
}

bool SpeculativeEngine::init_model() {
  if (!engine_->init_model()) {
    return false;
  }

  model_args_ = engine_->model_args();

  if (use_draft_engine_) {
    if (!draft_engine_->init_model()) {
      return false;
    }

    // Draft tokens are detokenized by the target, so the draft engine needs no
    // tokenizer and no draft/target vocab-compatibility check (not universal;
    // see llm_engine.cpp).

    // check if the max context length are the same
    const auto& draft_model_args = draft_engine_->model_args();
    if (model_args_.max_position_embeddings() !=
        draft_model_args.max_position_embeddings()) {
      LOG(WARNING) << "draft and target models have different max context "
                      "lengths, draft max_position_embeddings: "
                   << draft_model_args.max_position_embeddings()
                   << ", target max_position_embeddings: "
                   << model_args_.max_position_embeddings()
                   << ", using the minimum between them";
      model_args_.max_position_embeddings() =
          std::min(model_args_.max_position_embeddings(),
                   draft_model_args.max_position_embeddings());
    }
  }

  dtype_ = util::parse_dtype(model_args_.dtype(), options_.devices()[0]);
  return true;
}

bool SpeculativeEngine::allocate_kv_cache() {
  KVCacheCapacity target_kv_cache_cap = engine_->estimate_kv_cache_capacity();

  if (!use_draft_engine_) {
    return engine_->allocate_kv_cache(target_kv_cache_cap);
  }

  KVCacheCapacity draft_kv_cache_cap =
      draft_engine_->estimate_kv_cache_capacity();

  if (target_kv_cache_cap.c4_count() > 0 ||
      target_kv_cache_cap.c128_count() > 0) {
    draft_kv_cache_cap.n_blocks() = target_kv_cache_cap.n_blocks();
    return engine_->allocate_kv_cache(target_kv_cache_cap) &&
           draft_engine_->allocate_kv_cache(draft_kv_cache_cap);
  }

  const int64_t kv_cache_size =
      std::min(target_kv_cache_cap.cache_size_in_bytes(),
               draft_kv_cache_cap.cache_size_in_bytes());

  int64_t n_blocks = 0;
  // check if llm and ssm are using same device
  if (share_device_) {
    // on the same device, use the smaller kv cache size
    n_blocks = calculate_kv_cache(target_kv_cache_cap, draft_kv_cache_cap);
  } else {
    // on different devices, use the smaller number of blocks
    n_blocks =
        std::min(target_kv_cache_cap.n_blocks(), draft_kv_cache_cap.n_blocks());
  }
  CHECK_GT(n_blocks, 0) << "no memory for kv cache";

  // allocate kv cache
  target_kv_cache_cap.n_blocks() = n_blocks;
  target_kv_cache_cap.cache_size_in_bytes() = kv_cache_size;
  draft_kv_cache_cap.n_blocks() = n_blocks;
  draft_kv_cache_cap.cache_size_in_bytes() = kv_cache_size;
  return engine_->allocate_kv_cache(target_kv_cache_cap) &&
         draft_engine_->allocate_kv_cache(draft_kv_cache_cap);
}

// TODO: support dp batches later
ForwardOutput SpeculativeEngine::step(std::vector<Batch>& batches) {
  return engine_->step(batches);
}

int64_t SpeculativeEngine::calculate_kv_cache(
    const KVCacheCapacity& target_kv_cache_cap,
    const KVCacheCapacity& draft_kv_cache_cap) const {
  CHECK_GT(target_kv_cache_cap.cache_size_in_bytes(), 0)
      << "no memory for target kv cache";
  CHECK_GT(draft_kv_cache_cap.cache_size_in_bytes(), 0)
      << "no memory for draft kv cache";
  CHECK_EQ(target_kv_cache_cap.block_size(), draft_kv_cache_cap.block_size())
      << "target and draft kv cache block size must be the same";

  const int64_t block_size = target_kv_cache_cap.block_size();
  CHECK_GT(block_size, 0) << "kv cache block size must be greater than 0";

  const int64_t cache_size_in_bytes =
      std::min(target_kv_cache_cap.cache_size_in_bytes(),
               draft_kv_cache_cap.cache_size_in_bytes());
  const int64_t linear_cache_size_in_bytes =
      target_kv_cache_cap.linear_cache_size_in_bytes();
  CHECK_GT(cache_size_in_bytes, linear_cache_size_in_bytes)
      << "no memory left for speculative full-attention kv cache after "
         "reserving target linear state cache, cache_size: "
      << cache_size_in_bytes
      << ", linear_cache_size: " << linear_cache_size_in_bytes;

  const int64_t target_full_attention_slot_size =
      target_kv_cache_cap.slot_size() + target_kv_cache_cap.index_slot_size() +
      target_kv_cache_cap.scale_slot_size();
  const int64_t draft_full_attention_slot_size =
      draft_kv_cache_cap.slot_size() + draft_kv_cache_cap.index_slot_size() +
      draft_kv_cache_cap.scale_slot_size();
  const bool draft_body_uses_tp1 = options_.enable_mtp_draft_body_tp1();
  if (!draft_body_uses_tp1) {
    CHECK_LE(draft_full_attention_slot_size, target_full_attention_slot_size)
        << "draft full-attention kv cache slot size must not exceed target "
           "slot size because the current speculative worker allocates draft "
           "KV tensors with the target KVCacheShape";
  }
  const int64_t draft_allocated_full_attention_slot_size =
      draft_body_uses_tp1 ? draft_full_attention_slot_size
                          : target_full_attention_slot_size;
  CHECK_GT(target_full_attention_slot_size, 0)
      << "target full-attention kv cache slot size must be greater than 0";
  CHECK_GT(draft_allocated_full_attention_slot_size, 0)
      << "draft full-attention kv cache slot size must be greater than 0";

  const int64_t target_full_attention_layers =
      std::max<int64_t>(target_kv_cache_cap.num_full_attention_layers(), 1);
  // Draft model has no linear-attention layers in the current MTP/Eagle path.
  const int64_t draft_full_attention_layers = draft_kv_cache_cap.n_layers();
  const int64_t target_full_attention_block_size_in_bytes =
      block_size *
      (target_full_attention_layers * (target_kv_cache_cap.slot_size() +
                                       target_kv_cache_cap.scale_slot_size()) +
       target_kv_cache_cap.num_indexer_layers() *
           target_kv_cache_cap.index_slot_size());
  const int64_t draft_full_attention_block_size_in_bytes =
      draft_body_uses_tp1
          ? block_size * (draft_full_attention_layers *
                              (draft_kv_cache_cap.slot_size() +
                               draft_kv_cache_cap.scale_slot_size()) +
                          draft_kv_cache_cap.num_indexer_layers() *
                              draft_kv_cache_cap.index_slot_size())
          : block_size * draft_full_attention_layers *
                draft_allocated_full_attention_slot_size;
  const int64_t full_attention_block_size_in_bytes =
      target_full_attention_block_size_in_bytes +
      draft_full_attention_block_size_in_bytes;
  CHECK_GT(full_attention_block_size_in_bytes, 0)
      << "speculative kv cache block size in bytes must be greater than 0";

  return (cache_size_in_bytes - linear_cache_size_in_bytes) /
         full_attention_block_size_in_bytes;
}

void SpeculativeEngine::update_last_step_result(std::vector<Batch>& batch) {
  engine_->update_last_step_result(batch);
}

std::vector<int64_t> SpeculativeEngine::get_active_activation_memory() const {
  return engine_->get_active_activation_memory();
}

bool SpeculativeEngine::pull_kv_blocks(
    const int32_t src_dp_size,
    const int32_t src_dp_rank,
    const std::vector<uint64_t>& src_cluster_ids,
    const std::vector<std::string>& src_addrs,
    const std::vector<uint64_t>& src_blocks,
    const int32_t dst_dp_rank,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  return engine_->pull_kv_blocks(src_dp_size,
                                 src_dp_rank,
                                 src_cluster_ids,
                                 src_addrs,
                                 src_blocks,
                                 dst_dp_rank,
                                 dst_blocks,
                                 src_linear_state_ids,
                                 dst_linear_state_ids);
};

void SpeculativeEngine::get_cache_info(std::vector<uint64_t>& cluster_ids,
                                       std::vector<std::string>& addrs,
                                       std::vector<uint16_t>& ports) {
  engine_->get_cache_info(cluster_ids, addrs, ports);
};

bool SpeculativeEngine::link_cluster(const std::vector<uint64_t>& cluster_ids,
                                     const std::vector<std::string>& addrs,
                                     const std::vector<uint16_t>& ports,
                                     const int32_t src_dp_size,
                                     const int32_t src_kv_split_size) {
  return engine_->link_cluster(
      cluster_ids, addrs, ports, src_dp_size, src_kv_split_size);
};

bool SpeculativeEngine::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                       const std::vector<std::string>& addrs,
                                       const std::vector<uint16_t>& ports,
                                       const int32_t src_dp_size,
                                       const int32_t src_kv_split_size) {
  return engine_->unlink_cluster(
      cluster_ids, addrs, ports, src_dp_size, src_kv_split_size);
};
}  // namespace xllm
