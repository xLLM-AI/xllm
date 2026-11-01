/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "speculative_engine.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "common/metrics.h"
#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/eplb/eplb_controller.h"
#include "core/kv_cache/layout/kv_cache_estimation.h"
#include "core/kv_cache/layout/vmm_kv_cache_layout.h"
#include "framework/speculative/mtp_utils.h"
#include "llm_engine.h"
#include "util/utils.h"
#include "vlm_engine.h"

namespace xllm {

template <typename TargetEngine>
SpeculativeEngineBase<TargetEngine>::SpeculativeEngineBase(
    const runtime::Options& options)
    : SpeculativeEngineBase(options, /*use_draft_engine=*/true) {}

template <typename TargetEngine>
SpeculativeEngineBase<TargetEngine>::~SpeculativeEngineBase() = default;

template <typename TargetEngine>
SpeculativeEngineBase<TargetEngine>::SpeculativeEngineBase(
    const runtime::Options& options,
    bool use_draft_engine)
    : options_(options), use_draft_engine_(use_draft_engine) {
  CHECK_GT(options.num_speculative_tokens(), 0)
      << "speculative tokens should not be zero";

  runtime::Options dist_options = options;
  dist_options.num_decoding_tokens(options.num_speculative_tokens() + 1);
  dist_options.enable_speculative_decode(true);
  distributed_worker_manager_ =
      std::make_shared<DistributedWorkerManager>(dist_options);

  runtime::Options target_engine_options = options_;
  target_engine_options.num_decoding_tokens(options.num_speculative_tokens() +
                                            1);
  engine_ = std::make_unique<TargetEngine>(target_engine_options,
                                           distributed_worker_manager_);

  if (use_draft_engine_) {
    // draft engine
    // In vlm mtp case, we'd like to use llm backend instead of vlm backend for
    // draft engine.
    runtime::Options draft_engine_options = options_;
    // The draft engine always runs on the target's device list.
    draft_engine_options.model_path(options_.draft_model_path().value_or(""))
        .devices(options.devices())
        .backend("llm")
        .num_decoding_tokens(1)
        .enable_speculative_decode(/*enable_speculative_decode=*/false)
        .enable_graph(/*enable_graph=*/false)
        .is_draft_engine(true);
    if (KVCacheConfig::get_instance().enable_vmm_kvcache()) {
      draft_engine_options.model_id(options_.model_id() + "::mtp_draft");
    }
    draft_engine_ = std::make_unique<LLMEngine>(draft_engine_options,
                                                distributed_worker_manager_);
  }
}

SuffixSpeculativeEngine::SuffixSpeculativeEngine(
    const runtime::Options& options)
    : SpeculativeEngineBase<LLMEngine>(options, /*use_draft_engine=*/false) {}

template <typename TargetEngine>
bool SpeculativeEngineBase<TargetEngine>::init(MasterStatus master_status) {
  return init(master_status, /*prepare_model=*/{});
}

template <typename TargetEngine>
bool SpeculativeEngineBase<TargetEngine>::init(
    MasterStatus master_status,
    const LLMEngine::ModelInitCallback& prepare_model,
    std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator) {
  if (!init_model(master_status, prepare_model)) {
    return false;
  }

  if (!allocate_kv_cache(std::move(transfer_coordinator))) {
    return false;
  }

  return true;
}

template <typename TargetEngine>
bool SpeculativeEngineBase<TargetEngine>::init_model(
    MasterStatus master_status,
    const LLMEngine::ModelInitCallback& prepare_model) {
  if constexpr (std::is_same_v<TargetEngine, LLMEngine>) {
    if (!engine_->init_model(master_status, prepare_model)) {
      return false;
    }
  } else {
    if (!engine_->init_model(master_status)) {
      return false;
    }
  }

  model_args_ = engine_->model_args();

  if (LoadConfig::get_instance().enable_vmm_weights() &&
      draft_engine_ != nullptr && !should_skip_external_draft_kv_cache()) {
    LOG(ERROR) << "Virtual weight memory does not support an external "
                  "speculative draft engine.";
    return false;
  }
  if (KVCacheConfig::get_instance().enable_vmm_kvcache() &&
      draft_engine_ != nullptr &&
      (!SpeculativeConfig::is_mtp_algorithm(options_.speculative_algorithm()) ||
       should_skip_external_draft_kv_cache() || options_.enable_disagg_pd() ||
       options_.instance_role() != InstanceRole::DEFAULT)) {
    LOG(ERROR) << "Virtual KV cache supports co-located MTP target and draft "
                  "engines only; embedded Eagle3 and PD are unsupported.";
    return false;
  }

  if (use_draft_engine_) {
    if (!draft_engine_->init_model(/*master_status=*/MasterStatus::WAKEUP,
                                   prepare_model)) {
      return false;
    }

    // Draft tokens are detokenized by the target, so the draft engine needs no
    // tokenizer and no draft/target vocab-compatibility check (not universal;
    // see llm_engine.cpp).

    // check if the max context length are the same
    const auto& draft_model_args = draft_engine_->model_args();
    draft_cache_plan_ =
        resolve_draft_swa_cache_plan(draft_model_args, options_);
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

  if constexpr (std::is_same_v<TargetEngine, LLMEngine>) {
    engine_->eplb_controller_ = EplbController::create(
        engine_->model_args(),
        static_cast<int32_t>(engine_->worker_clients_num_),
        engine_->options_.ep_size());
    engine_->forward_input_factory_->set_eplb_controller(
        engine_->eplb_controller_.get());
  }
  dtype_ = util::parse_dtype(model_args_.dtype(), options_.devices()[0]);
  return true;
}

template <typename TargetEngine>
bool SpeculativeEngineBase<TargetEngine>::allocate_kv_cache(
    std::shared_ptr<KVCacheTransferCoordinatorBase> transfer_coordinator) {
  const bool shared_vmm = KVCacheConfig::get_instance().enable_vmm_kvcache() &&
                          use_draft_engine_ &&
                          !should_skip_external_draft_kv_cache();
  KVCacheCapacity target_kv_cache_cap =
      engine_->estimate_kv_cache_capacity(!shared_vmm);
  std::optional<VmmKVCacheLayout> shared_vmm_layout;
  const auto allocate_target_cache = [this,
                                      &transfer_coordinator,
                                      shared_vmm,
                                      &shared_vmm_layout](
                                         const KVCacheCapacity& capacity) {
    const VmmKVCacheLayout* layout = shared_vmm ? &*shared_vmm_layout : nullptr;
    if constexpr (std::is_same_v<TargetEngine, LLMEngine>) {
      return engine_->allocate_kv_cache(capacity, transfer_coordinator, layout);
    } else {
      return engine_->allocate_kv_cache(capacity, layout);
    }
  };

  if (!use_draft_engine_) {
    return allocate_target_cache(target_kv_cache_cap);
  }

  // Some MLA Eagle3 targets keep the draft worker inside the target engine and
  // allocate its full-attention KV cache with a separate shape. Do not compare
  // that shape with the target MLA cache or allocate the external draft here.
  if (should_skip_external_draft_kv_cache()) {
    return allocate_target_cache(target_kv_cache_cap);
  }

  KVCacheCapacity draft_kv_cache_cap =
      draft_engine_->estimate_kv_cache_capacity(!shared_vmm);

  int64_t shared_vmm_blocks = 0;
  if (shared_vmm) {
    shared_vmm_layout.emplace(target_kv_cache_cap,
                              engine_->model_args(),
                              engine_->dp_local_tp_size_,
                              dtype_);
    const int64_t draft_world_size = options_.enable_mtp_draft_body_tp1()
                                         ? 1
                                         : draft_engine_->dp_local_tp_size_;
    VmmKVCacheLayout draft_layout(draft_kv_cache_cap,
                                  draft_engine_->model_args(),
                                  draft_world_size,
                                  draft_engine_->dtype_,
                                  /*is_draft_engine=*/true);
    shared_vmm_layout->append(std::move(draft_layout));
    const int64_t reserved_state_bytes =
        target_kv_cache_cap.linear_cache_size_in_bytes() +
        draft_kv_cache_cap.linear_cache_size_in_bytes();
    auto& coordinator = DistributedVirtualMemoryCoordinator::get_instance();
    coordinator.prepare_kv_cache_pool(
        engine_->options_,
        engine_->dp_local_tp_size_,
        engine_->worker_clients_.size(),
        reserved_state_bytes,
        /*is_multimodal=*/std::is_same_v<TargetEngine, VLMEngine>);
    const KVCacheConfig& config = KVCacheConfig::get_instance();
    const size_t total_pages = coordinator.get_num_total_phy_pages();
    const int64_t pool_bytes =
        static_cast<int64_t>(total_pages) * config.phy_page_granularity_size();
    const size_t n_blocks = shared_vmm_layout->block_capacity(
        total_pages, config.phy_page_granularity_size());
    shared_vmm_blocks = static_cast<int64_t>(n_blocks);
    // Report the combined budget, while paged K/V block capacity depends only
    // on the pool. Persistent states have already been reserved outside it.
    target_kv_cache_cap.cache_size_in_bytes(pool_bytes + reserved_state_bytes);
    draft_kv_cache_cap.cache_size_in_bytes(pool_bytes + reserved_state_bytes);
  }

  if (target_kv_cache_cap.c4_count() > 0 ||
      target_kv_cache_cap.c128_count() > 0) {
    draft_kv_cache_cap.n_blocks() = target_kv_cache_cap.n_blocks();
    return allocate_target_cache(target_kv_cache_cap) &&
           draft_engine_->allocate_kv_cache(draft_kv_cache_cap,
                                            std::move(transfer_coordinator));
  }

  if (target_kv_cache_cap.kpool_tail_slot_size() > 0 &&
      draft_kv_cache_cap.kpool_tail_slot_size() > 0) {
    const SharedKPoolPlan plan =
        plan_shared_kpool(std::move(target_kv_cache_cap),
                          std::move(draft_kv_cache_cap),
                          draft_engine_->model_args());
    return allocate_target_cache(plan.target) &&
           draft_engine_->allocate_kv_cache(plan.draft,
                                            std::move(transfer_coordinator));
  }

  const int64_t kv_cache_size =
      std::min(target_kv_cache_cap.cache_size_in_bytes(),
               draft_kv_cache_cap.cache_size_in_bytes());

  // Target and draft run on the same device list, so they share a common KV
  // cache budget.
  const int64_t n_blocks =
      shared_vmm ? shared_vmm_blocks
                 : calculate_kv_cache(target_kv_cache_cap, draft_kv_cache_cap);
  CHECK_GT(n_blocks, 0) << "no memory for kv cache";

  // allocate kv cache
  target_kv_cache_cap.n_blocks() = n_blocks;
  target_kv_cache_cap.cache_size_in_bytes() = kv_cache_size;
  const int64_t draft_swa_count = draft_cache_plan_.num_blocks;
  if (draft_swa_count > 0) {
    target_kv_cache_cap.draft_swa_count(draft_swa_count)
        .draft_swa_window(draft_cache_plan_.window_size);
    draft_kv_cache_cap.draft_swa_count(draft_swa_count);
    LOG(INFO) << "Block draft SWA cache: target_blocks=" << n_blocks
              << ", draft_blocks=" << draft_swa_count
              << ", window=" << target_kv_cache_cap.draft_swa_window();
  }
  draft_kv_cache_cap.n_blocks() =
      draft_swa_count > 0 ? draft_swa_count : n_blocks;
  draft_kv_cache_cap.cache_size_in_bytes() = kv_cache_size;
  return allocate_target_cache(target_kv_cache_cap) &&
         draft_engine_->allocate_kv_cache(draft_kv_cache_cap,
                                          std::move(transfer_coordinator));
}

template <typename TargetEngine>
bool SpeculativeEngineBase<TargetEngine>::should_skip_external_draft_kv_cache()
    const {
  if (!use_draft_engine_ || draft_engine_ == nullptr) {
    return false;
  }
  return uses_embedded_eagle3_draft(options_.speculative_algorithm(),
                                    model_args_);
}

// TODO: support dp batches later
template <typename TargetEngine>
ForwardOutput SpeculativeEngineBase<TargetEngine>::step(BatchGroup& batches) {
  return engine_->step(batches);
}

template <typename TargetEngine>
int64_t SpeculativeEngineBase<TargetEngine>::calculate_kv_cache(
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
      target_kv_cache_cap.linear_cache_size_in_bytes() +
      draft_kv_cache_cap.linear_cache_size_in_bytes();
  CHECK_GT(cache_size_in_bytes, linear_cache_size_in_bytes)
      << "no memory left for speculative full-attention kv cache after "
         "reserving target and draft state caches, cache_size: "
      << cache_size_in_bytes
      << ", linear_cache_size: " << linear_cache_size_in_bytes;

  CHECK_GT(target_kv_cache_cap.slot_size() +
               target_kv_cache_cap.index_slot_size() +
               target_kv_cache_cap.scale_slot_size(),
           0)
      << "target full-attention kv cache slot size must be greater than 0";
  CHECK_GT(draft_kv_cache_cap.slot_size() +
               draft_kv_cache_cap.index_slot_size() +
               draft_kv_cache_cap.scale_slot_size(),
           0)
      << "draft full-attention kv cache slot size must be greater than 0";

  const int64_t target_full_attention_layers =
      std::max<int64_t>(target_kv_cache_cap.num_full_attention_layers(), 1);
  // Draft model has no linear-attention layers in the current MTP/Eagle path.
  const int64_t draft_full_attention_layers = draft_kv_cache_cap.n_layers();
  const int64_t target_full_attention_block_size_in_bytes =
      block_size * (target_full_attention_layers *
                    (target_kv_cache_cap.slot_size() +
                     target_kv_cache_cap.scale_slot_size())) +
      target_kv_cache_cap.num_indexer_layers() *
          target_kv_cache_cap.index_block_bytes();
  // The draft KV cache is allocated from the draft's own KVCacheShape, so its
  // per-block cost uses the draft's own slot size, which may exceed the
  // target's (e.g. a full-attention draft against an MLA target).
  const int64_t draft_full_attention_block_size_in_bytes =
      block_size * (draft_full_attention_layers *
                    (draft_kv_cache_cap.slot_size() +
                     draft_kv_cache_cap.scale_slot_size())) +
      draft_kv_cache_cap.num_indexer_layers() *
          draft_kv_cache_cap.index_block_bytes();
  const int32_t layerwise_split_size =
      options_.is_draft_engine()
          ? 1
          : ParallelConfig::get_instance().layerwise_split_size();
  const int64_t draft_swa_count = draft_cache_plan_.num_blocks;
  if (draft_swa_count > 0) {
    CHECK_EQ(layerwise_split_size, 1);
    const int64_t remaining_bytes =
        cache_size_in_bytes - linear_cache_size_in_bytes -
        draft_swa_count * draft_full_attention_block_size_in_bytes;
    CHECK_GT(remaining_bytes, 0) << "no KV memory after reserving draft SWA";
    return remaining_bytes / target_full_attention_block_size_in_bytes;
  }
  if (layerwise_split_size > 1) {
    return estimate_layerwise_split_block_count(
        model_args_,
        layerwise_split_size,
        target_kv_cache_cap,
        cache_size_in_bytes - linear_cache_size_in_bytes,
        draft_full_attention_block_size_in_bytes);
  }
  const int64_t full_attention_block_size_in_bytes =
      target_full_attention_block_size_in_bytes +
      draft_full_attention_block_size_in_bytes;
  CHECK_GT(full_attention_block_size_in_bytes, 0)
      << "speculative kv cache block size in bytes must be greater than 0";

  return (cache_size_in_bytes - linear_cache_size_in_bytes) /
         full_attention_block_size_in_bytes;
}

template <typename TargetEngine>
std::vector<RawForwardOutput>
SpeculativeEngineBase<TargetEngine>::collect_step_outputs(
    const BatchGroup& batch) {
  return engine_->collect_step_outputs(batch);
}

template class SpeculativeEngineBase<LLMEngine>;
template class SpeculativeEngineBase<VLMEngine>;

}  // namespace xllm
