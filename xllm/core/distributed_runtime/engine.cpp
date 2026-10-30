/* Copyright 2026 The xLLM Authors.

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

#include "core/distributed_runtime/engine.h"

#include "core/distributed_runtime/worker_fanout.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "framework/block/kv_cache_manager_factory.h"
#include "framework/model_loader/model_loader.h"
#include "models/model_registry.h"
#include "runtime/spec_runtime_helpers.h"
#include "util/utils.h"

namespace xllm {

bool Engine::dispatch_model_load(const ModelLoader& model_loader,
                                 const ModelArgs& model_args,
                                 MasterStatus master_status,
                                 const ModelInitCallback& prepare_model) {
  if (prepare_model &&
      !prepare_model(model_loader,
                     model_args.n_layers(),
                     /*dp_size=*/static_cast<int32_t>(dp_size()),
                     static_cast<int32_t>(dp_local_tp_size_),
                     master_status)) {
    return false;
  }
  return load_model(
      worker_clients_, model_loader.model_weights_path(), master_status);
}

bool Engine::init_draft_model(const ModelInitCallback& prepare_model) {
  const std::string& draft_model_path = options_.draft_model_path().value();
  auto model_loader = ModelLoader::create(draft_model_path);
  ModelArgs draft_args = model_loader->model_args();
  ModelRegistry::configure_mtp_args(
      draft_args,
      options_.speculative_algorithm(),
      /*is_draft_model=*/true,
      ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl()));
  LOG(INFO) << "Initializing draft model from: " << draft_model_path;

  // Draft tokens are detokenized by the target, so the draft model needs no
  // tokenizer and no draft/target vocab-compatibility check.
  if (args_.max_position_embeddings() != draft_args.max_position_embeddings()) {
    LOG(WARNING) << "draft and target models have different max context "
                    "lengths, draft max_position_embeddings: "
                 << draft_args.max_position_embeddings()
                 << ", target max_position_embeddings: "
                 << args_.max_position_embeddings()
                 << ", using the minimum between them";
    args_.max_position_embeddings() = std::min(
        args_.max_position_embeddings(), draft_args.max_position_embeddings());
  }

  if (!dispatch_model_load(
          *model_loader, draft_args, MasterStatus::WAKEUP, prepare_model)) {
    return false;
  }
  draft_args_ = std::move(draft_args);
  return true;
}

runtime::Options Engine::draft_estimate_options() const {
  runtime::Options draft_options = options_;
  spec_detail::mark_spec_draft_options(draft_options);
  draft_options.model_path(options_.draft_model_path().value())
      .backend("llm")
      .num_decoding_tokens(1)
      .enable_speculative_decode(/*value=*/false)
      .enable_graph(/*value=*/false);
  return draft_options;
}

std::optional<KVCacheCapacity> Engine::init_draft_and_estimate_capacity(
    const ModelInitCallback& prepare_model) {
  if (!SpeculativeConfig::loads_draft_model(options_)) {
    return KVCacheManagerFactory::estimate_speculative_capacity(
        {args_, options_, dtype_},
        /*draft=*/std::nullopt,
        dp_local_tp_size_,
        worker_clients_,
        is_multimodal(),
        reserved_memory_bytes_);
  }
  if (!init_draft_model(prepare_model)) {
    return std::nullopt;
  }

  // Keep the per-model dtype parsing for mixed-dtype drafts.
  const runtime::Options draft_options = draft_estimate_options();
  return KVCacheManagerFactory::estimate_speculative_capacity(
      {args_, options_, dtype_},
      {{draft_args_,
        draft_options,
        util::parse_dtype(draft_args_.dtype(), options_.devices()[0])}},
      dp_local_tp_size_,
      worker_clients_,
      is_multimodal(),
      reserved_memory_bytes_);
}

void Engine::setup_workers() {
  if (!distributed_worker_manager_) {
    distributed_worker_manager_ = std::make_shared<DistributedWorkerManager>(
        spec_detail::make_speculative_worker_options(options_));
  }
  worker_clients_ = distributed_worker_manager_->get_worker_clients();
}

}  // namespace xllm
