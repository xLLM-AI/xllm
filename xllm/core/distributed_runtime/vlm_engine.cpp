/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "vlm_engine.h"

#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/prefix/block_hasher.h"
#include "core/kv_cache/storage/kv_cache_utils.h"
#include "core/util/tensor_options.h"
#include "framework/block/kv_cache_manager_factory.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "platform/platform.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/vlm_forward_params.h"
#include "runtime/worker.h"
#include "util/env_var.h"
#include "util/tensor_helper.h"
#include "util/utils.h"
namespace xllm {

namespace {

void attach_embedding_output_metadata(const ModelArgs& args,
                                      const std::vector<Sequence*>& sequences,
                                      RawForwardOutput& output) {
  if (!ModelConfig::get_instance().enable_return_embedding_modality_tags()) {
    return;
  }
  if (ModelConfig::get_instance().task() == "mm_embed" ||
      !ModelConfig::get_instance().enable_return_mm_full_embeddings() ||
      output.outputs.size() != sequences.size()) {
    return;
  }

  for (size_t output_index = 0; output_index < output.outputs.size();
       ++output_index) {
    auto& sample_output = output.outputs[output_index];
    const auto& embeddings = sample_output.mm_embeddings;
    const auto& sequence = *sequences[output_index];
    if (embeddings.empty() ||
        !std::all_of(embeddings.begin(),
                     embeddings.end(),
                     [&sequence](const torch::Tensor& embedding) {
                       return embedding.defined() && embedding.dim() > 0 &&
                              embedding.size(0) ==
                                  static_cast<int64_t>(
                                      sequence.num_prompt_tokens());
                     })) {
      continue;
    }

    std::vector<int64_t> modality_tags;
    modality_tags.reserve(sequence.num_prompt_tokens());
    const auto tokens = sequence.tokens();
    for (size_t index = 0; index < sequence.num_prompt_tokens(); ++index) {
      const int32_t token_id = tokens[index];
      const bool is_vision_token = token_id == args.vision_start_token_id() ||
                                   token_id == args.vision_end_token_id() ||
                                   token_id == args.image_token_id() ||
                                   token_id == args.video_token_id();
      modality_tags.emplace_back(is_vision_token ? 0 : 1);
    }
    sample_output.mm_embeddings_metadata["token_tags"] =
        torch::tensor(modality_tags, torch::kInt64);
  }
}

}  // namespace

VLMEngine::VLMEngine(
    const runtime::Options& options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager)
    : Engine(options, std::move(distributed_worker_manager)) {
  auto master_node_addr = options.master_node_addr().value_or("");
  CHECK(!master_node_addr.empty())
      << " VLM need to set master node addr, Please set --master_node_addr.";
  const auto& devices = options_.devices();
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";
  }

  // setup all workers and create worker clients in nnode_rank=0 engine side.
  setup_workers();

  dp_local_tp_size_ = worker_clients_num() / dp_size();
  process_group_test();

  forward_input_factory_ =
      std::make_unique<VlmForwardInputFactory>(VlmForwardInputFactoryOptions{
          .dp_size = dp_size(),
          .enable_dp_global_json_object_active =
              dp_size() > 1 && options_.num_speculative_tokens() > 0});
}

bool VLMEngine::init(MasterStatus master_status,
                     const ModelInitCallback& prepare_model) {
  // The residency callback belongs to the draft only: plain VLM targets never
  // reserve weight pages, and routing it to the target would double-reserve
  // the draft's model_id in the page budget.
  if (!init_model(master_status)) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
  }

  const auto kv_cache_cap = init_draft_and_estimate_capacity(prepare_model);
  if (!kv_cache_cap.has_value()) {
    return false;
  }

  if (!allocate_kv_cache(*kv_cache_cap)) {
    LOG(ERROR) << "Failed to allocate kv cache";
    return false;
  }

  return true;
}

bool VLMEngine::init_model(MasterStatus master_status) {
  const std::string& model_path = options_.model_path();
  auto model_loader = ModelLoader::create(model_path);
  LOG(INFO) << "Initializing model from: " << model_path;

  tokenizer_ = model_loader->tokenizer();
  CHECK(tokenizer_ != nullptr);

  args_ = model_loader->model_args();
  tokenizer_args_ = model_loader->tokenizer_args();

  dtype_ = util::parse_dtype(args_.dtype(), options_.devices()[0]);
  LOG(INFO) << "Block info, block_size: " << options_.block_size()
            << ", n_layers: " << args_.n_layers() << ", dtype: " << dtype_;

  ::xllm::reconcile_vocab_size(args_,
                               static_cast<int64_t>(tokenizer_->vocab_size()));

  LOG(INFO) << "Initializing model with " << args_;
  LOG(INFO) << "Initializing model with quant args: "
            << model_loader->quant_args();
  LOG(INFO) << "Initializing model with tokenizer args: " << tokenizer_args_;
  LOG(INFO) << "Initializing model with random seed: "
            << ::xllm::ExecutionConfig::get_instance().random_seed();

  return dispatch_model_load(
      *model_loader, args_, MasterStatus::WAKEUP, /*prepare_model=*/{});
}

bool VLMEngine::allocate_kv_cache(const KVCacheCapacity& kv_cache_cap) {
  const bool enable_state_cache = has_linear_attention_layers(args_) ||
                                  kv_cache_cap.kpool_tail_slot_size() > 0;

  log_capacity_and_validate_chunked_prefill(kv_cache_cap, enable_state_cache);

  const int32_t block_size = static_cast<int32_t>(kv_cache_cap.block_size());

  // initialize block manager
  BlockManagerPool::Options options;
  options.num_blocks(kv_cache_cap.n_blocks())
      .kv_events_config(KVCacheConfig::get_instance().kv_events_config())
      .host_num_blocks(0)  // no host cache for vlm engine currently.
      .block_size(block_size)
      .enable_linear_state(enable_state_cache)
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_vmm_kvcache(KVCacheConfig::get_instance().enable_vmm_kvcache())
      .num_layers(args_.n_layers())
      .slot_size(kv_cache_cap.slot_size())
      .model_id(options_.model_id())
      .hasher_type(mtp_hasher_type(BlockHasherType::MM,
                                   options_.num_speculative_tokens(),
                                   options_.speculative_algorithm()))
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .num_speculative_tokens(options_.num_speculative_tokens())
      // DECODE-side prefix cache participation is per-leaf and gated by the
      // predicate in composite_block_manager.cpp; mirror llm_engine so a
      // linear-attention VLM decode instance disables the LINEAR prefix cache.
      .instance_is_decode(options_.instance_role() == InstanceRole::DECODE);
  if (enable_state_cache) {
    // The unified state slot pool spans all physical slots [0, N); it can
    // contain GDN state and KPool tail state.
    options.linear_state_num_slots(
        static_cast<int32_t>(kv_cache_cap.num_linear_state_blocks()));
  }
  auto factory_result =
      KVCacheManagerFactory::create(kv_cache_cap,
                                    args_,
                                    dp_local_tp_size_,
                                    dtype_,
                                    std::move(options),
                                    /*transfer_coordinator=*/nullptr,
                                    dp_size(),
                                    /*host_validation_options=*/std::nullopt);
  factory_result.shape.print_shapes();
  KVCacheShape kv_cache_shape = std::move(factory_result.shape);
  block_manager_pool_ = std::move(factory_result.manager);

  return allocate_kv_cache_on_workers(kv_cache_shape);
}

ForwardOutput VLMEngine::step(BatchGroup& batch) {
  if (worker_clients_.empty()) {
    // empty worker, return
    return {};
  }
  Timer timer;

  std::vector<VlmForwardInput> forward_inputs;
  forward_input_factory_->create_inputs(batch, args_, forward_inputs);

  DCHECK(dp_size() == forward_inputs.size())
      << "The processed forward inputs size " << forward_inputs.size()
      << " is not equal to dp size " << dp_size() << ".";

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  const int32_t num_workers = worker_clients_num();
  futures.reserve(num_workers);

  // update dp related global parameters and then execute model
  for (int32_t worker_rank = 0; worker_rank < num_workers; ++worker_rank) {
    int32_t dp_rank = worker_rank / dp_local_tp_size_;
    futures.emplace_back(worker_clients_[worker_rank]->step_remote_async(
        forward_inputs[dp_rank]));
  }

  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();

  assert(dp_size() == num_workers / dp_local_tp_size_);
  for (int32_t worker_rank = 0; worker_rank < num_workers;
       worker_rank += dp_local_tp_size_) {
    CHECK(results[worker_rank].hasValue())
        << "Failed to execute model, worker RPC failed";
  }
  std::vector<RawForwardOutput> outputs;
  outputs.reserve(dp_size());
  size_t dp_rank = 0;
  for (int32_t worker_rank = 0; worker_rank < num_workers;
       worker_rank += dp_local_tp_size_) {
    const bool empty_shard = batch[dp_rank].size() == 0 &&
                             (!forward_inputs[dp_rank].token_ids.defined() ||
                              forward_inputs[dp_rank].token_ids.numel() == 0);
    if (empty_shard) {
      outputs.emplace_back();
      ++dp_rank;
      continue;
    }
    CHECK(results[worker_rank].value().has_value())
        << "Failed to execute model, result has no value";
    auto& result = results[worker_rank].value();

    attach_embedding_output_metadata(
        args_, batch[dp_rank].get_sequences(), result.value());

    outputs.emplace_back(std::move(result.value()));
    ++dp_rank;
  }
  batch.apply_outputs(outputs,
                      options_.enable_schedule_overlap()
                          ? BatchOutputAction::RESERVE_PENDING
                          : BatchOutputAction::IMMEDIATE);

  COUNTER_ADD(engine_latency_seconds, timer.elapsed_seconds());
  return {};
}

std::vector<RawForwardOutput> VLMEngine::collect_step_outputs(
    const BatchGroup& batch) {
  CHECK_EQ(batch.size(), dp_size());
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  const int32_t num_workers = worker_clients_num();
  futures.reserve(num_workers);
  std::vector<RawForwardOutput> raw_forward_outputs;
  raw_forward_outputs.reserve(dp_size());

  // NOTE: We only need to get the output from the driver worker,
  // cause the output on other workers is the same as that on driver.
  // Under data parallelism (DP), we need to get dp_size outputs.
  // The `stride` means the workers num we can skip.
  int32_t stride = dp_local_tp_size_;
  for (int32_t worker_rank = 0; worker_rank < num_workers;
       worker_rank += stride) {
    futures.emplace_back(
        worker_clients_[worker_rank]->get_last_vlm_step_result_async());
  }
  // wait for the all future to complete
  auto last_step_results = folly::collectAll(futures).get();

  for (const auto& result : last_step_results) {
    CHECK(result.hasValue() && result.value().has_value())
        << "Failed to get last step results, result has no value";
  }

  for (int32_t worker_rank = 0; worker_rank < num_workers;
       worker_rank += dp_local_tp_size_) {
    auto& result = last_step_results[worker_rank / stride].value();
    raw_forward_outputs.emplace_back(std::move(result.value()));
  }

  for (size_t dp_rank = 0; dp_rank < batch.size(); ++dp_rank) {
    attach_embedding_output_metadata(
        args_, batch[dp_rank].get_sequences(), raw_forward_outputs[dp_rank]);
  }
  return raw_forward_outputs;
}

}  // namespace xllm
