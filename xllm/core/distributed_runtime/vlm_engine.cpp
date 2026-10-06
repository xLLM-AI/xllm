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
#include "core/common/global_flags.h"
#include "core/distributed_runtime/master.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/prefix_cache/block_hasher.h"
#include "framework/block/kv_cache_manager_factory.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "models/model_registry.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/params_utils.h"
#include "runtime/vlm_forward_params.h"
#include "runtime/worker.h"
#include "util/env_var.h"
#include "util/pretty_print.h"
#include "util/tensor_helper.h"
#include "util/utils.h"
namespace xllm {

VLMEngine::VLMEngine(
    const runtime::Options& options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager)
    : options_(options),
      distributed_worker_manager_(std::move(distributed_worker_manager)) {
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
#if defined(USE_NPU)
  FLAGS_enable_atb_comm_multiprocess =
      options.enable_offline_inference() || (options.nnodes() > 1);
#endif

  // setup all workers and create worker clients in nnode_rank=0 engine side.
  setup_workers(options);

  dp_size_ = options_.dp_size();
  worker_clients_num_ = worker_clients_.size();
  dp_local_tp_size_ = worker_clients_num_ / dp_size_;

  process_group_test();

  // init thread pool
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/16,
      /*cpu_binding=*/false,
      /*pool_name=*/"VLMEngine.forward_input");
}

void VLMEngine::process_group_test() {
#if !defined(USE_NPU)
  // In multi-node serving mode, only driver engine
  // create worker_clients_.
  if (worker_clients_num_ > 1) {
    // test process group
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(worker_clients_num_);
    for (auto& worker : worker_clients_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    // Wait for all futures to complete with a configurable timeout.
    // The timeout can be adjusted via the
    // XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS environment variable (default: 4
    // seconds). This is particularly important in multi-node multi-device
    // communication scenarios where network latency may require a longer
    // timeout period.
    const int timeout_seconds = util::get_process_group_test_timeout_seconds();
    folly::collectAll(futures)
        .within(std::chrono::seconds(timeout_seconds))
        .get();
  }
#endif
}

bool VLMEngine::init(MasterStatus master_status) {
  if (!init_model(master_status)) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
  }

  auto kv_cache_cap = estimate_kv_cache_capacity();

  if (!allocate_kv_cache(kv_cache_cap)) {
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
  quant_args_ = model_loader->quant_args();
  tokenizer_args_ = model_loader->tokenizer_args();

  dtype_ = util::parse_dtype(args_.dtype(), options_.devices()[0]);
  LOG(INFO) << "Block info, block_size: " << options_.block_size()
            << ", n_layers: " << args_.n_layers() << ", dtype: " << dtype_;

  const int64_t tokenizer_vocab_size =
      static_cast<int64_t>(tokenizer_->vocab_size());
  int64_t model_vocab_size = args_.vocab_size();
  if (tokenizer_vocab_size != model_vocab_size) {
    // use tokenizer vocab size if model vocab size is not set
    if (model_vocab_size <= 0) {
      LOG(WARNING) << "Model vocab size is not set, using tokenizer vocab "
                      "size: "
                   << tokenizer_vocab_size;
      args_.vocab_size(tokenizer_vocab_size);
    } else if (tokenizer_vocab_size > model_vocab_size) {
      LOG(WARNING) << "Unsafe vocab mismatch: tokenizer: "
                   << tokenizer_vocab_size << ", model: " << model_vocab_size;
    } else {
      LOG(INFO) << "Tokenizer/model vocab differ: tokenizer="
                << tokenizer_vocab_size << ", model=" << model_vocab_size;
    }
  }

  LOG(INFO) << "Initializing model with " << args_;
  LOG(INFO) << "Initializing model with quant args: " << quant_args_;
  LOG(INFO) << "Initializing model with tokenizer args: " << tokenizer_args_;
  LOG(INFO) << "Initializing model with random seed: "
            << ::xllm::ExecutionConfig::get_instance().random_seed();

  // init model for each worker in parallel
  // multiple workers, call async init
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->init_model_async(
        model_path,
        ::xllm::ExecutionConfig::get_instance().random_seed(),
        MasterStatus::WAKEUP));
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

KVCacheCapacity VLMEngine::estimate_kv_cache_capacity() {
  return KVCacheManagerFactory::estimate_capacity(args_,
                                                  options_,
                                                  dtype_,
                                                  dp_local_tp_size_,
                                                  worker_clients_,
                                                  /*is_multimodal=*/true);
}

bool VLMEngine::allocate_kv_cache(const KVCacheCapacity& kv_cache_cap) {
  LOG(INFO) << "kv cache capacity: "
            << readable_size(kv_cache_cap.cache_size_in_bytes())
            << ", blocks: " << kv_cache_cap.n_blocks()
            << ", slot_size: " << kv_cache_cap.slot_size()
            << ", index_slot_size: " << kv_cache_cap.index_slot_size()
            << ", indexer_layers: " << kv_cache_cap.num_indexer_layers()
            << ", scale_slot_size: " << kv_cache_cap.scale_slot_size()
            << ", linear_slot_size: " << kv_cache_cap.linear_slot_size()
            << ", linear_blocks: " << kv_cache_cap.num_linear_state_blocks()
            << ", linear_state_slots: "
            << (has_linear_attention_layers(args_)
                    ? kv_cache_cap.num_linear_state_blocks()
                    : 0)
            << ", max_linear_state_cache_slots: "
            << options_.max_linear_state_cache_slots()
            << ", reserved_linear_bytes: "
            << readable_size(kv_cache_cap.linear_cache_size_in_bytes())
            << ", n_layers: " << kv_cache_cap.n_layers()
            << ", kv_cache_dtype: " << options_.kv_cache_dtype();

  const int32_t block_size = static_cast<int32_t>(kv_cache_cap.block_size());
  const bool enable_linear_attention = has_linear_attention_layers(args_);

  if (options_.enable_prefix_cache() && enable_linear_attention) {
    const auto& scheduler_config = ::xllm::SchedulerConfig::get_instance();
    CHECK(scheduler_config.enable_chunked_prefill())
        << "Linear-attention prefix cache requires block-aligned chunked "
           "prefill to save matching linear states. Please set "
           "--enable_chunked_prefill=true in your config.";
    CHECK(scheduler_config.max_tokens_per_chunk_for_prefill() % block_size == 0)
        << "linear-attention prefix cache saves linear-state checkpoints at "
           "chunk-end boundaries, so max_tokens_per_chunk_for_prefill ("
        << scheduler_config.max_tokens_per_chunk_for_prefill()
        << ") must be a multiple of block_size (" << block_size << ").";
  }

  // initialize block manager
  BlockManagerPool::Options options;
  options.num_blocks(kv_cache_cap.n_blocks())
      .host_num_blocks(0)  // no host cache for vlm engine currently.
      .block_size(block_size)
      .enable_linear_state(enable_linear_attention)
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .hasher_type(mtp_hasher_type(BlockHasherType::MM,
                                   options_.num_speculative_tokens(),
                                   options_.speculative_algorithm()))
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .num_speculative_tokens(options_.num_speculative_tokens())
      // DECODE-side prefix cache participation is per-leaf and gated by the
      // predicate in composite_block_manager.cpp; mirror llm_engine so a
      // linear-attention VLM decode instance disables the LINEAR prefix cache.
      .instance_is_decode(options_.instance_role() == InstanceRole::DECODE);
  if (enable_linear_attention) {
    // The unified linear-state slot pool spans all physical slots [0, N);
    // id 0 is reserved as padding and ids [1, N) serve live and checkpoint
    // rows interchangeably under reference counting.
    options.linear_state_num_slots(
        static_cast<int32_t>(kv_cache_cap.num_linear_state_blocks()));
  }
  auto factory_result = KVCacheManagerFactory::create(kv_cache_cap,
                                                      args_,
                                                      dp_local_tp_size_,
                                                      std::move(options),
                                                      this,
                                                      dp_size_);
  factory_result.shape.print_shapes();
  KVCacheShape kv_cache_shape = std::move(factory_result.shape);
  kv_cache_manager_ = std::move(factory_result.manager);

  // init kv cache for each worker in parallel
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(worker_clients_.size());
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->allocate_kv_cache_async(kv_cache_shape));
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

ForwardOutput VLMEngine::step(BatchGroup& batch) {
  if (worker_clients_.empty()) {
    // empty worker, return
    return {};
  }
  Timer timer;
  DCHECK(dp_size_ == batch.size())
      << "Split DP batch failed with dp_size as " << dp_size_
      << " and actual batch size as " << batch.size() << ".";

  auto forward_inputs = prepare_inputs(batch);

  DCHECK(dp_size_ == forward_inputs.size())
      << "The processed forward inputs size " << forward_inputs.size()
      << " is not equal to dp size " << dp_size_ << ".";

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);

  // update dp related global parameters and then execute model
  for (int32_t worker_rank = 0; worker_rank < worker_clients_num_;
       ++worker_rank) {
    int32_t dp_rank = worker_rank / dp_local_tp_size_;
    futures.emplace_back(worker_clients_[worker_rank]->step_remote_async(
        forward_inputs[dp_rank]));
  }

  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();

  assert(dp_size_ == worker_clients_num_ / dp_local_tp_size_);
  size_t dp_rank = 0;
  for (int32_t worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += dp_local_tp_size_) {
    auto result = results[worker_rank].value();
    const bool empty_shard = batch[dp_rank].size() == 0 &&
                             (!forward_inputs[dp_rank].token_ids.defined() ||
                              forward_inputs[dp_rank].token_ids.numel() == 0);
    if (empty_shard) {
      ++dp_rank;
      continue;
    }
    if (result.has_value()) {
      // if src_seq_idxes is not empty, skip sample output processing and
      // process beam search output instead
      if (result.value().src_seq_idxes.size() == 0) {
        // set second input param enable_schedule_overlap to false,
        // if it's not enabled, process_sample_output will append the real
        // token, if it's enabled, this false here will append the fake token in
        // process_sample_output
        batch[dp_rank].process_sample_output(result.value(), false);
      } else {
        batch[dp_rank].process_beam_search_output(result.value(), false);
      }
      // Keep Batch::sequences_ aligned with SequencesGroup after beam updates.
      batch[dp_rank].refresh_sequences_from_groups();
    } else {
      LOG(FATAL) << "Failed to execute model, result has no value";
    }
    ++dp_rank;
  }

  COUNTER_ADD(engine_latency_seconds, timer.elapsed_seconds());
  return {};
}

void VLMEngine::update_last_step_result(BatchGroup& last_batch) {
  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);
  std::vector<RawForwardOutput> raw_forward_outputs;
  raw_forward_outputs.reserve(dp_size_);

  // NOTE: We only need to get the output from the driver worker,
  // cause the output on other workers is the same as that on driver.
  // Under data parallelism (DP), we need to get dp_size outputs.
  // The `stride` means the workers num we can skip.
  int32_t stride = dp_local_tp_size_;

  for (int32_t worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += stride) {
    futures.emplace_back(
        worker_clients_[worker_rank]->get_last_step_result_async());
  }
  // wait for the all future to complete
  auto last_step_results = folly::collectAll(futures).get();

  for (int32_t worker_rank = 0; worker_rank < worker_clients_num_;
       worker_rank += dp_local_tp_size_) {
    auto result = last_step_results[worker_rank / stride].value();
    if (result.has_value()) {
      raw_forward_outputs.emplace_back(std::move(result.value()));
    } else {
      throw std::runtime_error("Failed to get last step results.");
    }
  }

  for (size_t i = 0; i < last_batch.size(); ++i) {
    last_batch[i].process_sample_output(raw_forward_outputs[i],
                                        options_.enable_schedule_overlap());
    // Keep Batch::sequences_ aligned with SequencesGroup after beam updates.
    last_batch[i].refresh_sequences_from_groups();
  }
}

void VLMEngine::setup_workers(const runtime::Options& options) {
  if (!distributed_worker_manager_) {
    distributed_worker_manager_ =
        std::make_shared<DistributedWorkerManager>(options);
  }
  worker_clients_ = distributed_worker_manager_->get_worker_clients();
}

std::vector<int64_t> VLMEngine::get_active_activation_memory() const {
  // call worker to get active activation memory
  std::vector<folly::SemiFuture<int64_t>> futures;
  futures.reserve(worker_clients_num_);
  for (auto& worker : worker_clients_) {
    futures.push_back(worker->get_active_activation_memory_async());
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  std::vector<int64_t> active_activation_memories;
  active_activation_memories.reserve(worker_clients_num_);
  for (auto& result : results) {
    active_activation_memories.push_back(result.value());
  }
  return active_activation_memories;
}

std::vector<VlmForwardInput> VLMEngine::prepare_inputs(BatchGroup& batch) {
  std::vector<VlmForwardInput> batched_inputs;
  batched_inputs.reserve(dp_size_);
  // some dp related variables
  std::vector<int32_t> dp_global_token_nums(dp_size_);
  std::vector<int32_t> dp_global_sequence_nums(dp_size_);
  std::vector<int32_t> dp_global_kv_max_seq_lens(dp_size_);
  std::vector<int32_t> dp_is_decode(dp_size_, 0);
  // when enable dp, we need to check the forward type of each batch
  // and set the empty forward type of each batch to the same value as the first
  // batch
  BatchForwardType batch_forward_type;

  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    if (batch[dp_rank].empty()) {
      // Use value-initialization to zero primitive fields for empty shard.
      VlmForwardInput empty_input;
      empty_input.input_params.meta.batch_forward_type = BatchForwardType();
      empty_input.input_params.meta.batch_id = UNINITIALIZED_BATCH_ID;
      batched_inputs.emplace_back(std::move(empty_input));
    } else {
      batched_inputs.emplace_back(std::move(
          batch[dp_rank].prepare_vlm_forward_input(args_, threadpool_.get())));
    }
    dp_global_token_nums[dp_rank] =
        static_cast<int32_t>(batched_inputs[dp_rank].host_token_ids().numel());
    dp_global_sequence_nums[dp_rank] =
        batched_inputs[dp_rank].input_params.meta.num_sequences;
    dp_global_kv_max_seq_lens[dp_rank] =
        batched_inputs[dp_rank].input_params.meta.kv_max_seq_len;
    if (batch_forward_type.is_empty() &&
        !batched_inputs[dp_rank]
             .input_params.meta.batch_forward_type.is_empty()) {
      batch_forward_type =
          batched_inputs[dp_rank].input_params.meta.batch_forward_type;
    }
    dp_is_decode[dp_rank] =
        batched_inputs[dp_rank]
            .input_params.meta.batch_forward_type.is_decode() &&
        batched_inputs[dp_rank].input_params.meta.q_max_seq_len == 1;
  }

  // Empty DP ranks inherit decode below and use fake inputs in WorkerImpl.
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      batch_forward_type.is_decode()) {
    for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
      if (batched_inputs[dp_rank]
              .input_params.meta.batch_forward_type.is_empty() &&
          dp_global_token_nums[dp_rank] == 0) {
        dp_is_decode[dp_rank] = 1;
      }
    }
  }

  // update dp_global_token_nums and batch_forward_type
  for (int32_t dp_rank = 0; dp_rank < dp_size_; ++dp_rank) {
    batched_inputs[dp_rank].input_params.parallel.dp_global_token_nums =
        dp_global_token_nums;
    batched_inputs[dp_rank].input_params.parallel.dp_global_sequence_nums =
        dp_global_sequence_nums;
    batched_inputs[dp_rank].input_params.parallel.raw_dp_global_token_nums =
        dp_global_token_nums;
    batched_inputs[dp_rank].input_params.parallel.dp_global_kv_max_seq_lens =
        dp_global_kv_max_seq_lens;
    batched_inputs[dp_rank].input_params.parallel.dp_is_decode = dp_is_decode;
    if (batched_inputs[dp_rank]
            .input_params.meta.batch_forward_type.is_empty()) {
      batched_inputs[dp_rank].input_params.meta.batch_forward_type =
          batch_forward_type;
    }
  }

  return batched_inputs;
}

}  // namespace xllm
