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

#include "worker.h"

#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/runtime/task_execution_pipeline.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "runtime/dflash2_worker_impl.h"
#include "runtime/dflash_worker_impl.h"
#include "runtime/dit_worker_impl.h"
#include "runtime/dspark_worker_impl.h"
#include "runtime/eagle3_worker_impl.h"
#include "runtime/embed_vlm_worker_impl.h"
#include "runtime/embed_worker_impl.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/mm_embed_vlm_worker_impl.h"
#include "runtime/mtp_worker_impl.h"
#include "runtime/rec_worker_impl.h"
#include "runtime/suffix_worker_impl.h"
#include "runtime/vlm_worker_impl.h"
#include "util/timer.h"

namespace xllm {
Worker::Worker(const ParallelArgs& parallel_args,
               const torch::Device& device,
               const runtime::Options& options,
               WorkerType worker_type)
    : enable_task_pipeline_(options.enable_task_pipeline()) {
  if (enable_task_pipeline_) {
    CHECK(ModelConfig::is_python_model_impl(
        ModelConfig::get_instance().model_impl()))
        << "Task pipeline requires the Python model implementation.";
    CHECK(worker_type == WorkerType::LLM && options.task_type() == "generate" &&
          !options.enable_graph() && !options.enable_speculative_decode() &&
          !options.enable_prefill_piecewise_graph() &&
          !options.enable_disagg_pd() && options.host_blocks_factor() <= 1.0 &&
          !options.enable_kvcache_store() &&
          !options.enable_offline_inference() &&
          !EPLBConfig::get_instance().enable_eplb() &&
          !KVCacheConfig::get_instance().enable_xtensor() &&
          !LoadConfig::get_instance().enable_rolling_load() &&
          parallel_args.cp_size() == 1 &&
          ParallelConfig::get_instance().kv_split_size_effective() == 1 &&
          ParallelConfig::get_instance().layerwise_split_size() == 1)
        << "Task pipeline requires ordinary Python LLM with CP/KV/layerwise "
           "splits of one, without offload or disaggregation.";
  }
  if (options.enable_speculative_decode()) {
    const std::string& algorithm = options.speculative_algorithm();
    LOG(INFO) << "Speculative decode is enabled, algorithm: " << algorithm;
    if (algorithm == "Eagle3") {
      impl_ = new Eagle3WorkerImpl(parallel_args, device, options, worker_type);
    } else if (algorithm == "DFlash") {
      impl_ = new DFlashWorkerImpl(parallel_args, device, options);
    } else if (SpeculativeConfig::is_dflash2_algorithm(algorithm)) {
#if !defined(USE_NPU)
      LOG(FATAL) << "DFlash2 speculative decoding is only supported on NPU.";
#endif
      impl_ = new DFlash2WorkerImpl(parallel_args, device, options);
    } else if (algorithm == "DSpark") {
      impl_ = new DSparkWorkerImpl(parallel_args, device, options);
    } else if (algorithm == "Suffix") {
      impl_ = new SuffixWorkerImpl(parallel_args, device, options);
    } else if (SpeculativeConfig::is_mtp_algorithm(algorithm)) {
      impl_ = new MTPWorkerImpl(parallel_args, device, options, worker_type);
    } else {
      LOG(FATAL) << "Unsupported speculative decoding algorithm: " << algorithm;
    }
  } else if (worker_type == WorkerType::LLM) {
    impl_ = new LLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::VLM) {
    impl_ = new VLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::ELM) {
    impl_ = new EmbedWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::EVLM) {
    impl_ = new EmbedVLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::REC) {
    impl_ = new RecWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::MMEVLM) {
    impl_ = new MMEmbedVLMWorkerImpl(parallel_args, device, options);
  } else if (worker_type == WorkerType::DIT) {
    impl_ = new DiTWorkerImpl(parallel_args, device, options);
  } else {
    LOG(ERROR) << "Unknown worker type, please check logic";
  }
}

Worker::~Worker() {
  if (enable_task_pipeline_) {
    folly::Promise<folly::Unit> promise;
    auto future = promise.getFuture();
    threadpool_.schedule([promise = std::move(promise)]() mutable {
      promise.setValue(folly::unit);
    });
    std::move(future).get();
    task_pipeline_.reset();
  }
  delete impl_;
}

bool Worker::initialize_task_pipeline() {
  if (!enable_task_pipeline_) {
    return true;
  }
  CHECK(task_pipeline_ == nullptr);
  const Status status = impl_->create_task_pipeline(task_pipeline_);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
    return false;
  }
  return true;
}

bool Worker::init_model(const std::string& model_weights_path,
                        int32_t random_seed,
                        MasterStatus master_status) {
  CHECK(!enable_task_pipeline_ || master_status == MasterStatus::WAKEUP);
  return impl_->init_model(model_weights_path, random_seed, master_status) &&
         initialize_task_pipeline();
}

bool Worker::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache(kv_cache_shape);
}

bool Worker::set_speculative_validate_time_predictor(
    const SpeculativeProfileRegistry::ValidateTimePredictor& predictor) {
  SpeculativeProfileRegistry::get_instance().set_validate_time_predictor(
      predictor);
  return true;
}

void Worker::get_cache_info(uint64_t& cluster_id,
                            std::string& addr,
                            uint16_t& port) {
  impl_->get_cache_info(cluster_id, addr, port);
}

bool Worker::link_cluster(const std::vector<uint64_t>& cluster_ids,
                          const std::vector<std::string>& addrs,
                          const std::vector<uint16_t>& ports) {
  return impl_->link_cluster(cluster_ids, addrs, ports);
}

bool Worker::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<uint16_t>& ports) {
  return impl_->unlink_cluster(cluster_ids, addrs, ports);
}

bool Worker::link_p2p(const std::string& remote_addr) {
  return impl_->link_p2p(remote_addr);
}

bool Worker::unlink_p2p(const std::string& remote_addr) {
  return impl_->unlink_p2p(remote_addr);
}

std::tuple<int64_t, int64_t> Worker::estimate_kv_cache_capacity() {
  return impl_->estimate_kv_cache_capacity();
}

ForwardInput Worker::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

std::optional<ForwardOutput> Worker::step(const ForwardInput& inputs) {
  if (enable_task_pipeline_) {
    return std::move(step_async(inputs)).get();
  }
  return impl_->step(inputs);
}

const bool Worker::is_driver() { return impl_->is_driver(); }

folly::SemiFuture<std::tuple<int64_t, int64_t>>
Worker::estimate_kv_cache_capacity_async() {
  return impl_->estimate_kv_cache_capacity_async();
}

folly::SemiFuture<std::optional<ForwardOutput>> Worker::step_async(
    const ForwardInput& inputs) {
  if (enable_task_pipeline_) {
    CHECK(task_pipeline_ != nullptr);
    const TaskSubmission submission = task_pipeline_->submit(inputs);
    CHECK(submission.status.ok()) << submission.status.message();
    if (impl_->enable_schedule_overlap()) {
      // PrepareAck releases all caller views. GetLast consumes the FIFO later.
      return folly::makeSemiFuture(std::optional<ForwardOutput>{});
    }
    return task_pipeline_->take_result_async(submission.task_id)
        .thenValue([](TaskResult result) -> std::optional<ForwardOutput> {
          CHECK(result.status.ok()) << result.status.message();
          return std::move(result.output);
        })
        .semi();
  }
  return impl_->step_async(inputs);
}

folly::SemiFuture<folly::Unit> Worker::process_group_test_async() {
  return impl_->process_group_test_async();
}

// initialize model, cache manager. async call
folly::SemiFuture<bool> Worker::init_model_async(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  CHECK(!enable_task_pipeline_ || master_status == MasterStatus::WAKEUP);
  if (!enable_task_pipeline_) {
    return impl_->init_model_async(
        model_weights_path, random_seed, master_status);
  }
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        model_weights_path,
                        random_seed,
                        master_status,
                        promise = std::move(promise)]() mutable {
    const bool loaded =
        std::move(impl_->init_model_async(
                      model_weights_path, random_seed, master_status))
            .get();
    promise.setValue(loaded && initialize_task_pipeline());
  });
  return future;
}

folly::SemiFuture<bool> Worker::allocate_kv_cache_async(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache_async(kv_cache_shape);
}

folly::SemiFuture<bool> Worker::allocate_kv_cache_with_transfer_async(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache_with_transfer_async(kv_cache_shape);
}

folly::SemiFuture<bool> Worker::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  return impl_->pull_kv_blocks_async(src_cluster_id, src_addr, mappings);
}

uint32_t Worker::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  return impl_->transfer_kv_blocks(batch_id, block_transfer_info);
}

uint32_t Worker::transfer_kv_blocks(
    const uint64_t batch_id,
    Slice<BlockTransferInfo>& block_transfer_info) {
  return impl_->transfer_kv_blocks(batch_id, block_transfer_info);
}

std::vector<uint8_t> Worker::prefetch_kv_blocks(
    Slice<BlockTransferInfo>& block_transfer_info) {
  return impl_->prefetch_kv_blocks(block_transfer_info);
}

const torch::Device& Worker::device() const { return impl_->device(); }

folly::SemiFuture<std::optional<ForwardOutput>>
Worker::get_last_step_result_async() {
  if (enable_task_pipeline_) {
    CHECK(impl_->enable_schedule_overlap())
        << "Task results without scheduler overlap are returned by step_async.";
    CHECK(task_pipeline_ != nullptr);
    return task_pipeline_->take_result_async()
        .thenValue([](TaskResult result) -> std::optional<ForwardOutput> {
          CHECK(result.status.ok()) << result.status.message();
          return std::move(result.output);
        })
        .semi();
  }
  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(impl_->get_last_step_result());
  });
  return future;
}

int64_t Worker::get_active_activation_memory() {
  return impl_->get_active_activation_memory();
}

folly::SemiFuture<int64_t> Worker::get_active_activation_memory_async() {
  folly::Promise<int64_t> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(impl_->get_active_activation_memory());
  });
  return future;
}

bool Worker::sleep(MasterStatus master_status) {
  if (enable_task_pipeline_) {
    LOG(ERROR) << "Task pipeline resource rebuild is not supported yet.";
    return false;
  }
  return impl_->sleep(master_status);
}

bool Worker::wakeup(const WakeupOptions& options) {
  if (enable_task_pipeline_) {
    LOG(ERROR) << "Task pipeline resource rebuild is not supported yet.";
    return false;
  }
  return impl_->wakeup(options);
}

folly::SemiFuture<bool> Worker::wakeup_async(const WakeupOptions& options) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, options, promise = std::move(promise)]() mutable {
    promise.setValue(this->wakeup(options));
  });
  return future;
}

bool Worker::start_profile() { return impl_->start_profile(); }

bool Worker::stop_profile() { return impl_->stop_profile(); }

folly::SemiFuture<bool> Worker::start_profile_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(this->start_profile());
  });
  return future;
}

folly::SemiFuture<bool> Worker::stop_profile_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(this->stop_profile());
  });
  return future;
}
}  // namespace xllm
