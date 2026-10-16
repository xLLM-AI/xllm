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

#include "dit_master.h"

#include <glog/logging.h>

#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/metrics.h"
#include "core/framework/config/parallel_config_validation.h"
#include "dit_engine.h"
#include "framework/request/dit_request.h"
#include "runtime/options.h"
#include "scheduler/scheduler_factory.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {
DiTMaster::DiTMaster(const Options& options) : Master(options) {
  const std::optional<std::string> cp_error = validate_context_parallel_config(
      options_, EngineType::DIT, /*model_type=*/"", options_.nnodes());
  CHECK(!cp_error.has_value()) << cp_error.value();
  validate_layerwise_split_size_startup_config(
      options_, /*model_type=*/"", options_.nnodes());
  CHECK(!options_.enable_task_pipeline())
      << "Task pipeline is only supported by the LLM master.";
  CHECK(!options_.enable_disagg_pd())
      << "Disaggregated PD is not supported by the DiT engine.";
  CHECK(options_.host_blocks_factor() <= 1.0)
      << "Basic host KV cache offload is not supported by the DiT engine.";

  runtime::Options engine_options;
  engine_options.model_path(options_.model_path())
      .model_id(options_.model_id())
      .devices(devices_)
      .backend(options_.backend())
      .max_memory_utilization(options_.max_memory_utilization())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .task_type(options_.task_type())
      .npu_kernel_backend(options_.npu_kernel_backend())
      .master_node_addr(options_.master_node_addr())
      .nnodes(options_.nnodes())
      .node_rank(options_.node_rank())
      .dp_size(options_.dp_size())
      .ep_size(options_.ep_size())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .enable_offline_inference(options_.enable_offline_inference())
      .disable_log_stats(options_.disable_log_stats())
      .spawn_worker_path(options_.spawn_worker_path())
      .enable_shm(options_.enable_shm())
      .input_shm_size(options_.input_shm_size() * 1024 * 1024)
      .output_shm_size(options_.output_shm_size() * 1024 * 1024)
      .is_local(options_.is_local())
      .tp_size(options_.tp_size())
      .sp_size(options_.sp_size())
      .cfg_size(options_.cfg_size())
      .vae_size(options_.vae_size())
      .text_encoder_tp_size(options_.text_encoder_tp_size());
  dit_engine_ = std::make_unique<DiTEngine>(engine_options);
  if (!is_leader()) {
    return;
  }

  CHECK(dit_engine_->init());

  DiTScheduler::Options scheduler_options;
  scheduler_options.max_request_per_batch(options.max_requests_per_batch())
      .disable_log_stats(options.disable_log_stats());

  scheduler_ = create_dit_scheduler(dit_engine_.get(), scheduler_options);
  LOG(INFO) << "created dit scheduler in DiTMaster.";

  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/options.num_request_handling_threads(),
      /*cpu_binding=*/false,
      /*pool_name=*/"DiTMaster.request");
  LOG(INFO) << "ThreadPool with " << options.num_request_handling_threads()
            << " threads created in DiTMaster.";
}

DiTMaster::~DiTMaster() {
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void DiTMaster::handle_request(DiTRequestParams params,
                               std::optional<Call*> call,
                               DiTOutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback)](const DiTRequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  // add into the queue
  threadpool_->schedule([this,
                         params = std::move(params),
                         callback = std::move(cb),
                         call]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // Guard the rate-limit slot acquired at the service entry. Dismissed
    // right before DiTRequest takes ownership; any early return releases it.
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });

    Timer timer;
    // verify the prompt
    if (!params.verify_params(callback)) {
      return;
    }
    DiTRequestState dit_state =
        DiTRequestState(std::move(params.input_params),
                        std::move(params.generation_params),
                        callback,
                        params.request_kind,
                        call);
    rate_limit_guard.dismiss();
    auto request = std::make_shared<DiTRequest>(params.request_id,
                                                params.x_request_id,
                                                params.x_request_time,
                                                std::move(dit_state),
                                                /*service_request_id=*/"",
                                                /*source_xservice_addr=*/"",
                                                get_rate_limiter());

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void DiTMaster::handle_batch_request(std::vector<DiTRequestParams> params_vec,
                                     BatchDiTOutputCallback callback) {
  const size_t num_requests = params_vec.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(params_vec[i]),
                   std::nullopt,
                   [i, callback](const DiTRequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void DiTMaster::run() {
  if (!is_leader()) {
    Master::run();
    return;
  }

  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "DiTMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      scheduler_->step(timeout);
    }
    LOG(INFO) << "DiTMaster loop thread exiting.";
    running_.store(false, std::memory_order_relaxed);
  });
}

void DiTMaster::generate() {
  LOG(INFO) << "into DiTMaster::generate";

  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "Generate is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  scheduler_->generate();
  running_.store(false, std::memory_order_relaxed);
}

}  // namespace xllm
