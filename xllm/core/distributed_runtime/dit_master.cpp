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

#include "dit_master.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <boost/algorithm/string.hpp>
#include <csignal>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/metrics.h"
#include "core/platform/device_name_utils.h"
#include "dit_engine.h"
#include "framework/request/dit_request.h"
#include "models/model_registry.h"
#include "scheduler/scheduler_factory.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {
volatile bool DiTAssistantMaster::running_ = false;

DiTMaster::DiTMaster(const Options& options)
    : Master(options, EngineType::DIT) {
  CHECK(engine_->init());

  DiTScheduler::Options scheduler_options;
  scheduler_options.max_request_per_batch(options.max_requests_per_batch())
      .disable_log_stats(options.disable_log_stats());

  scheduler_ = create_dit_scheduler(engine_.get(), scheduler_options);
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

    Timer timer;
    // verify the prompt
    if (!params.verify_params(callback)) {
      return;
    }
    DiTRequestState dit_state = DiTRequestState(params.input_params,
                                                params.generation_params,
                                                callback,
                                                nullptr,
                                                params.request_kind,
                                                call);
    auto request = std::make_shared<DiTRequest>(params.request_id,
                                                params.x_request_id,
                                                params.x_request_time,
                                                std::move(dit_state));

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void DiTMaster::handle_batch_request(std::vector<DiTRequestParams> params_vec,
                                     BatchDiTOutputCallback callback) {
  const size_t num_requests = params_vec.size();
  scheduler_->incr_pending_requests(num_requests);
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

DiTAssistantMaster::DiTAssistantMaster(const Options& options)
    : Master(options, EngineType::DIT) {
  // setup process workers
  auto master_node_addr = options_.master_node_addr().value_or("");
  // TODO: support local unix domain socket later.
  if (master_node_addr.empty()) {
    LOG(FATAL)
        << "MultiNodeEngine required master_node_addr, current value is empty.";
    return;
  }

  running_ = true;
}

DiTAssistantMaster::~DiTAssistantMaster() {
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void DiTAssistantMaster::run() {
  signal(SIGINT, DiTAssistantMaster::handle_signal);
  signal(SIGTERM, DiTAssistantMaster::handle_signal);

  loop_thread_ = std::thread([this]() {
    while (running_) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
}

}  // namespace xllm
