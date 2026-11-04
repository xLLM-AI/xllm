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

#include "core/distributed_runtime/dit_engine.h"

#include <glog/logging.h>
#include <sys/sysinfo.h>

#include <utility>

#include "common/types.h"
#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/platform/device.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "runtime/worker.h"
#include "util/env_var.h"
#include "util/timer.h"

namespace xllm {
DiTForwardOutput select_dit_driver_output(
    std::vector<folly::Try<std::optional<RawForwardOutput>>>& results) {
  if (results.empty()) {
    throw DiTForwardExecutionError({StatusCode::UNKNOWN, "no DiT ranks"});
  }
  for (size_t rank = 0; rank < results.size(); ++rank) {
    try {
      const auto& result = results[rank].value();
      if (!result) {
        throw DiTForwardExecutionError(
            {StatusCode::UNKNOWN, "missing transport result"});
      }
      if (!result->dit_execution_status.ok()) {
        throw DiTForwardExecutionError(result->dit_execution_status);
      }
    } catch (const DiTForwardExecutionError& error) {
      throw DiTForwardExecutionError(
          {error.status().code(),
           "rank=" + std::to_string(rank) + ": " + error.what()});
    } catch (const std::exception& error) {
      throw DiTForwardExecutionError(
          {StatusCode::UNKNOWN,
           "rank=" + std::to_string(rank) + ": " + error.what()});
    } catch (...) {
      throw DiTForwardExecutionError(
          {StatusCode::UNKNOWN,
           "rank=" + std::to_string(rank) + ": unknown DiT execution error"});
    }
  }
  return std::move(results.front().value()->dit_forward_output);
}

DiTEngine::DiTEngine(
    const runtime::Options& options,
    std::shared_ptr<DistributedWorkerManager> distributed_worker_manager)
    : options_(options),
      distributed_worker_manager_(std::move(distributed_worker_manager)) {
  auto master_node_addr = options.master_node_addr().value_or("");
  CHECK(!master_node_addr.empty())
      << " DIT need to set master node addr, Please set --master_node_addr.";

  const auto& devices = options_.devices();
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (size_t i = 0; i < devices.size(); ++i) {
    CHECK(devices[i].type() == device_type)
        << "All devices should be the same type";
  }

  // setup all workers and create worker clients in nnode_rank=0 engine side.
  setup_workers(options);
  worker_clients_num_ = worker_clients_.size();

  forward_input_factory_ = std::make_unique<DiTForwardInputFactory>();
}

void DiTEngine::setup_workers(const runtime::Options& options) {
  if (!distributed_worker_manager_) {
    distributed_worker_manager_ =
        std::make_shared<DistributedWorkerManager>(options);
  }
  worker_clients_ = distributed_worker_manager_->get_worker_clients();
}

bool DiTEngine::init() {
  if (!init_model()) {
    LOG(ERROR) << "Failed to init model from: " << options_.model_path();
    return false;
  }
  return true;
}

bool DiTEngine::init_model() {
  const std::string& model_path = options_.model_path();

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
  LOG(INFO) << "All workers completed model initialization.";
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }

  LOG(INFO) << "All workers successfully initialized the model.";
  return true;
}

// TODO : change to ForwardOutput?
DiTForwardOutput DiTEngine::step(std::vector<DiTBatch>& batches) {
  if (worker_clients_.empty()) {
    throw DiTForwardExecutionError({StatusCode::UNKNOWN, "no DiT workers"});
  }

  Timer timer;
  auto forward_input = forward_input_factory_->create_input(batches[0]);
  COUNTER_ADD(prepare_input_latency_seconds, timer.elapsed_seconds());

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);

  for (auto worker_rank = 0; worker_rank < worker_clients_num_; ++worker_rank) {
    futures.emplace_back(
        worker_clients_[worker_rank]->step_remote_async(forward_input));
  }

  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();

  DiTForwardOutput output = select_dit_driver_output(results);
  const size_t request_count = batches[0].size();
  const bool valid_output =
      !output.encoded_media.empty()
          ? valid_dit_output_mode(output)
          : (!output.text_output.empty()
                 ? output.text_output.size() == request_count
                 : output.tensors.size() == request_count &&
                       (output.audio_tensors.empty() ||
                        output.audio_tensors.size() == output.tensors.size()));
  if (!valid_output) {
    throw DiTForwardExecutionError(
        {StatusCode::UNKNOWN,
         "DiT driver output does not match request count"});
  }
  batches[0].process_forward_output(output);
  return output;
}

}  // namespace xllm
