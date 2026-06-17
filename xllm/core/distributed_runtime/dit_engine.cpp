/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_engine.h"

#include <glog/logging.h>
#include <sys/sysinfo.h>

#include "common/device_monitor.h"
#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/distributed_runtime/master.h"
#include "core/framework/config/execution_config.h"
#include "core/platform/device.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "runtime/worker.h"
#include "util/env_var.h"
#include "util/timer.h"

namespace xllm {
DiTEngine::DiTEngine(const runtime::Options& options,
                     std::shared_ptr<DistManager> dist_manager)
    : options_(options), dist_manager_(dist_manager) {
  auto master_node_addr = options.master_node_addr().value_or("");
  CHECK(!master_node_addr.empty())
      << " DIT need to set master node addr, Please set --master_node_addr.";

  const auto& devices = options_.devices();
  // initialize device monitor
  DeviceMonitor::get_instance().initialize(devices);
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  CHECK(!devices[0].is_cpu()) << "CPU device is not supported";
  const auto device_type = devices[0].type();
  for (size_t i = 0; i < devices.size(); ++i) {
    CHECK(devices[i].type() == device_type)
        << "All devices should be the same type";

#if defined(USE_NPU)
    FLAGS_enable_atb_comm_multiprocess =
        options.enable_offline_inference() || (options.nnodes() > 1);
#endif
  }

  // setup all workers and create worker clients in nnode_rank=0 engine side.
  setup_workers(options);
  worker_clients_num_ = worker_clients_.size();

  // init thread pool
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/16,
      /*cpu_binding=*/false,
      /*pool_name=*/"DiTEngine.forward_input");
}

void DiTEngine::setup_workers(const runtime::Options& options) {
  if (!dist_manager_) {
    dist_manager_ = std::make_shared<DistManager>(options);
  }
  worker_clients_ = dist_manager_->get_worker_clients();
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
    // empty worker, return
    return {};
  }

  Timer timer;
  auto dit_forward_input = batches[0].prepare_forward_input();
  ForwardInput forward_input;
  forward_input.input_params.dit_forward_input = dit_forward_input;
  COUNTER_ADD(prepare_input_latency_seconds, timer.elapsed_seconds());

  std::vector<folly::SemiFuture<std::optional<RawForwardOutput>>> futures;
  futures.reserve(worker_clients_num_);

  for (auto worker_rank = 0; worker_rank < worker_clients_num_; ++worker_rank) {
    futures.emplace_back(
        worker_clients_[worker_rank]->step_remote_async(forward_input));
  }

  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();

  // return the result from the driver
  auto forward_output = results.front().value();
  DCHECK(forward_output.has_value()) << "Failed to execute model";
  batches[0].process_forward_output(forward_output.value().dit_forward_output);
  return forward_output.value().dit_forward_output;
}

std::vector<int64_t> DiTEngine::get_active_activation_memory() const {
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
}  // namespace xllm
