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

#include "dit_engine.h"

#include <glog/logging.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <chrono>
#include <exception>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "common/device_monitor.h"
#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/distributed_runtime/master.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/execution_config.h"
#include "core/platform/device.h"
#include "distributed_runtime/comm_channel.h"
#include "distributed_runtime/remote_worker.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "runtime/worker.h"
#include "util/env_var.h"
#include "util/timer.h"

namespace xllm {

namespace {

int64_t monotonic_time_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

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
  setup_vae_workers();

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

void DiTEngine::setup_vae_workers() {
  const auto& config = DiTConfig::get_instance();
  if (config.dit_instance_role() != "dit") {
    return;
  }

  const std::string& model_id = options_.model_id();
  const bool is_flux_model = model_id == "flux" || model_id == "flux-dev" ||
                             model_id.find("flux-dev-") == 0;
  CHECK(is_flux_model)
      << "Separate DiT/VAE instances currently support Flux only, got model: "
      << model_id;

  CHECK(!config.dit_vae_service_addresses().empty())
      << "dit_vae_service_addresses must be set for a dit instance.";

  std::stringstream addresses(config.dit_vae_service_addresses());
  std::unordered_set<std::string> configured_addresses;
  std::string address;
  int32_t rank = 0;
  while (std::getline(addresses, address, ',')) {
    const size_t first = address.find_first_not_of(" \t\n\r");
    const size_t last = address.find_last_not_of(" \t\n\r");
    if (first == std::string::npos) {
      continue;
    }
    address = address.substr(first, last - first + 1);
    CHECK(configured_addresses.insert(address).second)
        << "Duplicate VAE service address: " << address;
    auto channel = std::make_unique<CommChannel>();
    CHECK(channel->init_brpc(address, config.dit_vae_request_timeout_ms()))
        << "Failed to connect to VAE service: " << address;
    auto worker_state = std::make_unique<VaeWorkerState>();
    worker_state->worker = std::make_shared<RemoteWorker>(
        rank++, address, options_.devices().front(), std::move(channel));
    vae_workers_.emplace_back(std::move(worker_state));
  }
  CHECK(!vae_workers_.empty())
      << "No valid VAE service address was configured.";
  const size_t route_seed = static_cast<size_t>(getpid()) ^
                            static_cast<size_t>(config.dit_worker_port());
  next_vae_worker_.store(route_seed, std::memory_order_relaxed);
  LOG(INFO) << "Configured " << vae_workers_.size()
            << " VAE service instance(s) for DiT routing.";
}

size_t DiTEngine::select_vae_worker(
    const std::vector<bool>& attempted_workers) {
  CHECK_EQ(attempted_workers.size(), vae_workers_.size());

  const size_t start_index =
      next_vae_worker_.fetch_add(1, std::memory_order_relaxed) %
      vae_workers_.size();
  for (size_t offset = 0; offset < vae_workers_.size(); ++offset) {
    const size_t worker_index = (start_index + offset) % vae_workers_.size();
    if (attempted_workers[worker_index]) {
      continue;
    }
    return worker_index;
  }

  LOG(FATAL) << "No untried VAE worker is available.";
  return 0;
}

DiTForwardOutput DiTEngine::decode_with_vae(
    const DiTForwardInput& input,
    const DiTForwardOutput& latent_output) {
  if (latent_output.tensors.empty()) {
    LOG(ERROR) << "DiT instance returned no latent tensors.";
    return {};
  }

  DiTForwardInput vae_input = input;
  vae_input.prompts.clear();
  vae_input.prompts_2.clear();
  vae_input.negative_prompts.clear();
  vae_input.negative_prompts_2.clear();
  vae_input.prompt_embeds = torch::Tensor();
  vae_input.pooled_prompt_embeds = torch::Tensor();
  vae_input.negative_prompt_embeds = torch::Tensor();
  vae_input.negative_pooled_prompt_embeds = torch::Tensor();
  vae_input.images = torch::Tensor();
  vae_input.images_list.clear();
  vae_input.mask_images = torch::Tensor();
  vae_input.control_image = torch::Tensor();
  vae_input.masked_image_latents = torch::Tensor();
  vae_input.last_images = torch::Tensor();
  if (latent_output.tensors.size() == 1) {
    vae_input.latents = latent_output.tensors.front();
  } else {
    vae_input.latents = torch::cat(latent_output.tensors, 0);
  }

  ForwardInput forward_input;
  forward_input.input_params.dit_forward_input = std::move(vae_input);
  std::vector<bool> attempted_workers(vae_workers_.size(), false);
  const bool debug_print = DiTConfig::get_instance().dit_debug_print();
  Timer decode_timer;
  for (size_t offset = 0; offset < vae_workers_.size(); ++offset) {
    const size_t worker_index = select_vae_worker(attempted_workers);
    attempted_workers[worker_index] = true;
    auto& worker_state = *vae_workers_[worker_index];
    const auto& config = DiTConfig::get_instance();
    const int64_t now_ms = monotonic_time_ms();
    if (!worker_state.healthy.load(std::memory_order_relaxed) &&
        now_ms <
            worker_state.next_health_check_ms.load(std::memory_order_relaxed)) {
      continue;
    }
    if (!worker_state.healthy.load(std::memory_order_relaxed) &&
        !worker_state.worker->check_health(
            config.dit_vae_health_check_timeout_ms())) {
      worker_state.next_health_check_ms.store(
          now_ms + config.dit_vae_health_check_interval_ms(),
          std::memory_order_relaxed);
      continue;
    }
    worker_state.healthy.store(true, std::memory_order_relaxed);
    worker_state.next_health_check_ms.store(0, std::memory_order_relaxed);
    Timer rpc_timer;
    std::optional<RawForwardOutput> result;
    try {
      result = vae_workers_[worker_index]
                   ->worker->step_remote_async(forward_input)
                   .get();
    } catch (const std::exception& exception) {
      worker_state.healthy.store(false, std::memory_order_relaxed);
      worker_state.next_health_check_ms.store(
          monotonic_time_ms() + config.dit_vae_health_check_interval_ms(),
          std::memory_order_relaxed);
      LOG(WARNING) << "VAE worker " << worker_index
                   << " threw while decoding latent output: "
                   << exception.what() << "; trying next worker.";
      continue;
    } catch (...) {
      worker_state.healthy.store(false, std::memory_order_relaxed);
      worker_state.next_health_check_ms.store(
          monotonic_time_ms() + config.dit_vae_health_check_interval_ms(),
          std::memory_order_relaxed);
      LOG(WARNING) << "VAE worker " << worker_index
                   << " threw an unknown exception while decoding latent "
                      "output; trying next worker.";
      continue;
    }
    if (!result.has_value()) {
      worker_state.healthy.store(false, std::memory_order_relaxed);
      worker_state.next_health_check_ms.store(
          monotonic_time_ms() + config.dit_vae_health_check_interval_ms(),
          std::memory_order_relaxed);
      LOG(WARNING) << "VAE worker " << worker_index
                   << " failed to decode latent output, trying next worker.";
      continue;
    }
    const auto& output = result->dit_forward_output;
    if (output.tensors.size() != input.batch_size) {
      worker_state.healthy.store(false, std::memory_order_relaxed);
      worker_state.next_health_check_ms.store(
          monotonic_time_ms() + config.dit_vae_health_check_interval_ms(),
          std::memory_order_relaxed);
      LOG(WARNING) << "VAE worker " << worker_index
                   << " returned an invalid tensor count: "
                   << output.tensors.size() << ", expected " << input.batch_size
                   << ".";
      continue;
    }
    if (debug_print) {
      LOG(INFO) << "VAE worker " << worker_index
                << " decode rpc latency: " << rpc_timer.elapsed_seconds()
                << " s, total latency: " << decode_timer.elapsed_seconds()
                << " s.";
    }
    return output;
  }

  LOG(ERROR) << "All VAE workers failed to decode latent output.";
  return {};
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
  for (const auto& result : results) {
    if (result.hasException() || !result.value().has_value()) {
      LOG(ERROR) << "At least one DiT worker failed to execute the request.";
      batches[0].process_forward_error(
          Status(StatusCode::UNAVAILABLE,
                 "A DiT worker failed to execute the request."));
      return {};
    }
  }
  auto forward_output = results.front().value();
  DiTForwardOutput output = forward_output.value().dit_forward_output;
  if (DiTConfig::get_instance().dit_instance_role() == "dit") {
    output = decode_with_vae(dit_forward_input, output);
    if (output.tensors.empty()) {
      batches[0].process_forward_error(
          Status(StatusCode::UNAVAILABLE,
                 "All configured VAE workers failed to decode the request."));
      return output;
    }
  }
  batches[0].process_forward_output(output);
  return output;
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
