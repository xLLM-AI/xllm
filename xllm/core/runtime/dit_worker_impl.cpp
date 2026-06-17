/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "dit_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/dit_model_loader.h"
#include "core/platform/device.h"
#include "framework/dit_cache/dit_cache.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {
DiTCacheConfig parse_dit_cache_from_flags() {
  DiTCacheConfig cache_config;
  if (::xllm::DiTConfig::get_instance().dit_cache_policy() == "FBCache") {
    cache_config.selected_policy = PolicyType::FBCache;
    cache_config.fbcache.warmup_steps =
        ::xllm::DiTConfig::get_instance().dit_cache_warmup_steps();
    cache_config.fbcache.residual_diff_threshold =
        ::xllm::DiTConfig::get_instance().dit_cache_residual_diff_threshold();
  } else if (::xllm::DiTConfig::get_instance().dit_cache_policy() ==
             "TaylorSeer") {
    cache_config.selected_policy = PolicyType::TaylorSeer;
    cache_config.taylorseer.n_derivatives =
        ::xllm::DiTConfig::get_instance().dit_cache_n_derivatives();
    cache_config.taylorseer.skip_interval_steps =
        ::xllm::DiTConfig::get_instance().dit_cache_skip_interval_steps();
    cache_config.taylorseer.warmup_steps =
        ::xllm::DiTConfig::get_instance().dit_cache_warmup_steps();
  } else if (::xllm::DiTConfig::get_instance().dit_cache_policy() ==
             "FBCacheTaylorSeer") {
    cache_config.selected_policy = PolicyType::FBCacheTaylorSeer;
    cache_config.fbcachetaylorseer.n_derivatives =
        ::xllm::DiTConfig::get_instance().dit_cache_n_derivatives();
    cache_config.fbcachetaylorseer.warmup_steps =
        ::xllm::DiTConfig::get_instance().dit_cache_warmup_steps();
    cache_config.fbcachetaylorseer.residual_diff_threshold =
        ::xllm::DiTConfig::get_instance().dit_cache_residual_diff_threshold();
  } else if (::xllm::DiTConfig::get_instance().dit_cache_policy() ==
             "ResidualCache") {
    cache_config.selected_policy = PolicyType::ResidualCache;
    cache_config.residual_cache.dit_cache_start_steps =
        ::xllm::DiTConfig::get_instance().dit_cache_start_steps();
    cache_config.residual_cache.dit_cache_end_steps =
        ::xllm::DiTConfig::get_instance().dit_cache_end_steps();
    cache_config.residual_cache.dit_cache_start_blocks =
        ::xllm::DiTConfig::get_instance().dit_cache_start_blocks();
    cache_config.residual_cache.dit_cache_end_blocks =
        ::xllm::DiTConfig::get_instance().dit_cache_end_blocks();
    cache_config.residual_cache.skip_interval_steps =
        ::xllm::DiTConfig::get_instance().dit_cache_skip_interval_steps();
  } else if (::xllm::DiTConfig::get_instance().dit_cache_policy() == "None") {
    cache_config.selected_policy = PolicyType::None;
  }
  return cache_config;
}
}  // namespace

DiTWorkerImpl::DiTWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
}

bool DiTWorkerImpl::init_model(ModelContext& context) {
  LOG(ERROR)
      << "init model with model_context was not implemented for dit models";
  return false;
}

bool DiTWorkerImpl::init_model(const std::string& model_weights_path,
                               int32_t random_seed,
                               MasterStatus master_status) {
  CHECK(dit_model_ == nullptr) << "Model is already initialized.";

  // set same random seed for all worker
  device_.set_seed(random_seed);

  auto loader = std::make_unique<DiTModelLoader>(model_weights_path);
  dtype_ = util::parse_dtype(loader->get_torch_dtype(), device_);

  auto tensor_options = torch::dtype(dtype_).device(device_);
  DiTCacheConfig cache_config = parse_dit_cache_from_flags();

  auto model_type = loader->get_model_type();

  if (!ModelRegistry::has_dit_model_factory(model_type)) {
    LOG(WARNING) << "could not find model_type: " << model_type
                 << ", using model_id: " << options_.model_id() << " instead.";
    model_type = options_.model_id();
  }

  dit_context_ = DiTModelContext(parallel_args_,
                                 std::move(loader->get_model_args()),
                                 std::move(loader->get_quant_args()),
                                 tensor_options,
                                 cache_config,
                                 model_type);

  dit_model_ = create_dit_model(dit_context_);
  CHECK(dit_model_ != nullptr) << "Failed to create model.";
  dit_model_->load_model(std::move(loader));

  dit_model_executor_ =
      std::make_unique<DiTExecutor>(dit_model_.get(), options_);

  DiTCache::get_instance().init(cache_config);

  return true;
}

folly::SemiFuture<bool> DiTWorkerImpl::init_model_async(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  auto promise = std::make_shared<folly::Promise<bool>>();
  auto future = promise->getSemiFuture();
  threadpool_.schedule([this,
                        model_weights_path,
                        random_seed,
                        master_status,
                        promise]() mutable {
    bool status =
        this->init_model(model_weights_path, random_seed, master_status);
    promise->setValue(status);
  });
  return future;
}

std::optional<ForwardOutput> DiTWorkerImpl::step(const ForwardInput& inputs) {
  torch::DeviceGuard device_guard(device_);
  Timer timer;
  auto output = dit_model_executor_->forward(
      inputs.input_params.dit_forward_input.to(device_, dtype_));

  auto ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  ForwardOutput forward_output;
  forward_output.dit_forward_output = output;
  return forward_output;
}

folly::SemiFuture<std::optional<ForwardOutput>> DiTWorkerImpl::step_async(
    const ForwardInput& inputs) {
  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        input = std::move(inputs),
                        promise = std::move(promise)]() mutable {
    auto output = this->step(input);
    promise.setValue(output);
  });
  return future;
}

void DiTWorkerImpl::process_group_test() {
  // create random tensors
  const auto options = torch::dtype(torch::kHalf).device(device_);
  torch::Tensor tensor = torch::randn({10, 10}, options);
  // There would be conflicts when using the DiT process group, as it doesn't
  // contain a HCCL/NCCL process_group, but uses HCCL communication functions
  // directly. In the new framework of DiT, this will be solved.
  // Temporarily, we won't call communication functions.
  // The following communication functions are temporarily disabled.

  // call allreduce
  // parallel_state::reduce(tensor,
  // context_.get_parallel_args().process_group_);
  // call allgather
  // parallel_state::gather(tensor,
  // context_.get_parallel_args().process_group_);
}

folly::SemiFuture<folly::Unit> DiTWorkerImpl::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    this->process_group_test();
    promise.setValue();
  });
  return future;
}

// prepare input for execution
DiTForwardInput DiTWorkerImpl::prepare_inputs(DiTBatch& batch) {
  return dit_model_executor_->prepare_inputs(batch);
}

int64_t DiTWorkerImpl::get_active_activation_memory() {
  return DeviceMonitor::get_instance()
      .get_device_stats(device_.index())
      .active_activation_memory;
}

}  // namespace xllm
