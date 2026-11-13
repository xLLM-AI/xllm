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

#include "core/runtime/dit_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "core/common/types.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/model_loader/dit_model_loader.h"
#include "core/framework/request/minimax_h3_conditioning.h"
#include "core/platform/device.h"
#include "core/util/tensor_options.h"
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
  } else if (DiTConfig::get_instance().dit_cache_policy() == "CacheDiT") {
    const auto& config = DiTConfig::get_instance();
    if (config.dit_h3_execution_profile() != kMiniMaxH3ExecutionProfile) {
      throw std::invalid_argument(
          "CacheDiT requires the qualified MiniMax-H3 execution profile");
    }
    cache_config.selected_policy = PolicyType::CACHE_DIT;
    auto& options = cache_config.cache_dit;
    options.warmup_steps = config.dit_cachedit_warmup_steps();
    options.residual_diff_threshold =
        static_cast<float>(config.dit_cachedit_residual_diff_threshold());
    options.max_cached_steps = config.dit_cachedit_max_cached_steps();
    options.max_consecutive_hits = config.dit_cachedit_max_consecutive_hits();
    options.front_blocks = config.dit_cachedit_front_blocks();
    options.back_blocks = config.dit_cachedit_back_blocks();
    options.linear_residual_prediction =
        config.dit_cachedit_linear_residual_prediction();
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

  const DiTConfig& config = DiTConfig::get_instance();
  const bool qualified = config.dit_h3_execution_profile() != "native";
  if (config.dit_h3_rainfusion_enabled() && !qualified) {
    LOG(ERROR)
        << "MiniMax-H3 RainFusion requires the qualified execution profile";
    return false;
  }
  if (qualified) {
    if (LoadConfig::get_instance().enable_rolling_load()) {
      LOG(ERROR) << "Qualified MiniMax-H3 requires resident weights; "
                    "rolling load is unsupported";
      return false;
    }
#if !defined(USE_NPU)
    LOG(ERROR) << "MiniMax-H3 qualified execution requires an Ascend build";
    return false;
#else
    if (config.dit_h3_execution_profile() != kMiniMaxH3ExecutionProfile ||
        device_.unwrap().type() != torch::kPrivateUse1 ||
        config.dit_h3_task_type() != "ref2va" ||
        config.max_requests_per_batch() != 1) {
      LOG(ERROR) << "MiniMax-H3 qualified execution requires the A3 16-card "
                    "ref2va profile, an NPU device and batch limit 1";
      return false;
    }
    // Only the qualified CacheDiT policy is enabled at this stage.
    if ((config.dit_cache_policy() != "None" &&
         config.dit_cache_policy() != "CacheDiT") ||
        config.dit_sp_communication_overlap() !=
            config.dit_sp_qkv_communication_overlap() ||
        config.dit_sparse_attention_enabled() ||
        config.dit_laser_attention_enabled()) {
      LOG(ERROR) << "MiniMax-H3 requires cache=None/CacheDiT, paired QKV "
                    "overlap flags, and sparse/laser attention disabled";
      return false;
    }
#endif
  }

  // set same random seed for all worker
  device_.set_seed(random_seed);

  auto loader = std::make_unique<DiTModelLoader>(model_weights_path);
  dtype_ = util::parse_dtype(loader->get_torch_dtype(), device_);

  auto tensor_options = torch::dtype(dtype_).device(device_);
  DiTCacheConfig cache_config = parse_dit_cache_from_flags();

  auto model_type = loader->get_model_type();
  if (qualified && model_type != "MiniMaxH3ModularPipeline") {
    LOG(ERROR) << "MiniMax-H3 qualified execution requires a "
                  "MiniMaxH3ModularPipeline model index";
    return false;
  }

  if (!ModelRegistry::has_dit_model_factory(model_type)) {
    LOG(WARNING) << "could not find model_type: " << model_type
                 << ", using model_id: " << options_.model_id() << " instead.";
    model_type = options_.model_id();
  }

  // H3 was dense before cache integration. Preserve that default while
  // respecting explicit CLI and JSON policy choices, including TaylorSeer.
  const auto& json_config = config::get_parsed_json_config();
  if (model_type == "MiniMaxH3ModularPipeline" &&
      !config::is_flag_specified("dit_cache_policy") &&
      !(json_config &&
        json_config->value<std::string>("dit_cache_policy").has_value())) {
    cache_config.selected_policy = PolicyType::None;
  }

  dit_context_ = DiTModelContext(parallel_args_,
                                 loader->get_model_args(),
                                 loader->get_quant_args(),
                                 tensor_options,
                                 cache_config,
                                 model_type);

  dit_model_ = create_dit_model(dit_context_);
  CHECK(dit_model_ != nullptr) << "Failed to create model.";
  dit_model_->load_model(std::move(loader));

  dit_model_executor_ = std::make_unique<DiTExecutor>(dit_model_.get());

  DiTCache::get_instance().init(cache_config, parallel_args_);

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
    try {
      promise->setValue(
          this->init_model(model_weights_path, random_seed, master_status));
    } catch (...) {
      promise->setException(folly::exception_wrapper(std::current_exception()));
    }
  });
  return future;
}

std::optional<ForwardOutput> DiTWorkerImpl::step(
    const LlmForwardInput& /*inputs*/) {
  LOG(FATAL) << "DiT worker requires native DiTForwardInput";
  return std::nullopt;
}

std::optional<ForwardOutput> DiTWorkerImpl::step(
    const DiTForwardInput& inputs) {
  torch::DeviceGuard device_guard(device_);
  Timer timer;
  DiTForwardInput input_on_device = inputs.to(device_);
  DiTCache::get_instance().set_context({});
  DiTForwardOutput output =
      dit_model_executor_->forward(input_on_device.model_params);
  DiTCache::get_instance().set_context({});

  const int32_t ret = device_.synchronize_default_stream();
  if (ret != 0) {
    throw DiTForwardExecutionError(
        {StatusCode::UNKNOWN,
         "DiT device synchronization failed: " + std::to_string(ret)});
  }
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  // DiT currently does not support DP, so is_driver() identifies global rank 0.
  if (!is_driver()) {
    return std::nullopt;
  }

  ForwardOutput forward_output;
  forward_output.dit_forward_output = std::move(output);
  return forward_output;
}

folly::SemiFuture<std::optional<ForwardOutput>> DiTWorkerImpl::step_async(
    const DiTForwardInput& inputs) {
  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        input = inputs,
                        promise = std::move(promise)]() mutable {
    try {
      promise.setValue(this->step(input));
    } catch (...) {
      promise.setException(folly::exception_wrapper(std::current_exception()));
    }
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
    try {
      this->process_group_test();
      promise.setValue();
    } catch (...) {
      promise.setException(folly::exception_wrapper(std::current_exception()));
    }
  });
  return future;
}

// prepare input for execution

}  // namespace xllm
