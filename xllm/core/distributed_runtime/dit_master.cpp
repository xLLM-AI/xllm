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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/metrics.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config_validation.h"
#include "core/framework/model_loader/hf_model_config.h"
#include "core/util/tensor_helper.h"
#include "dit_engine.h"
#include "framework/request/dit_request.h"
#include "models/model_registry.h"
#include "runtime/options.h"
#include "scheduler/scheduler_factory.h"
#include "util/scope_guard.h"
#include "util/timer.h"

namespace xllm {
namespace {

const char* python_h3_request_error(const DiTRequestParams& params) {
  const auto& generation = params.generation_params;
  if (params.request_kind != DiTRequestKind::kVideo ||
      !params.input_params.media_sources.entries().empty() ||
      generation.num_videos_per_prompt != 1) {
    return "Python MiniMax-H3 supports T2VA with one video per request";
  }
  if (generation.audio_sampling_rate != 32000 || generation.num_frames <= 0 ||
      !std::isfinite(generation.video_fps) || generation.video_fps <= 0.0) {
    return "Python MiniMax-H3 requires 32 kHz audio, positive frames and fps";
  }
  // The serving checkpoint uses 17-frame clips and rounds up by at most 16.
  const double frames = static_cast<double>(generation.num_frames);
  if (frames / generation.video_fps < 5.0 ||
      (frames + 16.0) / generation.video_fps > 15.0) {
    return "Python MiniMax-H3 duration must be between 5 and 15 seconds";
  }
  const auto source =
      params.input_params.tensor_sources.get_named_tensor("prompt_embed");
  if (!source.has_value()) {
    return "Python MiniMax-H3 requires prompt_embed from the H3 encoder";
  }
  const auto& embedding = source->get().tensor;
  if (embedding.dim() != 2 || embedding.size(0) == 0 ||
      embedding.size(1) != 5120 || !embedding.is_floating_point()) {
    return "Python MiniMax-H3 requires a floating point [L, 5120] prompt_embed";
  }
  const auto* tags = get_tensor_parameter<std::vector<int64_t>>(
      source->get().parameters, "prompt_token_tags");
  if (tags == nullptr ||
      tags->size() != static_cast<size_t>(embedding.size(0)) ||
      !std::all_of(
          tags->begin(), tags->end(), [](int64_t tag) { return tag == 1; })) {
    return "Python MiniMax-H3 T2VA requires L text-only prompt_token_tags";
  }
  return nullptr;
}

}  // namespace

DiTMaster::DiTMaster(const Options& options) : Master(options) {
  const std::optional<std::string> cp_error = validate_context_parallel_config(
      options_, EngineType::DIT, /*model_type=*/"", options_.nnodes());
  CHECK(!cp_error.has_value()) << cp_error.value();
  validate_layerwise_split_size_startup_config(
      options_, /*model_type=*/"", options_.nnodes());
  CHECK(!options_.enable_disagg_pd())
      << "Disaggregated PD is not supported by the DiT engine.";
  CHECK(options_.host_blocks_factor() <= 1.0)
      << "Basic host KV cache offload is not supported by the DiT engine.";
  const std::string& model_impl = ModelConfig::get_instance().model_impl();
  if (ModelConfig::is_python_model_impl(model_impl)) {
    const std::string model_type =
        get_model_type(options_.model_path(), options_.backend());
    is_python_minimax_h3_ =
        model_type == "MiniMaxH3ModularPipeline" ||
        (options_.model_id() == "MiniMaxH3ModularPipeline" &&
         !ModelRegistry::has_dit_model_factory(model_type));
    if (is_python_minimax_h3_) {
      CHECK_EQ(options_.max_requests_per_batch(), 1);
    }
  }

  runtime::Options engine_options;
  engine_options.model_path(options_.model_path())
      .model_id(options_.model_id())
      .model_impl(model_impl)
      .extension_impl(options_.extension_impl())
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
      .max_tokens_per_chunk_for_prefill(
          options_.max_tokens_per_chunk_for_prefill())
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
    if (is_python_minimax_h3_) {
      if (const char* error = python_h3_request_error(params)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, error);
        return;
      }
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
