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

#include "rec_master.h"

#include <absl/time/time.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "common/macros.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/framework/multimodal/mm_data.h"
#include "models/model_registry.h"
#include "rec_engine.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "util/rec_model_utils.h"
#include "util/scope_guard.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

namespace {

RecType get_rec_type(const ModelArgs& model_args) {
  const auto kind = get_rec_model_kind(model_args.model_type());
  switch (kind) {
    case RecModelKind::kOneRec:
      return RecType::kOneRec;
    case RecModelKind::kLlmRec:
      return RecType::kLlmRec;
    case RecModelKind::kNone:
      return RecType::kNone;
  }
  return RecType::kNone;
}

}  // namespace

RecMaster::RecMaster(const Options& options)
    : Master(options, EngineType::REC) {
  if (!is_leader()) {
    // RecEngine does not create DistManager in its constructor. LlmRec
    // starts workers in init(); skip that on non-leaders but still host
    // the local WorkerServer so rank 0 can collect the cluster.
    auto* rec_engine = dynamic_cast<RecEngine*>(engine_.get());
    CHECK(rec_engine != nullptr);
    rec_engine->setup_distributed_workers();
    return;
  }

  // Initialize with Rec engine type
  // The rest of the initialization follows the same pattern as LLMMaster
  CHECK(engine_->init());

  model_args_ = engine_->model_args();
  rec_type_ = get_rec_type(model_args_);
  if (rec_type_ == RecType::kNone) {
    LOG(ERROR) << "Unsupported rec model_type: " << model_args_.model_type();
  }

  if (options_.enable_service_routing()) {
    XServiceClient* xservice_client = XServiceClient::get_instance();
    if (!xservice_client->init(options_.etcd_addr().value_or(""),
                               options_.instance_name().value_or(""),
                               engine_->block_manager_pool(),
                               options_.etcd_namespace().value_or(""))) {
      LOG(FATAL) << "XServiceClient init fail!";
      return;
    }
  }

  ContinuousScheduler::Options scheduler_options;
  scheduler_options.max_tokens_per_batch(options_.max_tokens_per_batch())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options_.max_tokens_per_chunk_for_prefill())
      .num_speculative_tokens(options_.num_speculative_tokens())
      .dp_size(options_.dp_size())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing())
      .disable_log_stats(options_.disable_log_stats())
      .rec_worker_max_concurrency(options_.rec_worker_max_concurrency());
  scheduler_ = create_fixed_steps_scheduler(engine_.get(), scheduler_options);

  chat_template_ = nullptr;
  // Initialize chat template and tokenizer for LlmRec (Qwen3).
  if (rec_type_ == RecType::kLlmRec) {
    chat_template_ =
        std::make_unique<JinjaChatTemplate>(engine_->tokenizer_args());
    tokenizer_ = engine_->tokenizer()->clone();
  } else {
    tokenizer_ = nullptr;
  }
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/options_.num_request_handling_threads(),
      /*cpu_binding=*/false,
      /*pool_name=*/"RecMaster.request");

  // Create the request factory with the pipeline selected from the model kind.
  auto rec_model_kind = get_rec_model_kind(model_args_.model_type());
  CHECK(rec_model_kind != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_args_.model_type();
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind);
  request_factory_ = std::make_unique<RecRequestFactory>(&model_args_,
                                                         tokenizer_.get(),
                                                         &options_,
                                                         get_rate_limiter(),
                                                         rec_type_,
                                                         pipeline_type);
}

void RecMaster::run() {
  if (!is_leader()) {
    Master::run();
    return;
  }

  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "RecMaster is already running.";
    return;
  }
  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(5);
    while (!stopped_.load(std::memory_order_relaxed)) {
      // move scheduler forward
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });

  // Engine run method is not available, remove this call
}

RecMaster::~RecMaster() {
  // set stop flag
  stopped_.store(true, std::memory_order_relaxed);

  // Drain and join the request thread pool before any of the members its
  // scheduled closures touch are destroyed. Those closures dereference
  // request_factory_ (as well as scheduler_), but request_factory_ is declared
  // after threadpool_ in the header, so member destruction would otherwise free
  // the factory while pool tasks are still in flight. ~ThreadPool only
  // signals/joins its workers when the pool is destroyed, so reset it here
  // explicitly while all dependencies are alive. Done before joining
  // loop_thread_ so the scheduler keeps advancing while the pool drains.
  threadpool_.reset();

  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void RecMaster::handle_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    RequestParams sp,
    OutputCallback callback) {
  // This interface supports both OneRec and LlmRec (qwen3 without mm_data)
  if (rec_type_ != RecType::kOneRec && rec_type_ != RecType::kLlmRec) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Unsupported rec type for this interface");
    return;
  }
  schedule_request(std::move(sp),
                   std::move(callback),
                   [this,
                    prompt = std::move(prompt),
                    prompt_tokens = std::move(prompt_tokens),
                    input_tensors = std::move(input_tensors)](
                       const RequestParams& params, OutputCallback cb) mutable {
                     return request_factory_->create(std::move(prompt),
                                                     std::move(prompt_tokens),
                                                     std::move(input_tensors),
                                                     params,
                                                     std::move(cb));
                   });
}

void RecMaster::handle_request(
    std::vector<Message> messages,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    RequestParams sp,
    OutputCallback callback) {
  if (rec_type_ != RecType::kLlmRec) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Chat is only supported for LLMRec models");
    return;
  }

  if (!chat_template_) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Chat template is not initialized");
    return;
  }

  Timer timer;

  std::optional<std::string> prompt;
  prompt = chat_template_->apply(messages, sp.tools, sp.chat_template_kwargs);

  if (!prompt.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to construct prompt from messages");
    LOG(ERROR) << "Failed to construct prompt from messages";
    return;
  }

  COUNTER_ADD(chat_template_latency_seconds, timer.elapsed_seconds());

  schedule_request(std::move(sp),
                   std::move(callback),
                   [this,
                    prompt = std::move(prompt.value()),
                    prompt_tokens = std::move(prompt_tokens),
                    input_tensors = std::move(input_tensors)](
                       const RequestParams& params, OutputCallback cb) mutable {
                     return request_factory_->create(std::move(prompt),
                                                     std::move(prompt_tokens),
                                                     std::move(input_tensors),
                                                     params,
                                                     std::move(cb));
                   });
}

void RecMaster::handle_request(const std::vector<int>& prompt_tokens,
                               std::optional<MMData> mm_data,
                               RequestParams sp,
                               OutputCallback callback) {
  if (rec_type_ != RecType::kLlmRec) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "LLMRec should use raw input interface");
    return;
  }

  schedule_request(std::move(sp),
                   std::move(callback),
                   [this,
                    prompt_tokens = std::move(prompt_tokens),
                    mm_data = std::move(mm_data)](const RequestParams& params,
                                                  OutputCallback cb) mutable {
                     return request_factory_->create(std::move(prompt_tokens),
                                                     std::move(mm_data),
                                                     params,
                                                     std::move(cb));
                   });
}

void RecMaster::schedule_request(RequestParams sp,
                                 OutputCallback callback,
                                 RequestBuilder build_request) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };
  threadpool_->schedule([this,
                         sp = std::move(sp),
                         callback = std::move(cb),
                         build_request = std::move(build_request)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    Timer timer;
    if (!sp.verify_params(callback)) {
      return;
    }

    auto request = build_request(sp, std::move(callback));
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

}  // namespace xllm
