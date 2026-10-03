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

#include "vlm_master.h"

#include <glog/logging.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/common/message.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config_validation.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#if defined(USE_NPU)
#include "models/model_registry.h"
#endif
#include "runtime/options.h"
#include "runtime/xservice_client.h"
#include "scheduler/scheduler_factory.h"
#include "server/xllm_server_registry.h"
#include "util/model_config_utils.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "util/utils.h"
#include "vlm_engine.h"

namespace xllm {

namespace {

bool should_use_vlm_speculative_engine(const Options& options) {
  return options.speculative_algorithm() != "Suffix" &&
         !options.draft_model_path().value_or("").empty();
}

template <typename Function>
bool dispatch_engine(VLMEngine* vlm_engine,
                     SpeculativeEngineBase<VLMEngine>* speculative_engine,
                     Function&& function) {
  if (vlm_engine != nullptr) {
    return function(*vlm_engine);
  }
  if (speculative_engine != nullptr) {
    return function(*speculative_engine);
  }
  return false;
}

void configure_disaggregated_pd_options(Options* options) {
  CHECK(options != nullptr);
  if (!options->enable_disagg_pd()) {
    return;
  }

  options->enable_service_routing(true);
  if (options->instance_role() == InstanceRole::PREFILL) {
    options->enable_schedule_overlap(false);
    LOG(WARNING) << "Force to disable schedule overlap for prefill instance "
                    "in disagg pd mode.";
  }
}

std::vector<Message> build_user_messages_from_image_urls(
    std::string prompt,
    const std::vector<std::string>& image_urls) {
  MMContentVec contents;
  contents.reserve(image_urls.size() + 1);
  for (const auto& url : image_urls) {
    contents.emplace_back("image_url", ImageURL{url});
  }
  contents.emplace_back("text", std::move(prompt));

  std::vector<Message> messages;
  messages.emplace_back("user", std::move(contents));
  return messages;
}

// Extracts the final prompt from a single message, concatenating its text
// contents in order and skipping non-text (image/video/audio) contents.
//
// The offline "prompt as-is" contract expects the caller to hand in exactly
// one message whose text is the final prompt already assembled with the
// chat template (images may ride along as non-text contents). Multi-turn
// conversations must go through the chat template path instead; joining
// their texts would silently mash the roles together, so any other message
// shape returns std::nullopt and fails the request.
std::optional<std::string> join_message_texts(
    const std::vector<Message>& messages) {
  if (messages.size() != 1) {
    return std::nullopt;
  }
  const auto& message = messages.front();
  if (auto text = std::get_if<std::string>(&message.content)) {
    return *text;
  }
  if (auto contents = std::get_if<MMContentVec>(&message.content)) {
    std::string prompt;
    for (const auto& content : *contents) {
      if (content.type == "text") {
        prompt += content.text;
      }
    }
    return prompt;
  }
  return std::nullopt;
}

// Throws std::invalid_argument describing the violation when messages do not
// satisfy the offline "prompt as-is" contract (see join_message_texts).
// Offline-only: raised on the caller thread so pybind surfaces it to the
// Python caller at the call site, instead of returning an empty prompt that
// is hard to trace.
[[noreturn]] void throw_prompt_as_is_error(
    const std::vector<Message>& messages) {
  std::string roles;
  for (const auto& message : messages) {
    if (!roles.empty()) {
      roles += ", ";
    }
    roles += message.role;
  }
  throw std::invalid_argument(
      "use_prompt_as_is expects exactly 1 message holding the final prompt "
      "(images allowed as non-text contents), got " +
      std::to_string(messages.size()) + " messages [" + roles +
      "]; apply the chat template for multi-turn conversations");
}

}  // namespace

VLMMaster::VLMMaster(const Options& options) : Master(options) {
  options_.enable_mla(util::should_enable_mla(
      std::filesystem::path(options_.model_path()), options_.backend()));
#if defined(USE_NPU)
  resolve_npu_kernel_backend(&options_);
#endif
  configure_disaggregated_pd_options(&options_);
  const bool use_speculative_engine =
      should_use_vlm_speculative_engine(options_);
  const std::string model_type =
      util::get_model_type(options_.model_path(), options_.backend());
  const std::optional<std::string> cp_error = validate_context_parallel_config(
      options_,
      use_speculative_engine ? EngineType::VLMSSM : EngineType::VLM,
      model_type,
      options_.nnodes());
  CHECK(!cp_error.has_value()) << cp_error.value();
  const std::optional<std::string> speculative_error =
      ModelConfig::validate_python_speculative_decode(
          ModelConfig::get_instance().model_impl(),
          model_type,
          options_.num_speculative_tokens(),
          options_.speculative_algorithm());
  CHECK(!speculative_error.has_value()) << speculative_error.value();
  validate_layerwise_split_size_startup_config(
      options_, model_type, options_.nnodes());
  CHECK(!options_.enable_task_pipeline())
      << "Task pipeline is only supported by the LLM master.";
  CHECK(options_.host_blocks_factor() <= 1.0)
      << "Basic host KV cache offload supports the LLM engine and "
         "model-based speculative engines only.";

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
      .cp_size(options_.cp_size())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .enable_offline_inference(options_.enable_offline_inference())
      .disable_log_stats(options_.disable_log_stats())
      .spawn_worker_path(options_.spawn_worker_path())
      .enable_shm(options_.enable_shm())
      .input_shm_size(options_.input_shm_size() * 1024 * 1024)
      .output_shm_size(options_.output_shm_size() * 1024 * 1024)
      .is_local(options_.is_local())
      .block_size(options_.block_size())
      .max_cache_size(options_.max_cache_size())
      .max_encoder_cache_size(options_.max_encoder_cache_size())
      .max_processor_cache_items(options_.max_processor_cache_items())
      .max_linear_state_cache_slots(options_.max_linear_state_cache_slots())
      .enable_mla(options_.enable_mla())
      .enable_flashcomm1(options_.enable_flashcomm1())
      .flashcomm1_min_prefill_tokens(options_.flashcomm1_min_prefill_tokens())
      .enable_mmrs_fusion(options_.enable_mmrs_fusion())
      .mmrs_comm_mode(options_.mmrs_comm_mode())
      .instance_role(options_.instance_role())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .max_tokens_per_batch(options_.max_tokens_per_batch())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .enable_graph(options_.enable_graph())
      .enable_graph_mode_decode_no_padding(
          options_.enable_graph_mode_decode_no_padding())
      .enable_prefill_piecewise_graph(options_.enable_prefill_piecewise_graph())
      .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
      .max_tokens_per_chunk_for_prefill(
          options_.max_tokens_per_chunk_for_prefill());

  if (use_speculative_engine) {
    CHECK(!options_.enable_disagg_pd())
        << "VLM speculative decoding does not support disaggregated PD";
    CHECK(!options_.enable_service_routing())
        << "VLM speculative decoding does not support service routing";
    CHECK(!options_.enable_adaptive_speculative_decode())
        << "VLM speculative decoding does not support adaptive speculative "
           "decode";
    const std::string draft_model_path =
        options_.draft_model_path().value_or("");
    CHECK(!draft_model_path.empty())
        << "draft model path is required for VLM speculative decoding";
    engine_options.draft_model_path(draft_model_path)
        .num_speculative_tokens(options_.num_speculative_tokens())
        .speculative_algorithm(options_.speculative_algorithm())
        .draft_sampling_mode(options_.draft_sampling_mode())
        .enable_mtp_draft_body_tp1(options_.enable_mtp_draft_body_tp1())
        .speculative_suffix_cache_max_depth(
            options_.speculative_suffix_cache_max_depth())
        .speculative_suffix_max_spec_factor(
            options_.speculative_suffix_max_spec_factor())
        .speculative_suffix_max_spec_offset(
            options_.speculative_suffix_max_spec_offset())
        .speculative_suffix_min_token_prob(
            options_.speculative_suffix_min_token_prob())
        .speculative_suffix_max_cached_requests(
            options_.speculative_suffix_max_cached_requests())
        .speculative_suffix_use_tree_spec(
            options_.speculative_suffix_use_tree_spec())
        .enable_adaptive_speculative_decode(
            options_.enable_adaptive_speculative_decode())
        .adaptive_speculative_min_gain(options_.adaptive_speculative_min_gain())
        .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
        .transfer_listen_port(options_.transfer_listen_port())
        .host_blocks_factor(options_.host_blocks_factor())
        .enable_kvcache_store(options_.enable_kvcache_store())
        .store_protocol(options_.store_protocol())
        .store_rdma_devices(options_.store_rdma_devices())
        .store_master_server_address(options_.store_master_server_address())
        .store_metadata_server(options_.store_metadata_server())
        .store_local_hostname(options_.store_local_hostname())
        .prefetch_batch_size(options_.prefetch_batch_size())
        .prefetch_timeout(options_.prefetch_timeout())
        .layers_wise_copy_batchs(options_.layers_wise_copy_batchs())
        .kv_cache_dtype(options_.kv_cache_dtype());
    speculative_engine_ =
        std::make_unique<SpeculativeEngineBase<VLMEngine>>(engine_options);
  } else {
    vlm_engine_ = std::make_unique<VLMEngine>(engine_options);
  }
  if (!is_leader()) {
    return;
  }

  auto initialize_engine = [this](auto* engine) {
    CHECK(engine->init(options_.master_status()));
    model_args_ = engine->model_args();
    if (options_.enable_service_routing()) {
      XServiceClient* xservice_client = XServiceClient::get_instance();
      CHECK(xservice_client->init(options_.etcd_addr().value_or(""),
                                  options_.instance_name().value_or(""),
                                  engine->block_manager_pool(),
                                  options_.etcd_namespace().value_or("")))
          << "XServiceClient init fail!";
    }
  };
  if (use_speculative_engine) {
    initialize_engine(speculative_engine_.get());
  } else {
    initialize_engine(vlm_engine_.get());
  }

  SchedulerOptions scheduler_options;
  scheduler_options.max_tokens_per_batch(options.max_tokens_per_batch())
      .max_seqs_per_batch(options.max_seqs_per_batch())
      .request_queue_size(options.request_queue_size())
      .max_tokens_per_chunk_for_prefill(
          options.max_tokens_per_chunk_for_prefill())
      .num_speculative_tokens(options_.num_speculative_tokens())
      .dp_size(options_.dp_size())
      .cp_size(options_.cp_size())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_name(options_.instance_name())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing())
      .disable_log_stats(options_.disable_log_stats())
      .disable_ttft_profiling(options_.disable_ttft_profiling())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .server_idx(options_.server_idx());
  if (use_speculative_engine) {
    scheduler_ = create_continuous_scheduler(speculative_engine_.get(),
                                             scheduler_options);
  } else {
    scheduler_ =
        create_continuous_scheduler(vlm_engine_.get(), scheduler_options);
  }

  if (options_.enable_service_routing()) {
    auto& instance_info = scheduler_->get_instance_info();
    XServiceClient::get_instance()->register_instance(instance_info);
  }

  auto initialize_processor = [this](auto* engine) {
    chat_template_ =
        std::make_unique<JinjaChatTemplate>(engine->tokenizer_args());
    tokenizer_ = engine->tokenizer()->clone();
    processor_ =
        create_multimodal_processor(model_args_,
                                    tokenizer_,
                                    options_.max_processor_cache_items(),
                                    engine->tokenizer_args());
  };
  if (use_speculative_engine) {
    initialize_processor(speculative_engine_.get());
  } else {
    initialize_processor(vlm_engine_.get());
  }

  request_factory_ = std::make_unique<VLMRequestFactory>(processor_.get(),
                                                         chat_template_.get(),
                                                         tokenizer_.get(),
                                                         &model_args_,
                                                         &options_,
                                                         get_rate_limiter());

  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/options_.num_request_handling_threads(),
      /*cpu_binding=*/false,
      /*pool_name=*/"VLMMaster.request");
}

VLMMaster::~VLMMaster() {
  stoped_.store(true, std::memory_order_relaxed);

  // Drain and join the request thread pool before any of the members its
  // worker lambdas touch are destroyed. Those lambdas dereference
  // request_factory_ (as well as scheduler_ and the rate limiter), but
  // request_factory_ is declared after threadpool_ in the header, so member
  // destruction would otherwise free the factory while pool workers are still
  // in flight. ~ThreadPool only signals/joins its workers when the pool is
  // destroyed, so reset it here explicitly while all dependencies are alive.
  // Done before joining loop_thread_ so the scheduler keeps advancing while the
  // pool drains.
  threadpool_.reset();

  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void VLMMaster::handle_request(std::string prompt,
                               MMData mm_data,
                               RequestParams sp,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  threadpool_->schedule([this,
                         prompt = std::move(prompt),
                         mm_data = std::move(mm_data),
                         sp = std::move(sp),
                         callback = std::move(cb)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // Guard the rate-limit slot acquired at the service entry.
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    rate_limit_guard.dismiss();
    auto request = request_factory_->create(
        std::move(prompt), std::move(mm_data), sp, std::move(callback));
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void VLMMaster::handle_request(std::vector<Message> messages,
                               RequestParams sp,
                               std::string payload,
                               OutputCallback callback,
                               bool use_prompt_as_is) {
  // Offline-only guard: fail loudly on the caller thread (pybind surfaces
  // the exception to the Python caller) before any scheduling side effect.
  if (use_prompt_as_is && !join_message_texts(messages).has_value()) {
    throw_prompt_as_is_error(messages);
  }

  scheduler_->incr_pending_requests(1);
  auto cb = [callback = std::move(callback),
             scheduler = scheduler_.get()](const RequestOutput& output) {
    output.log_request_status();
    return callback(output);
  };

  threadpool_->schedule([this,
                         messages = std::move(messages),
                         sp = std::move(sp),
                         payload = std::move(payload),
                         use_prompt_as_is,
                         callback = std::move(cb)]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_chat);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // Guard the rate-limit slot acquired at the service entry.
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });

    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    rate_limit_guard.dismiss();
    auto request = request_factory_->create(std::move(messages),
                                            sp,
                                            std::move(payload),
                                            std::move(callback),
                                            use_prompt_as_is);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
    }
  });
}

void VLMMaster::handle_batch_request(std::vector<std::string> prompts,
                                     std::vector<MMData> mm_datas,
                                     std::vector<RequestParams> sps,
                                     BatchOutputCallback callback) {
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(prompts[i]),
                   std::move(mm_datas[i]),
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

// Offline batch entry aligned with the vllm-ascend offline behavior: each
// prompt is handed in as the final prompt and is used as-is, without applying
// the chat template again (double assembly duplicates the vision placeholders
// and crashes prompt processing). This only affects offline inference; the
// online serving path still applies the chat template.
void VLMMaster::handle_batch_request_with_image_urls(
    std::vector<std::string> prompts,
    std::vector<std::vector<std::string>> image_urls,
    std::vector<RequestParams> sps,
    BatchOutputCallback callback) {
  CHECK(prompts.size() == image_urls.size())
      << "Number of prompts and image urls should be the same";
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  std::vector<std::vector<Message>> conversations;
  conversations.reserve(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    conversations.push_back(build_user_messages_from_image_urls(
        std::move(prompts[i]), image_urls[i]));
  }

  std::string payload;
  for (size_t i = 0; i < num_requests; ++i) {
    // Offline vLLM-style requests hand in a final prompt. The text-only
    // offline path never applies the chat template either, so use the
    // prompt as-is instead of re-assembling it (double assembly duplicates
    // the vision placeholders and crashes prompt processing).
    handle_request(
        std::move(conversations[i]),
        // the sampling parameter may be shared
        sps.size() == 1 ? sps[0] : std::move(sps[i]),
        std::move(payload),
        [i, callback](const RequestOutput& output) {
          output.log_request_status();
          return callback(i, output);
        },
        /*use_prompt_as_is=*/true);
  }
}

void VLMMaster::handle_batch_request(
    std::vector<std::vector<Message>> conversations,
    std::vector<RequestParams> sps,
    BatchOutputCallback callback) {
  CHECK(conversations.size() == sps.size() || sps.size() == 1)
      << "Number of conversations and sampling parameters should be the same";

  std::string payload;
  const size_t num_requests = conversations.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(conversations[i]),
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   std::move(payload),
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void VLMMaster::run() {
  if (!is_leader()) {
    Master::run();
    return;
  }

  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "VLMMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    running_.store(true, std::memory_order_relaxed);
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });
}

void VLMMaster::generate() {
  DCHECK(options_.enable_schedule_overlap())
      << "Mode generate does not support schedule overlap yet.";
  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "Generate is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  scheduler_->generate();
  running_.store(false, std::memory_order_relaxed);
}

bool VLMMaster::start_profile() {
  return dispatch_engine(vlm_engine_.get(),
                         speculative_engine_.get(),
                         [](auto& engine) { return engine.start_profile(); });
}

bool VLMMaster::stop_profile() {
  return dispatch_engine(vlm_engine_.get(),
                         speculative_engine_.get(),
                         [](auto& engine) { return engine.stop_profile(); });
}

}  // namespace xllm
