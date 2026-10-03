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

#include "llm_master.h"

#include <glog/logging.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "api_service/call.h"
#include "common/metrics.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config_validation.h"
#include "core/framework/config/speculative_config.h"
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
#include "util/net.h"
#include "util/scope_guard.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {

bool should_use_ssm_engine(const Options& options) {
  return !options.draft_model_path().value_or("").empty() ||
         (options.speculative_algorithm() == "Suffix" &&
          options.num_speculative_tokens() > 0);
}

template <typename Function>
bool dispatch_engine(LLMEngine* llm_engine,
                     SuffixSpeculativeEngine* suffix_engine,
                     SpeculativeEngineBase<LLMEngine>* speculative_engine,
                     Function&& function) {
  if (llm_engine != nullptr) {
    return function(*llm_engine);
  }
  if (suffix_engine != nullptr) {
    return function(*suffix_engine);
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

}  // namespace

LLMMaster::LLMMaster(const Options& options)
    : Master(options), master_status_(options.master_status()) {
  options_.enable_mla(util::should_enable_mla(
      std::filesystem::path(options_.model_path()), options_.backend()));
#if defined(USE_NPU)
  resolve_npu_kernel_backend(&options_);
#endif
  configure_disaggregated_pd_options(&options_);
  const bool use_ssm_engine = should_use_ssm_engine(options_);
  const EngineType engine_type =
      use_ssm_engine ? EngineType::SSM : EngineType::LLM;
  const std::string model_type =
      util::get_model_type(options_.model_path(), options_.backend());
  const std::optional<std::string> cp_error = validate_context_parallel_config(
      options_, engine_type, model_type, options_.nnodes());
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

  if (options_.enable_task_pipeline()) {
    const std::string& algorithm = options_.speculative_algorithm();
    const bool supported_speculation =
        SpeculativeConfig::is_mtp_algorithm(algorithm) ||
        algorithm == "DFlash" ||
        SpeculativeConfig::is_dflash2_algorithm(algorithm);
    CHECK((engine_type == EngineType::LLM ||
           (engine_type == EngineType::SSM && supported_speculation)) &&
          options_.task_type() == "generate" &&
          !options_.enable_offline_inference())
        << "Task pipeline requires online LLM, MTP, DFlash or DFlash2 "
           "generation.";
    CHECK(master_status_ == MasterStatus::WAKEUP)
        << "Task pipeline must start with loaded weights.";
  }
  if (options_.host_blocks_factor() > 1.0) {
    const bool supports_host_offload =
        engine_type == EngineType::LLM ||
        SpeculativeConfig::supports_host_kv_cache(
            options_.speculative_algorithm());
    CHECK(supports_host_offload)
        << "Basic host KV cache offload supports the LLM engine and "
           "model-based speculative engines only.";
  }

  if (!use_ssm_engine &&
      (options_.task_type() == "embed" || options_.task_type() == "mm_embed")) {
    options_.enable_schedule_overlap(false);
    LOG(WARNING) << "Force to disable schedule overlap for embedding model, "
                    "avoiding performance degradation.";
  }

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
      .max_linear_state_cache_slots(options_.max_linear_state_cache_slots())
      .enable_mla(options_.enable_mla())
      .enable_flashcomm1(options_.enable_flashcomm1())
      .flashcomm1_min_prefill_tokens(options_.flashcomm1_min_prefill_tokens())
      .enable_mmrs_fusion(options_.enable_mmrs_fusion())
      .mmrs_comm_mode(options_.mmrs_comm_mode())
      .instance_role(options_.instance_role())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .transfer_listen_port(options_.transfer_listen_port())
      .enable_service_routing(options_.enable_service_routing())
      .server_idx(options_.server_idx())
      .enable_task_pipeline(options_.enable_task_pipeline())
      .enable_graph(options_.enable_graph())
      .enable_graph_mode_decode_no_padding(
          options_.enable_graph_mode_decode_no_padding())
      .enable_prefill_piecewise_graph(options_.enable_prefill_piecewise_graph())
      .max_tokens_for_graph_mode(options_.max_tokens_for_graph_mode())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .max_tokens_per_batch(options_.max_tokens_per_batch())
      .max_tokens_per_chunk_for_prefill(
          options_.max_tokens_per_chunk_for_prefill())
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

  if (!use_ssm_engine) {
    llm_engine_ = std::make_unique<LLMEngine>(engine_options);
  } else {
    const std::string draft_model_path =
        options_.draft_model_path().value_or("");
    const bool use_suffix_spec = options_.speculative_algorithm() == "Suffix";
    CHECK(use_suffix_spec || !draft_model_path.empty())
        << "draft model path is required unless --speculative_algorithm=Suffix";
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
        .adaptive_speculative_min_gain(
            options_.adaptive_speculative_min_gain());
    if (use_suffix_spec) {
      suffix_engine_ =
          std::make_unique<SuffixSpeculativeEngine>(engine_options);
    } else {
      speculative_engine_ =
          std::make_unique<SpeculativeEngineBase<LLMEngine>>(engine_options);
    }
  }
  if (!is_leader()) {
    return;
  }

  auto initialize_engine = [this](auto* engine) {
    CHECK(engine->init(master_status_));
    model_args_ = engine->model_args();
    if (options_.enable_service_routing()) {
      xservice_client_ = XServiceClient::get_instance();
      CHECK(xservice_client_->init(options_.etcd_addr().value_or(""),
                                   options_.instance_name().value_or(""),
                                   engine->block_manager_pool(),
                                   options_.etcd_namespace().value_or("")))
          << "XServiceClient init fail!";
    }
  };
  if (!use_ssm_engine) {
    initialize_engine(llm_engine_.get());
  } else if (options_.speculative_algorithm() == "Suffix") {
    initialize_engine(suffix_engine_.get());
  } else {
    initialize_engine(speculative_engine_.get());
  }
  task_type_ = options_.task_type();

  SchedulerOptions scheduler_options;
  scheduler_options.max_tokens_per_batch(options_.max_tokens_per_batch())
      .max_seqs_per_batch(options_.max_seqs_per_batch())
      .enable_task_pipeline(options_.enable_task_pipeline())
      .request_queue_size(options_.request_queue_size())
      .max_tokens_per_chunk_for_prefill(
          options_.max_tokens_per_chunk_for_prefill())
      .num_speculative_tokens(options_.num_speculative_tokens())
      .nnodes(options_.nnodes())
      .dp_size(options_.dp_size())
      .cp_size(options_.cp_size())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_schedule_overlap(options_.enable_schedule_overlap())
      .enable_chunked_prefill(options_.enable_chunked_prefill())
      .instance_name(options_.instance_name())
      .instance_role(options_.instance_role())
      .kv_cache_transfer_mode(options_.kv_cache_transfer_mode())
      .enable_service_routing(options_.enable_service_routing())
      .disable_log_stats(options_.disable_log_stats())
      .priority_strategy(options_.priority_strategy())
      .enable_profile_step_time(options_.enable_profile_step_time())
      .enable_profile_token_budget(options_.enable_profile_token_budget())
      .enable_latency_aware_schedule(options_.enable_latency_aware_schedule())
      .profile_max_prompt_length(options_.profile_max_prompt_length())
      .enable_profile_kv_blocks(options_.enable_profile_kv_blocks())
      .disable_ttft_profiling(options_.disable_ttft_profiling())
      .max_global_ttft_ms(options_.max_global_ttft_ms())
      .max_global_tpot_ms(options_.max_global_tpot_ms())
      .server_idx(options_.server_idx())
      .rec_worker_max_concurrency(options_.rec_worker_max_concurrency());
  if (!use_ssm_engine) {
    scheduler_ =
        create_continuous_scheduler(llm_engine_.get(), scheduler_options);
  } else if (options_.speculative_algorithm() == "Suffix") {
    scheduler_ =
        create_continuous_scheduler(suffix_engine_.get(), scheduler_options);
  } else {
    scheduler_ = create_continuous_scheduler(speculative_engine_.get(),
                                             scheduler_options);
  }

  if (options_.enable_service_routing()) {
    auto& instance_info = scheduler_->get_instance_info();
    XServiceClient::get_instance()->register_instance(instance_info);
  }

  auto initialize_tokenizer = [this](auto* engine) {
    chat_template_ = ChatTemplate::create(engine->tokenizer_args(),
                                          model_args_.model_type());
    tokenizer_ = engine->tokenizer()->clone();
  };
  if (!use_ssm_engine) {
    initialize_tokenizer(llm_engine_.get());
  } else if (options_.speculative_algorithm() == "Suffix") {
    initialize_tokenizer(suffix_engine_.get());
  } else {
    initialize_tokenizer(speculative_engine_.get());
  }
  Tokenizer* request_tokenizer = tokenizer_.get();
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/options_.num_request_handling_threads(),
      [request_tokenizer]() { request_tokenizer->warmup(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"LLMMaster.request");

  request_factory_ = std::make_unique<LLMRequestFactory>(
      tokenizer_.get(),
      chat_template_.get(),
      &model_args_,
      &options_,
      get_rate_limiter(),
      task_type_,
      [this](const std::vector<RequestOutput>& outputs) {
        return handle_rpc_responses(outputs);
      });
}

LLMMaster::~LLMMaster() {
  LOG(INFO) << "LLMMaster stopping...";

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
  // Stop accepting work after request handlers have finished admission. The
  // loop still drains prefetch callbacks before scheduler/engine destruction.
  stoped_.store(true, std::memory_order_release);

  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void LLMMaster::count_chat_tokens(
    std::vector<Message> messages,
    RequestParams params,
    std::function<void(Status, int32_t)> callback) {
  threadpool_->schedule([this,
                         messages = std::move(messages),
                         params = std::move(params),
                         callback = std::move(callback)]() {
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });
    std::optional<std::string> prompt = chat_template_->apply(
        messages, params.tools, params.chat_template_kwargs);
    if (!prompt.has_value()) {
      callback(Status(StatusCode::INVALID_ARGUMENT,
                      "Failed to construct prompt from messages"),
               0);
      return;
    }
    std::vector<int32_t> tokens;
    if (!tokenizer_->encode(
            prompt.value(), &tokens, params.add_special_tokens)) {
      callback(Status(StatusCode::INVALID_ARGUMENT, "Failed to encode prompt"),
               0);
      return;
    }
    callback(Status(), static_cast<int32_t>(tokens.size()));
  });
}

void LLMMaster::handle_batch_request(std::vector<std::string> prompts,
                                     std::vector<RequestParams> sps,
                                     BatchOutputCallback callback) {
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(prompts[i]),
                   std::nullopt,
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   std::nullopt,
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void LLMMaster::handle_batch_request(
    std::vector<std::vector<Message>> conversations,
    std::vector<RequestParams> sps,
    BatchOutputCallback callback) {
  CHECK(conversations.size() == sps.size() || sps.size() == 1)
      << "Number of conversations and sampling parameters should be the same";

  const size_t num_requests = conversations.size();
  for (size_t i = 0; i < num_requests; ++i) {
    handle_request(std::move(conversations[i]),
                   std::nullopt,
                   // the sampling parameter may be shared
                   sps.size() == 1 ? sps[0] : std::move(sps[i]),
                   std::nullopt,
                   [i, callback](const RequestOutput& output) {
                     output.log_request_status();
                     return callback(i, output);
                   });
  }
}

void LLMMaster::handle_request(std::string prompt,
                               std::optional<std::vector<int>> prompt_tokens,
                               RequestParams sp,
                               std::optional<Call*> call,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  // add into the queue
  threadpool_->schedule([this,
                         prompt = std::move(prompt),
                         prompt_token = std::move(prompt_tokens),
                         sp = std::move(sp),
                         callback = std::move(callback),
                         call]() mutable {
    AUTO_COUNTER(request_handling_latency_seconds_completion);

    // remove the pending request after scheduling
    SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

    // Guard the rate-limit slot acquired at the service entry. If we bail
    // before the factory has a chance to create the Request, this
    // releases the slot; otherwise Request itself takes ownership.
    xllm::ScopeGuard rate_limit_guard(
        [this] { get_rate_limiter()->decrease_one_request(); });

    Timer timer;
    // verify the prompt
    if (!sp.verify_params(callback)) {
      return;
    }

    rate_limit_guard.dismiss();
    auto request = request_factory_->create(
        std::move(prompt), std::move(prompt_token), sp, call, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request",
                          sp.service_request_id,
                          sp.source_xservice_addr);
    }
  });
}

void LLMMaster::handle_request(std::vector<Message> messages,
                               std::optional<std::vector<int>> prompt_tokens,
                               RequestParams sp,
                               std::optional<Call*> call,
                               OutputCallback callback) {
  scheduler_->incr_pending_requests(1);
  // add into the queue
  threadpool_->schedule([this,
                         messages = std::move(messages),
                         prompt_token = std::move(prompt_tokens),
                         sp = std::move(sp),
                         callback = std::move(callback),
                         call]() mutable {
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
    auto request = request_factory_->create(
        messages, std::move(prompt_token), sp, call, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->add_request(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request",
                          sp.service_request_id,
                          sp.source_xservice_addr);
    }
  });
}

void LLMMaster::run() {
  if (!is_leader()) {
    Master::run();
    return;
  }

  const bool already_running = running_.load(std::memory_order_relaxed);
  if (already_running) {
    LOG(WARNING) << "LLMMaster is already running.";
    return;
  }

  running_.store(true, std::memory_order_relaxed);
  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_acquire) ||
           scheduler_->has_pending_prefetch()) {
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });
}

void LLMMaster::generate() {
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

bool LLMMaster::handle_rpc_response(const RequestOutput& output) {
  // response to xllm service to avoid the redirect cost.
  if (xservice_client_ == nullptr) return false;
  auto return_status = xservice_client_->generations({output});
  CHECK_EQ(return_status.size(), 1)
      << "return size of generations is not equal to 1";
  return return_status[0];
}

std::vector<bool> LLMMaster::handle_rpc_responses(
    const std::vector<RequestOutput>& outputs) {
  // response to xllm service to avoid the redirect cost.
  if (xservice_client_ == nullptr)
    return std::vector<bool>(outputs.size(), false);
  return xservice_client_->generations(outputs);
}

bool LLMMaster::sleep() {
  return dispatch_engine(
      llm_engine_.get(),
      suffix_engine_.get(),
      speculative_engine_.get(),
      [this](auto& engine) { return engine.sleep(master_status_); });
}

bool LLMMaster::wakeup() {
  WakeupOptions options;
  options.master_status = master_status_;
  return dispatch_engine(
      llm_engine_.get(),
      suffix_engine_.get(),
      speculative_engine_.get(),
      [&options](auto& engine) { return engine.wakeup(options); });
}

bool LLMMaster::wakeup(const WakeupOptions& options) {
  WakeupOptions opts = options;
  opts.master_status = master_status_;
  return dispatch_engine(llm_engine_.get(),
                         suffix_engine_.get(),
                         speculative_engine_.get(),
                         [&opts](auto& engine) { return engine.wakeup(opts); });
}

bool LLMMaster::link_p2p(const std::vector<std::string>& remote_addrs) {
  return dispatch_engine(
      llm_engine_.get(),
      suffix_engine_.get(),
      speculative_engine_.get(),
      [&remote_addrs](auto& engine) { return engine.link_p2p(remote_addrs); });
}

bool LLMMaster::unlink_p2p(const std::vector<std::string>& remote_addrs) {
  return dispatch_engine(llm_engine_.get(),
                         suffix_engine_.get(),
                         speculative_engine_.get(),
                         [&remote_addrs](auto& engine) {
                           return engine.unlink_p2p(remote_addrs);
                         });
}

bool LLMMaster::start_profile() {
  return dispatch_engine(llm_engine_.get(),
                         suffix_engine_.get(),
                         speculative_engine_.get(),
                         [](auto& engine) { return engine.start_profile(); });
}

bool LLMMaster::stop_profile() {
  return dispatch_engine(llm_engine_.get(),
                         suffix_engine_.get(),
                         speculative_engine_.get(),
                         [](auto& engine) { return engine.stop_profile(); });
}

}  // namespace xllm
