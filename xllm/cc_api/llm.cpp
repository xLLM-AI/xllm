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

#include "llm.h"

#include <folly/Unit.h>
#include <folly/experimental/coro/Timeout.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <pthread.h>

#include <atomic>
#include <exception>

#include "core/framework/config/beam_search_config.h"
#include "internal.h"

namespace xllm {
namespace {
static std::atomic<bool> g_glog_inited = false;
static pthread_mutex_t g_log_init_mutex = PTHREAD_MUTEX_INITIALIZER;

void InitGlog(const std::string& log_dir) {
  pthread_mutex_lock(&g_log_init_mutex);
  if (!g_glog_inited) {
    google::InitGoogleLogging("xllm");
    google::SetLogDestination(google::INFO,
                              (log_dir + "/xllm.log.INFO.").c_str());
    google::SetLogDestination(google::WARNING,
                              (log_dir + "/xllm.log.WARNING.").c_str());
    google::SetLogDestination(google::ERROR,
                              (log_dir + "/xllm.log.ERROR.").c_str());
    google::SetStderrLogging(google::FATAL);

    g_glog_inited = true;
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}
}  // namespace

LLM::LLM() = default;
LLM::~LLM() {
  if (nullptr != llm_core_) {
    delete llm_core_;
    llm_core_ = nullptr;
  }
}

bool LLM::Initialize(const std::string& model_path,
                     const XLLM_InitLLMOptions& init_options) {
  if (!init_options.log_dir.empty()) {
    InitGlog(init_options.log_dir);
  }

  if (!std::filesystem::exists(model_path)) {
    LOG(ERROR) << "model path[" << model_path << "] does not exist";
    return false;
  }

  try {
    Options options;
    options.model_path(model_path)
        .task_type(init_options.task)
        .draft_model_path(init_options.draft_model)
        .backend("llm")
        .block_size(init_options.block_size)
        .max_cache_size(init_options.max_cache_size)
        .max_memory_utilization(init_options.max_memory_utilization)
        .enable_prefix_cache(init_options.enable_prefix_cache)
        .max_tokens_per_batch(init_options.max_tokens_per_batch)
        .max_seqs_per_batch(init_options.max_seqs_per_batch)
        .max_tokens_per_chunk_for_prefill(
            init_options.max_tokens_per_chunk_for_prefill)
        .num_speculative_tokens(init_options.num_speculative_tokens)
        .num_request_handling_threads(init_options.num_request_handling_threads)
        .communication_backend(init_options.communication_backend)
        .rank_tablefile(init_options.rank_tablefile)
        .expert_parallel_degree(init_options.expert_parallel_degree)
        .enable_chunked_prefill(init_options.enable_chunked_prefill)
        .cp_size(init_options.cp_size)
        .master_node_addr(init_options.master_node_addr)
        .transfer_listen_port(init_options.transfer_listen_port)
        .nnodes(init_options.nnodes)
        .node_rank(init_options.node_rank)
        .dp_size(init_options.dp_size)
        .ep_size(init_options.ep_size)
        .instance_name(init_options.instance_name)
        .enable_disagg_pd(init_options.enable_disagg_pd)
        .enable_schedule_overlap(init_options.enable_schedule_overlap)
        .enable_pd_ooc(init_options.enable_pd_ooc)
        .kv_cache_transfer_mode(init_options.kv_cache_transfer_mode)
        .disable_ttft_profiling(init_options.disable_ttft_profiling)
        .enable_forward_interruption(init_options.enable_forward_interruption)
        .enable_shm(init_options.enable_shm)
        .input_shm_size(init_options.input_shm_size)
        .output_shm_size(init_options.output_shm_size)
        .is_local(init_options.is_local)
        .server_idx(init_options.server_idx);

#if !defined(USE_NPU) && !defined(USE_CUDA)
    BeamSearchConfig::get_instance().enable_block_copy_kernel(false);
#endif
    llm_core_ = new LLMCore();
    llm_core_->master = std::make_unique<LLMMaster>(options);
    llm_core_->master->run();

    size_t cpu_cores = std::thread::hardware_concurrency();
    size_t thread_num = std::clamp((cpu_cores == 0) ? 8 : cpu_cores / 2,
                                   static_cast<size_t>(4),
                                   static_cast<size_t>(16));
    llm_core_->executor =
        std::make_unique<folly::CPUThreadPoolExecutor>(thread_num);

    std::filesystem::path model_path_fs =
        std::filesystem::path(model_path).lexically_normal();
    std::string model_id;
    if (model_path_fs.has_filename()) {
      model_id = model_path_fs.filename().string();
    } else if (!model_path_fs.empty()) {
      model_id = model_path_fs.string();
    } else {
      model_id = "default";
    }
    llm_core_->model_ids.emplace_back(model_id);

    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "LLM initialization failed: " << e.what();
    if (nullptr != llm_core_) {
      delete llm_core_;
      llm_core_ = nullptr;
    }

    return false;
  }
}

XLLM_Response LLM::Completions(const std::string& model_id,
                               const std::string& prompt,
                               uint32_t timeout_ms,
                               const XLLM_RequestParams& request_params) {
  return detail::handle_inference_request(llm_core_,
                                          model_id,
                                          prompt,
                                          timeout_ms,
                                          request_params,
                                          detail::InterfaceType::COMPLETIONS);
}

XLLM_Response LLM::ChatCompletions(
    const std::string& model_id,
    const std::vector<XLLM_ChatMessage>& messages,
    uint32_t timeout_ms,
    const XLLM_RequestParams& request_params) {
  std::vector<Message> internal_messages;
  internal_messages.reserve(messages.size());
  for (const auto& msg : messages) {
    internal_messages.emplace_back(msg.role, msg.content);
  }

  return detail::handle_inference_request(
      llm_core_,
      model_id,
      internal_messages,
      timeout_ms,
      request_params,
      detail::InterfaceType::CHAT_COMPLETIONS);
}
}  // namespace xllm
