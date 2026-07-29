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

#include "c_api/llm.h"

#include <folly/Unit.h>
#include <folly/experimental/coro/Timeout.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <pthread.h>

#include <atomic>
#include <cstring>
#include <exception>
#include <stdexcept>

#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/execution_config.h"
#include "helper.h"

XLLM_CAPI_EXPORT XLLM_LLM_Handler* xllm_llm_create(void) {
  XLLM_LLM_Handler* handler = new XLLM_LLM_Handler();
  CHECK(nullptr != handler);

  handler->initialized = false;

  return handler;
}

XLLM_CAPI_EXPORT void xllm_llm_destroy(XLLM_LLM_Handler* handler) {
  if (!handler) return;

  handler->master.reset();
  handler->executor.reset();
  handler->model_ids.clear();
  handler->initialized = false;

  delete handler;
}

XLLM_CAPI_EXPORT void xllm_llm_init_options_default(
    XLLM_InitOptions* init_options) {
  if (nullptr == init_options) return;
  *init_options = XLLM_INIT_LLM_OPTIONS_DEFAULT;
}

XLLM_CAPI_EXPORT bool xllm_llm_initialize(
    XLLM_LLM_Handler* handler,
    const char* model_path,
    const XLLM_InitOptions* init_options) {
  if (!handler || !model_path) return false;

  try {
    XLLM_InitOptions xllm_init_options;
    xllm::helper::set_init_options(
        xllm::helper::BackendType::LLM, init_options, &xllm_init_options);

    std::string log_dir(xllm_init_options.log_dir);
    if (!log_dir.empty()) {
      xllm::helper::init_log(xllm_init_options.log_dir);
    }

    if (!std::filesystem::exists(model_path)) {
      LOG(ERROR) << "model path[" << model_path << "] does not exist";
      return false;
    }

    xllm::Options options;
    options.model_path(model_path)
        .task_type(xllm_init_options.task)
        .draft_model_path(xllm_init_options.draft_model)
        .backend("llm")
        .block_size(xllm_init_options.block_size)
        .max_cache_size(xllm_init_options.max_cache_size)
        .max_memory_utilization(xllm_init_options.max_memory_utilization)
        .enable_prefix_cache(xllm_init_options.enable_prefix_cache)
        .max_tokens_per_batch(xllm_init_options.max_tokens_per_batch)
        .max_seqs_per_batch(xllm_init_options.max_seqs_per_batch)
        .max_tokens_per_chunk_for_prefill(
            xllm_init_options.max_tokens_per_chunk_for_prefill)
        .num_speculative_tokens(xllm_init_options.num_speculative_tokens)
        .num_request_handling_threads(
            xllm_init_options.num_request_handling_threads)
        .communication_backend(xllm_init_options.communication_backend)
        .expert_parallel_degree(xllm_init_options.expert_parallel_degree)
        .enable_chunked_prefill(xllm_init_options.enable_chunked_prefill)
        .cp_size(static_cast<int32_t>(xllm_init_options.cp_size))
        .master_node_addr(xllm_init_options.master_node_addr)
        .transfer_listen_port(xllm_init_options.transfer_listen_port)
        .nnodes(xllm_init_options.nnodes)
        .node_rank(xllm_init_options.node_rank)
        .dp_size(xllm_init_options.dp_size)
        .ep_size(xllm_init_options.ep_size)
        .instance_name(xllm_init_options.instance_name)
        .enable_disagg_pd(xllm_init_options.enable_disagg_pd)
        .enable_schedule_overlap(xllm_init_options.enable_schedule_overlap)
        .enable_pd_ooc(xllm_init_options.enable_pd_ooc)
        .kv_cache_transfer_mode(xllm_init_options.kv_cache_transfer_mode)
        .enable_shm(xllm_init_options.enable_shm)
        .is_local(true)
        .server_idx(xllm_init_options.server_idx);

    options.enable_graph(
        ::xllm::ExecutionConfig::get_instance().enable_graph());

#if !defined(USE_NPU) && !defined(USE_CUDA)
    xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel(false);
#endif
    handler->master = std::make_unique<xllm::LLMMaster>(options);
    handler->master->run();

    size_t cpu_cores = std::thread::hardware_concurrency();
    size_t thread_num = std::clamp((cpu_cores == 0) ? 8 : cpu_cores / 2,
                                   static_cast<size_t>(4),
                                   static_cast<size_t>(16));
    handler->executor =
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
    handler->model_ids.clear();
    handler->model_ids.emplace_back(model_id);

    handler->initialized = true;

    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "LLM initialization failed: " << e.what();
  }

  handler->master.reset();
  handler->executor.reset();
  handler->model_ids.clear();
  handler->initialized = false;

  return false;
}

XLLM_CAPI_EXPORT void xllm_llm_request_params_default(
    XLLM_RequestParams* request_params) {
  if (nullptr == request_params) return;
  *request_params = XLLM_LLM_REQUEST_PARAMS_DEFAULT;
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_llm_completions(
    XLLM_LLM_Handler* handler,
    const char* model_id,
    const char* prompt,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !prompt ||
      *prompt == '\0') {
    return xllm::helper::build_error_response(
        "", XLLM_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  return xllm::helper::handle_inference_request(
      handler,
      xllm::helper::InferenceType::LLM_COMPLETIONS,
      model_id,
      prompt,
      nullptr,
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_llm_chat_completions(
    XLLM_LLM_Handler* handler,
    const char* model_id,
    const XLLM_ChatMessage* messages,
    size_t messages_count,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !messages ||
      messages_count == 0) {
    return xllm::helper::build_error_response(
        "", XLLM_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  std::vector<xllm::Message> xllm_messages;
  xllm_messages.reserve(messages_count);
  for (int i = 0; i < messages_count; i++) {
    xllm_messages.emplace_back(messages[i].role, messages[i].content);
  }

  return xllm::helper::handle_inference_request(
      handler,
      xllm::helper::InferenceType::LLM_CHAT_COMPLETIONS,
      model_id,
      xllm_messages,
      nullptr,
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT void xllm_llm_free_response(XLLM_Response* resp) {
  return xllm::helper::xllm_free_response(resp);
}
