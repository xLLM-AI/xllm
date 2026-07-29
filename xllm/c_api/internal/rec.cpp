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

#include "c_api/rec.h"

#include <folly/Unit.h>
#include <folly/experimental/coro/Timeout.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <pthread.h>

#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model_loader.h"
#include "core/util/cpu_affinity.h"
#include "helper.h"

XLLM_CAPI_EXPORT XLLM_REC_Handler* xllm_rec_create(void) {
  XLLM_REC_Handler* handler = new XLLM_REC_Handler();
  CHECK(nullptr != handler);

  handler->initialized = false;

  return handler;
}

XLLM_CAPI_EXPORT void xllm_rec_destroy(XLLM_REC_Handler* handler) {
  if (!handler) return;

  handler->master.reset();
  handler->executor.reset();
  handler->model_ids.clear();
  handler->initialized = false;

  delete handler;
}

XLLM_CAPI_EXPORT void xllm_rec_init_options_default(
    XLLM_InitOptions* init_options) {
  if (nullptr == init_options) return;
  *init_options = XLLM_INIT_REC_OPTIONS_DEFAULT;
}

XLLM_CAPI_EXPORT bool xllm_rec_initialize(
    XLLM_REC_Handler* handler,
    const char* model_path,
    const XLLM_InitOptions* init_options) {
  if (!handler || !model_path) return false;

  try {
    xllm::CpuAffinity::get_instance().set_cpu_affinity(
        init_options->cpu_affinity);

    XLLM_InitOptions xllm_init_options;
    xllm::helper::set_init_options(
        xllm::helper::BackendType::REC, init_options, &xllm_init_options);

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
        .backend("rec")
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

    // @TODO: Currently, gflags are configured through hard coding, which needs
    // to be improved in the future. For example, a separate gflags
    // configuration file can be provided to the so for setting gflags.
    //
    // REC so still has two configuration paths:
    // - some request/runtime code reads FLAGS_* directly
    // - master/worker construction reads xllm::Options
    //
    // The fields copied from init options below are read from FLAGS_* today.
    // beam_width/block_size/max_tokens/max_seqs are also represented in
    // Options, so duplicated values must stay aligned.
    FLAGS_beam_width = xllm_init_options.beam_width;
    FLAGS_max_decode_rounds = xllm_init_options.max_decode_rounds;
    FLAGS_max_seqs_per_batch = xllm_init_options.max_seqs_per_batch;
    FLAGS_max_tokens_per_batch = xllm_init_options.max_tokens_per_batch;
    FLAGS_block_size = xllm_init_options.block_size;
    if (xllm_init_options.flashinfer_workspace_buffer_size >
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      LOG(ERROR) << "flashinfer_workspace_buffer_size["
                 << xllm_init_options.flashinfer_workspace_buffer_size
                 << "] exceeds supported int32 range";
      return false;
    }
    xllm::BeamSearchConfig::get_instance()
        .beam_width(xllm_init_options.beam_width)
        .enable_block_copy_kernel(xllm_init_options.enable_block_copy_kernel)
        .enable_topk_sorted(xllm_init_options.enable_topk_sorted);
    xllm::RecConfig::get_instance()
        .max_decode_rounds(xllm_init_options.max_decode_rounds)
        .enable_rec_prefill_only(xllm_init_options.enable_rec_prefill_only)
        .enable_rec_fast_sampler(xllm_init_options.enable_rec_fast_sampler)
        .enable_xattention_one_stage(
            xllm_init_options.enable_xattention_one_stage)
        .rec_worker_max_concurrency(
            xllm_init_options.rec_worker_max_concurrency);
    if (xllm_init_options.request_queue_size > 0) {
      xllm::RecConfig::get_instance().request_queue_size(
          xllm_init_options.request_queue_size);
    }
    xllm::SchedulerConfig::get_instance()
        .max_seqs_per_batch(xllm_init_options.max_seqs_per_batch)
        .max_tokens_per_batch(xllm_init_options.max_tokens_per_batch)
        .max_tokens_per_chunk_for_prefill(
            xllm_init_options.max_tokens_per_chunk_for_prefill)
        .enable_schedule_overlap(xllm_init_options.enable_schedule_overlap)
        .enable_chunked_prefill(xllm_init_options.enable_chunked_prefill);
    xllm::KVCacheConfig::get_instance()
        .block_size(xllm_init_options.block_size)
        .enable_prefix_cache(xllm_init_options.enable_prefix_cache);
    xllm::ModelConfig::get_instance().flashinfer_workspace_buffer_size(
        static_cast<int32_t>(
            xllm_init_options.flashinfer_workspace_buffer_size));
    xllm::ExecutionConfig::get_instance()
        .enable_graph(xllm_init_options.enable_graph)
        .enable_prefill_piecewise_graph(
            xllm_init_options.enable_prefill_piecewise_graph)
        .enable_graph_mode_decode_no_padding(
            xllm_init_options.enable_graph_mode_decode_no_padding);

#if !defined(USE_NPU) && !defined(USE_CUDA)
    xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel(false);
#endif
    // Keep dual-source settings aligned with the Config values above.
    options.enable_graph(::xllm::ExecutionConfig::get_instance().enable_graph())
        .beam_width(::xllm::BeamSearchConfig::get_instance().beam_width())
        .rec_worker_max_concurrency(
            ::xllm::RecConfig::get_instance().rec_worker_max_concurrency());
    LOG(INFO)
        << "REC C API runtime config:"
        << ", enable_rec_prefill_only="
        << ::xllm::RecConfig::get_instance().enable_rec_prefill_only()
        << ", enable_constrained_decoding="
        << ::xllm::RecConfig::get_instance().enable_constrained_decoding()
        << ", enable_prefix_cache="
        << ::xllm::KVCacheConfig::get_instance().enable_prefix_cache()
        << ", enable_schedule_overlap="
        << ::xllm::SchedulerConfig::get_instance().enable_schedule_overlap()
        << ", enable_chunked_prefill="
        << ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()
        << ", enable_rec_fast_sampler="
        << ::xllm::RecConfig::get_instance().enable_rec_fast_sampler()
        << ", max_decode_rounds="
        << ::xllm::RecConfig::get_instance().max_decode_rounds();

    handler->master = std::make_unique<xllm::RecMaster>(options);
    handler->master->run();

    size_t available_cpu_cores_count =
        static_cast<size_t>(xllm::CpuAffinity::get_available_cpu_cores_count());
    LOG(INFO) << "Available CPU cores count " << available_cpu_cores_count;
    size_t thread_num = std::clamp(
        (available_cpu_cores_count == 0) ? 8 : available_cpu_cores_count / 2,
        static_cast<size_t>(8),
        static_cast<size_t>(16));

    auto thread_factory = std::make_shared<xllm::CpuAffinityThreadFactory>(
        /*prefix=*/"XllmRecExec");
    handler->executor = std::make_unique<folly::CPUThreadPoolExecutor>(
        thread_num, std::move(thread_factory));

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

XLLM_CAPI_EXPORT void xllm_rec_request_params_default(
    XLLM_RequestParams* request_params) {
  if (nullptr == request_params) return;
  *request_params = XLLM_REC_REQUEST_PARAMS_DEFAULT;
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_text_completions(
    XLLM_REC_Handler* handler,
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
      xllm::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      prompt,
      nullptr,
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_token_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const int32_t* token_ids,
    size_t token_size,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !token_ids ||
      token_size == 0) {
    return xllm::helper::build_error_response(
        "", XLLM_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  std::vector<int> token_ids_vec;
  for (int i = 0; i < token_size; i++) {
    token_ids_vec.push_back(token_ids[i]);
  }

  return xllm::helper::handle_inference_request(
      handler,
      xllm::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      token_ids_vec,
      nullptr,
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_multimodal_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const int32_t* token_ids,
    size_t token_size,
    const XLLM_MM_Data* mm_data,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  if (!handler || !model_id || *model_id == '\0' || !token_ids ||
      token_size == 0) {
    return xllm::helper::build_error_response(
        "", XLLM_StatusCode::kInvalidRequest, "Invalid input parameters");
  }

  if (!mm_data) {
    return xllm_rec_token_completions(
        handler, model_id, token_ids, token_size, timeout_ms, request_params);
  }

  xllm::MMData internal_mm_data;
  try {
    bool ret = xllm::helper::convert_xllm_mm_data_to_internal(mm_data,
                                                              internal_mm_data);
    if (!ret) {
      return xllm::helper::build_error_response(
          "", XLLM_StatusCode::kInternalError, "Fail in mm_data conversion");
    }
  } catch (const std::exception& e) {
    return xllm::helper::build_error_response(
        "",
        XLLM_StatusCode::kInternalError,
        "Critical error in mm_data conversion: " + std::string(e.what()));
  }

  std::vector<int> token_ids_vec;
  for (int i = 0; i < token_size; i++) {
    token_ids_vec.push_back(token_ids[i]);
  }

  return xllm::helper::handle_inference_request(
      handler,
      xllm::helper::InferenceType::REC_COMPLETIONS,
      model_id,
      token_ids_vec,
      static_cast<void*>(&internal_mm_data),
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_chat_completions(
    XLLM_REC_Handler* handler,
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
      xllm::helper::InferenceType::REC_CHAT_COMPLETIONS,
      model_id,
      xllm_messages,
      nullptr,
      timeout_ms,
      request_params);
}

XLLM_CAPI_EXPORT void xllm_rec_free_response(XLLM_Response* resp) {
  return xllm::helper::xllm_free_response(resp);
}
