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

#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/futures/Promise.h>
#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "c_api/default.h"
#include "c_api/types.h"
#include "core/common/instance_name.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/rec_master.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/util/rec_model_utils.h"

/**
 * @brief Opaque handle for LLM inference instance
 */
struct XLLM_LLM_Handler {
  /** Flag indicating if LLM instance is initialized and ready for inference */
  bool initialized{false};

  /** List of loaded model IDs (for model existence validation) */
  std::vector<std::string> model_ids;

  /** Core controller for LLM runtime management */
  std::unique_ptr<xllm::LLMMaster> master;

  /** Thread pool for asynchronous inference task scheduling */
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor;
};

/**
 * @brief Opaque handle for REC (Recommendation) inference instance
 */
struct XLLM_REC_Handler {
  /** Flag indicating if REC instance is initialized and ready for inference */
  bool initialized{false};

  /** Selected REC pipeline type for the loaded model */
  xllm::RecPipelineType pipeline_type{xllm::RecPipelineType::kLlmRecDefault};

  /** List of loaded recommendation model IDs */
  std::vector<std::string> model_ids;

  /** Core controller for REC runtime management */
  std::unique_ptr<xllm::RecMaster> master;

  /** Thread pool for asynchronous recommendation task scheduling */
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor;
};

namespace xllm {
namespace helper {

enum class BackendType { LLM = 0, VLM = 1, REC = 2 };

enum class InferenceType {
  LLM_COMPLETIONS = 0,
  LLM_CHAT_COMPLETIONS = 1,
  REC_COMPLETIONS = 2,
  REC_CHAT_COMPLETIONS = 3,
  REC_TOKENID_COMPLETIONS = 4,
};

#define XLLM_SET_META_STRING_FIELD(DST, SRC_STR)                              \
  do {                                                                        \
    static_assert(sizeof(DST) > 1, "Destination buffer is too small");        \
    strncpy(                                                                  \
        (char*)(DST), (SRC_STR).c_str(), XLLM_META_STRING_FIELD_MAX_LEN - 1); \
    (DST)[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';                         \
  } while (0)

/**
 * @brief Thread-safe glog initialization for xLLM framework
 * @note This API is idempotent (multiple calls have same effect as single call)
 * @note Thread-safe: protected by pthread mutex to prevent race condition
 * @param log_dir Directory to store log files (empty = current directory)
 */
void init_log(const std::string& log_dir);

/**
 * @brief Safely shutdown glog and release resources
 * @note Call this function before program exit (optional but recommended)
 */
void shutdown_log();

/**
 * @brief Set init options, merge default options
 */
void set_init_options(BackendType backend_type,
                      const XLLM_InitOptions* init_options,
                      XLLM_InitOptions* xllm_init_options);

/**
 * @brief Transfer C API request params to xLLM internal request params
 */
void transfer_request_params(InferenceType inference_type,
                             const XLLM_RequestParams* request_params,
                             xllm::RequestParams* xllm_request_params);

/**
 * @brief Build error response for failed inference requests
 */
XLLM_Response* build_error_response(const std::string& request_id,
                                    XLLM_StatusCode status_code,
                                    const std::string& error_info);

/**
 * @brief Build success response for completed inference requests
 */
XLLM_Response* build_success_response(const InferenceType& inference_type,
                                      const xllm::RequestOutput& output,
                                      xllm::RecPipelineType rec_pipeline_type,
                                      const std::string& request_id,
                                      int64_t created_time,
                                      const std::string& model);

/**
 * @brief Generic inference request handler (template function)
 */
template <typename HandlerType, typename InputType>
XLLM_Response* handle_inference_request(
    HandlerType* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const InputType& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

/**
 * @brief Safely free all memory allocated in XLLM_Response
 */
void xllm_free_response(XLLM_Response* resp);

/**
 * @brief Generate unique request ID for tracing
 */
std::string generate_request_id();

torch::ScalarType xllm_dtype_to_torch_scalar_type(XLLM_DataType dtype);

torch::Tensor convert_xllm_tensor_to_torch(const XLLM_Tensor& xllm_tensor);

xllm::MMDataItem convert_xllm_mm_item_to_internal(
    const XLLM_MM_Item& xllm_item);

bool convert_xllm_mm_data_to_internal(const XLLM_MM_Data* mm_data,
                                      xllm::MMData& internal_mm_data);
}  // namespace helper
}  // namespace xllm
