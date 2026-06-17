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

#include "helper.h"

#include <glog/logging.h>
#include <pthread.h>
#include <torch/torch.h>

#include <atomic>
#include <string>

#include "core/common/global_flags.h"
#include "core/framework/config/rec_config.h"
#include "core/util/env_var.h"
#include "core/util/rec_model_utils.h"
#include "core/util/uuid.h"

namespace xllm {
namespace helper {
namespace {
thread_local ShortUUID short_uuid;
static std::atomic<bool> g_glog_inited = false;
static pthread_mutex_t g_log_init_mutex = PTHREAD_MUTEX_INITIALIZER;
}  // namespace

std::string generate_request_id() {
  return "xllm-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

void init_log(const std::string& log_dir) {
  if (g_glog_inited.load(std::memory_order_acquire)) {
    return;
  }

  pthread_mutex_lock(&g_log_init_mutex);
  if (!g_glog_inited.load(std::memory_order_relaxed)) {
    google::InitGoogleLogging("xllm");

    std::string log_prefix = log_dir.empty() ? "./" : log_dir + "/";
    google::SetLogDestination(google::INFO,
                              (log_prefix + "xllm.log.INFO.").c_str());
    google::SetLogDestination(google::WARNING,
                              (log_prefix + "xllm.log.WARNING.").c_str());
    google::SetLogDestination(google::ERROR,
                              (log_prefix + "xllm.log.ERROR.").c_str());
    google::SetStderrLogging(google::FATAL);
    g_glog_inited.store(true, std::memory_order_release);
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}

void shutdown_log() {
  if (!g_glog_inited.load(std::memory_order_acquire)) {
    return;
  }

  pthread_mutex_lock(&g_log_init_mutex);
  if (g_glog_inited.load(std::memory_order_relaxed)) {
    google::ShutdownGoogleLogging();
    g_glog_inited.store(false, std::memory_order_release);
  }
  pthread_mutex_unlock(&g_log_init_mutex);
}

void set_init_options(BackendType backend_type,
                      const XLLM_InitOptions* init_options,
                      XLLM_InitOptions* xllm_init_options) {
  if (init_options == nullptr) {
    if (backend_type == BackendType::LLM) {
      memcpy(xllm_init_options,
             &XLLM_INIT_LLM_OPTIONS_DEFAULT,
             sizeof(XLLM_InitOptions));
    } else if (backend_type == BackendType::REC) {
      memcpy(xllm_init_options,
             &XLLM_INIT_REC_OPTIONS_DEFAULT,
             sizeof(XLLM_InitOptions));
    }
  } else {
    memcpy(xllm_init_options, init_options, sizeof(XLLM_InitOptions));
  }

  return;
}

void transfer_request_params(InferenceType inference_type,
                             const XLLM_RequestParams* request_params,
                             xllm::RequestParams* xllm_request_params) {
  XLLM_RequestParams final_request_params;
  if (nullptr == request_params) {
    if (inference_type == InferenceType::LLM_COMPLETIONS ||
        inference_type == InferenceType::LLM_CHAT_COMPLETIONS) {
      memcpy(&final_request_params,
             &XLLM_LLM_REQUEST_PARAMS_DEFAULT,
             sizeof(XLLM_RequestParams));
    } else if (inference_type == InferenceType::REC_COMPLETIONS ||
               inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
      memcpy(&final_request_params,
             &XLLM_REC_REQUEST_PARAMS_DEFAULT,
             sizeof(XLLM_RequestParams));
    }
  } else {
    memcpy(&final_request_params, request_params, sizeof(XLLM_RequestParams));
  }

  xllm_request_params->echo = final_request_params.echo;
  xllm_request_params->offline = final_request_params.offline;
  xllm_request_params->logprobs = final_request_params.logprobs;
  xllm_request_params->ignore_eos = final_request_params.ignore_eos;

  xllm_request_params->best_of = final_request_params.best_of;
  xllm_request_params->top_k = final_request_params.top_k;
  xllm_request_params->top_p = final_request_params.top_p;
  xllm_request_params->n = final_request_params.n;
  xllm_request_params->max_tokens = final_request_params.max_tokens;
  xllm_request_params->frequency_penalty =
      final_request_params.frequency_penalty;
  xllm_request_params->presence_penalty = final_request_params.presence_penalty;
  xllm_request_params->repetition_penalty =
      final_request_params.repetition_penalty;
  xllm_request_params->beam_width = final_request_params.beam_width;
  xllm_request_params->num_return_sequences =
      final_request_params.num_return_sequences;
  xllm_request_params->top_logprobs = final_request_params.top_logprobs;
  xllm_request_params->temperature = final_request_params.temperature;
  xllm_request_params->request_id = final_request_params.request_id;
  xllm_request_params->ttlt_slo_ms = final_request_params.ttlt_slo_ms;
  xllm_request_params->ttft_slo_ms = final_request_params.ttft_slo_ms;
  xllm_request_params->tpot_slo_ms = final_request_params.tpot_slo_ms;

  return;
}

XLLM_Response* build_error_response(const std::string& request_id,
                                    XLLM_StatusCode status_code,
                                    const std::string& error_info) {
  XLLM_Response* response = new XLLM_Response();
  CHECK(nullptr != response);

  response->status_code = status_code;
  strncpy(
      response->error_info, error_info.c_str(), XLLM_ERROR_INFO_MAX_LEN - 1);
  response->error_info[XLLM_ERROR_INFO_MAX_LEN - 1] = '\0';

  XLLM_SET_META_STRING_FIELD(response->id, request_id);

  LOG(ERROR) << "Request [" << request_id << "] error: " << error_info
             << " (code: " << static_cast<int>(response->status_code) << ")";

  return response;
}

XLLM_Response* build_success_response(const InferenceType& inference_type,
                                      const RequestOutput& output,
                                      RecPipelineType rec_pipeline_type,
                                      const std::string& request_id,
                                      int64_t created_time,
                                      const std::string& model) {
  XLLM_Response* response = new XLLM_Response();
  CHECK(nullptr != response);

  response->status_code = XLLM_StatusCode::kSuccess;
  response->created = created_time;
  XLLM_SET_META_STRING_FIELD(response->id, request_id);
  XLLM_SET_META_STRING_FIELD(response->model, model);

  if (inference_type == InferenceType::LLM_COMPLETIONS ||
      inference_type == InferenceType::REC_COMPLETIONS) {
    snprintf(response->object, sizeof(response->object), "text_completion");
  } else if (inference_type == InferenceType::LLM_CHAT_COMPLETIONS ||
             inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
    snprintf(response->object, sizeof(response->object), "chat.completion");
  }

  response->choices.entries_size = output.outputs.size();
  response->choices.entries = new XLLM_Choice[response->choices.entries_size]();
  CHECK(nullptr != response->choices.entries);
  const bool is_rec_inference =
      inference_type == InferenceType::REC_COMPLETIONS ||
      inference_type == InferenceType::REC_CHAT_COMPLETIONS;
  const bool is_onerec_pipeline =
      is_rec_inference && is_onerec_pipeline_type(rec_pipeline_type);
  if (is_onerec_pipeline) {
    response->rec_outputs.entries_size = output.outputs.size();
    response->rec_outputs.entries =
        new XLLM_RecOutput[response->rec_outputs.entries_size]();
    CHECK(nullptr != response->rec_outputs.entries);
  }

  int32_t total_item_count = 0;
  const int32_t total_threshold =
      ::xllm::RecConfig::get_instance().total_conversion_threshold();

  for (int i = 0; i < output.outputs.size(); i++) {
    const auto& seq_output = output.outputs[i];
    XLLM_Choice& choice = response->choices.entries[i];
    choice.index = seq_output.index;
    XLLM_RecOutput* rec_output = nullptr;
    if (response->rec_outputs.entries != nullptr) {
      rec_output = &response->rec_outputs.entries[i];
      rec_output->index = seq_output.index;
    }

    if (inference_type == InferenceType::LLM_COMPLETIONS ||
        inference_type == InferenceType::REC_COMPLETIONS) {
      size_t text_len = seq_output.text.length();
      choice.text = new char[text_len + 1];
      CHECK(nullptr != choice.text);
      strncpy(choice.text, seq_output.text.c_str(), text_len + 1);
      choice.text[text_len] = '\0';
    } else if (inference_type == InferenceType::LLM_CHAT_COMPLETIONS ||
               inference_type == InferenceType::REC_CHAT_COMPLETIONS) {
      choice.message = new XLLM_ChatMessage();
      CHECK(nullptr != choice.message);

      snprintf(choice.message->role, sizeof(choice.message->role), "assistant");
      size_t text_len = seq_output.text.length();
      choice.message->content = new char[text_len + 1];
      CHECK(nullptr != choice.message->content);
      strncpy(choice.message->content, seq_output.text.c_str(), text_len + 1);
      choice.message->content[text_len] = '\0';
    }

    if (seq_output.finish_reason.has_value()) {
      XLLM_SET_META_STRING_FIELD(choice.finish_reason,
                                 seq_output.finish_reason.value());
    }

    if (seq_output.token_ids.size() > 0) {
      choice.token_size = seq_output.token_ids.size();
      choice.token_ids = new int32_t[choice.token_size];
      CHECK(nullptr != choice.token_ids);
      for (int j = 0; j < choice.token_size; j++) {
        choice.token_ids[j] = seq_output.token_ids[j];
      }
    }

    if (seq_output.logprobs.has_value()) {
      choice.logprobs.entries_size = seq_output.logprobs.value().size();
      choice.logprobs.entries =
          new XLLM_LogProb[choice.logprobs.entries_size]();
      CHECK(nullptr != choice.logprobs.entries);
      for (int j = 0; j < seq_output.logprobs.value().size(); j++) {
        const auto& logprob = seq_output.logprobs.value()[j];
        XLLM_LogProb& xllm_logprob = choice.logprobs.entries[j];

        xllm_logprob.token_id = logprob.token_id;
        xllm_logprob.logprob = logprob.logprob;
      }
    }

    if (is_onerec_pipeline &&
        ::xllm::RecConfig::get_instance().enable_convert_tokens_to_item() &&
        rec_output != nullptr) {
      size_t copied_item_count = 0;
      if (!seq_output.item_ids_list.empty()) {
        copied_item_count =
            std::min(seq_output.item_ids_list.size(),
                     static_cast<size_t>(
                         std::max(total_threshold - total_item_count, 0)));
        if (copied_item_count > 0) {
          rec_output->item_ids_size = copied_item_count;
          rec_output->item_ids = new int64_t[copied_item_count];
          CHECK(nullptr != rec_output->item_ids);
          for (size_t j = 0; j < copied_item_count; ++j) {
            rec_output->item_ids[j] = seq_output.item_ids_list[j];
          }
          total_item_count += static_cast<int32_t>(copied_item_count);
        }
      } else if (seq_output.item_ids.has_value() &&
                 total_item_count < total_threshold) {
        rec_output->item_ids_size = 1;
        rec_output->item_ids = new int64_t[1];
        CHECK(nullptr != rec_output->item_ids);
        rec_output->item_ids[0] = seq_output.item_ids.value();
        ++total_item_count;
      }
    }

    if (is_onerec_pipeline &&
        ::xllm::RecConfig::get_instance().enable_output_sku_logprobs() &&
        !seq_output.token_ids_logprobs.empty() && rec_output != nullptr) {
      rec_output->rec_token_logprobs_size =
          seq_output.token_ids_logprobs.size();
      rec_output->rec_token_logprobs =
          new float[rec_output->rec_token_logprobs_size];
      CHECK(nullptr != rec_output->rec_token_logprobs);
      for (size_t j = 0; j < rec_output->rec_token_logprobs_size; ++j) {
        if (seq_output.token_ids_logprobs[j].has_value()) {
          rec_output->rec_token_logprobs[j] =
              seq_output.token_ids_logprobs[j].value();
        } else {
          rec_output->rec_token_logprobs[j] = 0.0f;
        }
      }
    }
  }

  if (output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response->usage.prompt_tokens = usage.num_prompt_tokens;
    response->usage.completion_tokens = usage.num_generated_tokens;
    response->usage.total_tokens = usage.num_total_tokens;
    response->usage.cached_tokens = usage.num_cached_tokens;
  }

  return response;
}

template <typename HandlerType, typename InputType>
XLLM_Response* handle_inference_request(
    HandlerType* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const InputType& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params) {
  CHECK(nullptr != handler);

  std::string request_id;
  if (nullptr != request_params && strlen(request_params->request_id) > 0) {
    request_id = request_params->request_id;
  } else {
    request_id = generate_request_id();
  }

  if (!handler->initialized) {
    return build_error_response(
        request_id, XLLM_StatusCode::kNotInitialized, "LLM is not initialized");
  }

  if (std::find(handler->model_ids.begin(),
                handler->model_ids.end(),
                model_id) == handler->model_ids.end()) {
    return build_error_response(request_id,
                                XLLM_StatusCode::kModelNotFound,
                                "Specified model ID not loaded: " + model_id);
  }

  xllm::RequestParams xllm_request_params;
  transfer_request_params(inference_type, request_params, &xllm_request_params);
  xllm_request_params.request_id = request_id;
  RecPipelineType rec_pipeline_type = RecPipelineType::kLlmRecDefault;
  if constexpr (std::is_same_v<HandlerType, XLLM_REC_Handler>) {
    rec_pipeline_type = handler->pipeline_type;
    if (::xllm::RecConfig::get_instance().enable_output_sku_logprobs() &&
        is_onerec_pipeline_type(rec_pipeline_type)) {
      xllm_request_params.logprobs = true;
    }
  }

  const int64_t created_time = absl::ToUnixSeconds(absl::Now());

  try {
    auto promise_ptr = std::make_shared<folly::Promise<XLLM_Response*>>();
    auto future = promise_ptr->getSemiFuture();

    auto on_request_complete = [model_id,
                                request_id,
                                created_time,
                                inference_type,
                                rec_pipeline_type,
                                weak_promise = std::weak_ptr(promise_ptr)](
                                   const RequestOutput& req_output) -> bool {
      if (auto locked_promise = weak_promise.lock()) {
        try {
          if (req_output.status.has_value()) {
            if (req_output.status.value().ok()) {
              locked_promise->setValue(build_success_response(inference_type,
                                                              req_output,
                                                              rec_pipeline_type,
                                                              request_id,
                                                              created_time,
                                                              model_id));
            } else {
              locked_promise->setValue(build_error_response(
                  request_id,
                  XLLM_StatusCode::kInternalError,
                  "RequestOutput status is not ok, message: " +
                      req_output.status.value().message()));
            }
          } else {
            locked_promise->setValue(
                build_error_response(request_id,
                                     XLLM_StatusCode::kInternalError,
                                     "RequestOutput status has no value"));
          }
          return true;
        } catch (const std::exception& e) {
          LOG(ERROR) << "Build response failed: " << e.what();
          locked_promise->setValue(build_error_response(
              request_id,
              XLLM_StatusCode::kInternalError,
              "Build response failed: " + std::string(e.what())));
        }
      }
      return false;
    };

    if constexpr (std::is_same_v<HandlerType, XLLM_LLM_Handler>) {
      handler->master->handle_request(input,
                                      std::nullopt,
                                      xllm_request_params,
                                      std::nullopt,
                                      on_request_complete);
    } else if constexpr (std::is_same_v<HandlerType, XLLM_REC_Handler>) {
      if constexpr (std::is_same_v<InputType, std::vector<int>>) {
        if (nullptr != extra) {
          xllm::MMData* mm_data =
              dynamic_cast<xllm::MMData*>(static_cast<xllm::MMData*>(extra));
          CHECK(nullptr != mm_data);

          std::optional<xllm::MMData> opt_mm_data = std::move(*mm_data);
          handler->master->handle_request(
              input, opt_mm_data, xllm_request_params, on_request_complete);

        } else {
          handler->master->handle_request("",
                                          input,
                                          std::nullopt,
                                          xllm_request_params,
                                          on_request_complete);
        }
      } else {
        handler->master->handle_request(input,
                                        std::nullopt,
                                        std::nullopt,
                                        xllm_request_params,
                                        on_request_complete);
      }
    } else {
      CHECK(false);
    }

    return std::move(future)
        .via(handler->executor.get())
        .within(std::chrono::milliseconds(timeout_ms))
        .thenTry([request_id](
                     folly::Try<XLLM_Response*>&& result) -> XLLM_Response* {
          if (result.hasValue()) return std::move(result).value();

          std::string error_msg;
          XLLM_StatusCode code = XLLM_StatusCode::kInternalError;
          try {
            result.throwUnlessValue();
          } catch (const folly::FutureTimeout& e) {
            error_msg = "Request timed out: " + std::string(e.what());
            code = XLLM_StatusCode::kTimeout;
          } catch (const std::exception& e) {
            error_msg = "Inference failed: " + std::string(e.what());
          } catch (...) {
            error_msg = "Inference failed with unknown exception";
          }
          return build_error_response(request_id, code, error_msg);
        })
        .get();

  } catch (...) {
    return build_error_response(request_id,
                                XLLM_StatusCode::kInternalError,
                                "Critical error in inference pipeline");
  }
}

void xllm_free_response(XLLM_Response* resp) {
  if (nullptr == resp) {
    return;
  }

  if (nullptr != resp->choices.entries) {
    for (int i = 0; i < resp->choices.entries_size; ++i) {
      XLLM_Choice& choice = resp->choices.entries[i];

      if (nullptr != choice.text) {
        delete[] choice.text;
        choice.text = nullptr;
      }

      if (nullptr != choice.message) {
        if (nullptr != choice.message->content) {
          delete[] choice.message->content;
          choice.message->content = nullptr;
        }
        delete choice.message;
        choice.message = nullptr;
      }

      if (nullptr != choice.token_ids) {
        delete[] choice.token_ids;
        choice.token_ids = nullptr;
        choice.token_size = 0;
      }

      if (nullptr != choice.logprobs.entries) {
        delete[] choice.logprobs.entries;
        choice.logprobs.entries = nullptr;
      }
      choice.logprobs.entries_size = 0;
    }

    delete[] resp->choices.entries;
    resp->choices.entries = nullptr;
  }

  resp->choices.entries_size = 0;
  if (nullptr != resp->rec_outputs.entries) {
    for (size_t i = 0; i < resp->rec_outputs.entries_size; ++i) {
      XLLM_RecOutput& rec_output = resp->rec_outputs.entries[i];
      if (nullptr != rec_output.item_ids) {
        delete[] rec_output.item_ids;
        rec_output.item_ids = nullptr;
        rec_output.item_ids_size = 0;
      }
      if (nullptr != rec_output.rec_token_logprobs) {
        delete[] rec_output.rec_token_logprobs;
        rec_output.rec_token_logprobs = nullptr;
        rec_output.rec_token_logprobs_size = 0;
      }
    }
    delete[] resp->rec_outputs.entries;
    resp->rec_outputs.entries = nullptr;
  }
  resp->rec_outputs.entries_size = 0;
  delete resp;

  return;
}

torch::ScalarType xllm_dtype_to_torch_scalar_type(XLLM_DataType dtype) {
  switch (dtype) {
    case XLLM_DTYPE_UNDEFINED:
      throw std::runtime_error(
          "XLLM_DTYPE_UNDEFINED is not a valid dtype for tensor conversion");
    case XLLM_DTYPE_FLOAT16:
      return torch::kFloat16;
    case XLLM_DTYPE_FLOAT32:
      return torch::kFloat32;
    case XLLM_DTYPE_FLOAT64:
      return torch::kFloat64;
    case XLLM_DTYPE_BFLOAT16:
      return torch::kBFloat16;
    case XLLM_DTYPE_INT8:
      return torch::kInt8;
    case XLLM_DTYPE_INT16:
      return torch::kInt16;
    case XLLM_DTYPE_INT32:
      return torch::kInt32;
    case XLLM_DTYPE_INT64:
      return torch::kInt64;
    case XLLM_DTYPE_BOOL:
      return torch::kBool;
    case XLLM_DTYPE_STRING:
      throw std::runtime_error(
          "String dtype is not supported for torch::Tensor");
    default:
      throw std::runtime_error("Unsupported XLLM_DataType: " +
                               std::to_string(dtype));
  }
}

torch::Tensor convert_xllm_tensor_to_torch(const XLLM_Tensor& xllm_tensor) {
  if (xllm_tensor.data == nullptr) {
    throw std::runtime_error("XLLM_Tensor data pointer is null");
  }

  torch::ScalarType scalar_type =
      xllm_dtype_to_torch_scalar_type(xllm_tensor.dtype);

  std::vector<int64_t> shape;
  for (int i = 0; i < xllm_tensor.dims.rank; ++i) {
    int dim = xllm_tensor.dims.dim[i];
    if (dim > 0) {
      shape.push_back(dim);
    }
  }

  if (shape.empty()) {
    throw std::runtime_error("XLLM_Tensor all dimensions are invalid value");
  }

  torch::Tensor tensor =
      torch::from_blob(const_cast<void*>(xllm_tensor.data), shape, scalar_type)
          .clone();

  return tensor;
}

xllm::MMDataItem convert_xllm_mm_item_to_internal(
    const XLLM_MM_Item& xllm_item) {
  uint32_t xllm_type_val = static_cast<uint32_t>(xllm_item.type);
  xllm::MMType::Value internal_val = xllm::MMType::NONE;

  switch (xllm_type_val) {
    case XLLM_MM_TYPE_EMBEDDING:
      internal_val = xllm::MMType::EMBEDDING;
      break;
    case XLLM_MM_TYPE_IMAGE:
      internal_val = xllm::MMType::IMAGE;
      break;
    case XLLM_MM_TYPE_VIDEO:
      internal_val = xllm::MMType::VIDEO;
      break;
    case XLLM_MM_TYPE_AUDIO:
      internal_val = xllm::MMType::AUDIO;
      break;
    case XLLM_MM_TYPE_NONE:
      internal_val = xllm::MMType::NONE;
      break;
    default:
      throw std::runtime_error(std::string("Unsupported XLLM_MM_Type: ") +
                               std::to_string(xllm_type_val));
  }

  xllm::MMType item_type(internal_val);
  xllm::MMDataItem internal_item(item_type);

  xllm::MMItemState& state = internal_item.mutable_state();
  xllm::MMItemState::TokenPos& token_pos = state.mutable_token_pos();
  token_pos.offset = xllm_item.state.token_pos.offset;
  token_pos.length = xllm_item.state.token_pos.length;

  if (xllm_item.data.is_single_tensor) {
    torch::Tensor tensor =
        convert_xllm_tensor_to_torch(xllm_item.data.data.tensor);
    internal_item.add("tensor", tensor);
  } else {
    std::vector<torch::Tensor> tensor_list;
    const XLLM_Tensors& xllm_tensors = xllm_item.data.data.tensors;
    for (size_t i = 0; i < xllm_tensors.entries_size; ++i) {
      tensor_list.push_back(
          convert_xllm_tensor_to_torch(xllm_tensors.entries[i]));
    }
    internal_item.add("tensor_list", tensor_list);
  }

  return internal_item;
}

bool convert_xllm_mm_data_to_internal(const XLLM_MM_Data* mm_data,
                                      xllm::MMData& internal_mm_data) {
  if (mm_data == nullptr || mm_data->type_mask == XLLM_MM_TYPE_NONE) {
    return false;
  }

  xllm::MMType::Value internal_val =
      static_cast<xllm::MMType::Value>(mm_data->type_mask);
  xllm::MMType mm_type(internal_val);

  if (mm_data->is_dict) {
    const XLLM_MM_Dict& xllm_dict = mm_data->data.dict;
    xllm::MMDict internal_dict;

    for (size_t i = 0; i < xllm_dict.entries_size; ++i) {
      const XLLM_MM_DictEntry& xllm_entry = xllm_dict.entries[i];
      xllm::MMKey key(xllm_entry.key);

      const XLLM_MM_Value& xllm_value = xllm_entry.value;
      if (xllm_value.is_single_tensor) {
        torch::Tensor tensor =
            convert_xllm_tensor_to_torch(xllm_value.data.tensor);
        internal_dict.insert({key, tensor});
      } else {
        std::vector<torch::Tensor> tensor_list;
        const XLLM_Tensors& xllm_tensors = xllm_value.data.tensors;
        for (size_t j = 0; j < xllm_tensors.entries_size; ++j) {
          tensor_list.push_back(
              convert_xllm_tensor_to_torch(xllm_tensors.entries[j]));
        }
        internal_dict.insert({key, tensor_list});
      }
    }

    internal_mm_data.set<xllm::MMDict>(mm_type, internal_dict);
  } else {
    const XLLM_MM_Items& xllm_items = mm_data->data.items;
    xllm::MMItemVec internal_item_vec;

    for (size_t i = 0; i < xllm_items.entries_size; ++i) {
      const XLLM_MM_Item& xllm_item = xllm_items.entries[i];

      xllm::MMDataItem internal_item =
          convert_xllm_mm_item_to_internal(xllm_item);
      internal_item_vec.push_back(std::move(internal_item));
    }

    internal_mm_data.set<xllm::MMItemVec>(mm_type, internal_item_vec);
  }

  return true;
}

// 1. LLM Handler + const char* (text completions)
template XLLM_Response* handle_inference_request<XLLM_LLM_Handler, const char*>(
    XLLM_LLM_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const char* const& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 2. LLM Handler + std::vector<xllm::Message> (chat completions)
template XLLM_Response*
handle_inference_request<XLLM_LLM_Handler, std::vector<xllm::Message>>(
    XLLM_LLM_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<xllm::Message>& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 3. REC Handler + const char* (REC completions)
template XLLM_Response* handle_inference_request<XLLM_REC_Handler, const char*>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const char* const& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 4. REC Handler + std::vector<xllm::Message> (REC chat completions)
template XLLM_Response*
handle_inference_request<XLLM_REC_Handler, std::vector<xllm::Message>>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<xllm::Message>& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

// 5. REC Handler + std::vector<int> (chat completions)
template XLLM_Response*
handle_inference_request<XLLM_REC_Handler, std::vector<int>>(
    XLLM_REC_Handler* handler,
    InferenceType inference_type,
    const std::string& model_id,
    const std::vector<int>& input,
    void* extra,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);
}  // namespace helper
}  // namespace xllm
