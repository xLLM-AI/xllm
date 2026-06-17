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

#include "rec_master.h"

#include <absl/strings/str_join.h>
#include <absl/time/time.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <limits>
#include <string>
#include <unordered_set>
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
#include "util/utils.h"

namespace xllm {

namespace {

constexpr int32_t kDefaultPlaceholderToken = 20152019;
constexpr const char* kOneRecSparseEmbeddingName = "sparse_embedding";
constexpr const char* kOneRecDecoderContextEmbeddingName =
    "decoder_context_embedding";

std::string format_tensor_shape(const proto::InferInputTensor& tensor) {
  std::vector<std::string> dims;
  dims.reserve(tensor.shape_size());
  for (int i = 0; i < tensor.shape_size(); ++i) {
    dims.emplace_back(std::to_string(tensor.shape(i)));
  }
  return "[" + absl::StrJoin(dims, ", ") + "]";
}

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

bool process_onerec_inputs(
    const std::optional<std::vector<int>>& prompt_tokens,
    const std::optional<std::vector<proto::InferInputTensor>>& input_tensors,
    const ModelArgs& model_args,
    std::vector<int32_t>* local_prompt_tokens,
    MMData* processed_mm_data,
    OutputCallback callback) {
  if (prompt_tokens.has_value() && input_tensors.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "prompt_tokens and input_tensors cannot both be set");
    return false;
  }

  if (prompt_tokens.has_value()) {
    local_prompt_tokens->assign(prompt_tokens.value().begin(),
                                prompt_tokens.value().end());
  }

  if (input_tensors.has_value()) {
    if (input_tensors->empty()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "OneRec input_tensors cannot be empty");
      return false;
    }

    MMDict mm_dict;
    mm_dict.reserve(input_tensors->size());
    bool has_sparse_embedding = false;
    bool has_decoder_context_embedding = false;
    int64_t sparse_embedding_hidden_size = -1;
    int64_t decoder_context_hidden_size = -1;

    for (const auto& tensor : input_tensors.value()) {
      const auto& tensor_name = tensor.name();
      if (tensor_name != kOneRecSparseEmbeddingName &&
          tensor_name != kOneRecDecoderContextEmbeddingName) {
        CALLBACK_WITH_ERROR(
            StatusCode::INVALID_ARGUMENT,
            "OneRec input_tensors only supports 'sparse_embedding' and "
            "'decoder_context_embedding', got '" +
                tensor_name + "'");
        return false;
      }
      if (mm_dict.find(tensor_name) != mm_dict.end()) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Duplicate OneRec input tensor: " + tensor_name);
        return false;
      }
      if (!tensor.has_contents()) {
        CALLBACK_WITH_ERROR(
            StatusCode::INVALID_ARGUMENT,
            "OneRec input tensor '" + tensor_name + "' has no contents");
        return false;
      }
      if (tensor.data_type() != proto::DataType::FLOAT) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "OneRec input tensor '" + tensor_name +
                                "' must use FLOAT(fp32), got " +
                                proto::DataType_Name(tensor.data_type()));
        return false;
      }
      if (tensor.shape_size() != 2) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "OneRec input tensor '" + tensor_name +
                                "' must be 2-D [len, hidden], got " +
                                format_tensor_shape(tensor));
        return false;
      }

      const int64_t len = tensor.shape(0);
      const int64_t hidden = tensor.shape(1);
      if (len <= 0 || hidden <= 0) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "OneRec input tensor '" + tensor_name +
                                "' must have positive shape, got " +
                                format_tensor_shape(tensor));
        return false;
      }
      if (hidden != model_args.hidden_size()) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "OneRec input tensor '" + tensor_name +
                                "' hidden size mismatch, expected " +
                                std::to_string(model_args.hidden_size()) +
                                ", got " + std::to_string(hidden));
        return false;
      }

      if (len > std::numeric_limits<int64_t>::max() / hidden) {
        CALLBACK_WITH_ERROR(
            StatusCode::INVALID_ARGUMENT,
            "OneRec input tensor '" + tensor_name + "' shape is too large");
        return false;
      }

      const int64_t expected_numel = len * hidden;
      const int64_t actual_numel =
          static_cast<int64_t>(tensor.contents().fp32_contents_size());
      if (expected_numel != actual_numel) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "OneRec input tensor '" + tensor_name +
                                "' fp32 contents size mismatch, expected " +
                                std::to_string(expected_numel) + ", got " +
                                std::to_string(actual_numel));
        return false;
      }

      try {
        mm_dict[tensor_name] =
            util::convert_rec_tensor_to_torch(tensor).to(torch::kBFloat16);
      } catch (const std::exception& e) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to parse OneRec input tensor '" +
                                tensor_name + "': " + e.what());
        return false;
      }

      if (tensor_name == kOneRecSparseEmbeddingName) {
        has_sparse_embedding = true;
        sparse_embedding_hidden_size = hidden;
      } else {
        has_decoder_context_embedding = true;
        decoder_context_hidden_size = hidden;
      }
    }

    if (!has_sparse_embedding) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "OneRec input_tensors must include 'sparse_embedding'");
      return false;
    }
    if (has_decoder_context_embedding &&
        sparse_embedding_hidden_size != decoder_context_hidden_size) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "OneRec tensor hidden size mismatch: sparse_embedding=" +
              std::to_string(sparse_embedding_hidden_size) +
              ", decoder_context_embedding=" +
              std::to_string(decoder_context_hidden_size));
      return false;
    }

    *processed_mm_data = MMData(MMType::EMBEDDING, mm_dict);
  }

  if (local_prompt_tokens->empty() && !processed_mm_data->valid()) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Rec model requires prompt_tokens or input_tensors to be provided");
    return false;
  }

  return true;
}

bool process_llmrec_with_mm_data_inputs(
    const std::vector<int>& prompt_tokens,
    std::optional<MMData> mm_data,
    std::vector<int32_t>* local_prompt_tokens,
    MMData* processed_mm_data,
    int32_t hidden_size,
    OutputCallback callback) {
  local_prompt_tokens->assign(prompt_tokens.begin(), prompt_tokens.end());

  if (!mm_data.has_value()) {
    return true;
  }

  std::vector<torch::Tensor> all_indices_list;
  std::vector<torch::Tensor> all_values_list;

  MMData mm_data_s = mm_data.value();
  if (!mm_data_s.hold<MMItemVec>()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "MMData need be item vec type");
    return false;
  }

  int64_t last_end = 0;
  MMItemVec mm_items = mm_data_s.items<MMItemVec>();
  for (auto& mm_item : mm_items) {
    const MMItemState::TokenPos token_pos = mm_item.state().token_pos();
    std::optional<torch::Tensor> mm_value =
        mm_item.get<torch::Tensor>("tensor");

    if (!mm_value.has_value()) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "Rec model requires embedding in mm data to be provided");
      return false;
    }

    int64_t start = static_cast<int64_t>(token_pos.offset);
    int64_t end = static_cast<int64_t>(token_pos.offset + token_pos.length);
    torch::Tensor tensor = mm_value.value();

    if (start < last_end || end >= local_prompt_tokens->size()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Token pos in mm data is wrong");
      return false;
    }

    last_end = end;

    if (tensor.dim() != 2) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Embedding in mm data is invalid");
      return false;
    }

    if (tensor.size(0) != token_pos.length) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Token length is not match to embedding");
      return false;
    }

    if (tensor.size(1) != hidden_size) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Embedding size is not match to model hidden size");
      return false;
    }

    torch::Tensor range_indices = torch::arange(start, end);
    all_indices_list.push_back(range_indices);
    all_values_list.push_back(tensor);
  }

  torch::Tensor total_indices = torch::cat(all_indices_list, 0);
  torch::Tensor total_values = torch::cat(all_values_list, 0);

  MMDict mm_dict;
  mm_dict["MULTI_MODAL_INDICES"] = total_indices;
  mm_dict["MULTI_MODAL_VALUES"] = total_values;
  *processed_mm_data = MMData(MMType::EMBEDDING, mm_dict);

  return true;
}

}  // namespace

// ============================================================
// RecMasterPipeline base class default implementations
// ============================================================
std::shared_ptr<Request> RecMaster::RecMasterPipeline::generate_request(
    std::string /*prompt*/,
    std::optional<std::vector<int>> /*prompt_tokens*/,
    std::optional<std::vector<proto::InferInputTensor>> /*input_tensors*/,
    const RequestParams& /*sp*/,
    OutputCallback callback) {
  CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                      "This pipeline does not support prompt-based input");
  return nullptr;
}

std::shared_ptr<Request> RecMaster::RecMasterPipeline::generate_request(
    const std::vector<int>& prompt_tokens,
    std::optional<MMData> mm_data,
    const RequestParams& sp,
    OutputCallback callback) {
  CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                      "This pipeline does not support raw input");
  return nullptr;
}

std::shared_ptr<Request>
RecMaster::RecMasterPipeline::generate_onerec_request_common(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    const RequestParams& sp,
    OutputCallback callback,
    bool build_stop_checker) {
  Timer timer;
  std::vector<int32_t> local_prompt_tokens;
  MMData processed_mm_data;

  if (!process_onerec_inputs(prompt_tokens,
                             input_tensors,
                             master_.model_args_,
                             &local_prompt_tokens,
                             &processed_mm_data,
                             callback)) {
    return nullptr;
  }

  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  return master_.build_request_common(std::move(prompt),
                                      std::move(local_prompt_tokens),
                                      std::move(processed_mm_data),
                                      sp,
                                      callback,
                                      build_stop_checker);
}

// ============================================================
// LlmRecMasterPipeline implementation (pure qwen3, no mm_data)
// ============================================================
RecMaster::LlmRecMasterPipeline::LlmRecMasterPipeline(RecMaster& master)
    : RecMasterPipeline(master) {}

std::shared_ptr<Request> RecMaster::LlmRecMasterPipeline::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> /*input_tensors*/,
    const RequestParams& sp,
    OutputCallback callback) {
  Timer timer;
  std::vector<int32_t> local_prompt_tokens;
  MMData processed_mm_data;

  // LlmRec without mm_data: use prompt_tokens or tokenize prompt string
  if (prompt_tokens.has_value()) {
    local_prompt_tokens.assign(prompt_tokens.value().begin(),
                               prompt_tokens.value().end());
  } else if (!prompt.empty()) {
    // Tokenize prompt string if prompt_tokens not provided
    if (!master_.tokenizer_) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Tokenizer is required for prompt-based input");
      return nullptr;
    }
    std::vector<int> tmp_tokens;
    if (!master_.tokenizer_->encode(
            prompt, &tmp_tokens, sp.add_special_tokens)) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "Failed to tokenize prompt");
      return nullptr;
    }
    local_prompt_tokens.assign(tmp_tokens.begin(), tmp_tokens.end());
  }

  if (local_prompt_tokens.empty()) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "LlmRec requires prompt or prompt_tokens to be provided");
    return nullptr;
  }

  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());

  return master_.build_request_common(std::move(prompt),
                                      std::move(local_prompt_tokens),
                                      std::move(processed_mm_data),
                                      sp,
                                      callback,
                                      /*build_stop_checker=*/true);
}

// ============================================================
// LlmRecWithMmDataMasterPipeline implementation (qwen3 with embedding)
// ============================================================
RecMaster::LlmRecWithMmDataMasterPipeline::LlmRecWithMmDataMasterPipeline(
    RecMaster& master)
    : RecMasterPipeline(master) {}

std::shared_ptr<Request>
RecMaster::LlmRecWithMmDataMasterPipeline::generate_request(
    const std::vector<int>& prompt_tokens,
    std::optional<MMData> mm_data,
    const RequestParams& sp,
    OutputCallback callback) {
  std::vector<int32_t> local_prompt_tokens;
  MMData processed_mm_data;

  int32_t hidden_size = master_.model_args_.hidden_size();
  bool ret = process_llmrec_with_mm_data_inputs(prompt_tokens,
                                                mm_data,
                                                &local_prompt_tokens,
                                                &processed_mm_data,
                                                hidden_size,
                                                callback);
  if (!ret) {
    return nullptr;
  }

  return master_.build_request_common(std::string(""),
                                      std::move(local_prompt_tokens),
                                      std::move(processed_mm_data),
                                      sp,
                                      callback,
                                      /*build_stop_checker=*/true);
}

// ============================================================
// OneRecPrefillOnlyMasterPipeline implementation (OneRec with input_tensors)
// ============================================================
RecMaster::OneRecPrefillOnlyMasterPipeline::OneRecPrefillOnlyMasterPipeline(
    RecMaster& master)
    : RecMasterPipeline(master) {}

std::shared_ptr<Request>
RecMaster::OneRecPrefillOnlyMasterPipeline::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    const RequestParams& sp,
    OutputCallback callback) {
  return generate_onerec_request_common(std::move(prompt),
                                        std::move(prompt_tokens),
                                        std::move(input_tensors),
                                        sp,
                                        callback,
                                        /*build_stop_checker=*/false);
}

std::shared_ptr<Request>
RecMaster::OneRecXAttentionMasterPipeline::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    const RequestParams& sp,
    OutputCallback callback) {
  return generate_onerec_request_common(std::move(prompt),
                                        std::move(prompt_tokens),
                                        std::move(input_tensors),
                                        sp,
                                        callback,
                                        /*build_stop_checker=*/true);
}

// ============================================================
// RecMaster pipeline factory (static method)
// ============================================================
std::unique_ptr<RecMaster::RecMasterPipeline> RecMaster::create_pipeline(
    RecPipelineType type,
    RecMaster& master) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return std::make_unique<LlmRecMasterPipeline>(master);
    case RecPipelineType::kLlmRecWithMmData:
      return std::make_unique<LlmRecWithMmDataMasterPipeline>(master);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecPrefillOnlyMasterPipeline>(master);
    case RecPipelineType::kOneRecXAttentionPipeline:
      return std::make_unique<OneRecXAttentionMasterPipeline>(master);
    default:
      LOG(FATAL) << "Unknown RecMaster pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

RecMaster::RecMaster(const Options& options)
    : Master(options, EngineType::REC) {
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

  // Create pipelines based on rec_type
  auto rec_model_kind = get_rec_model_kind(model_args_.model_type());
  CHECK(rec_model_kind != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_args_.model_type();
  auto pipeline_type = get_rec_pipeline_type(rec_model_kind);
  pipeline_ = create_pipeline(pipeline_type, *this);

  // For LlmRec, also create mm_data pipeline for raw input interface
  if (rec_type_ == RecType::kLlmRec) {
    mm_data_pipeline_ =
        create_pipeline(RecPipelineType::kLlmRecWithMmData, *this);
  }
}

void RecMaster::run() {
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
                     return pipeline_->generate_request(
                         std::move(prompt),
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
                     return pipeline_->generate_request(
                         std::move(prompt),
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
                     return mm_data_pipeline_->generate_request(
                         std::move(prompt_tokens),
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

std::shared_ptr<Request> RecMaster::build_request_common(
    std::string prompt,
    std::vector<int32_t> prompt_tokens,
    MMData mm_data,
    const RequestParams& sp,
    OutputCallback callback,
    bool build_stop_checker) {
  int32_t max_context_len = model_args_.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    int32_t max_tokens_per_req = options_.max_tokens_per_batch();
    if (rec_type_ == RecType::kLlmRec && is_rec_multi_round_mode()) {
      CHECK_GT(options_.max_seqs_per_batch(), 0)
          << "max_seqs_per_batch must be greater than 0 in multi-round mode";
      max_tokens_per_req /= options_.max_seqs_per_batch();
    }
    max_context_len = std::min(max_context_len, max_tokens_per_req);
  }
  if (prompt_tokens.size() >= max_context_len) {
    LOG(ERROR) << "Prompt is too long: " << prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is too long");
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 5120;
    max_tokens = kDefaultMaxTokens;
  }

  size_t capacity = prompt_tokens.size() + max_tokens +
                    options_.num_speculative_tokens() + /*bonus_token*/ 1;
  if (options_.enable_schedule_overlap()) {
    capacity += options_.num_speculative_tokens() + 1;
  }
  const size_t best_of = sp.best_of.value_or(sp.n);

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = sp.frequency_penalty;
  sampling_param.presence_penalty = sp.presence_penalty;
  sampling_param.repetition_penalty = sp.repetition_penalty;
  sampling_param.temperature = sp.temperature;
  sampling_param.top_p = sp.top_p;
  sampling_param.top_k = sp.top_k;
  sampling_param.logprobs = sp.logprobs;
  sampling_param.top_logprobs = sp.top_logprobs;
  sampling_param.is_embeddings = sp.is_embeddings;
  sampling_param.beam_width = sp.beam_width;
  sampling_param.num_return_sequences = sp.num_return_sequences;
  if (best_of > sp.n) {
    sampling_param.logprobs = true;
  }

  bool stream = sp.streaming;
  if (best_of != sp.n) {
    stream = false;
  }

  StoppingChecker stopping_checker;
  if (build_stop_checker) {
    std::unordered_set<int32_t> stop_tokens;
    if (sp.stop_token_ids.has_value()) {
      const auto& stop_token_ids = sp.stop_token_ids.value();
      stop_tokens.insert(stop_token_ids.begin(), stop_token_ids.end());
    } else {
      stop_tokens = model_args_.stop_token_ids();
    }

    std::vector<std::vector<int32_t>> stop_sequences;
    if (sp.stop.has_value()) {
      if (!tokenizer_) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Tokenizer is required for stop sequences");
        return nullptr;
      }
      for (const auto& s : sp.stop.value()) {
        std::vector<int> tmp_tokens;
        if (!tokenizer_->encode(s, &tmp_tokens)) {
          CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                              "Failed to encode stop sequence");
          LOG(ERROR) << "Failed to encode stop sequence: " << s;
          return nullptr;
        }
        stop_sequences.push_back(std::move(tmp_tokens));
      }
    }

    stopping_checker =
        StoppingChecker(max_tokens,
                        max_context_len - options_.num_speculative_tokens(),
                        model_args_.eos_token_id(),
                        sp.ignore_eos,
                        std::move(stop_tokens),
                        std::move(stop_sequences));
  }

  RequestState req_state(std::move(prompt),
                         std::move(prompt_tokens),
                         std::move(mm_data),
                         std::move(sampling_param),
                         std::move(stopping_checker),
                         capacity,
                         sp.n,
                         best_of,
                         sp.logprobs,
                         stream,
                         sp.echo,
                         sp.skip_special_tokens,
                         options_.enable_schedule_overlap(),
                         callback,
                         nullptr,
                         sp.decode_address);
  req_state.rec_type = rec_type_;
  req_state.bos_token_id = model_args_.bos_token_id();
  auto request = std::make_shared<Request>(sp.request_id,
                                           sp.x_request_id,
                                           sp.x_request_time,
                                           std::move(req_state),
                                           sp.service_request_id,
                                           sp.source_xservice_addr);
  return request;
}

}  // namespace xllm
