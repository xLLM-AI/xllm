/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "anthropic.pb.h"
#include "chat.pb.h"
#include "common.pb.h"
#include "completion.pb.h"
#include "core/common/macros.h"
#include "core/common/types.h"
#include "core/framework/request/request.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/sample_slot.h"
#include "embedding.pb.h"
#include "multimodal.pb.h"
#include "rerank.pb.h"

namespace xllm {

enum class ResponseFormatType : int8_t {
  NONE = 0,
  JSON_OBJECT = 1,
};

// Defined in core/framework/sampling/sampling_params.h and
// core/framework/request/request_state.h respectively. Forward-declared here so
// the projection helpers below don't drag those (torch-heavy) headers into
// every translation unit that only needs RequestParams.
struct RequestSamplingParam;
struct SchedulerParam;
class Tokenizer;

class RequestParams final {
 public:
  RequestParams() = default;
  RequestParams(const proto::CompletionRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::ChatRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::MMChatRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::EmbeddingRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::MMEmbeddingRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::RerankRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);
  RequestParams(const proto::AnthropicMessagesRequest& request,
                const std::string& x_rid,
                const std::string& x_rtime);

  void set_x_request_id_if_absent(const std::string& fallback) {
    if (x_request_id.empty()) {
      x_request_id = fallback;
    }
  }

  bool verify_params(OutputCallback callback) const;

  // Projects the shared sampling-related fields into a RequestSamplingParam.
  // `best_of` is the effective best_of (== best_of.value_or(n)); when it
  // exceeds `n`, logprobs are forced on so a per-sequence logprob can be
  // produced. Model-specific fields/rules are intentionally left out and
  // layered on by the individual request factories: json_object (LLM), and the
  // beam-search fields beam_width/num_return_sequences plus their logprob
  // normalization, which differ per model (LLM normalizes, REC copies both, VLM
  // omits them).
  RequestSamplingParam to_sampling_param(size_t best_of) const;

  std::optional<std::string> prepare_sampling_constraints(
      RequestSamplingParam& sampling_param,
      const Tokenizer* tokenizer,
      int64_t vocab_size,
      int32_t eos_token_id,
      const std::unordered_set<int32_t>& model_stop_token_ids) const;

  // Projects the scheduler-related fields into a SchedulerParam. SLO/priority
  // weights are only meaningful for online requests, so they are left at their
  // defaults when `offline` is set.
  SchedulerParam to_scheduler_param() const;

  // request id
  std::string request_id;
  std::string service_request_id = "";
  std::string source_xservice_addr = "";
  std::string x_request_id;
  std::string x_request_time;

  bool streaming = false;

  // number of tokens to generate. truncated to model's max context length.
  uint32_t max_tokens = 16;

  // number of sequences to generate for each prompt.
  uint32_t n = 1;

  // number of sequences to generate for each prompt and select n best among.
  std::optional<uint32_t> best_of;

  // whether to include the original prompt in the completion response.
  bool echo = false;

  // frequency penalty to reduce the likelihood of generating the same word
  // multiple times. values between [-2.0, 2.0]. 0.0 means no penalty. default =
  // 0.0 Positive values penalize new tokens based on their existing frequency
  // in the text.
  float frequency_penalty = 0.0;

  // Presence penalty based on tokens in generated text, in [-2.0, 2.0].
  // Prompt tokens do not contribute to presence or frequency penalties.
  float presence_penalty = 0.0;

  // repetition penalty to penalize new tokens based on their occurrence in the
  // text. values > 1.0 encourage the model to use new tokens, while values
  // < 1.0 encourage the model to repeat tokens. default = 1.0
  float repetition_penalty = 1.0;

  // Finite, non-negative sampling temperature. Zero selects greedy decoding.
  // higher value will make the output more random.
  float temperature = 1.0;

  // top_p sampling cutoff, in (0.0, 1.0]. default = 1.0
  float top_p = 1.0;

  // top_k sampling cutoff. Zero and -1 disable the cutoff.
  int64_t top_k = 0;

  // Minimum probability relative to the largest probability, in [0, 1].
  float min_p = 0.0;

  std::optional<int64_t> seed;
  uint32_t min_tokens = 0;
  std::unordered_map<int32_t, float> logit_bias;
  std::optional<std::vector<int32_t>> allowed_token_ids;
  std::vector<std::string> bad_words;

  // whether to return the log probabilities of the tokens. default = false.
  bool logprobs = false;

  // number of top log probabilities to return. default = 0.
  int64_t top_logprobs = 0;

  // whether to skip special tokens in the output text. default = true.
  bool skip_special_tokens = true;

  // whether to include stop strings or stop tokens in the output text.
  // default = false.
  bool include_stop_str_in_output = false;

  // whether to ignore the end of sequence token. default = false.
  bool ignore_eos = false;

  // whether to get the embeddings of the tokens. used by embeddings model.
  bool is_embeddings = false;

  // the list of strings to stop generating further tokens.
  std::optional<std::vector<std::string>> stop;

  // the list of token ids to stop generating further tokens.
  std::optional<std::vector<int32_t>> stop_token_ids;

  // decode address.
  std::string decode_address;

  // JSON-based tools (replacing proto_tools)
  std::vector<xllm::JsonTool> tools;

  std::string tool_choice = "auto";

  bool offline = false;

  int32_t ttlt_slo_ms = std::numeric_limits<int32_t>::max();

  int32_t ttft_slo_ms = std::numeric_limits<int32_t>::max();

  int32_t tpot_slo_ms = std::numeric_limits<int32_t>::max();

  int32_t tpot_priority_weight = 1;

  int32_t ttft_priority_weight = 1;

  int32_t ttlt_priority_weight = 1;

  int32_t priority_weight = 1;

  RequestPriority priority = RequestPriority::NORMAL;

  // beam search
  int32_t beam_width = 0;
  int32_t num_return_sequences = 0;

  bool add_special_tokens = false;

  nlohmann::json chat_template_kwargs = nlohmann::json::object();

  ResponseFormatType response_format = ResponseFormatType::NONE;
  std::string response_format_error;

  bool is_sample_request = false;

  std::vector<SampleSlot> sample_slots;
};

}  // namespace xllm
