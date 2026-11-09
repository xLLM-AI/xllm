/* Copyright 2026 The xLLM Authors.

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

#include <algorithm>
#include <cstdint>
#include <utility>

#include "core/common/macros.h"
#include "core/common/metrics.h"
#include "core/common/types.h"
#include "core/distributed_runtime/worker_model_codec.h"
#include "core/framework/config/eplb_config.h"
#include "core/runtime/forward_params.h"
#include "core/util/tensor_codec.h"
#include "core/util/tensor_helper.h"
#include "core/util/timer.h"
#include "core/util/utils.h"
#include "tensor.pb.h"
#include "worker.pb.h"

namespace xllm {
namespace {

bool valid_host_tensor(const torch::Tensor& tensor,
                       torch::ScalarType scalar_type) {
  return !tensor.defined() ||
         (tensor.device().is_cpu() && tensor.is_contiguous() &&
          tensor.scalar_type() == scalar_type);
}

bool valid_token_output(const torch::Tensor& next_tokens,
                        const torch::Tensor& logprobs,
                        const torch::Tensor& top_tokens,
                        const torch::Tensor& top_logprobs,
                        const torch::Tensor& embeddings) {
  if (next_tokens.defined() &&
      (!next_tokens.device().is_cpu() || !next_tokens.is_contiguous() ||
       (next_tokens.scalar_type() != torch::kInt32 &&
        next_tokens.scalar_type() != torch::kInt64) ||
       (next_tokens.dim() != 1 && next_tokens.dim() != 2))) {
    return false;
  }
  if (!valid_host_tensor(logprobs, torch::kFloat32) ||
      !valid_host_tensor(top_tokens, torch::kInt64) ||
      !valid_host_tensor(top_logprobs, torch::kFloat32) ||
      !valid_host_tensor(embeddings, torch::kFloat32)) {
    return false;
  }
  if (top_tokens.defined() != top_logprobs.defined() ||
      (top_tokens.defined() && !logprobs.defined())) {
    return false;
  }
  if (logprobs.defined() &&
      (!next_tokens.defined() || logprobs.sizes() != next_tokens.sizes())) {
    return false;
  }
  if (top_tokens.defined()) {
    if (!next_tokens.defined() || top_tokens.dim() != next_tokens.dim() + 1 ||
        top_tokens.sizes() != top_logprobs.sizes()) {
      return false;
    }
    for (int64_t dim = 0; dim < next_tokens.dim(); ++dim) {
      if (top_tokens.size(dim) != next_tokens.size(dim)) {
        return false;
      }
    }
  }
  if (embeddings.defined()) {
    const int64_t expected_rank =
        next_tokens.defined() && next_tokens.dim() == 2 ? 3 : 2;
    if (embeddings.dim() != expected_rank || embeddings.numel() == 0) {
      return false;
    }
    if (next_tokens.defined() && next_tokens.dim() == 2 &&
        embeddings.size(1) != next_tokens.size(1)) {
      return false;
    }
  }
  return true;
}

void append_token(const Token& token,
                  const torch::Tensor& token_embeddings,
                  proto::SquenceOutput* sequence_output) {
  proto::Token* pb_token = sequence_output->add_tokens();
  pb_token->set_id(token.id);
  if (token.logprob.has_value()) {
    pb_token->set_logprob(token.logprob.value());
  } else {
    pb_token->set_empty(true);
  }
  pb_token->mutable_top_tokens()->Reserve(
      static_cast<int32_t>(token.top_tokens.size()));
  for (const int64_t top_token : token.top_tokens) {
    pb_token->add_top_tokens(top_token);
  }
  pb_token->mutable_top_logprobs()->Reserve(
      static_cast<int32_t>(token.top_logprobs.size()));
  for (const float top_logprob : token.top_logprobs) {
    pb_token->add_top_logprobs(top_logprob);
  }
  if (token_embeddings.defined()) {
    Slice<float> embedding_slice = tensor_slice<float>(token_embeddings);
    ADD_VECTOR_TO_PROTO(pb_token->mutable_embeddings()->mutable_vals(),
                        embedding_slice);
  }
}

template <typename Response>
bool decode_token_output(const Response& response,
                         RawForwardOutput& output,
                         const char* model_name) {
  Timer timer;
  RawForwardOutput decoded_output;
  decoded_output.expert_load_data.assign(response.expert_load_data().begin(),
                                         response.expert_load_data().end());
  decoded_output.json_object_errors.reserve(response.json_object_errors_size());
  for (const proto::JsonObjectOutputError& error :
       response.json_object_errors()) {
    decoded_output.json_object_errors.emplace_back(
        JsonObjectOutputError{error.sample_sequence_id(), error.message()});
  }
  decoded_output.prepared_token = response.prepared_token();
  decoded_output.outputs.reserve(response.outputs_size());
  for (const proto::SquenceOutput& sequence : response.outputs()) {
    RawSampleOutput sample;
    sample.tokens.reserve(sequence.tokens_size());
    for (const proto::Token& token : sequence.tokens()) {
      if (token.top_tokens_size() != token.top_logprobs_size()) {
        LOG(ERROR) << "Mismatched " << model_name
                   << " top-k token and logprob counts";
        return false;
      }
      RawToken decoded_token;
      decoded_token.id = token.id();
      if (token.lp_case() == proto::Token::kLogprob) {
        decoded_token.logprob = token.logprob();
      }
      decoded_token.top_tokens.assign(token.top_tokens().begin(),
                                      token.top_tokens().end());
      decoded_token.top_logprobs.assign(token.top_logprobs().begin(),
                                        token.top_logprobs().end());
      decoded_token.embeddings.assign(token.embeddings().vals().begin(),
                                      token.embeddings().vals().end());
      sample.tokens.emplace_back(std::move(decoded_token));
    }
    sample.mm_embeddings.reserve(sequence.mm_embeddings().tensors_size());
    for (const proto::Tensor& encoded_tensor :
         sequence.mm_embeddings().tensors()) {
      torch::Tensor tensor = util::proto_to_torch(encoded_tensor);
      if (!tensor.defined()) {
        LOG(ERROR) << "Failed to deserialize " << model_name
                   << " multimodal embedding tensor";
        return false;
      }
      sample.mm_embeddings.emplace_back(std::move(tensor));
    }
    sample.speculative_token_stats.accepted_tokens =
        sequence.speculative_token_stats().accepted_tokens();
    sample.speculative_token_stats.proposed_tokens =
        sequence.speculative_token_stats().proposed_tokens();
    sample.speculative_token_stats.verified_rounds =
        sequence.speculative_token_stats().verified_rounds();
    decoded_output.outputs.emplace_back(std::move(sample));
  }
  output = std::move(decoded_output);
  COUNTER_ADD(proto_latency_seconds_proto2o, timer.elapsed_seconds());
  return true;
}

template <typename Response>
bool encode_token_output(
    const ForwardOutput& output,
    const std::vector<SpeculativeTokenStats>& speculative_token_stats,
    Response* response,
    const char* model_name) {
  const auto& next_tokens = output.sample_output.next_tokens;
  const auto& logprobs = output.sample_output.logprobs;
  const auto& top_tokens = output.sample_output.top_tokens;
  const auto& top_logprobs = output.sample_output.top_logprobs;
  const auto& embeddings = output.sample_output.embeddings;
  const auto& mm_embeddings = output.sample_output.mm_embeddings;
  if (response == nullptr) {
    return false;
  }
  Response encoded_output;
  Response* pb_forward_output = &encoded_output;
  Timer timer;
  if (!valid_token_output(
          next_tokens, logprobs, top_tokens, top_logprobs, embeddings) ||
      !valid_host_tensor(output.expert_load_data, torch::kInt64)) {
    LOG(ERROR) << "Invalid forward output tensor layout.";
    return false;
  }
  int32_t num_seqs =
      next_tokens.defined() ? static_cast<int32_t>(next_tokens.size(0)) : 0;
  if (embeddings.defined() && embeddings.numel() > 0) {
    num_seqs = std::max(num_seqs, static_cast<int32_t>(embeddings.size(0)));
  }
  if (!mm_embeddings.empty()) {
    num_seqs = std::max(num_seqs, static_cast<int32_t>(mm_embeddings.size()));
  }
  if (!speculative_token_stats.empty() &&
      speculative_token_stats.size() != static_cast<size_t>(num_seqs)) {
    LOG(ERROR) << "Speculative token stats must match forward output rows.";
    return false;
  }
  if (embeddings.defined() &&
      embeddings.size(0) < static_cast<int64_t>(num_seqs)) {
    LOG(ERROR) << "Embedding rows do not match forward output rows.";
    return false;
  }
  if (next_tokens.defined() && embeddings.defined() &&
      next_tokens.size(0) != embeddings.size(0)) {
    LOG(ERROR) << "Embedding rows do not match token rows.";
    return false;
  }
  if (next_tokens.defined() &&
      (next_tokens.dim() == 2 || next_tokens.numel() > 0) &&
      next_tokens.size(0) < static_cast<int64_t>(num_seqs)) {
    LOG(ERROR) << "Token rows do not match forward output rows.";
    return false;
  }

  pb_forward_output->mutable_outputs()->Reserve(num_seqs);
  for (int32_t output_idx = 0; output_idx < num_seqs; ++output_idx) {
    proto::SquenceOutput* pb_seq_out = pb_forward_output->add_outputs();
    if (next_tokens.defined() && next_tokens.dim() == 2) {
      const auto curr_embeddings =
          embeddings.defined() ? embeddings[output_idx] : embeddings;
      const int32_t num_tokens = static_cast<int32_t>(next_tokens.size(1));
      pb_seq_out->mutable_tokens()->Reserve(num_tokens);
      for (int32_t i = 0; i < num_tokens; ++i) {
        const Token token = output.sample_output.token_at(output_idx, i);
        if (token.id < 0) {
          break;
        }
        const auto token_embeddings =
            curr_embeddings.defined() ? curr_embeddings[i] : curr_embeddings;
        append_token(token, token_embeddings, pb_seq_out);
      }
    } else {
      pb_seq_out->mutable_tokens()->Reserve(1);
      Token token(-1);

      // Embedding-only results use a placeholder token.
      if (next_tokens.defined() && next_tokens.numel() > 0) {
        token = output.sample_output.token_at(output_idx);
      }
      const auto token_embeddings =
          embeddings.defined() ? embeddings[output_idx] : embeddings;
      append_token(token, token_embeddings, pb_seq_out);
    }
    if (output_idx < static_cast<int32_t>(mm_embeddings.size())) {
      const auto& tensors = mm_embeddings[output_idx];
      pb_seq_out->mutable_mm_embeddings()->mutable_tensors()->Reserve(
          static_cast<int32_t>(tensors.size()));
      for (const torch::Tensor& tensor : tensors) {
        if (!tensor.defined() || !tensor.device().is_cpu() ||
            !tensor.is_contiguous() ||
            !util::torch_to_proto(
                tensor, pb_seq_out->mutable_mm_embeddings()->add_tensors())) {
          return false;
        }
      }
    }
    if (!speculative_token_stats.empty()) {
      const SpeculativeTokenStats& stats =
          speculative_token_stats[static_cast<size_t>(output_idx)];
      proto::SpeculativeTokenStats* pb_stats =
          pb_seq_out->mutable_speculative_token_stats();
      pb_stats->set_accepted_tokens(stats.accepted_tokens);
      pb_stats->set_proposed_tokens(stats.proposed_tokens);
      pb_stats->set_verified_rounds(stats.verified_rounds);
    }
  }

  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    pb_forward_output->set_prepared_token(output.prepared_token);
    if (output.expert_load_data.defined()) {
      Slice<int64_t> expert_load_data_flattened_slice =
          tensor_slice<int64_t>(output.expert_load_data.view({-1}));
      ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_expert_load_data(),
                          expert_load_data_flattened_slice);
    }
  }

  for (const JsonObjectOutputError& error : output.json_object_errors) {
    proto::JsonObjectOutputError* pb_error =
        pb_forward_output->add_json_object_errors();
    pb_error->set_sample_sequence_id(error.sample_sequence_id);
    pb_error->set_message(error.message);
  }
  response->Swap(pb_forward_output);
  COUNTER_ADD(proto_latency_seconds_o2proto, timer.elapsed_seconds());
  return true;
}

}  // namespace

bool encode_llm_output(const ForwardOutput& output,
                       const std::vector<SpeculativeTokenStats>& stats,
                       proto::LLMForwardResponse* response) {
  return encode_token_output(output, stats, response, "llm");
}

bool decode_llm_output(const proto::LLMForwardResponse& response,
                       RawForwardOutput& output) {
  return decode_token_output(response, output, "llm");
}

bool encode_vlm_output(const ForwardOutput& output,
                       const std::vector<SpeculativeTokenStats>& stats,
                       proto::VLMForwardResponse* response) {
  return encode_token_output(output, stats, response, "vlm");
}

bool decode_vlm_output(const proto::VLMForwardResponse& response,
                       RawForwardOutput& output) {
  return decode_token_output(response, output, "vlm");
}

}  // namespace xllm
