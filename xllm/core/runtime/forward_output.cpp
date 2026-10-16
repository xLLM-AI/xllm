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

#include "common/macros.h"
#include "common/metrics.h"
#include "core/framework/config/eplb_config.h"
#include "runtime/forward_params.h"
#include "util/tensor_helper.h"
#include "util/timer.h"
#include "util/utils.h"
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

}  // namespace

bool RawForwardOutput::from_proto(const proto::ForwardOutput& pb_output) {
  Timer timer;
  RawForwardOutput decoded_output;
  size_t seq_nums = pb_output.outputs().size();
  decoded_output.outputs.reserve(seq_nums);
  size_t expert_load_data_size = pb_output.expert_load_data().size();
  decoded_output.expert_load_data.reserve(expert_load_data_size);
  decoded_output.expert_load_data.assign(pb_output.expert_load_data().begin(),
                                         pb_output.expert_load_data().end());
  decoded_output.src_seq_idxes.reserve(pb_output.src_seq_idxes().size());
  decoded_output.src_seq_idxes.assign(pb_output.src_seq_idxes().begin(),
                                      pb_output.src_seq_idxes().end());
  decoded_output.out_tokens.reserve(pb_output.out_tokens().size());
  decoded_output.out_tokens.assign(pb_output.out_tokens().begin(),
                                   pb_output.out_tokens().end());
  decoded_output.out_logprobs.reserve(pb_output.out_logprobs().size());
  decoded_output.out_logprobs.assign(pb_output.out_logprobs().begin(),
                                     pb_output.out_logprobs().end());
  const size_t beam_src_seq_count = decoded_output.src_seq_idxes.size();
  const size_t beam_token_count = decoded_output.out_tokens.size();
  const size_t beam_logprob_count = decoded_output.out_logprobs.size();
  if ((beam_src_seq_count != 0 || beam_token_count != 0 ||
       beam_logprob_count != 0) &&
      (beam_src_seq_count != beam_token_count ||
       beam_src_seq_count != beam_logprob_count)) {
    LOG(ERROR) << "Mismatched beam output field counts";
    return false;
  }
  decoded_output.json_object_errors.reserve(
      pb_output.json_object_errors_size());
  for (const proto::JsonObjectOutputError& pb_error :
       pb_output.json_object_errors()) {
    decoded_output.json_object_errors.push_back(
        {pb_error.sample_sequence_id(), pb_error.message()});
  }
  decoded_output.prepared_token = pb_output.prepared_token();
  for (size_t i = 0; i < seq_nums; ++i) {
    const proto::SquenceOutput& pb_seq_out = pb_output.outputs()[i];
    RawSampleOutput s;
    size_t token_nums = pb_seq_out.tokens().size();
    s.tokens.reserve(token_nums);
    for (size_t j = 0; j < token_nums; ++j) {
      RawToken t;
      const proto::Token& pb_token = pb_seq_out.tokens()[j];
      t.id = pb_token.id();
      switch (pb_token.lp_case()) {
        case proto::Token::kEmpty:
          break;
        case proto::Token::kLogprob:
          t.logprob = pb_token.logprob();
          break;
        default:
          break;
      }
      if (pb_token.top_tokens_size() != pb_token.top_logprobs_size()) {
        LOG(ERROR) << "Mismatched top-k token and logprob counts";
        return false;
      }
      t.top_tokens.assign(pb_token.top_tokens().begin(),
                          pb_token.top_tokens().end());
      t.top_logprobs.assign(pb_token.top_logprobs().begin(),
                            pb_token.top_logprobs().end());
      t.embeddings.assign(pb_token.embeddings().vals().begin(),
                          pb_token.embeddings().vals().end());
      s.tokens.emplace_back(std::move(t));
    }
    s.mm_embeddings.reserve(pb_seq_out.mm_embeddings().tensors_size());
    for (const auto& pb_tensor : pb_seq_out.mm_embeddings().tensors()) {
      torch::Tensor tensor = util::proto_to_torch(pb_tensor);
      if (!tensor.defined()) {
        LOG(ERROR) << "Failed to deserialize multimodal embedding tensor";
        return false;
      }
      s.mm_embeddings.emplace_back(std::move(tensor));
    }
    s.speculative_token_stats.accepted_tokens =
        pb_seq_out.speculative_token_stats().accepted_tokens();
    s.speculative_token_stats.proposed_tokens =
        pb_seq_out.speculative_token_stats().proposed_tokens();
    decoded_output.outputs.emplace_back(std::move(s));
  }
  if (!decoded_output.dit_forward_output.from_proto(
          pb_output.dit_forward_output())) {
    return false;
  }
  *this = std::move(decoded_output);
  COUNTER_ADD(proto_latency_seconds_proto2o, timer.elapsed_seconds());
  return true;
}

bool ForwardOutput::to_proto(
    const std::vector<SpeculativeTokenStats>& speculative_token_stats,
    proto::ForwardOutput* output) const {
  const auto& next_tokens = sample_output.next_tokens;
  const auto& logprobs = sample_output.logprobs;
  const auto& top_tokens = sample_output.top_tokens;
  const auto& top_logprobs = sample_output.top_logprobs;
  const auto& embeddings = sample_output.embeddings;
  const auto& mm_embeddings = sample_output.mm_embeddings;
  const auto& src_seq_idxes = beam_search_output.src_seq_idxes;
  const auto& out_tokens = beam_search_output.out_tokens;
  const auto& out_logprobs = beam_search_output.out_logprobs;
  if (output == nullptr) {
    return false;
  }
  proto::ForwardOutput encoded_output;
  proto::ForwardOutput* pb_forward_output = &encoded_output;
  Timer timer;
  if (!valid_token_output(
          next_tokens, logprobs, top_tokens, top_logprobs, embeddings) ||
      !valid_host_tensor(expert_load_data, torch::kInt64) ||
      !valid_host_tensor(src_seq_idxes, torch::kInt32) ||
      !valid_host_tensor(out_tokens, torch::kInt32) ||
      !valid_host_tensor(out_logprobs, torch::kFloat32)) {
    LOG(ERROR) << "Invalid forward output tensor layout.";
    return false;
  }
  // LLM decode fills next_tokens; DiT text diffusion (e.g. Cola-DLM) may leave
  // it undefined and only populate dit_text_output. Guard before
  // .size()/.dim().
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
  if (next_tokens.defined() && next_tokens.numel() > 0 &&
      next_tokens.size(0) < static_cast<int64_t>(num_seqs)) {
    LOG(ERROR) << "Token rows do not match forward output rows.";
    return false;
  }

  const auto append_token = [](const Token& token,
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
  };

  pb_forward_output->mutable_outputs()->Reserve(num_seqs);
  for (int32_t output_idx = 0; output_idx < num_seqs; ++output_idx) {
    proto::SquenceOutput* pb_seq_out = pb_forward_output->add_outputs();
    if (next_tokens.defined() && next_tokens.dim() == 2) {
      const auto curr_embeddings =
          embeddings.defined() ? embeddings[output_idx] : embeddings;
      int32_t num_tokens = static_cast<int32_t>(next_tokens.size(1));
      pb_seq_out->mutable_tokens()->Reserve(num_tokens);
      for (int32_t i = 0; i < num_tokens; ++i) {
        const Token token = sample_output.token_at(output_idx, i);
        if (token.id == -1) {
          break;
        }
        const auto token_embeddings =
            curr_embeddings.defined() ? curr_embeddings[i] : curr_embeddings;
        append_token(token, token_embeddings, pb_seq_out);
      }
    } else {
      pb_seq_out->mutable_tokens()->Reserve(1);
      Token token(-1);

      // Handle case where next_tokens might be empty but embeddings have data
      if (next_tokens.defined() && next_tokens.numel() > 0) {
        token = sample_output.token_at(output_idx);
      }

      const auto token_embeddings =
          embeddings.defined() ? embeddings[output_idx] : embeddings;
      append_token(token, token_embeddings, pb_seq_out);
      if (output_idx < static_cast<int32_t>(mm_embeddings.size())) {
        for (const auto& tensor : mm_embeddings[output_idx]) {
          if ((!tensor.defined() || !tensor.device().is_cpu() ||
               !tensor.is_contiguous() ||
               !util::torch_to_proto(
                   tensor,
                   pb_seq_out->mutable_mm_embeddings()->add_tensors()))) {
            return false;
          }
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
    }
  }

  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    pb_forward_output->set_prepared_token(prepared_token);

    if (expert_load_data.defined()) {
      Slice<int64_t> expert_load_data_flattened_slice =
          tensor_slice<int64_t>(expert_load_data.view({-1}));
      ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_expert_load_data(),
                          expert_load_data_flattened_slice);
    }
  }

  if (src_seq_idxes.defined() && src_seq_idxes.numel() > 0) {
    Slice<int32_t> src_seq_idxes_slice = tensor_slice(src_seq_idxes);
    ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_src_seq_idxes(),
                        src_seq_idxes_slice);
  }
  if (out_tokens.defined() && out_tokens.numel() > 0) {
    Slice<int32_t> out_tokens_slice = tensor_slice(out_tokens);
    ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_out_tokens(),
                        out_tokens_slice);
  }
  if (out_logprobs.defined() && out_logprobs.numel() > 0) {
    Slice<float> out_logprobs_slice = tensor_slice<float>(out_logprobs);
    ADD_VECTOR_TO_PROTO(pb_forward_output->mutable_out_logprobs(),
                        out_logprobs_slice);
  }
  if ((!dit_forward_output.tensors.empty() ||
       !dit_forward_output.audio_tensors.empty() ||
       !dit_forward_output.text_output.empty()) &&
      !dit_forward_output.to_proto(
          pb_forward_output->mutable_dit_forward_output())) {
    return false;
  }
  for (const JsonObjectOutputError& error : json_object_errors) {
    proto::JsonObjectOutputError* pb_error =
        pb_forward_output->add_json_object_errors();
    pb_error->set_sample_sequence_id(error.sample_sequence_id);
    pb_error->set_message(error.message);
  }
  output->Swap(pb_forward_output);
  COUNTER_ADD(proto_latency_seconds_o2proto, timer.elapsed_seconds());
  return true;
}

}  // namespace xllm
