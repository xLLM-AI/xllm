/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "embedding_cache.h"

#include <glog/logging.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "util/utils.h"

namespace xllm {
namespace {

torch::Tensor to_cpu_int64_contiguous(const torch::Tensor& tensor) {
  torch::Tensor cpu_tensor = safe_to(tensor, torch::kCPU).contiguous();
  if (cpu_tensor.scalar_type() != torch::kInt64) {
    cpu_tensor = cpu_tensor.to(torch::kInt64);
  }
  return cpu_tensor;
}

}  // namespace

EmbeddingCache::EmbeddingCache(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No embeddings to allocate";
  decode_tails_.resize(total_nums);
}

void EmbeddingCache::write_prefill_target_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& next_tokens,
    const torch::Tensor& embeddings,
    const torch::Tensor& selected_token_idxes) {
  CHECK(next_tokens.defined()) << "prefill target tokens are undefined";
  CHECK(embeddings.defined()) << "prefill target embeddings are undefined";
  CHECK_EQ(next_tokens.dim(), 1) << "prefill target tokens should be [batch]";
  CHECK_EQ(embeddings.dim(), 2)
      << "prefill target embeddings should be [batch, hidden]";
  CHECK_EQ(next_tokens.size(0), static_cast<int64_t>(ids.size()))
      << "prefill target token count mismatch";
  CHECK(request_ids.empty() || request_ids.size() == ids.size())
      << "prefill target request id count mismatch";

  torch::Tensor target_embeddings = embeddings;
  if (target_embeddings.size(0) != static_cast<int64_t>(ids.size())) {
    CHECK(selected_token_idxes.defined())
        << "prefill target embedding selection index is undefined";
    CHECK_EQ(selected_token_idxes.numel(), static_cast<int64_t>(ids.size()))
        << "prefill target embedding selection count mismatch";
    torch::Tensor embedding_idxes = selected_token_idxes.to(
        torch::dtype(torch::kLong).device(target_embeddings.device()));
    target_embeddings =
        target_embeddings.index_select(/*dim=*/0, embedding_idxes);
  }
  CHECK_EQ(target_embeddings.size(0), static_cast<int64_t>(ids.size()))
      << "prefill target embedding count mismatch";

  torch::Tensor next_tokens_cpu = to_cpu_int64_contiguous(next_tokens);
  const int64_t* next_tokens_data = next_tokens_cpu.const_data_ptr<int64_t>();
  const int32_t num_ids = static_cast<int32_t>(ids.size());
  for (int32_t i = 0; i < num_ids; ++i) {
    const int64_t token = next_tokens_data[i];
    CHECK_GE(token, 0) << "prefill target token should be valid";
    CHECK_LE(token, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        << "prefill target token overflow";

    DecodeState state;
    state.valid = true;
    if (!request_ids.empty()) {
      state.request_id = request_ids[i];
    }
    state.all_draft_accepted = false;
    state.token_id = static_cast<int32_t>(token);
    state.position_offset = 0;
    state.embedding = target_embeddings.select(/*dim=*/0, i).detach();

    DecodeState& tail = mutable_tail(ids[i]);
    tail = std::move(state);
  }
}

void EmbeddingCache::write_mtp_bootstrap_context(
    int32_t embedding_id,
    const std::string& request_id,
    int32_t token_id,
    const torch::Tensor& embedding) {
  CHECK(embedding.defined()) << "MTP bootstrap embedding is undefined";
  CHECK_GE(token_id, 0) << "MTP bootstrap token should be valid";

  DecodeState state;
  state.valid = true;
  state.request_id = request_id;
  state.all_draft_accepted = false;
  state.token_id = token_id;
  state.position_offset = 0;
  state.embedding = embedding.detach();

  DecodeState& tail = mutable_tail(embedding_id);
  tail = std::move(state);
}

void EmbeddingCache::write_target_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    int32_t num_speculative_tokens) {
  CHECK(accepted_tokens.defined()) << "accepted target tokens are undefined";
  CHECK(accepted_embeddings.defined())
      << "accepted target embeddings are undefined";
  CHECK_EQ(accepted_tokens.dim(), 2)
      << "accepted target tokens should be [batch, width]";
  CHECK_EQ(accepted_embeddings.dim(), 3)
      << "accepted target embeddings should be [batch, width, hidden]";
  CHECK_EQ(accepted_tokens.size(0), static_cast<int64_t>(ids.size()))
      << "accepted token batch mismatch";
  CHECK(request_ids.empty() || request_ids.size() == ids.size())
      << "accepted request id count mismatch";
  CHECK_EQ(accepted_embeddings.size(0), static_cast<int64_t>(ids.size()))
      << "accepted embedding batch mismatch";
  CHECK_EQ(accepted_tokens.size(1), accepted_embeddings.size(1))
      << "accepted token/embedding width mismatch";
  CHECK_GE(num_speculative_tokens, 0) << "invalid speculative token count";

  torch::Tensor accepted_tokens_cpu = to_cpu_int64_contiguous(accepted_tokens);
  const int64_t* accepted_tokens_data =
      accepted_tokens_cpu.const_data_ptr<int64_t>();
  const int32_t num_ids = static_cast<int32_t>(ids.size());
  const int32_t token_width = static_cast<int32_t>(accepted_tokens_cpu.size(1));
  for (int32_t i = 0; i < num_ids; ++i) {
    int32_t accepted_len = 0;
    int32_t last_token_id = -1;
    int32_t correction_token = -1;
    int32_t correction_offset = -1;
    const int64_t row_offset = static_cast<int64_t>(i) * token_width;
    for (int32_t j = 0; j < token_width; ++j) {
      const int64_t token = accepted_tokens_data[row_offset + j];
      if (token < 0) {
        break;
      }
      CHECK_LE(token, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
          << "accepted token overflow";
      last_token_id = static_cast<int32_t>(token);
      correction_token = static_cast<int32_t>(token);
      correction_offset = j;
      ++accepted_len;
    }
    CHECK_GT(accepted_len, 0)
        << "each sequence must have at least one accepted target token";

    const int32_t last_idx = accepted_len - 1;
    DecodeState state;
    state.valid = true;
    if (!request_ids.empty()) {
      state.request_id = request_ids[i];
    }
    state.all_draft_accepted = accepted_len == num_speculative_tokens + 1;
    state.token_id = last_token_id;
    state.position_offset = last_idx;
    state.correction_token_id = correction_token;
    state.correction_position_offset = correction_offset;
    state.embedding = accepted_embeddings.select(/*dim=*/0, i)
                          .select(/*dim=*/0, last_idx)
                          .detach();
    if (last_idx > 0) {
      const int64_t prev_token =
          accepted_tokens_data[row_offset + last_idx - 1];
      state.prev_token_id = static_cast<int32_t>(prev_token);
      state.prev_embedding = accepted_embeddings.select(/*dim=*/0, i)
                                 .select(/*dim=*/0, last_idx - 1)
                                 .detach();
    }

    DecodeState& tail = mutable_tail(ids[i]);
    tail = std::move(state);
  }
}

void EmbeddingCache::set_placeholder(
    const torch::Tensor& embedding_placeholder) {
  embedding_placeholder_ = embedding_placeholder;
}

const torch::Tensor& EmbeddingCache::embedding_placeholder() const {
  return embedding_placeholder_;
}

std::vector<EmbeddingCache::DecodeState> EmbeddingCache::read_decode_states(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK(!ids.empty()) << "decode ids should not be empty";
  CHECK(request_ids.empty() || request_ids.size() == ids.size())
      << "decode request id count mismatch";
  std::vector<DecodeState> states;
  states.reserve(ids.size());
  for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) {
    const int32_t id = ids[i];
    const DecodeState& cached_state = get_tail(id);
    DecodeState state = cached_state;
    if (state.valid && !request_ids.empty() &&
        state.request_id != request_ids[i]) {
      state = DecodeState();
    }
    if (!state.valid) {
      state.token_id = 0;
      state.position_offset = 0;
      state.all_draft_accepted = false;
    } else {
      CHECK_GE(state.token_id, 0) << "decode entry missing target token id";
      CHECK(state.embedding.defined())
          << "decode entry missing target embedding";
      if (state.prev_token_id >= 0) {
        CHECK(state.prev_embedding.defined())
            << "decode entry missing previous target embedding";
      }
    }
    states.emplace_back(std::move(state));
  }
  return states;
}

std::vector<int32_t> EmbeddingCache::read_accepted_prefix_lengths(
    const std::vector<int32_t>& ids) const {
  CHECK(!ids.empty()) << "decode ids should not be empty";
  std::vector<int32_t> accepted_prefix_lengths;
  accepted_prefix_lengths.reserve(ids.size());
  for (int32_t id : ids) {
    const DecodeState& state = get_tail(id);
    CHECK_GE(state.correction_token_id, 0)
        << "decode entry missing correction token id";
    accepted_prefix_lengths.emplace_back(state.correction_position_offset + 1);
  }
  return accepted_prefix_lengths;
}

void EmbeddingCache::clear(const std::vector<int32_t>& ids) {
  for (int32_t id : ids) {
    DecodeState& tail = mutable_tail(id);
    tail = DecodeState();
  }
}

EmbeddingCache::DecodeState& EmbeddingCache::mutable_tail(
    int32_t embedding_id) {
  CHECK_GE(embedding_id, 0);
  CHECK_LT(static_cast<size_t>(embedding_id), decode_tails_.size());
  return decode_tails_[embedding_id];
}

const EmbeddingCache::DecodeState& EmbeddingCache::get_tail(
    int32_t embedding_id) const {
  CHECK_GE(embedding_id, 0);
  CHECK_LT(static_cast<size_t>(embedding_id), decode_tails_.size());
  return decode_tails_[embedding_id];
}

}  // namespace xllm
