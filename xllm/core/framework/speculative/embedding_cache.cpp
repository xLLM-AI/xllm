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

#include "core/framework/speculative/embedding_cache.h"

#include <glog/logging.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "util/tensor_helper.h"

namespace xllm {

EmbeddingCache::EmbeddingCache(int32_t total_nums, bool retain_replay_span)
    : retain_replay_span_(retain_replay_span) {
  CHECK_GT(total_nums, 0) << "No embeddings to allocate";
  decode_tails_.resize(total_nums);
}

void EmbeddingCache::write_prefill_target_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& next_tokens,
    const torch::Tensor& embeddings) {
  CHECK(next_tokens.defined()) << "prefill target tokens are undefined";
  CHECK(embeddings.defined()) << "prefill target embeddings are undefined";
  CHECK_EQ(next_tokens.dim(), 1) << "prefill target tokens should be [batch]";
  CHECK_EQ(embeddings.dim(), 2)
      << "prefill target embeddings should be [batch, hidden]";
  CHECK_EQ(next_tokens.size(0), static_cast<int64_t>(ids.size()))
      << "prefill target token count mismatch";
  CHECK(request_ids.empty() || request_ids.size() == ids.size())
      << "prefill target request id count mismatch";
  CHECK_EQ(embeddings.size(0), static_cast<int64_t>(ids.size()))
      << "prefill target embedding count mismatch";

  const torch::Tensor next_tokens_cpu =
      to_cpu_contiguous(next_tokens, torch::kInt64);
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
    state.embedding =
        clone_contiguous_detached_tensor(embeddings.select(/*dim=*/0, i));
    if (retain_replay_span_) {
      // The prefill tail is a single emitted position; seed the replay span
      // from the already-detached clone so later producer reuse cannot alias.
      state.replay_token_ids = {state.token_id};
      state.replay_embeddings = state.embedding.unsqueeze(0);
    }

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
  DecodeState& tail = mutable_tail(embedding_id);
  if (tail.valid && !request_id.empty() && tail.request_id == request_id) {
    return;
  }
  DecodeState state;
  state.valid = true;
  state.request_id = request_id;
  state.token_id = token_id;
  state.embedding = clone_contiguous_detached_tensor(embedding);
  if (retain_replay_span_) {
    state.replay_token_ids = {state.token_id};
    state.replay_embeddings = state.embedding.unsqueeze(0);
  }
  tail = std::move(state);
}

void EmbeddingCache::write_target_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    int32_t num_speculative_tokens,
    bool retain_previous_embedding) {
  write_target_context(ids,
                       request_ids,
                       accepted_tokens,
                       accepted_embeddings,
                       torch::Tensor(),
                       num_speculative_tokens,
                       false,
                       retain_previous_embedding);
}

void EmbeddingCache::write_target_context(
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& accepted_count,
    int32_t num_speculative_tokens,
    bool allow_immutable_compact_view,
    bool retain_previous_embedding) {
  CHECK(accepted_tokens.defined()) << "accepted target tokens are undefined";
  CHECK(accepted_embeddings.defined())
      << "accepted target embeddings are undefined";
  CHECK_EQ(accepted_tokens.dim(), 2)
      << "accepted target tokens should be [batch, width]";
  const bool compact = accepted_embeddings.dim() == 2;
  const bool borrow_compact_view =
      allow_immutable_compact_view && compact && ids.size() == 1;
  CHECK(compact || accepted_embeddings.dim() == 3)
      << "target embeddings must be compact [2*batch, hidden] or full [batch, "
         "width, hidden]";
  CHECK_EQ(accepted_tokens.size(0), static_cast<int64_t>(ids.size()))
      << "accepted token batch mismatch";
  CHECK(request_ids.empty() || request_ids.size() == ids.size())
      << "accepted request id count mismatch";
  if (accepted_count.defined()) {
    CHECK(accepted_count.device().is_cpu()) << "accepted count must be on CPU";
    CHECK_EQ(accepted_count.dim(), 1) << "accepted count should be a vector";
    CHECK_EQ(accepted_count.size(0), static_cast<int64_t>(ids.size()))
        << "accepted count batch mismatch";
    CHECK(accepted_count.scalar_type() == torch::kInt ||
          accepted_count.scalar_type() == torch::kLong)
        << "accepted count must be int32 or int64";
  }
  CHECK_EQ(accepted_embeddings.size(0),
           static_cast<int64_t>(ids.size()) * (compact ? 2 : 1))
      << "accepted embedding batch mismatch";
  if (!compact) {
    CHECK_EQ(accepted_tokens.size(1), accepted_embeddings.size(1))
        << "accepted token/embedding width mismatch";
  }
  const torch::Tensor embedding_rows =
      compact ? accepted_embeddings.view({static_cast<int64_t>(ids.size()),
                                          2,
                                          accepted_embeddings.size(-1)})
              : accepted_embeddings;
  CHECK_GE(num_speculative_tokens, 0) << "invalid speculative token count";

  const torch::Tensor accepted_tokens_cpu =
      to_cpu_contiguous(accepted_tokens, torch::kInt64);
  const int64_t* accepted_tokens_data =
      accepted_tokens_cpu.const_data_ptr<int64_t>();
  const int32_t num_ids = static_cast<int32_t>(ids.size());
  const int32_t token_width = static_cast<int32_t>(accepted_tokens.size(1));
  for (int32_t i = 0; i < num_ids; ++i) {
    int32_t accepted_len = 0;
    if (accepted_count.defined()) {
      accepted_len =
          static_cast<int32_t>(accepted_count.index({i}).item<int64_t>()) + 1;
      CHECK_GE(accepted_len, 1);
      CHECK_LE(accepted_len, num_speculative_tokens + 1);
    }
    int32_t last_token_id = -1;
    int32_t correction_token = -1;
    int32_t correction_offset = -1;
    const int64_t row_offset = static_cast<int64_t>(i) * token_width;
    if (accepted_count.defined()) {
      correction_offset = accepted_len - 1;
      const int64_t token =
          accepted_tokens_data[row_offset + correction_offset];
      CHECK_GE(token, 0) << "accepted target token is missing at graph count";
      CHECK_LE(token, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
          << "accepted token overflow";
      last_token_id = static_cast<int32_t>(token);
      correction_token = last_token_id;
    } else {
      for (int32_t j = 0; j < token_width; ++j) {
        const int64_t token = accepted_tokens_data[row_offset + j];
        if (token < 0) {
          break;
        }
        CHECK_LE(token,
                 static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
            << "accepted token overflow";
        last_token_id = static_cast<int32_t>(token);
        correction_token = static_cast<int32_t>(token);
        correction_offset = j;
        ++accepted_len;
      }
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
    if (retain_replay_span_) {
      // Accepted-span replay needs every verified KV position, so the full
      // [batch, width, hidden] layout is mandatory; the compact two-row view
      // only covers the last two positions.
      CHECK(!compact) << "replay-span retention requires full-span embeddings";
      state.replay_token_ids.assign(
          accepted_tokens_data + row_offset,
          accepted_tokens_data + row_offset + accepted_len);
      state.replay_embeddings = clone_contiguous_detached_tensor(
          embedding_rows.select(/*dim=*/0, i).narrow(0, 0, accepted_len));
    }
    // Replay retention already owns an immutable copy of the accepted span;
    // select legacy last/previous views from it instead of cloning again.
    const torch::Tensor row = retain_replay_span_
                                  ? state.replay_embeddings
                                  : embedding_rows.select(/*dim=*/0, i);
    if (retain_previous_embedding && last_idx > 0) {
      const int64_t previous_token =
          accepted_tokens_data[row_offset + last_idx - 1];
      state.prev_token_id = static_cast<int32_t>(previous_token);
      // One request-sized snapshot keeps both rows alive without pinning a
      // whole batch output. Subsequent replay may overwrite its source.
      const torch::Tensor pair =
          borrow_compact_view || retain_replay_span_
              ? row.narrow(/*dim=*/0,
                           /*start=*/compact ? 0 : last_idx - 1,
                           /*length=*/2)
                    .detach()
              : clone_contiguous_detached_tensor(row.narrow(
                    /*dim=*/0,
                    /*start=*/compact ? 0 : last_idx - 1,
                    /*length=*/2));
      state.prev_embedding = pair.select(/*dim=*/0, /*index=*/0);
      state.embedding = pair.select(/*dim=*/0, /*index=*/1);
    } else {
      const torch::Tensor embedding =
          row.select(/*dim=*/0, /*index=*/compact ? 1 : last_idx);
      state.embedding = borrow_compact_view || retain_replay_span_
                            ? embedding
                            : clone_contiguous_detached_tensor(embedding);
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
    const std::vector<int32_t>& ids,
    const std::vector<std::string>& request_ids) const {
  CHECK(!ids.empty()) << "decode ids should not be empty";
  CHECK(request_ids.empty() || request_ids.size() == ids.size())
      << "embedding_id / request_id count mismatch";
  std::vector<int32_t> accepted_prefix_lengths;
  accepted_prefix_lengths.reserve(ids.size());
  for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) {
    const DecodeState& state = get_tail(ids[i]);
    // A slot that never received target output, or one whose request_id no
    // longer matches (embedding_id recycled by a later request), carries no
    // usable correction offset — fall back to a single accepted token so the
    // previous request's offset cannot leak into this sequence's spec-verify
    // metadata. An empty request_ids skips the request_id match.
    int32_t accepted_length = 1;
    if (state.valid &&
        (request_ids.empty() || state.request_id == request_ids[i])) {
      CHECK_GE(state.correction_token_id, 0)
          << "decode entry missing correction token id";
      accepted_length = state.correction_position_offset + 1;
    }
    accepted_prefix_lengths.emplace_back(accepted_length);
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
