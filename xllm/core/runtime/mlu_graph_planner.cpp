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

#include "core/runtime/mlu_graph_planner.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "core/common/constants.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/seq_lens.h"
#include "runtime/decode_graph_bucket.h"

namespace {
using xllm::mlu::get_block_capacity;
using xllm::mlu::GraphLayout;

uint32_t get_bucket_num_tokens(uint32_t num_tokens,
                               const xllm::runtime::Options& options) {
  return static_cast<uint32_t>(xllm::runtime::get_decode_graph_token_bucket(
      num_tokens, options.enable_graph_mode_decode_no_padding()));
}

uint32_t align_tokens(uint32_t tokens, uint32_t align) {
  uint32_t rem = tokens % align;
  return rem == 0 ? tokens : tokens + align - rem;
}

bool has_zero_tokens(const std::vector<int32_t>& dp_token_nums) {
  return std::any_of(dp_token_nums.begin(),
                     dp_token_nums.end(),
                     [](int32_t token_num) { return token_num == 0; });
}

bool dp_tokens_equal(const std::vector<int32_t>& dp_token_nums) {
  return dp_token_nums.empty() ||
         std::all_of(
             dp_token_nums.begin(),
             dp_token_nums.end(),
             [first_token_num = dp_token_nums.front()](int32_t token_num) {
               return token_num == first_token_num;
             });
}

bool allows_graph_mode(const xllm::runtime::Options& options,
                       const xllm::MtpModelCapabilities& capabilities,
                       const xllm::ModelParamsView& params) {
  if (options.is_draft_model()) {
    return params.meta.batch_forward_type.is_decode();
  }
  if (options.enable_speculative_decode() &&
      xllm::SpeculativeConfig::is_mtp_algorithm(
          options.speculative_algorithm()) &&
      capabilities.supports_grouped_graph) {
    return params.is_spec_verify &&
           params.meta.batch_forward_type.is_chunked_prefill();
  }
  return !params.is_spec_verify && params.meta.batch_forward_type.is_decode();
}

uint32_t get_tp_size(const xllm::runtime::Options& options) {
  int32_t world_size = options.world_size();
  int32_t dp_size = options.dp_size();
  if (world_size <= 1 || dp_size <= 1 || world_size < dp_size ||
      world_size % dp_size != 0) {
    return 1;
  }
  return static_cast<uint32_t>(world_size / dp_size);
}

uint32_t get_graph_dp_tokens(int64_t actual_tokens,
                             const std::vector<int32_t>& token_nums,
                             const xllm::runtime::Options& options) {
  if (token_nums.size() <= 1) {
    return get_bucket_num_tokens(static_cast<uint32_t>(actual_tokens), options);
  }
  const auto max_token_num =
      *std::max_element(token_nums.begin(), token_nums.end());
  const uint32_t tp_size = get_tp_size(options);
  return align_tokens(
      std::max(
          get_bucket_num_tokens(static_cast<uint32_t>(max_token_num), options),
          tp_size),
      tp_size);
}

bool supports_ordinary_graph(const xllm::runtime::Options& options,
                             const xllm::ModelParamsView& params) {
  if (params.is_spec_verify || options.is_draft_model() ||
      !params.meta.batch_forward_type.is_decode() ||
      params.meta.q_max_seq_len == 0 ||
      has_zero_tokens(params.parallel.dp_global_token_nums)) {
    return false;
  }
  const auto& token_nums = params.parallel.dp_global_token_nums;
  const auto& decode_flags = params.parallel.dp_is_decode;
  if (options.dp_size() > 1 &&
      token_nums.size() != static_cast<std::size_t>(options.dp_size())) {
    return false;
  }
  if (token_nums.size() <= 1) {
    return true;
  }
  return decode_flags.size() == token_nums.size() &&
         std::find(decode_flags.begin(), decode_flags.end(), 0) ==
             decode_flags.end() &&
         (dp_tokens_equal(token_nums) || params.meta.q_max_seq_len == 1);
}

bool supports_grouped_graph(const xllm::runtime::Options& options,
                            const xllm::ModelParamsView& params) {
  if (options.cp_size() > 1 || params.meta.q_max_seq_len <= 0 ||
      params.meta.num_sequences <= 0 || params.prefill_without_cache ||
      !(params.meta.batch_forward_type.is_decode() ||
        (params.is_spec_verify &&
         params.meta.batch_forward_type.is_chunked_prefill()))) {
    return false;
  }
  if (!dp_tokens_equal(params.parallel.dp_global_token_nums) ||
      has_zero_tokens(params.parallel.dp_global_token_nums)) {
    return false;
  }
  if (options.dp_size() > 1 &&
      (params.parallel.dp_global_token_nums.size() !=
           static_cast<std::size_t>(options.dp_size()) ||
       params.parallel.dp_is_decode.size() !=
           params.parallel.dp_global_token_nums.size() ||
       std::find(params.parallel.dp_is_decode.begin(),
                 params.parallel.dp_is_decode.end(),
                 0) != params.parallel.dp_is_decode.end())) {
    return false;
  }
  return params.attention.device.block_tables.defined() &&
         params.attention.device.q_seq_lens.defined() &&
         params.attention.device.kv_seq_lens.defined() &&
         params.attention.device.new_cache_slots.defined();
}

bool supports_graph_history(const xllm::ModelParamsView& params,
                            int64_t graph_max_kv_seq_len,
                            int32_t block_size) {
  if (!params.attention.device.block_tables.defined()) {
    return true;
  }
  if (graph_max_kv_seq_len <= 0 || params.meta.kv_max_seq_len < 0 ||
      params.meta.kv_max_seq_len > graph_max_kv_seq_len) {
    return false;
  }
  return params.meta.kv_max_seq_len == 0 ||
         params.attention.device.block_tables.size(1) >=
             get_block_capacity(params.meta.kv_max_seq_len, block_size);
}

std::optional<GraphLayout> get_graph_layout(
    int64_t tokens,
    const xllm::ModelParamsView& params,
    const xllm::runtime::Options& options,
    bool grouped,
    bool state_required) {
  const auto& q_seq_lens = params.attention.host.q_seq_lens;
  if (tokens <= 0 || q_seq_lens.empty()) {
    return std::nullopt;
  }
  const auto& spans = params.attention.host.kpool_query_lens;
  const int64_t tokens_per_request =
      spans.empty() ? q_seq_lens.front() : spans.front();
  if (tokens_per_request <= 0 || tokens % tokens_per_request != 0) {
    return std::nullopt;
  }
  const int64_t bucket_units =
      grouped ? get_bucket_num_tokens(
                    static_cast<uint32_t>(tokens / tokens_per_request), options)
              : get_graph_dp_tokens(
                    tokens, params.parallel.dp_global_token_nums, options);
  const int64_t tp = grouped && params.parallel.dp_global_token_nums.size() > 1
                         ? get_tp_size(options)
                         : 1;
  GraphLayout layout;
  layout.tokens_per_request = tokens_per_request;
  layout.attention_row_query_len = q_seq_lens.front();
  if (layout.attention_row_query_len <= 0 ||
      params.meta.q_max_seq_len != layout.attention_row_query_len ||
      tokens_per_request % layout.attention_row_query_len != 0 ||
      static_cast<int64_t>(q_seq_lens.size()) !=
          tokens / layout.attention_row_query_len) {
    return std::nullopt;
  }
  if (!std::all_of(
          q_seq_lens.begin(), q_seq_lens.end(), [&layout](int32_t len) {
            return len == layout.attention_row_query_len;
          })) {
    return std::nullopt;
  }
  layout.num_reqs = tokens / tokens_per_request;
  if (!spans.empty() &&
      (static_cast<int64_t>(spans.size()) != layout.num_reqs ||
       !std::all_of(
           spans.begin(), spans.end(), [tokens_per_request](int32_t span) {
             return span == tokens_per_request;
           }))) {
    return std::nullopt;
  }
  if (grouped) {
    const int64_t alignment = tp / std::gcd(tp, tokens_per_request);
    layout.padded_num_reqs =
        ((std::max(layout.num_reqs, bucket_units) + alignment - 1) /
         alignment) *
        alignment;
    layout.padded_num_tokens = layout.padded_num_reqs * tokens_per_request;
    layout.cache_pad_slot = xllm::kPadSlotId;
  } else {
    layout.padded_num_tokens = bucket_units;
    if (layout.padded_num_tokens < tokens ||
        layout.padded_num_tokens % tokens_per_request != 0) {
      return std::nullopt;
    }
    layout.padded_num_reqs = layout.padded_num_tokens / tokens_per_request;
  }
  if (grouped &&
      ((state_required && params.embedding.linear_state_ids.empty()) ||
       (!params.embedding.linear_state_ids.empty() &&
        params.embedding.linear_state_ids.size() !=
            static_cast<std::size_t>(layout.num_reqs)) ||
       (params.is_spec_verify &&
        (!params.num_accepted_tokens.defined() ||
         params.num_accepted_tokens.numel() != layout.num_reqs)))) {
    return std::nullopt;
  }
  const auto& attention = params.attention.device;
  if (grouped &&
      (attention.q_seq_lens.numel() !=
           static_cast<int64_t>(q_seq_lens.size()) ||
       attention.kv_seq_lens.numel() !=
           static_cast<int64_t>(q_seq_lens.size()) ||
       attention.block_tables.size(0) !=
           tokens / layout.attention_row_query_len ||
       attention.new_cache_slots.numel() != tokens ||
       (params.embedding.linear_state_indices.defined() &&
        params.embedding.linear_state_indices.numel() != layout.num_reqs))) {
    return std::nullopt;
  }
  return layout;
}

}  // namespace

namespace xllm::mlu {

int64_t get_block_capacity(int64_t max_kv_seq_len, int32_t block_size) {
  CHECK_GT(block_size, 0);
  return (max_kv_seq_len + block_size - 1) / block_size;
}

std::size_t GraphKeyHash::operator()(const GraphKey& key) const {
  std::size_t seed = 0;
  const auto hash_value = [&seed](int64_t value) {
    seed ^=
        std::hash<int64_t>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  };
  hash_value(key.padded_num_tokens);
  hash_value(key.padded_num_reqs);
  hash_value(key.attention_row_query_len);
  hash_value(key.has_input_embedding);
  hash_value(key.grouped);
  for (int64_t column_count : key.multi_block_table_column_counts) {
    hash_value(column_count);
  }
  hash_value(static_cast<int64_t>(key.multi_block_table_column_counts.size()));
  return seed;
}

int64_t get_graph_token_capacity(const runtime::Options& options,
                                 const MtpModelCapabilities& capabilities) {
  int64_t width =
      options.is_draft_model() ? 1 : options.num_speculative_tokens() + 1;
  const int64_t max_requests =
      std::max<int64_t>(options.max_seqs_per_batch(), 1);
  int64_t capacity = get_bucket_num_tokens(
      static_cast<uint32_t>(max_requests * width), options);
  const uint32_t tp_size = get_tp_size(options);
  capacity = align_tokens(
      static_cast<uint32_t>(std::max<int64_t>(capacity, tp_size)), tp_size);
  if (!capabilities.supports_grouped_graph) {
    return capacity;
  }
  width = options.num_speculative_tokens() + 1;
  if (!options.is_draft_model() &&
      !SpeculativeConfig::is_mtp_algorithm(options.speculative_algorithm())) {
    width = 1;
  }
  const uint32_t requests =
      get_bucket_num_tokens(static_cast<uint32_t>(max_requests), options);
  for (uint32_t span = 1; span <= static_cast<uint32_t>(width); ++span) {
    const uint32_t alignment = tp_size / std::gcd(tp_size, span);
    capacity =
        std::max<int64_t>(capacity, align_tokens(requests, alignment) * span);
  }
  return capacity;
}

MluGraphPlanner::MluGraphPlanner(GraphPlannerConfig config)
    : config_(std::move(config)) {}

GraphDecision MluGraphPlanner::plan(int64_t tokens,
                                    const ModelParamsView& params) const {
  const runtime::Options& options = config_.options;
  if (!allows_graph_mode(options, config_.capabilities, params)) {
    return EagerReason::NON_DECODE;
  }
  const bool grouped = config_.capabilities.supports_grouped_graph &&
                       (config_.state_required || options.is_draft_model() ||
                        params.is_spec_verify);
  if (params.mtp_topk_state != nullptr ||
      (grouped ? !supports_grouped_graph(options, params)
               : !supports_ordinary_graph(options, params))) {
    return EagerReason::UNSUPPORTED_INPUT;
  }
  std::optional<GraphLayout> layout = get_graph_layout(
      tokens, params, options, grouped, config_.state_required);
  if (!layout) {
    return EagerReason::INVALID_LAYOUT;
  }
  if (layout->padded_num_tokens > config_.max_tokens) {
    return EagerReason::TOKEN_LIMIT;
  }
  if (config_.capabilities.graph_context_policy ==
      GraphContextPolicy::QWEN_FULL_ATTENTION) {
    layout->cache_pad_slot = xllm::kPadSlotId;
  }
  const int64_t capacity = params.attention.device.block_tables.defined()
                               ? config_.history_capacity
                               : 0;
  if (!supports_graph_history(params, capacity, options.block_size())) {
    return EagerReason::HISTORY_LIMIT;
  }
  GraphKey key;
  key.padded_num_tokens = layout->padded_num_tokens;
  key.padded_num_reqs = layout->padded_num_reqs;
  key.attention_row_query_len = layout->attention_row_query_len;
  key.has_input_embedding = params.embedding.input_embedding.defined();
  key.grouped = grouped;
  key.multi_block_table_column_counts.reserve(params.multi_block_tables.size());
  for (const torch::Tensor& table : params.multi_block_tables) {
    CHECK_EQ(table.dim(), 2);
    key.multi_block_table_column_counts.push_back(table.size(1));
  }
  const int64_t main_columns =
      capacity > 0 ? get_block_capacity(capacity, options.block_size()) +
                         (config_.capabilities.graph_context_policy ==
                                  GraphContextPolicy::QWEN_FULL_ATTENTION
                              ? 1
                              : 0)
                   : 0;
  return GraphPlan{*layout, std::move(key), capacity, main_columns};
}

}  // namespace xllm::mlu
