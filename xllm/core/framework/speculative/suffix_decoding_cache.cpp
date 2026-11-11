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

#include "core/framework/speculative/suffix_decoding_cache.h"

#include <iterator>
#include <stdexcept>
#include <utility>

namespace xllm {

namespace {

constexpr int32_t kMaxInt32SeqId = 0x7FFFFFFF;

}  // namespace

SuffixDecodingCache::SuffixDecodingCache(int32_t max_tree_depth,
                                         int32_t max_cached_requests)
    : max_tree_depth_(max_tree_depth),
      max_cached_requests_(max_cached_requests),
      global_tree_(max_tree_depth) {}

SuffixTree* SuffixDecodingCache::start_request(
    const std::string& req_id,
    std::span<const int32_t> prompt_token_ids) {
  auto it = local_trees_.find(req_id);
  if (it != local_trees_.end()) {
    it->second->extend(/*seq_id=*/0, prompt_token_ids);
    return it->second.get();
  }

  auto tree = std::make_unique<SuffixTree>(max_tree_depth_);
  tree->extend(/*seq_id=*/0, prompt_token_ids);
  auto [emplace_it, _] = local_trees_.emplace(req_id, std::move(tree));

  if (max_cached_requests_ != 0) {
    evict(req_id);
    generate_seq_id(req_id);
  }
  return emplace_it->second.get();
}

void SuffixDecodingCache::stop_request(const std::string& req_id) {
  local_trees_.erase(req_id);
}

void SuffixDecodingCache::add_active_response(
    const std::string& req_id,
    std::span<const int32_t> token_ids) {
  auto tree_it = local_trees_.find(req_id);
  if (tree_it == local_trees_.end()) {
    throw std::invalid_argument("Request '" + req_id + "' is not active");
  }
  tree_it->second->extend(/*seq_id=*/0, token_ids);

  auto cached_it = req_to_seq_id_.find(req_id);
  if (cached_it != req_to_seq_id_.end()) {
    global_tree_.extend(cached_it->second.seq_id, token_ids);
  }
}

void SuffixDecodingCache::evict(const std::string& req_id) {
  const auto it = req_to_seq_id_.find(req_id);
  if (it == req_to_seq_id_.end()) {
    return;
  }
  const int32_t seq_id = it->second.seq_id;
  const auto order_it = it->second.order_it;
  req_to_seq_id_.erase(it);
  seq_to_req_id_.erase(seq_id);
  global_tree_.remove(seq_id);
  cache_order_.erase(order_it);
}

Draft SuffixDecodingCache::speculate(const std::string& req_id,
                                     std::span<const int32_t> seed_tokens,
                                     int32_t max_tokens) {
  auto it = local_trees_.find(req_id);
  SuffixTree* local_tree = it == local_trees_.end()
                               ? start_request(req_id, seed_tokens)
                               : it->second.get();
  const std::span<const int32_t> context =
      local_tree->tail(/*seq_id=*/0, max_tree_depth_);

  Draft draft_local = local_tree->speculate(context, max_tokens);

  Draft draft_global = global_tree_.speculate(context, max_tokens);

  return draft_local.score >= draft_global.score ? std::move(draft_local)
                                                 : std::move(draft_global);
}

void SuffixDecodingCache::generate_seq_id(const std::string& req_id) {
  int32_t seq_id = 0;
  while (true) {
    seq_id = next_seq_id_;
    next_seq_id_ = (next_seq_id_ + 1) & kMaxInt32SeqId;

    auto seq_it = seq_to_req_id_.find(seq_id);
    if (seq_it == seq_to_req_id_.end()) {
      break;
    }

    if (local_trees_.find(seq_it->second) == local_trees_.end()) {
      evict(seq_it->second);
      break;
    }
  }

  cache_order_.push_back(req_id);
  req_to_seq_id_[req_id] = {seq_id, std::prev(cache_order_.end())};
  seq_to_req_id_.emplace(seq_id, req_id);
  maybe_evict_requests();
}

void SuffixDecodingCache::maybe_evict_requests() {
  if (max_cached_requests_ <= 0) {
    return;
  }

  while (static_cast<int32_t>(req_to_seq_id_.size()) > max_cached_requests_) {
    evict(cache_order_.front());
  }
}

}  // namespace xllm
