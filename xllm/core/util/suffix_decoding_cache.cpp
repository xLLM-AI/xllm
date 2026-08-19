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

#include "suffix_decoding_cache.h"

#include <algorithm>
#include <stdexcept>

namespace xllm {

namespace {

constexpr int32_t kMaxInt32SeqId = 0x7FFFFFFF;

}  // namespace

SuffixDecodingDraft SuffixDecodingDraft::from_native(const Draft& draft) {
  return {
      .token_ids = draft.token_ids,
      .parents = draft.parents,
      .probs = draft.probs,
      .score = draft.score,
      .match_len = draft.match_len,
  };
}

SuffixDecodingCache::SuffixDecodingCache(int32_t max_tree_depth,
                                         int32_t max_cached_requests)
    : max_tree_depth_(max_tree_depth),
      max_cached_requests_(max_cached_requests),
      global_tree_(max_tree_depth) {
  if (max_cached_requests > kMaxInt32SeqId) {
    throw std::invalid_argument("max_cached_requests must be at most 2^31");
  }
}

std::vector<std::string> SuffixDecodingCache::active_requests() const {
  std::vector<std::string> requests;
  requests.reserve(local_trees_.size());
  for (const auto& [req_id, tree] : local_trees_) {
    requests.push_back(req_id);
  }
  return requests;
}

std::vector<std::string> SuffixDecodingCache::cached_requests() const {
  std::vector<std::string> requests;
  requests.reserve(req_to_seq_id_.size());
  for (const auto& [req_id, seq_id] : req_to_seq_id_) {
    (void)seq_id;
    requests.push_back(req_id);
  }
  return requests;
}

bool SuffixDecodingCache::has_active_request(const std::string& req_id) const {
  return local_trees_.find(req_id) != local_trees_.end();
}

bool SuffixDecodingCache::has_cached_request(const std::string& req_id) const {
  return req_to_seq_id_.find(req_id) != req_to_seq_id_.end();
}

void SuffixDecodingCache::start_request(
    const std::string& req_id,
    std::span<const int32_t> prompt_token_ids) {
  if (has_active_request(req_id)) {
    throw std::invalid_argument("Request '" + req_id + "' is already active");
  }

  auto tree = std::make_unique<SuffixTree>(max_tree_depth_);
  tree->extend(/*seq_id=*/0, prompt_token_ids);
  local_trees_.emplace(req_id, std::move(tree));

  if (max_cached_requests_ != 0) {
    if (has_cached_request(req_id)) {
      evict_cached_response(req_id);
    }
    generate_seq_id(req_id);
  }
}

void SuffixDecodingCache::stop_request(const std::string& req_id) {
  if (!has_active_request(req_id)) {
    throw std::invalid_argument("Request '" + req_id + "' is not active");
  }
  local_trees_.erase(req_id);
}

void SuffixDecodingCache::add_active_prompt(
    const std::string& req_id,
    std::span<const int32_t> token_ids) {
  SuffixTree* local_tree = find_local_tree(req_id);
  if (local_tree == nullptr) {
    throw std::invalid_argument("Request '" + req_id + "' is not active");
  }

  local_tree->extend(/*seq_id=*/0, token_ids);
}

void SuffixDecodingCache::add_active_response(
    const std::string& req_id,
    std::span<const int32_t> token_ids) {
  SuffixTree* local_tree = find_local_tree(req_id);
  if (local_tree == nullptr) {
    throw std::invalid_argument("Request '" + req_id + "' is not active");
  }

  local_tree->extend(/*seq_id=*/0, token_ids);

  auto it = req_to_seq_id_.find(req_id);
  if (it != req_to_seq_id_.end()) {
    global_tree_.extend(it->second, token_ids);
  }
}

void SuffixDecodingCache::evict_cached_response(const std::string& req_id) {
  auto it = req_to_seq_id_.find(req_id);
  if (it == req_to_seq_id_.end()) {
    throw std::invalid_argument("Request '" + req_id + "' is not cached");
  }

  int32_t seq_id = it->second;
  req_to_seq_id_.erase(it);
  seq_to_req_id_.erase(seq_id);
  global_tree_.remove(seq_id);

  if (!cache_order_.empty()) {
    auto pos = std::find(cache_order_.begin(), cache_order_.end(), req_id);
    if (pos != cache_order_.end()) {
      cache_order_.erase(pos);
    }
  }
}

SuffixDecodingDraft SuffixDecodingCache::speculate(
    const std::string& req_id,
    std::span<const int32_t> context,
    std::optional<int32_t> max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {
  SuffixTree* local_tree = find_local_tree(req_id);
  if (local_tree == nullptr) {
    throw std::invalid_argument("Request '" + req_id + "' is not active");
  }

  int32_t max_tokens = max_spec_tokens.value_or(max_tree_depth_);

  if (context.size() > static_cast<size_t>(max_tree_depth_)) {
    context = context.subspan(context.size() - max_tree_depth_);
  }

  Draft draft_local = local_tree->speculate(context,
                                            max_tokens,
                                            max_spec_factor,
                                            max_spec_offset,
                                            min_token_prob,
                                            use_tree_spec);

  Draft draft_global = global_tree_.speculate(context,
                                              max_tokens,
                                              max_spec_factor,
                                              max_spec_offset,
                                              min_token_prob,
                                              use_tree_spec);

  return SuffixDecodingDraft::from_native(
      draft_local.score >= draft_global.score ? draft_local : draft_global);
}

int32_t SuffixDecodingCache::generate_seq_id(const std::string& req_id) {
  int32_t seq_id = 0;
  while (true) {
    seq_id = next_seq_id_;
    next_seq_id_ = (next_seq_id_ + 1) & kMaxInt32SeqId;

    auto seq_it = seq_to_req_id_.find(seq_id);
    if (seq_it == seq_to_req_id_.end()) {
      break;
    }

    const std::string& mapped_req = seq_it->second;
    if (!has_active_request(mapped_req)) {
      break;
    }
  }

  auto seq_it = seq_to_req_id_.find(seq_id);
  if (seq_it != seq_to_req_id_.end()) {
    const std::string old_req_id = seq_it->second;
    req_to_seq_id_.erase(old_req_id);
    seq_to_req_id_.erase(seq_id);
    global_tree_.remove(seq_id);

    auto pos = std::find(cache_order_.begin(), cache_order_.end(), old_req_id);
    if (pos != cache_order_.end()) {
      cache_order_.erase(pos);
    }
  }

  req_to_seq_id_[req_id] = seq_id;
  seq_to_req_id_.emplace(seq_id, req_id);
  cache_order_.push_back(req_id);
  maybe_evict_requests(seq_id);
  return seq_id;
}

void SuffixDecodingCache::maybe_evict_requests(int32_t new_seq_id) {
  if (max_cached_requests_ < 0) {
    return;
  }

  if (max_cached_requests_ == 0) {
    return;
  }

  while (static_cast<int32_t>(req_to_seq_id_.size()) > max_cached_requests_) {
    bool evicted = false;

    for (auto it = cache_order_.cbegin(); it != cache_order_.cend(); ++it) {
      const std::string& req_id = *it;
      auto req_it = req_to_seq_id_.find(req_id);
      if (req_it == req_to_seq_id_.end()) {
        continue;
      }
      if (req_it->second == new_seq_id) {
        continue;
      }

      const std::string req_copy = req_id;
      evict_cached_response(req_copy);
      evicted = true;
      break;
    }

    if (!evicted) {
      break;
    }
  }
}

SuffixTree* SuffixDecodingCache::find_local_tree(const std::string& req_id) {
  auto it = local_trees_.find(req_id);
  if (it == local_trees_.end()) {
    return nullptr;
  }
  return it->second.get();
}

const SuffixTree* SuffixDecodingCache::find_local_tree(
    const std::string& req_id) const {
  auto it = local_trees_.find(req_id);
  if (it == local_trees_.end()) {
    return nullptr;
  }
  return it->second.get();
}

}  // namespace xllm
