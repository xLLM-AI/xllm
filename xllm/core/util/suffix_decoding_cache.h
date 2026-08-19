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

// ref to:
// https://
// github.com/snowflakedb/ArcticInference/blob/main/arctic_inference/suffix_decoding/cache.py

#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "suffix_tree.h"

namespace xllm {

struct SuffixDecodingDraft {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> parents;
  std::vector<float> probs;
  float score = 0.0;
  int32_t match_len = 0;

  static SuffixDecodingDraft from_native(const Draft& draft);
};

class SuffixDecodingCache {
 public:
  SuffixDecodingCache(int32_t max_tree_depth = 64,
                      int32_t max_cached_requests = -1);

  int32_t max_tree_depth() const { return max_tree_depth_; }

  int32_t max_cached_requests() const { return max_cached_requests_; }

  std::vector<std::string> active_requests() const;

  std::vector<std::string> cached_requests() const;

  bool has_active_request(const std::string& req_id) const;

  bool has_cached_request(const std::string& req_id) const;

  void start_request(const std::string& req_id,
                     std::span<const int32_t> prompt_token_ids);

  void stop_request(const std::string& req_id);

  // Append prompt tokens for an active request. Different from
  // add_active_response, this only updates the local tree and does not pollute
  // global response cache.
  void add_active_prompt(const std::string& req_id,
                         std::span<const int32_t> token_ids);

  void add_active_response(const std::string& req_id,
                           std::span<const int32_t> token_ids);

  void evict_cached_response(const std::string& req_id);

  SuffixDecodingDraft speculate(
      const std::string& req_id,
      std::span<const int32_t> context,
      std::optional<int32_t> max_spec_tokens = std::nullopt,
      float max_spec_factor = 1.0f,
      float max_spec_offset = 0.0f,
      float min_token_prob = 0.1f,
      bool use_tree_spec = false);

 private:
  int32_t generate_seq_id(const std::string& req_id);

  void maybe_evict_requests(int32_t new_seq_id);

  SuffixTree* find_local_tree(const std::string& req_id);

  const SuffixTree* find_local_tree(const std::string& req_id) const;

 private:
  int32_t max_tree_depth_;
  int32_t max_cached_requests_;

  SuffixTree global_tree_;
  std::unordered_map<std::string, std::unique_ptr<SuffixTree>> local_trees_;

  std::unordered_map<std::string, int32_t> req_to_seq_id_;
  Int32Map<std::string> seq_to_req_id_;
  int32_t next_seq_id_ = 0;

  std::deque<std::string> cache_order_;
};

}  // namespace xllm
