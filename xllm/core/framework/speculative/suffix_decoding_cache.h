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
#include <list>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>

#include "core/framework/speculative/suffix_tree.h"

namespace xllm {

class SuffixDecodingCache final {
 public:
  explicit SuffixDecodingCache(int32_t max_tree_depth,
                               int32_t max_cached_requests);

  SuffixTree* start_request(const std::string& req_id,
                            std::span<const int32_t> prompt_token_ids);

  void stop_request(const std::string& req_id);

  void add_active_response(const std::string& req_id,
                           std::span<const int32_t> token_ids);

  Draft speculate(const std::string& req_id,
                  std::span<const int32_t> seed_tokens,
                  int32_t max_tokens);

 private:
  struct CacheEntry {
    int32_t seq_id = 0;
    std::list<std::string>::iterator order_it;
  };

  void generate_seq_id(const std::string& req_id);

  void maybe_evict_requests();

  void evict(const std::string& req_id);

  int32_t max_tree_depth_;
  int32_t max_cached_requests_;

  SuffixTree global_tree_;
  std::unordered_map<std::string, std::unique_ptr<SuffixTree>> local_trees_;

  std::unordered_map<std::string, CacheEntry> req_to_seq_id_;
  Int32Map<std::string> seq_to_req_id_;
  int32_t next_seq_id_ = 0;

  std::list<std::string> cache_order_;
};

}  // namespace xllm
