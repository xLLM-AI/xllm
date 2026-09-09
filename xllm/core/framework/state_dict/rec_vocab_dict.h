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

#include <boost/functional/hash.hpp>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"
#include "common/types.h"
#include "util/slice.h"

namespace xllm {

struct RecConstraintTables {
  std::vector<int32_t> first_token_ids;

  // prefix1_values[prefix1_offsets[t0]:prefix1_offsets[t0 + 1]] contains
  // valid t1 tokens after prefix [t0].
  std::vector<int32_t> prefix1_offsets;
  std::vector<int32_t> prefix1_values;
  // Globally sorted keys aligned with prefix1_values and prefix2_value_offsets.
  // key = int64_t(t0) * vocab_size + int64_t(t1).
  std::vector<int64_t> prefix1_pair_keys;

  // prefix2_value_offsets[i:i + 1] is aligned with prefix1_values[i].
  // It contains valid t2 tokens after prefix [t0, prefix1_values[i]].
  std::vector<int32_t> prefix2_value_offsets;
  std::vector<int32_t> prefix2_values;

  int32_t vocab_size = 0;
  int32_t max_first_degree = 0;
  int32_t max_prefix1_degree = 0;
  int32_t max_prefix2_degree = 0;
};

// A vocab dictionary in generative recommendation scenarios, used for mapping
// token IDs and item IDs, currently updated with the model version, and
// real-time updates are not supported.
class RecVocabDict final {
 public:
  RecVocabDict() = default;

  ~RecVocabDict() {
    initialized_ = false;
    item_to_tokens_map_.clear();
    tokens_to_item_infos_map_.clear();
    prefix_tokens_to_next_tokens_map_.clear();
  }

  /**
   * @brief Initialize instance, parse vocab file
   * @param vocab_file vocab file, need full path
   * @return true represents successful initialization, false represents failed
   * initialization
   */
  bool initialize(const std::string& vocab_file);

  /**
   * @brief Get the corresponding item ID list through a token ID triplet
   * @param token_ids, a token ID triplet, so token_ids size must be three
   * @param item_ids, output mapping item id list
   * @return true represents successful gain, false represents failed gain
   */
  bool get_items_by_tokens(const RecTokenTriple& rec_token_triple,
                           std::vector<int64_t>* item_ids) const;

  bool get_item_infos_by_tokens(const RecTokenTriple& rec_token_triple,
                                std::vector<RecItemInfo>* item_infos) const;

  /**
   * @brief Get the corresponding token ID triplet through a item id
   * @param item_ids, input item id
   * @param token_ids, output mapping token id triplet, so token_ids size will
   * be three
   * @return true represents successful gain, false represents failed gain
   */
  bool get_tokens_by_item(int64_t item_id,
                          std::vector<int32_t>* token_ids) const;

  /**
   * @brief Get all next token id list through the prefix token id list, for
   * example, in the vocab file, there are these token id triplets, 1-2-3,
   * 1-2-4, 7-8-9, if prefix the token id is [1], then the next token id list
   * is [2], if the prefix token id is [1,2], then the next token id list is
   * [3,4]
   * @param prefix_token_ids, prefix token id list, the size must be less then
   * three
   * @attention if prefix_token_ids size is zero, will return all first token of
   * the token triplets
   * @return  next token id list
   */
  const std::unordered_set<int32_t>& get_next_tokens_by_prefix_tokens(
      const Slice<int32_t>& prefix_token_ids) const;

  RecConstraintTables build_constraint_tables(int32_t vocab_size) const;

 private:
  // Check if initialization has been successful
  bool initialized_ = false;

  // Convert token to item map, key: token id triplet, value: item info list,
  // there is a token id triplet corresponding to multiple items, and
  // boost::hash<RecTokenTriple> will generate ordered triplet hash value.
  std::unordered_map<RecTokenTriple,
                     std::vector<RecItemInfo>,
                     boost::hash<RecTokenTriple>>
      tokens_to_item_infos_map_;

  // Convert item to tokens map, key: item id, value: token triplet, there is a
  // item id corresponding to a token id triplet
  std::unordered_map<int64_t, RecTokenTriple> item_to_tokens_map_;

  // Convert prifix tokens to next tokens map, key: prefix token id list, value:
  // next token id list
  std::unordered_map<std::vector<int32_t>,
                     std::unordered_set<int32_t>,
                     boost::hash<std::vector<int32_t>>>
      prefix_tokens_to_next_tokens_map_;
};
}  // namespace xllm
