#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>

#include "common/macros.h"
#include "common/types.h"
#include "util/slice.h"

// Pure C++ hash replacement for Boost::hash
namespace xllm {
namespace pure_cpp {
  struct RecTokenTripleHash {
    size_t operator()(const RecTokenTriple& t) const {
      size_t h = 0;
      std::hash<int64_t> hasher;
      h ^= hasher(t.t0) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= hasher(t.t1) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h ^= hasher(t.t2) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };
  struct VectorInt32Hash {
    size_t operator()(const std::vector<int32_t>& v) const {
      size_t h = 0;
      std::hash<int32_t> hasher;
      for (int32_t i : v) {
        h ^= hasher(i) + 0x9e3779b9 + (h << 6) + (h >> 2);
      }
      return h;
    }
  };
} // namespace pure_cpp
} // namespace xllm

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
};

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
  // pure C++ hash replacement for Boost::hash<RecTokenTriple>
  std::unordered_map<RecTokenTriple,
                     std::vector<RecItemInfo>,
                     pure_cpp::RecTokenTripleHash>
      tokens_to_item_infos_map_;

  // Convert item to tokens map, key: item id, value: token triplet, there is a
  // item id corresponding to a token id triplet
  std::unordered_map<int64_t, RecTokenTriple> item_to_tokens_map_;

  // Convert prifix tokens to next tokens map, key: prefix token id list, value:
  // next token id list
  std::unordered_map<std::vector<int32_t>,
                     std::unordered_set<int32_t>,
                     pure_cpp::VectorInt32Hash>
      prefix_tokens_to_next_tokens_map_;
};
}  // namespace xllm
