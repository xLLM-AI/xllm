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

#include "framework/state_dict/rec_vocab_dict.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

#include "core/framework/config/rec_config.h"

namespace xllm {
namespace {

class ScopedConstrainedDecodingFlag final {
 public:
  explicit ScopedConstrainedDecodingFlag(bool value)
      : old_value_(RecConfig::get_instance().enable_constrained_decoding()) {
    RecConfig::get_instance().enable_constrained_decoding(value);
  }

  ~ScopedConstrainedDecodingFlag() {
    RecConfig::get_instance().enable_constrained_decoding(old_value_);
  }

 private:
  bool old_value_;
};

void write_vocab_file(
    const std::filesystem::path& path,
    const std::vector<std::pair<int64_t, RecTokenTriple>>& records) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  CHECK(ofs.is_open()) << "Failed to open test vocab file: " << path;
  for (const auto& record : records) {
    const int64_t item_id = record.first;
    const RecTokenTriple& tokens = record.second;
    ofs.write(reinterpret_cast<const char*>(&item_id), sizeof(item_id));
    ofs.write(reinterpret_cast<const char*>(tokens.data()),
              REC_TOKEN_SIZE * sizeof(int32_t));
  }
}

std::vector<int32_t> sorted_next_tokens(
    const std::unordered_set<int32_t>& token_set) {
  std::vector<int32_t> tokens(token_set.begin(), token_set.end());
  std::sort(tokens.begin(), tokens.end());
  return tokens;
}

std::vector<int32_t> prefix1_values_for_token(const RecConstraintTables& tables,
                                              int32_t t0) {
  const int32_t begin = tables.prefix1_offsets[static_cast<size_t>(t0)];
  const int32_t end = tables.prefix1_offsets[static_cast<size_t>(t0) + 1];
  return std::vector<int32_t>(tables.prefix1_values.begin() + begin,
                              tables.prefix1_values.begin() + end);
}

std::vector<int32_t> prefix2_values_for_tokens(
    const RecConstraintTables& tables,
    int32_t t0,
    int32_t t1) {
  const int32_t begin = tables.prefix1_offsets[static_cast<size_t>(t0)];
  const int32_t end = tables.prefix1_offsets[static_cast<size_t>(t0) + 1];
  for (int32_t index = begin; index < end; ++index) {
    if (tables.prefix1_values[static_cast<size_t>(index)] != t1) {
      continue;
    }
    const int32_t prefix2_begin =
        tables.prefix2_value_offsets[static_cast<size_t>(index)];
    const int32_t prefix2_end =
        tables.prefix2_value_offsets[static_cast<size_t>(index) + 1];
    return std::vector<int32_t>(tables.prefix2_values.begin() + prefix2_begin,
                                tables.prefix2_values.begin() + prefix2_end);
  }
  return {};
}

std::vector<int64_t> prefix1_pair_keys_for_token(
    const RecConstraintTables& tables,
    int32_t t0) {
  const int32_t begin = tables.prefix1_offsets[static_cast<size_t>(t0)];
  const int32_t end = tables.prefix1_offsets[static_cast<size_t>(t0) + 1];
  return std::vector<int64_t>(tables.prefix1_pair_keys.begin() + begin,
                              tables.prefix1_pair_keys.begin() + end);
}

}  // namespace

TEST(RecVocabDictTest, BuildConstraintTablesMatchesLegacyPrefixMap) {
  ScopedConstrainedDecodingFlag flag(/*value=*/true);
  const std::filesystem::path vocab_path =
      std::filesystem::path(::testing::TempDir()) / "rec_vocab_dict_test.bin";

  write_vocab_file(vocab_path,
                   {
                       {100, RecTokenTriple{1, 2, 3}},
                       {101, RecTokenTriple{1, 2, 4}},
                       {102, RecTokenTriple{1, 5, 6}},
                       {103, RecTokenTriple{7, 8, 9}},
                       {104, RecTokenTriple{7, 8, 10}},
                       {105, RecTokenTriple{1, 2, 3}},
                   });

  RecVocabDict vocab_dict;
  ASSERT_TRUE(vocab_dict.initialize(vocab_path.string()));

  const RecConstraintTables tables =
      vocab_dict.build_constraint_tables(/*vocab_size=*/16);

  EXPECT_EQ(tables.vocab_size, 16);
  EXPECT_EQ(tables.first_token_ids, std::vector<int32_t>({1, 7}));
  EXPECT_EQ(prefix1_values_for_token(tables, /*t0=*/1),
            std::vector<int32_t>({2, 5}));
  EXPECT_EQ(prefix1_values_for_token(tables, /*t0=*/7),
            std::vector<int32_t>({8}));
  EXPECT_EQ(prefix1_pair_keys_for_token(tables, /*t0=*/1),
            std::vector<int64_t>({18, 21}));
  EXPECT_EQ(prefix1_pair_keys_for_token(tables, /*t0=*/7),
            std::vector<int64_t>({120}));
  EXPECT_TRUE(std::is_sorted(tables.prefix1_pair_keys.begin(),
                             tables.prefix1_pair_keys.end()));
  EXPECT_EQ(tables.prefix1_pair_keys.size(), tables.prefix1_values.size());
  EXPECT_TRUE(prefix1_values_for_token(tables, /*t0=*/0).empty());
  EXPECT_EQ(prefix2_values_for_tokens(tables, /*t0=*/1, /*t1=*/2),
            std::vector<int32_t>({3, 4}));
  EXPECT_EQ(prefix2_values_for_tokens(tables, /*t0=*/1, /*t1=*/5),
            std::vector<int32_t>({6}));
  EXPECT_EQ(prefix2_values_for_tokens(tables, /*t0=*/7, /*t1=*/8),
            std::vector<int32_t>({9, 10}));
  EXPECT_EQ(tables.prefix2_value_offsets.size(),
            tables.prefix1_values.size() + 1);
  EXPECT_EQ(tables.max_first_degree, 2);
  EXPECT_EQ(tables.max_prefix1_degree, 2);
  EXPECT_EQ(tables.max_prefix2_degree, 2);

  std::vector<int32_t> empty_prefix;
  EXPECT_EQ(sorted_next_tokens(vocab_dict.get_next_tokens_by_prefix_tokens(
                Slice<int32_t>(empty_prefix))),
            tables.first_token_ids);

  std::vector<int32_t> prefix1{1};
  EXPECT_EQ(sorted_next_tokens(vocab_dict.get_next_tokens_by_prefix_tokens(
                Slice<int32_t>(prefix1))),
            prefix1_values_for_token(tables, /*t0=*/1));

  std::vector<int32_t> prefix2{1, 2};
  EXPECT_EQ(sorted_next_tokens(vocab_dict.get_next_tokens_by_prefix_tokens(
                Slice<int32_t>(prefix2))),
            prefix2_values_for_tokens(tables, /*t0=*/1, /*t1=*/2));

  std::filesystem::remove(vocab_path);
}

}  // namespace xllm
