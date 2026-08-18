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
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/global_flags.h"
#include "core/util/rec_model_utils.h"

namespace xllm {
namespace {

class ScopedConstrainedDecodingFlag final {
 public:
  explicit ScopedConstrainedDecodingFlag(bool value)
      : old_value_(FLAGS_enable_constrained_decoding) {
    FLAGS_enable_constrained_decoding = value;
  }

  ~ScopedConstrainedDecodingFlag() {
    FLAGS_enable_constrained_decoding = old_value_;
  }

 private:
  bool old_value_;
};

class ScopedInt32Flag final {
 public:
  ScopedInt32Flag(int32_t* flag, int32_t value)
      : flag_(flag), old_value_(*flag) {
    *flag_ = value;
  }

  ~ScopedInt32Flag() { *flag_ = old_value_; }

 private:
  int32_t* flag_;
  int32_t old_value_;
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

TEST(RecVocabDictTest, SamplesOneItemWhenConversionThresholdIsOne) {
  ScopedConstrainedDecodingFlag constrained(/*value=*/false);
  ScopedInt32Flag threshold(&FLAGS_each_conversion_threshold, 1);
  ScopedInt32Flag seed(&FLAGS_random_seed, 7);

  const std::filesystem::path vocab_path =
      std::filesystem::path(::testing::TempDir()) /
      "rec_vocab_dict_sample_one.bin";
  const RecTokenTriple triple{1, 2, 3};
  std::vector<std::pair<int64_t, RecTokenTriple>> records;
  records.reserve(100);
  std::unordered_set<int64_t> expected_ids;
  for (int64_t i = 0; i < 100; ++i) {
    records.emplace_back(1000 + i, triple);
    expected_ids.insert(1000 + i);
  }
  write_vocab_file(vocab_path, records);

  RecVocabDict vocab_dict;
  ASSERT_TRUE(vocab_dict.initialize(vocab_path.string()));

  std::vector<RecItemInfo> item_infos;
  ASSERT_TRUE(vocab_dict.get_item_infos_by_tokens(triple, &item_infos));
  ASSERT_EQ(item_infos.size(), 1);
  EXPECT_TRUE(expected_ids.count(item_infos.front().item_id));

  std::vector<RecItemInfo> again;
  ASSERT_TRUE(vocab_dict.get_item_infos_by_tokens(triple, &again));
  ASSERT_EQ(again.size(), 1);
  EXPECT_EQ(again.front().item_id, item_infos.front().item_id);

  std::vector<int64_t> item_ids;
  ASSERT_TRUE(vocab_dict.get_items_by_tokens(triple, &item_ids));
  ASSERT_EQ(item_ids.size(), 1);
  EXPECT_EQ(item_ids.front(), item_infos.front().item_id);

  std::filesystem::remove(vocab_path);
}

TEST(RecVocabDictTest, SamplesKItemsWithoutReturningFullCollisionList) {
  ScopedConstrainedDecodingFlag constrained(/*value=*/false);
  ScopedInt32Flag threshold(&FLAGS_each_conversion_threshold, 3);
  ScopedInt32Flag seed(&FLAGS_random_seed, 11);

  const std::filesystem::path vocab_path =
      std::filesystem::path(::testing::TempDir()) /
      "rec_vocab_dict_sample_k.bin";
  const RecTokenTriple triple{4, 5, 6};
  std::vector<std::pair<int64_t, RecTokenTriple>> records;
  std::unordered_set<int64_t> expected_ids;
  for (int64_t i = 0; i < 20; ++i) {
    records.emplace_back(2000 + i, triple);
    expected_ids.insert(2000 + i);
  }
  write_vocab_file(vocab_path, records);

  RecVocabDict vocab_dict;
  ASSERT_TRUE(vocab_dict.initialize(vocab_path.string()));

  std::vector<RecItemInfo> item_infos;
  ASSERT_TRUE(vocab_dict.get_item_infos_by_tokens(triple, &item_infos));
  ASSERT_EQ(item_infos.size(), 3);
  std::unordered_set<int64_t> sampled_ids;
  for (const RecItemInfo& info : item_infos) {
    EXPECT_TRUE(expected_ids.count(info.item_id));
    EXPECT_TRUE(sampled_ids.insert(info.item_id).second);
  }

  std::filesystem::remove(vocab_path);
}

TEST(RecVocabDictTest, ReturnsFullListWhenConversionThresholdDisabled) {
  ScopedConstrainedDecodingFlag constrained(/*value=*/false);
  ScopedInt32Flag threshold(&FLAGS_each_conversion_threshold, 0);

  const std::filesystem::path vocab_path =
      std::filesystem::path(::testing::TempDir()) /
      "rec_vocab_dict_sample_all.bin";
  const RecTokenTriple triple{7, 8, 9};
  write_vocab_file(vocab_path,
                   {
                       {1, triple},
                       {2, triple},
                       {3, triple},
                   });

  RecVocabDict vocab_dict;
  ASSERT_TRUE(vocab_dict.initialize(vocab_path.string()));

  std::vector<int64_t> item_ids;
  ASSERT_TRUE(vocab_dict.get_items_by_tokens(triple, &item_ids));
  EXPECT_EQ(item_ids, (std::vector<int64_t>{1, 2, 3}));

  std::filesystem::remove(vocab_path);
}

TEST(RecVocabDictTest, NormalizeSamplesWithoutKeepingFullList) {
  ScopedInt32Flag threshold(&FLAGS_each_conversion_threshold, 1);
  ScopedInt32Flag seed(&FLAGS_random_seed, 13);

  std::vector<RecItemInfo> raw;
  raw.reserve(50);
  for (int64_t i = 0; i < 50; ++i) {
    RecItemInfo info;
    info.item_id = i;
    raw.push_back(std::move(info));
  }

  const std::vector<RecItemInfo> sampled =
      normalize_rec_item_infos(raw, /*sequence_index=*/0);
  ASSERT_EQ(sampled.size(), 1);
  EXPECT_GE(sampled.front().item_id, 0);
  EXPECT_LT(sampled.front().item_id, 50);
}

}  // namespace xllm
