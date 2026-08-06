/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/tokenizer/rwkv_tokenizer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "framework/tokenizer/tokenizer_args.h"

namespace xllm {
namespace {

constexpr char kRwkvVocabFile[] = "rwkv_vocab_test.txt";

bool WriteTestRwkvVocab(const std::filesystem::path& filepath) {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    return false;
  }
  // Minimal trie vocab in the official rwkv_vocab_v20230424.txt line format:
  //   <id> <python_literal> <byte_length>
  file << "0 '<|bos|>' 7\n";
  file << "1 '<|eos|>' 7\n";
  file << "2 'hello' 5\n";
  file << "3 ' world' 6\n";
  file << "4 'test' 4\n";
  return file.good();
}

TokenizerArgs MakeRwkvArgs() {
  TokenizerArgs args;
  args.tokenizer_type() = "rwkv";
  args.vocab_file() = kRwkvVocabFile;
  return args;
}

}  // namespace

class RwkvTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() / "rwkv_tokenizer_test";
    std::filesystem::create_directories(test_dir_);

    vocab_path_ = test_dir_ / kRwkvVocabFile;
    ASSERT_TRUE(WriteTestRwkvVocab(vocab_path_));
  }

  void TearDown() override {
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  std::filesystem::path test_dir_;
  std::filesystem::path vocab_path_;
};

TEST_F(RwkvTokenizerTest, EncodeDecodeRoundTrip) {
  RwkvTokenizer tokenizer(test_dir_.string(), MakeRwkvArgs());
  EXPECT_EQ(tokenizer.vocab_size(), 5U);

  std::vector<int32_t> ids;
  ASSERT_TRUE(
      tokenizer.encode("hello world", &ids, /*add_special_tokens=*/false));
  EXPECT_EQ(ids, std::vector<int32_t>({2, 3}));

  const std::string decoded =
      tokenizer.decode(ids, /*skip_special_tokens=*/false);
  EXPECT_EQ(decoded, "hello world");
}

TEST_F(RwkvTokenizerTest, CloneUsesModelDirWithRelativeVocabFile) {
  RwkvTokenizer tokenizer(test_dir_.string(), MakeRwkvArgs());

  auto cloned = tokenizer.clone();
  ASSERT_NE(cloned, nullptr);
  EXPECT_EQ(cloned->vocab_size(), tokenizer.vocab_size());

  std::vector<int32_t> ids;
  ASSERT_TRUE(cloned->encode("test", &ids, /*add_special_tokens=*/false));
  EXPECT_EQ(ids, std::vector<int32_t>({4}));
}

TEST_F(RwkvTokenizerTest, AddBosToken) {
  TokenizerArgs args = MakeRwkvArgs();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = false;

  RwkvTokenizer tokenizer(test_dir_.string(), args);

  std::vector<int32_t> ids;
  ASSERT_TRUE(
      tokenizer.encode("hello world", &ids, /*add_special_tokens=*/true));
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids.front(), 0);
  EXPECT_GT(ids.size(), 1U);
}

TEST_F(RwkvTokenizerTest, AddEosToken) {
  TokenizerArgs args = MakeRwkvArgs();
  args.add_bos_token() = false;
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  RwkvTokenizer tokenizer(test_dir_.string(), args);

  std::vector<int32_t> ids;
  ASSERT_TRUE(
      tokenizer.encode("hello world", &ids, /*add_special_tokens=*/true));
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids.back(), 1);
  EXPECT_GT(ids.size(), 1U);
}

TEST_F(RwkvTokenizerTest, DecodeSkipSpecialTokens) {
  TokenizerArgs args = MakeRwkvArgs();
  args.bos_token() = "<|bos|>";
  args.eos_token() = "<|eos|>";

  RwkvTokenizer tokenizer(test_dir_.string(), args);

  const std::vector<int32_t> ids = {0, 2, 3, 1};
  const std::string decoded =
      tokenizer.decode(ids, /*skip_special_tokens=*/true);
  EXPECT_EQ(decoded, "hello world");
}

TEST_F(RwkvTokenizerTest, TokenToIdAndIdToToken) {
  RwkvTokenizer tokenizer(test_dir_.string(), MakeRwkvArgs());

  const std::optional<int32_t> hello_id = tokenizer.token_to_id("hello");
  ASSERT_TRUE(hello_id.has_value());
  EXPECT_EQ(hello_id.value(), 2);
  EXPECT_EQ(tokenizer.id_to_token(hello_id.value()), "hello");
}

TEST_F(RwkvTokenizerTest, EncodeFailsOnUnknownByte) {
  RwkvTokenizer tokenizer(test_dir_.string(), MakeRwkvArgs());

  std::vector<int32_t> ids;
  EXPECT_FALSE(
      tokenizer.encode("hello\x01", &ids, /*add_special_tokens=*/false));
  EXPECT_TRUE(ids.empty());
}

}  // namespace xllm
