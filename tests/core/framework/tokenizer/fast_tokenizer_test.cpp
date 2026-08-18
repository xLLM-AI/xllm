/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "framework/tokenizer/fast_tokenizer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "framework/tokenizer/tokenizer_args.h"
#include "framework/tokenizer/tokenizer_proxy.h"

namespace xllm {

std::string CreateTestTokenizerJson(const std::string& filepath,
                                    const std::string& truncation_json,
                                    const std::string& padding_json);

// Helper function to create a minimal valid tokenizer.json file for testing
// This creates a simple BPE tokenizer with a small vocabulary
std::string CreateTestTokenizerJson(const std::string& filepath) {
  return CreateTestTokenizerJson(filepath, "null", "null");
}

std::string CreateByteLevelTokenizerJson(const std::string& filepath) {
  const std::string tokenizer_json = R"({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": false
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": true,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {
      "{": 0,
      "}": 1,
      "\"": 2,
      "\u0120": 3,
      "\u00e4\u00b8\u0122": 4,
      "\u00e5\u012e\u0139": 5,
      "city": 6,
      "\u00c3": 7,
      "\u00a9": 8
    },
    "merges": []
  }
})";

  std::ofstream file(filepath);
  if (!file.is_open()) {
    return "";
  }
  file << tokenizer_json;
  file.close();
  return filepath;
}

std::string CreateNonByteLevelTokenizerJson(const std::string& filepath) {
  CreateTestTokenizerJson(filepath);
  std::ifstream input(filepath);
  nlohmann::json tokenizer_json = nlohmann::json::parse(input);
  tokenizer_json["decoder"] = {
      {"type", "WordPiece"}, {"prefix", "##"}, {"cleanup", true}};
  tokenizer_json["model"]["vocab"]["\u00a1"] = 15;

  std::ofstream output(filepath);
  output << tokenizer_json.dump();
  return filepath;
}

std::string CreateTestTokenizerJson(const std::string& filepath,
                                    const std::string& truncation_json,
                                    const std::string& padding_json) {
  // Minimal valid tokenizer.json for testing
  // This is a simplified BPE tokenizer configuration compatible with
  // HuggingFace tokenizers
  const std::string tokenizer_json = R"({
  "version": "1.0",
  "truncation": __TRUNCATION__,
  "padding": __PADDING__,
  "added_tokens": [
    {"id": 0, "content": "<|bos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|eos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "hello", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 3, "content": "world", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 4, "content": "test", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": {
    "type": "NFC"
  },
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true,
    "use_regex": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {
      "<|bos|>": 0,
      "<|eos|>": 1,
      "hello": 2,
      "world": 3,
      "test": 4,
      "h": 5,
      "e": 6,
      "l": 7,
      "o": 8,
      "w": 9,
      "r": 10,
      "d": 11,
      "t": 12,
      "s": 13,
      " ": 14
    },
    "merges": []
  }
})";
  std::string rendered = tokenizer_json;
  size_t pos = rendered.find("__TRUNCATION__");
  rendered.replace(pos, std::string("__TRUNCATION__").size(), truncation_json);
  pos = rendered.find("__PADDING__");
  rendered.replace(pos, std::string("__PADDING__").size(), padding_json);

  std::ofstream file(filepath);
  if (!file.is_open()) {
    return "";
  }
  file << rendered;
  file.close();
  return filepath;
}

class FastTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for test files
    test_dir_ = std::filesystem::temp_directory_path() / "fast_tokenizer_test";
    std::filesystem::create_directories(test_dir_);

    // Create test tokenizer.json file
    tokenizer_json_path_ = test_dir_ / "tokenizer.json";
    CreateTestTokenizerJson(tokenizer_json_path_.string());
  }

  void TearDown() override {
    // Clean up test files
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  std::filesystem::path test_dir_;
  std::filesystem::path tokenizer_json_path_;
};

// Test that BOS token is added when add_bos_token is true
TEST_F(FastTokenizerTest, AddBosToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // Check that BOS token (ID 0) is at the beginning
  EXPECT_EQ(ids[0], 0) << "BOS token should be at the beginning";

  // Verify that the rest of the tokens are present
  // The tokenizer should encode "hello world" into token IDs
  EXPECT_GT(ids.size(), 1) << "Should have more than just BOS token";
}

// Test that EOS token is added when add_eos_token is true
TEST_F(FastTokenizerTest, AddEosToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // Check that EOS token (ID 1) is at the end
  EXPECT_EQ(ids.back(), 1) << "EOS token should be at the end";

  // Verify that the rest of the tokens are present
  EXPECT_GT(ids.size(), 1) << "Should have more than just EOS token";
}

// Test that both BOS and EOS tokens are added when both flags are true
TEST_F(FastTokenizerTest, AddBothBosAndEosTokens) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_GE(ids.size(), 2) << "Should have at least BOS and EOS tokens";

  // Check that BOS token (ID 0) is at the beginning
  EXPECT_EQ(ids[0], 0) << "BOS token should be at the beginning";

  // Check that EOS token (ID 1) is at the end
  EXPECT_EQ(ids.back(), 1) << "EOS token should be at the end";

  // Verify that there are tokens in between
  EXPECT_GT(ids.size(), 2) << "Should have tokens between BOS and EOS";
}

// Test that no special tokens are added when both flags are false
TEST_F(FastTokenizerTest, NoSpecialTokens) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // Check that BOS token (ID 0) is NOT at the beginning
  EXPECT_NE(ids[0], 0) << "BOS token should not be present";

  // Check that EOS token (ID 1) is NOT at the end
  EXPECT_NE(ids.back(), 1) << "EOS token should not be present";
}

TEST_F(FastTokenizerTest, CallLevelFalseSuppressesConfiguredSpecialTokens) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);
  std::vector<int32_t> ids_without_special_tokens;
  std::vector<int32_t> ids_with_special_tokens;
  ASSERT_TRUE(tokenizer.encode("hello",
                               &ids_without_special_tokens,
                               /*add_special_tokens=*/false));
  ASSERT_TRUE(tokenizer.encode("hello",
                               &ids_with_special_tokens,
                               /*add_special_tokens=*/true));

  ASSERT_FALSE(ids_without_special_tokens.empty());
  ASSERT_FALSE(ids_with_special_tokens.empty());
  EXPECT_NE(ids_without_special_tokens.front(), 0);
  EXPECT_NE(ids_without_special_tokens.back(), 1);
  EXPECT_EQ(ids_with_special_tokens.front(), 0);
  EXPECT_EQ(ids_with_special_tokens.back(), 1);
}

TEST_F(FastTokenizerTest, DecodeTokenPreservesByteLevelPieces) {
  const std::filesystem::path byte_level_path =
      test_dir_ / "byte_level_tokenizer.json";
  CreateByteLevelTokenizerJson(byte_level_path.string());

  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = byte_level_path.string();
  FastTokenizer tokenizer(args);

  EXPECT_EQ(tokenizer.decode_token(0), "{");
  EXPECT_EQ(tokenizer.decode_token(2), "\"");
  EXPECT_EQ(tokenizer.decode_token(3), " ");
  EXPECT_EQ(tokenizer.decode_token(4), "\xE4\xB8\x80");
  EXPECT_EQ(tokenizer.decode_token(5), "\xE5\x8C\x97");
  EXPECT_EQ(tokenizer.decode_token(7), "\xC3");
  EXPECT_EQ(tokenizer.decode_token(8), "\xA9");

  const std::vector<int32_t> split_utf8_ids = {7, 8};
  EXPECT_EQ(tokenizer.decode(split_utf8_ids, /*skip_special_tokens=*/false),
            "\xC3\xA9");
}

TEST_F(FastTokenizerTest, DecodeTokenKeepsNonByteLevelPieces) {
  const std::filesystem::path non_byte_level_path =
      test_dir_ / "non_byte_level_tokenizer.json";
  CreateNonByteLevelTokenizerJson(non_byte_level_path.string());

  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = non_byte_level_path.string();
  FastTokenizer tokenizer(args);

  EXPECT_EQ(tokenizer.decode_token(15), "\xC2\xA1");
}

TEST_F(FastTokenizerTest, ProxyForwardsDecodeToken) {
  const std::filesystem::path byte_level_path =
      test_dir_ / "byte_level_proxy_tokenizer.json";
  CreateByteLevelTokenizerJson(byte_level_path.string());

  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = byte_level_path.string();
  std::unique_ptr<Tokenizer> tokenizer = std::make_unique<FastTokenizer>(args);
  TokenizerProxy proxy(std::move(tokenizer));

  EXPECT_EQ(proxy.decode_token(3), " ");
  EXPECT_EQ(proxy.decode_token(4), "\xE4\xB8\x80");
}

// Test that BOS token is not added when bos_token is empty
TEST_F(FastTokenizerTest, AddBosTokenWithEmptyToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "";  // Empty token
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // BOS token should not be added because bos_token is empty
  EXPECT_NE(ids[0], 0)
      << "BOS token should not be added when bos_token is empty";
}

// Test that EOS token is not added when eos_token is empty
TEST_F(FastTokenizerTest, AddEosTokenWithEmptyToken) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = true;
  args.eos_token() = "";  // Empty token

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  bool success = tokenizer.encode("hello world", &ids);

  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // EOS token should not be added because eos_token is empty
  EXPECT_NE(ids.back(), 1)
      << "EOS token should not be added when eos_token is empty";
}

// Test that BOS token is not duplicated when it already exists
// This simulates the case where the underlying tokenizer already added BOS
TEST_F(FastTokenizerTest, SkipBosTokenWhenAlreadyPresent) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = false;

  FastTokenizer tokenizer(args);

  // First encode a text that doesn't start with BOS
  std::vector<int32_t> ids1;
  bool success1 = tokenizer.encode("hello world", &ids1);
  ASSERT_TRUE(success1);
  ASSERT_FALSE(ids1.empty());

  // Verify BOS token was added
  EXPECT_EQ(ids1[0], 0) << "BOS token should be added";
  size_t size_with_bos = ids1.size();

  // Now encode text that starts with BOS token directly
  // We'll encode the BOS token itself, which should result in BOS being the
  // first token
  std::vector<int32_t> ids2;
  bool success2 = tokenizer.encode("<|bos|> hello world", &ids2);
  ASSERT_TRUE(success2);
  ASSERT_FALSE(ids2.empty());

  // The BOS token should be present, but we should not have added it twice
  // Count how many times BOS token (ID 0) appears at the beginning
  int bos_count_at_start = 0;
  for (size_t i = 0; i < ids2.size() && ids2[i] == 0; ++i) {
    bos_count_at_start++;
  }

  // Should have at most one BOS token at the beginning
  EXPECT_LE(bos_count_at_start, 1)
      << "BOS token should not be duplicated when already present";
}

// Test that EOS token is not duplicated when it already exists
TEST_F(FastTokenizerTest, SkipEosTokenWhenAlreadyPresent) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = false;
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  // First encode a text that doesn't end with EOS
  std::vector<int32_t> ids1;
  bool success1 = tokenizer.encode("hello world", &ids1);
  ASSERT_TRUE(success1);
  ASSERT_FALSE(ids1.empty());

  // Verify EOS token was added
  EXPECT_EQ(ids1.back(), 1) << "EOS token should be added";
  size_t size_with_eos = ids1.size();

  // Now encode text that ends with EOS token directly
  std::vector<int32_t> ids2;
  bool success2 = tokenizer.encode("hello world <|eos|>", &ids2);
  ASSERT_TRUE(success2);
  ASSERT_FALSE(ids2.empty());

  // The EOS token should be present, but we should not have added it twice
  // Count how many times EOS token (ID 1) appears at the end
  int eos_count_at_end = 0;
  for (int i = ids2.size() - 1; i >= 0 && ids2[i] == 1; --i) {
    eos_count_at_end++;
  }

  // Should have at most one EOS token at the end
  EXPECT_LE(eos_count_at_end, 1)
      << "EOS token should not be duplicated when already present";
}

// Test that both BOS and EOS tokens are not duplicated when both already exist
TEST_F(FastTokenizerTest, SkipBothBosAndEosTokensWhenAlreadyPresent) {
  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = tokenizer_json_path_.string();
  args.add_bos_token() = true;
  args.bos_token() = "<|bos|>";
  args.add_eos_token() = true;
  args.eos_token() = "<|eos|>";

  FastTokenizer tokenizer(args);

  // Encode text that already contains both BOS and EOS tokens
  std::vector<int32_t> ids;
  bool success = tokenizer.encode("<|bos|> hello world <|eos|>", &ids);
  ASSERT_TRUE(success);
  ASSERT_FALSE(ids.empty());

  // Count BOS tokens at the beginning
  int bos_count_at_start = 0;
  for (size_t i = 0; i < ids.size() && ids[i] == 0; ++i) {
    bos_count_at_start++;
  }

  // Count EOS tokens at the end
  int eos_count_at_end = 0;
  for (int i = ids.size() - 1; i >= 0 && ids[i] == 1; --i) {
    eos_count_at_end++;
  }

  // Should have at most one BOS token at the beginning
  EXPECT_LE(bos_count_at_start, 1)
      << "BOS token should not be duplicated when already present";

  // Should have at most one EOS token at the end
  EXPECT_LE(eos_count_at_end, 1)
      << "EOS token should not be duplicated when already present";
}

TEST_F(FastTokenizerTest, IgnoreTokenizerJsonPaddingAndTruncation) {
  std::filesystem::path padded_tokenizer_path =
      test_dir_ / "tokenizer_with_padding.json";
  ASSERT_FALSE(
      CreateTestTokenizerJson(
          padded_tokenizer_path.string(),
          R"({"direction":"Right","max_length":4,"strategy":"LongestFirst","stride":0})",
          R"({"strategy":{"Fixed":4},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"<|bos|>"})")
          .empty());

  TokenizerArgs args;
  args.tokenizer_type() = "fast";
  args.vocab_file() = padded_tokenizer_path.string();

  FastTokenizer tokenizer(args);

  std::vector<int32_t> ids;
  ASSERT_TRUE(tokenizer.encode("hello", &ids));
  EXPECT_EQ(ids.size(), 1)
      << "FastTokenizer should ignore tokenizer.json padding/truncation.";
  EXPECT_EQ(ids[0], 2);
}

}  // namespace xllm
