/* Copyright 2026 The xLLM Authors.

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

#include "processors/vision_token_scanner.h"

#include <gtest/gtest.h>

#include <string>
#include <utility>

namespace xllm {
namespace {

using TokenType = VisionTokenScanner::TokenType;

TEST(VisionTokenScannerTest, PreservesMixedTokenOrderAndOffsets) {
  const std::string prompt = "a<img>b<vid><img>tail";
  VisionTokenScanner scanner(prompt, "<img>", "<vid>");
  EXPECT_EQ(scanner.find_next(0), std::make_pair(TokenType::IMAGE, size_t{1}));
  EXPECT_EQ(scanner.find_next(6), std::make_pair(TokenType::VIDEO, size_t{7}));
  EXPECT_EQ(scanner.find_next(12),
            std::make_pair(TokenType::IMAGE, size_t{12}));
  EXPECT_EQ(scanner.find_next(17),
            std::make_pair(TokenType::INVALID, std::string::npos));
}

TEST(VisionTokenScannerTest, HandlesOneModalityAndExhaustion) {
  const std::string prompt = "<img><img>tail";
  VisionTokenScanner scanner(prompt, "<img>", "<vid>");
  EXPECT_EQ(scanner.find_next(0), std::make_pair(TokenType::IMAGE, size_t{0}));
  EXPECT_EQ(scanner.find_next(5), std::make_pair(TokenType::IMAGE, size_t{5}));
  EXPECT_EQ(scanner.find_next(10),
            std::make_pair(TokenType::INVALID, std::string::npos));
  EXPECT_EQ(scanner.find_next(prompt.size()),
            std::make_pair(TokenType::INVALID, std::string::npos));
}

TEST(VisionTokenScannerTest, SkipsConsumedWrapperAndHandlesVideoOnlyPrompt) {
  const std::string prompt = "<start><vid><end><start><vid><end>";
  VisionTokenScanner scanner(prompt, "<img>", "<vid>");
  EXPECT_EQ(scanner.find_next(0), std::make_pair(TokenType::VIDEO, size_t{7}));
  EXPECT_EQ(scanner.find_next(17),
            std::make_pair(TokenType::VIDEO, size_t{24}));
  EXPECT_EQ(scanner.find_next(prompt.size()),
            std::make_pair(TokenType::INVALID, std::string::npos));
}

TEST(VisionTokenScannerTest, HandlesEmptyAndTextOnlyPrompts) {
  VisionTokenScanner empty("", "<img>", "<vid>");
  EXPECT_EQ(empty.find_next(0),
            std::make_pair(TokenType::INVALID, std::string::npos));
  VisionTokenScanner text("ordinary text", "<img>", "<vid>");
  EXPECT_EQ(text.find_next(0),
            std::make_pair(TokenType::INVALID, std::string::npos));
}

}  // namespace
}  // namespace xllm
