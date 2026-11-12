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

#include <gtest/gtest.h>

#include "core/util/partial_json_parser.h"

namespace xllm::partial_json {
namespace {

TEST(PartialJsonLoadsTest, CompletesNestedArguments) {
  const std::string input = R"({"arguments":{"city":"Shang)";
  auto [value, consumed] = partial_json_loads(input, Allow::ALL);
  EXPECT_EQ(value, nlohmann::json({{"arguments", {{"city", "Shang"}}}}));
  EXPECT_EQ(consumed, static_cast<int32_t>(input.size()));
  EXPECT_FALSE(is_complete_json(input));
  EXPECT_TRUE(is_complete_json(value.dump()));
}

TEST(PartialJsonLoadsTest, HonorsIncompleteStringPolicy) {
  const std::string input = R"({"name":"wea)";
  auto [allowed, allowed_end] = partial_json_loads(input, Allow::ALL);
  auto [restricted, restricted_end] =
      partial_json_loads(input, Allow::ALL & ~Allow::STR);
  EXPECT_EQ(allowed, nlohmann::json({{"name", "wea"}}));
  EXPECT_EQ(restricted, nlohmann::json::object());
  EXPECT_EQ(allowed_end, static_cast<int32_t>(input.size()));
  EXPECT_EQ(restricted_end, static_cast<int32_t>(input.size()));
}

TEST(PartialJsonLoadsTest, PreservesCompleteValues) {
  const std::string input = R"([null,true,42,{"text":"ok"}])";
  auto [value, consumed] = partial_json_loads(input, Allow::ALL);
  EXPECT_EQ(value, nlohmann::json::parse(input));
  EXPECT_EQ(consumed, static_cast<int32_t>(input.size()));
  EXPECT_TRUE(is_complete_json(input));
  EXPECT_FALSE(is_complete_json(input + " trailing"));
}

TEST(PartialJsonLoadsTest, PropagatesInvalidInput) {
  EXPECT_THROW(partial_json_loads("invalid", Allow::ALL),
               nlohmann::json::parse_error);
  EXPECT_THROW(partial_json_loads("", Allow::ALL), nlohmann::json::parse_error);
  EXPECT_FALSE(is_complete_json("invalid"));
  EXPECT_FALSE(is_complete_json(""));
}

}  // namespace
}  // namespace xllm::partial_json
