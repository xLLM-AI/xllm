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

#include "core/distributed_runtime/spawn_worker_server/spawn_worker_protocol.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

namespace xllm::spawn_worker_protocol {
namespace {

std::vector<char*> mutable_argv(std::vector<std::string>* arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments->size());
  for (std::string& argument : *arguments) {
    argv.emplace_back(argument.data());
  }
  return argv;
}

TEST(SpawnWorkerProtocolTest, UsesExplicitDtypeFromCurrentArguments) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "int8";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), "int8");
}

TEST(SpawnWorkerProtocolTest, UsesDefaultDtypeFromLegacyArguments) {
  std::vector<std::string> arguments(kMinimumArgumentCount, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kMinimumArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), kDefaultIndexerCacheDtype);
}

TEST(SpawnWorkerProtocolTest, RejectsArgumentsBelowLegacyMinimum) {
  std::vector<std::string> arguments(kMinimumArgumentCount - 1, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_FALSE(parse_indexer_cache_dtype(kMinimumArgumentCount - 1, argv.data())
                   .has_value());
}

TEST(SpawnWorkerProtocolTest, IgnoresUnknownTrailingArguments) {
  std::vector<std::string> arguments(kArgumentCount + 1, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "int8";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount + 1, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), "int8");
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitEmptyDtype) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_TRUE(indexer_cache_dtype->empty());
}

}  // namespace
}  // namespace xllm::spawn_worker_protocol
