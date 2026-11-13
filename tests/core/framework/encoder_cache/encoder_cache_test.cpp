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

#include "framework/encoder_cache/encoder_cache.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <optional>
#include <string_view>

#include "util/hash_util.h"

namespace xllm {
namespace {

XXH3Key cache_key(std::string_view value) {
  return hash_string(value, /*seed=*/1024);
}

int64_t tensor_bytes(const torch::Tensor& tensor) {
  return tensor.numel() * static_cast<int64_t>(tensor.element_size());
}

torch::Tensor make_embedding(float value) {
  return torch::full({2, 3}, value, torch::kFloat32);
}

TEST(EncoderCacheTest, InsertAndLookup) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding));

  cache.insert(cache_key("IMAGE_0"), embedding);
  std::optional<torch::Tensor> cached = cache.lookup(cache_key("IMAGE_0"));

  ASSERT_TRUE(cached.has_value());
  EXPECT_TRUE(torch::equal(cached.value(), embedding));
}

TEST(EncoderCacheTest, LookupOnEmpty) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding));

  std::optional<torch::Tensor> cached = cache.lookup(cache_key("IMAGE_0"));

  EXPECT_FALSE(cached.has_value());
}

TEST(EncoderCacheTest, LruEvictsLeastRecentlyUsed) {
  torch::Tensor embedding = make_embedding(1.0);
  const int64_t bytes = tensor_bytes(embedding);
  EncoderCache cache(/*max_bytes=*/bytes * 2);

  cache.insert(cache_key("IMAGE_0"), make_embedding(1.0));
  cache.insert(cache_key("IMAGE_1"), make_embedding(2.0));
  cache.insert(cache_key("IMAGE_2"), make_embedding(3.0));

  EXPECT_FALSE(cache.lookup(cache_key("IMAGE_0")).has_value());
  EXPECT_TRUE(cache.lookup(cache_key("IMAGE_1")).has_value());
  EXPECT_TRUE(cache.lookup(cache_key("IMAGE_2")).has_value());
}

TEST(EncoderCacheTest, LookupTouchesEntry) {
  torch::Tensor embedding = make_embedding(1.0);
  const int64_t bytes = tensor_bytes(embedding);
  EncoderCache cache(/*max_bytes=*/bytes * 2);

  cache.insert(cache_key("IMAGE_0"), make_embedding(1.0));
  cache.insert(cache_key("IMAGE_1"), make_embedding(2.0));
  ASSERT_TRUE(cache.lookup(cache_key("IMAGE_0")).has_value());
  cache.insert(cache_key("IMAGE_2"), make_embedding(3.0));

  EXPECT_TRUE(cache.lookup(cache_key("IMAGE_0")).has_value());
  EXPECT_FALSE(cache.lookup(cache_key("IMAGE_1")).has_value());
  EXPECT_TRUE(cache.lookup(cache_key("IMAGE_2")).has_value());
}

TEST(EncoderCacheTest, SingleEntryLargerThanCapacityIsSkipped) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding) - 1);

  cache.insert(cache_key("IMAGE_0"), embedding);

  EXPECT_FALSE(cache.lookup(cache_key("IMAGE_0")).has_value());
}

TEST(EncoderCacheTest, OversizedEntryDoesNotEvictCachedEntries) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding));
  cache.insert(cache_key("retained"), embedding);
  cache.insert(cache_key("oversized"), torch::ones({100}, torch::kFloat32));
  EXPECT_TRUE(cache.lookup(cache_key("retained")).has_value());
  EXPECT_FALSE(cache.lookup(cache_key("oversized")).has_value());
}

TEST(EncoderCacheTest, RetainsIndependentCopyOfNoncontiguousView) {
  torch::Tensor source = torch::arange(24, torch::kFloat32).reshape({4, 6});
  torch::Tensor view =
      source.slice(/*dim=*/1, /*start=*/0, /*end=*/6, /*step=*/2);
  torch::Tensor expected = view.clone();
  EncoderCache cache(/*max_bytes=*/tensor_bytes(view));
  cache.insert(cache_key("view"), view);
  source.zero_();
  const auto cached = cache.lookup(cache_key("view"));
  ASSERT_TRUE(cached.has_value());
  EXPECT_TRUE(torch::equal(cached.value(), expected));
}

TEST(EncoderCacheTest, ClearResetsState) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding) * 2);
  cache.insert(cache_key("IMAGE_0"), make_embedding(1.0));
  cache.insert(cache_key("IMAGE_1"), make_embedding(2.0));

  cache.clear();

  EXPECT_FALSE(cache.lookup(cache_key("IMAGE_0")).has_value());
  EXPECT_FALSE(cache.lookup(cache_key("IMAGE_1")).has_value());
  cache.insert(cache_key("IMAGE_2"), make_embedding(3.0));
  EXPECT_TRUE(cache.lookup(cache_key("IMAGE_2")).has_value());
}

}  // namespace
}  // namespace xllm
