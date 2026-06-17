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

#include "framework/encoder_cache/encoder_cache.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <optional>

#include "util/hash_util.h"

namespace xllm {
namespace {

int64_t tensor_bytes(const torch::Tensor& tensor) {
  return tensor.numel() * static_cast<int64_t>(tensor.element_size());
}

torch::Tensor make_embedding(float value) {
  return torch::full({2, 3}, value, torch::kFloat32);
}

TEST(EncoderCacheTest, InsertAndLookup) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding));

  cache.insert(hash_string("IMAGE_0"), embedding);
  std::optional<torch::Tensor> cached = cache.lookup(hash_string("IMAGE_0"));

  ASSERT_TRUE(cached.has_value());
  EXPECT_TRUE(torch::equal(cached.value(), embedding));
}

TEST(EncoderCacheTest, LookupOnEmpty) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding));

  std::optional<torch::Tensor> cached = cache.lookup(hash_string("IMAGE_0"));

  EXPECT_FALSE(cached.has_value());
}

TEST(EncoderCacheTest, LruEvictsLeastRecentlyUsed) {
  torch::Tensor embedding = make_embedding(1.0);
  const int64_t bytes = tensor_bytes(embedding);
  EncoderCache cache(/*max_bytes=*/bytes * 2);

  cache.insert(hash_string("IMAGE_0"), make_embedding(1.0));
  cache.insert(hash_string("IMAGE_1"), make_embedding(2.0));
  cache.insert(hash_string("IMAGE_2"), make_embedding(3.0));

  EXPECT_FALSE(cache.lookup(hash_string("IMAGE_0")).has_value());
  EXPECT_TRUE(cache.lookup(hash_string("IMAGE_1")).has_value());
  EXPECT_TRUE(cache.lookup(hash_string("IMAGE_2")).has_value());
}

TEST(EncoderCacheTest, LookupTouchesEntry) {
  torch::Tensor embedding = make_embedding(1.0);
  const int64_t bytes = tensor_bytes(embedding);
  EncoderCache cache(/*max_bytes=*/bytes * 2);

  cache.insert(hash_string("IMAGE_0"), make_embedding(1.0));
  cache.insert(hash_string("IMAGE_1"), make_embedding(2.0));
  ASSERT_TRUE(cache.lookup(hash_string("IMAGE_0")).has_value());
  cache.insert(hash_string("IMAGE_2"), make_embedding(3.0));

  EXPECT_TRUE(cache.lookup(hash_string("IMAGE_0")).has_value());
  EXPECT_FALSE(cache.lookup(hash_string("IMAGE_1")).has_value());
  EXPECT_TRUE(cache.lookup(hash_string("IMAGE_2")).has_value());
}

TEST(EncoderCacheTest, SingleEntryLargerThanCapacityIsSkipped) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding) - 1);

  cache.insert(hash_string("IMAGE_0"), embedding);

  EXPECT_FALSE(cache.lookup(hash_string("IMAGE_0")).has_value());
}

TEST(EncoderCacheTest, ClearResetsState) {
  torch::Tensor embedding = make_embedding(1.0);
  EncoderCache cache(/*max_bytes=*/tensor_bytes(embedding) * 2);
  cache.insert(hash_string("IMAGE_0"), make_embedding(1.0));
  cache.insert(hash_string("IMAGE_1"), make_embedding(2.0));

  cache.clear();

  EXPECT_FALSE(cache.lookup(hash_string("IMAGE_0")).has_value());
  EXPECT_FALSE(cache.lookup(hash_string("IMAGE_1")).has_value());
  cache.insert(hash_string("IMAGE_2"), make_embedding(3.0));
  EXPECT_TRUE(cache.lookup(hash_string("IMAGE_2")).has_value());
}

}  // namespace
}  // namespace xllm
