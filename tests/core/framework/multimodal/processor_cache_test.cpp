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

#include "core/framework/multimodal/processor_cache.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <optional>

#include "util/hash_util.h"

namespace xllm {
namespace {

MMDataItem make_item(int64_t value) {
  MMDict data;
  data["value"] = torch::tensor({value}, torch::kInt64);
  return MMDataItem(MMType::IMAGE, data);
}

int64_t item_value(const MMDataItem& item) {
  return item.get<torch::Tensor>("value")->item<int64_t>();
}

TEST(ProcessorCacheTest, LookupTouchesEntryBeforeEviction) {
  ProcessorCache cache(/*max_items=*/2);
  const XXH3Key key_a = hash_string("image-a");
  const XXH3Key key_b = hash_string("image-b");
  const XXH3Key key_c = hash_string("image-c");

  cache.insert(key_a, make_item(1));
  cache.insert(key_b, make_item(2));
  ASSERT_TRUE(cache.lookup(key_a).has_value());
  cache.insert(key_c, make_item(3));

  std::optional<MMDataItem> cached_a = cache.lookup(key_a);
  std::optional<MMDataItem> cached_b = cache.lookup(key_b);
  std::optional<MMDataItem> cached_c = cache.lookup(key_c);
  ASSERT_TRUE(cached_a.has_value());
  EXPECT_EQ(item_value(cached_a.value()), 1);
  EXPECT_FALSE(cached_b.has_value());
  ASSERT_TRUE(cached_c.has_value());
  EXPECT_EQ(item_value(cached_c.value()), 3);
}

TEST(ProcessorCacheTest, ClearsStateAndDetachesTensorViews) {
  ProcessorCache cache(/*max_items=*/1);
  const XXH3Key key = hash_string("image-view");
  torch::Tensor batch = torch::arange(8, torch::kInt64).reshape({2, 4});
  torch::Tensor view = batch[0];
  MMDict data;
  data["pixel_values"] = view;
  MMDataItem item(MMType::IMAGE, data);
  item.mutable_state().mutable_token_pos().offset = 17;
  item.mutable_state().mutable_seq_index() = 3;
  item.mutable_state().mutable_schedule_data().key = key;

  cache.insert(key, item);
  batch[0].fill_(99);
  std::optional<MMDataItem> cached = cache.lookup(key);

  ASSERT_TRUE(cached.has_value());
  torch::Tensor cached_tensor =
      cached->get<torch::Tensor>("pixel_values").value();
  EXPECT_TRUE(torch::equal(cached_tensor, torch::arange(4, torch::kInt64)));
  EXPECT_EQ(cached_tensor.storage().nbytes(), cached_tensor.nbytes());
  EXPECT_EQ(cached->state().token_pos().offset, 0);
  EXPECT_EQ(cached->state().seq_index(), -1);
}

TEST(ProcessorCacheTest, KeepsOwnedTensorStorage) {
  ProcessorCache cache(/*max_items=*/1);
  const XXH3Key key = hash_string("video-owned");
  torch::Tensor owned = torch::ones({2, 2}, torch::kFloat32);
  MMDict data;
  data["video_values"] = owned;

  cache.insert(key, MMDataItem(MMType::VIDEO, data));
  std::optional<MMDataItem> cached = cache.lookup(key);

  ASSERT_TRUE(cached.has_value());
  torch::Tensor cached_tensor =
      cached->get<torch::Tensor>("video_values").value();
  EXPECT_EQ(cached_tensor.data_ptr(), owned.data_ptr());
}

}  // namespace
}  // namespace xllm
