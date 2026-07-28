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

#include "processors/caching_multimodal_processor.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "util/hash_util.h"

namespace xllm {
namespace {

class FakeMultimodalProcessor final : public MultimodalProcessorBase {
 public:
  FakeMultimodalProcessor() : MultimodalProcessorBase(/*tokenizer=*/nullptr) {}

  bool process_prompt(std::string& /*prompt*/,
                      MMData& /*mm_data*/,
                      std::vector<int32_t>& /*token_ids*/) override {
    ++prompt_call_count_;
    return true;
  }

  bool process_multimodal(const MMInput& inputs, MMData& data) const override {
    ++multimodal_call_count_;
    std::vector<std::string> raw_items;
    raw_items.reserve(inputs.size());
    MMItemVec output_items;
    output_items.reserve(inputs.size());
    uint32_t full_type = MMType::NONE;
    for (const MMInputItem& input : inputs.items()) {
      raw_items.emplace_back(input.raw_data);
      MMType type =
          input.has_type(MMType::VIDEO) ? MMType::VIDEO : MMType::IMAGE;
      int64_t value = 3;
      if (input.raw_data == "image-a") {
        value = 1;
      } else if (input.raw_data == "video-b") {
        value = 2;
      }
      MMDict item_data;
      item_data["value"] = torch::tensor({value}, torch::kInt64);
      MMDataItem item(type, item_data);
      item.mutable_state().mutable_token_pos().offset = 99;
      item.mutable_state().mutable_seq_index() = 7;
      full_type |= type;
      output_items.emplace_back(std::move(item));
    }
    processed_raw_items_.emplace_back(std::move(raw_items));
    data.set(full_type, std::move(output_items));
    hash_mm_items(inputs, data);
    return true;
  }

  int32_t prompt_call_count_ = 0;
  mutable int32_t multimodal_call_count_ = 0;
  mutable std::vector<std::vector<std::string>> processed_raw_items_;
};

MMInputItem make_raw_input(MMType type, std::string raw_data) {
  MMInputItem input;
  input.type = type;
  input.raw_data = std::move(raw_data);
  return input;
}

MMInputItem make_embedding_input() {
  MMInputItem input;
  input.type = MMType::IMAGE;
  input.embedding.embedding = torch::ones({1, 2}, torch::kFloat32);
  return input;
}

MMInputItem make_uuid_input(std::string raw_data, std::string uuid) {
  MMInputItem input = make_raw_input(MMType::IMAGE, std::move(raw_data));
  input.uuid = std::move(uuid);
  return input;
}

MMInput make_input(std::vector<MMInputItem> items) {
  MMInput input;
  input.insert(items);
  return input;
}

int64_t item_value(const MMDataItem& item) {
  return item.get<torch::Tensor>("value")->item<int64_t>();
}

TEST(CachingMultimodalProcessorTest, RecombinesMixedHitsInInputOrder) {
  auto inner = std::make_unique<FakeMultimodalProcessor>();
  FakeMultimodalProcessor* inner_ptr = inner.get();
  CachingMultimodalProcessor processor(std::move(inner),
                                       /*max_cache_items=*/4);

  MMData first_data;
  ASSERT_TRUE(processor.process_multimodal(
      make_input({make_raw_input(MMType::IMAGE, "image-a")}), first_data));
  ASSERT_EQ(inner_ptr->multimodal_call_count_, 1);

  MMData mixed_data;
  ASSERT_TRUE(processor.process_multimodal(
      make_input({make_raw_input(MMType::IMAGE, "image-a"),
                  make_embedding_input(),
                  make_raw_input(MMType::VIDEO, "video-b")}),
      mixed_data));

  ASSERT_EQ(inner_ptr->multimodal_call_count_, 2);
  ASSERT_EQ(inner_ptr->processed_raw_items_.size(), 2);
  EXPECT_EQ(inner_ptr->processed_raw_items_[1],
            (std::vector<std::string>{"", "video-b"}));
  EXPECT_EQ(mixed_data.type(),
            static_cast<uint32_t>(MMType::IMAGE) |
                static_cast<uint32_t>(MMType::VIDEO));
  const MMItemVec& mixed_items = mixed_data.items<MMItemVec>();
  ASSERT_EQ(mixed_items.size(), 3);
  EXPECT_EQ(item_value(mixed_items[0]), 1);
  EXPECT_EQ(item_value(mixed_items[1]), 3);
  EXPECT_EQ(item_value(mixed_items[2]), 2);
  EXPECT_EQ(mixed_items[0].state().token_pos().offset, 0);
  EXPECT_EQ(mixed_items[0].state().seq_index(), -1);
  EXPECT_EQ(mixed_items[0].state().schedule_data().key, hash_string("image-a"));
  EXPECT_EQ(mixed_items[1].state().token_pos().offset, 99);

  MMData hit_data;
  ASSERT_TRUE(processor.process_multimodal(
      make_input({make_raw_input(MMType::IMAGE, "image-a"),
                  make_raw_input(MMType::VIDEO, "video-b")}),
      hit_data));
  EXPECT_EQ(inner_ptr->multimodal_call_count_, 2);
  EXPECT_EQ(hit_data.type(),
            static_cast<uint32_t>(MMType::IMAGE) |
                static_cast<uint32_t>(MMType::VIDEO));
}

TEST(CachingMultimodalProcessorTest, ForwardsPromptProcessing) {
  auto inner = std::make_unique<FakeMultimodalProcessor>();
  FakeMultimodalProcessor* inner_ptr = inner.get();
  CachingMultimodalProcessor processor(std::move(inner),
                                       /*max_cache_items=*/1);
  std::string prompt = "prompt";
  MMData data;
  std::vector<int32_t> token_ids;

  EXPECT_TRUE(processor.process_prompt(prompt, data, token_ids));
  EXPECT_EQ(inner_ptr->prompt_call_count_, 1);
}

TEST(CachingMultimodalProcessorTest, PrefersUuidOverRawDataForCacheKey) {
  auto inner = std::make_unique<FakeMultimodalProcessor>();
  FakeMultimodalProcessor* inner_ptr = inner.get();
  CachingMultimodalProcessor processor(std::move(inner),
                                       /*max_cache_items=*/1);

  MMData first_data;
  ASSERT_TRUE(processor.process_multimodal(
      make_input({make_uuid_input("image-a", "shared-uuid")}), first_data));
  ASSERT_EQ(inner_ptr->multimodal_call_count_, 1);
  EXPECT_EQ(first_data.items<MMItemVec>()[0].state().schedule_data().key,
            hash_string("shared-uuid"));

  MMData second_data;
  ASSERT_TRUE(processor.process_multimodal(
      make_input({make_uuid_input("different-data", "shared-uuid")}),
      second_data));

  EXPECT_EQ(inner_ptr->multimodal_call_count_, 1);
  EXPECT_EQ(item_value(second_data.items<MMItemVec>()[0]), 1);
  EXPECT_EQ(second_data.items<MMItemVec>()[0].state().schedule_data().key,
            hash_string("shared-uuid"));
}

}  // namespace
}  // namespace xllm
