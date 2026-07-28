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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <utility>
#include <vector>

#include "processors/caching_multimodal_processor.h"
#include "util/hash_util.h"

namespace xllm {
namespace {

class FakeMultimodalProcessor final : public MultimodalProcessorBase {
 public:
  FakeMultimodalProcessor() : MultimodalProcessorBase(/*tokenizer=*/nullptr) {}

  bool process_prompt(std::string& /*prompt*/,
                      MMData& /*mm_data*/,
                      std::vector<int32_t>& /*token_ids*/) override {
    return true;
  }

  bool process_multimodal(const MMInput& inputs, MMData& data) const override {
    ++call_count_;
    std::vector<std::string> raw_items;
    raw_items.reserve(inputs.size());
    MMItemVec output_items;
    output_items.reserve(inputs.size());
    for (const MMInputItem& input : inputs.items()) {
      raw_items.emplace_back(input.raw_data);
      MMDict item_data;
      item_data["value"] = torch::tensor({next_value_++}, torch::kInt64);
      MMDataItem item(MMType::IMAGE, item_data);
      item.mutable_state().mutable_token_pos().offset = 99;
      output_items.emplace_back(std::move(item));
    }
    processed_raw_items_.emplace_back(std::move(raw_items));
    data.set(MMType::IMAGE, std::move(output_items));
    hash_mm_items(inputs, data);
    return true;
  }

  mutable int32_t call_count_ = 0;
  mutable int64_t next_value_ = 1;
  mutable std::vector<std::vector<std::string>> processed_raw_items_;
};

std::string make_png(uint8_t red, uint8_t green, uint8_t blue) {
  cv::Mat image(1, 1, CV_8UC3, cv::Scalar(blue, green, red));
  std::vector<uint8_t> encoded;
  CHECK(cv::imencode(".png", image, encoded));
  return std::string(reinterpret_cast<const char*>(encoded.data()),
                     encoded.size());
}

MMContent make_binary_image(size_t size, std::string uuid = "") {
  ImageURL image_url;
  image_url.url = "data:image/png;binary," + std::to_string(size);
  MMContent content("image_url", image_url);
  content.uuid = std::move(uuid);
  return content;
}

MMContent make_image_url(std::string url, std::string uuid = "") {
  ImageURL image_url;
  image_url.url = std::move(url);
  MMContent content("image_url", image_url);
  content.uuid = std::move(uuid);
  return content;
}

std::vector<Message> make_messages(MMContentVec contents) {
  std::vector<Message> messages;
  messages.emplace_back("user", std::move(contents));
  return messages;
}

int64_t item_value(const MMDataItem& item) {
  return item.get<torch::Tensor>("value")->item<int64_t>();
}

TEST(CachingMultimodalProcessorRequestTest, RecombinesUuidHitAndUuidMiss) {
  auto inner = std::make_unique<FakeMultimodalProcessor>();
  FakeMultimodalProcessor* inner_ptr = inner.get();
  CachingMultimodalProcessor processor(std::move(inner),
                                       /*max_cache_items=*/4);
  const std::string first_image =
      make_png(/*red=*/255, /*green=*/0, /*blue=*/0);
  const std::string second_image =
      make_png(/*red=*/0, /*green=*/255, /*blue=*/0);

  MMData warm_data;
  EXPECT_EQ(
      processor.process_multimodal_request(
          make_messages({make_binary_image(first_image.size(), "image-a")}),
          first_image,
          warm_data),
      MMErrCode::SUCCESS);
  ASSERT_EQ(inner_ptr->call_count_, 1);

  // Both contents carry a uuid, so the request stays on the uuid path. The
  // first hits the warmed cache (loaded lazily, not downloaded); the second
  // misses and is loaded from its binary payload.
  MMData mixed_data;
  EXPECT_EQ(
      processor.process_multimodal_request(
          make_messages({make_image_url("invalid://cached-image", "image-a"),
                         make_binary_image(second_image.size(), "image-b")}),
          second_image,
          mixed_data),
      MMErrCode::SUCCESS);

  ASSERT_EQ(inner_ptr->call_count_, 2);
  ASSERT_EQ(inner_ptr->processed_raw_items_.size(), 2);
  EXPECT_EQ(inner_ptr->processed_raw_items_[1],
            (std::vector<std::string>{second_image}));
  const MMItemVec& items = mixed_data.items<MMItemVec>();
  ASSERT_EQ(items.size(), 2);
  EXPECT_EQ(item_value(items[0]), 1);
  EXPECT_EQ(item_value(items[1]), 2);
  EXPECT_EQ(items[0].state().token_pos().offset, 0);
  EXPECT_EQ(items[0].state().schedule_data().key, hash_string("image-a"));
  EXPECT_EQ(items[1].state().schedule_data().key, hash_string("image-b"));
}

TEST(CachingMultimodalProcessorRequestTest, FallsBackToContentPathWithoutUuid) {
  auto inner = std::make_unique<FakeMultimodalProcessor>();
  FakeMultimodalProcessor* inner_ptr = inner.get();
  CachingMultimodalProcessor processor(std::move(inner),
                                       /*max_cache_items=*/4);
  const std::string first_image =
      make_png(/*red=*/255, /*green=*/0, /*blue=*/0);
  const std::string second_image =
      make_png(/*red=*/0, /*green=*/255, /*blue=*/0);

  // One content lacks a uuid, so the whole request takes the content-keyed
  // path: every item is loaded and processed, and cache keys fall back to the
  // raw data hash.
  MMData data;
  EXPECT_EQ(processor.process_multimodal_request(
                make_messages({make_binary_image(first_image.size(), "image-a"),
                               make_binary_image(second_image.size())}),
                first_image + second_image,
                data),
            MMErrCode::SUCCESS);

  ASSERT_EQ(inner_ptr->call_count_, 1);
  ASSERT_EQ(inner_ptr->processed_raw_items_.size(), 1);
  EXPECT_EQ(inner_ptr->processed_raw_items_[0],
            (std::vector<std::string>{first_image, second_image}));
  const MMItemVec& items = data.items<MMItemVec>();
  ASSERT_EQ(items.size(), 2);
  EXPECT_EQ(items[0].state().schedule_data().key, hash_string("image-a"));
  EXPECT_EQ(items[1].state().schedule_data().key, hash_string(second_image));
}

TEST(CachingMultimodalProcessorRequestTest, ConsumesPayloadForSkippedUuidHit) {
  auto inner = std::make_unique<FakeMultimodalProcessor>();
  FakeMultimodalProcessor* inner_ptr = inner.get();
  CachingMultimodalProcessor processor(std::move(inner),
                                       /*max_cache_items=*/4);
  const std::string first_image =
      make_png(/*red=*/255, /*green=*/0, /*blue=*/0);
  const std::string second_image =
      make_png(/*red=*/0, /*green=*/0, /*blue=*/255);

  MMData warm_data;
  ASSERT_EQ(
      processor.process_multimodal_request(
          make_messages({make_binary_image(first_image.size(), "image-a")}),
          first_image,
          warm_data),
      MMErrCode::SUCCESS);

  // Both contents carry a uuid (uuid path). The first hits the cache and is
  // skipped, but its binary bytes must still be consumed from the shared
  // payload so the second (miss) reads from the correct offset.
  MMData mixed_data;
  ASSERT_EQ(
      processor.process_multimodal_request(
          make_messages({make_binary_image(first_image.size(), "image-a"),
                         make_binary_image(second_image.size(), "image-b")}),
          first_image + second_image,
          mixed_data),
      MMErrCode::SUCCESS);

  ASSERT_EQ(inner_ptr->processed_raw_items_.size(), 2);
  EXPECT_EQ(inner_ptr->processed_raw_items_[1],
            (std::vector<std::string>{second_image}));
}

TEST(CachingMultimodalProcessorRequestTest, UuidMissUsesUrlLoadingPath) {
  auto inner = std::make_unique<FakeMultimodalProcessor>();
  FakeMultimodalProcessor* inner_ptr = inner.get();
  CachingMultimodalProcessor processor(std::move(inner),
                                       /*max_cache_items=*/4);

  MMData data;
  MMErrCode code = processor.process_multimodal_request(
      make_messages({make_image_url(
          "file:///xllm/nonexistent/uuid-cache-test.png", "missing-image")}),
      "",
      data);

  EXPECT_EQ(code, MMErrCode::INVALID_URL_ERR);
  EXPECT_EQ(inner_ptr->call_count_, 0);
}

}  // namespace
}  // namespace xllm
