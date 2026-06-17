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

#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/common/message.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "processors/audio_processor.h"
#include "processors/image_processor.h"
#include "processors/processor_util.h"
#include "processors/prompt_processor.h"
#include "processors/video_processor.h"

namespace xllm {

class ModelArgs;
class Tokenizer;

class MultimodalProcessorBase {
 public:
  virtual ~MultimodalProcessorBase();

  virtual bool process_prompt(std::string& prompt,
                              MMData& mm_data,
                              std::vector<int32_t>& token_ids) = 0;

  virtual bool process_multimodal(const MMInput& inputs,
                                  MMData& data) const = 0;

 protected:
  explicit MultimodalProcessorBase(std::shared_ptr<Tokenizer> tokenizer);

  bool tokenize(const std::string& prompt,
                std::vector<int32_t>& token_ids) const;

  void hash_mm_items(const MMInput& mm_input, MMData& mm_data) const;

 private:
  std::shared_ptr<Tokenizer> tokenizer_;
};

std::unique_ptr<MultimodalProcessorBase> create_multimodal_processor(
    const ModelArgs& model_args,
    std::shared_ptr<Tokenizer> tokenizer);

template <typename PromptProcessor,
          typename ImageProcessor = ImageNoneProcessor,
          typename VideoProcessor = VideoNoneProcessor,
          typename AudioProcessor = AudioNoneProcessor>
class MultimodalProcessor final : public MultimodalProcessorBase {
 public:
  MultimodalProcessor(const ModelArgs& model_args,
                      std::shared_ptr<Tokenizer> tokenizer)
      : MultimodalProcessorBase(std::move(tokenizer)),
        image_processor_(std::make_unique<ImageProcessor>(model_args)),
        video_processor_(std::make_unique<VideoProcessor>(model_args)),
        audio_processor_(std::make_unique<AudioProcessor>(model_args)),
        prompt_processor_(std::make_unique<PromptProcessor>(model_args)) {}

  ~MultimodalProcessor() override = default;

  bool process_prompt(std::string& prompt,
                      MMData& mm_data,
                      std::vector<int32_t>& token_ids) override {
    prompt_processor_->process(prompt, mm_data);
    if (!tokenize(prompt, token_ids)) {
      return false;
    }
    prompt_processor_->find_mm_spans(token_ids, mm_data);
    return true;
  }

  bool process_multimodal(const MMInput& inputs, MMData& data) const override {
    MMInputGatherVisitor gather;
    inputs.foreach (gather);

    std::vector<MMDataItem> image_items;
    if (!gather.images_.empty() &&
        !process_images(gather.images_, image_items)) {
      LOG(ERROR) << "Process image failed.";
      return false;
    }

    std::vector<MMDataItem> video_items;
    if (!gather.videos_.empty() &&
        !process_videos(gather.videos_, gather.video_metadata_, video_items)) {
      LOG(ERROR) << "Process video failed.";
      return false;
    }

    std::vector<MMDataItem> audio_items;
    if (!gather.audios_.empty() &&
        !process_audios(gather.audios_, gather.audio_metadata_, audio_items)) {
      LOG(ERROR) << "Process audio failed.";
      return false;
    }

    MMItemVec output_items =
        gather.finish(image_items, video_items, audio_items);
    data.set(gather.data_type_, std::move(output_items));

    hash_mm_items(inputs, data);
    return true;
  }

 private:
  bool process_images(const std::vector<torch::Tensor>& images,
                      std::vector<MMDataItem>& output_items) const {
    output_items.clear();
    output_items.assign(images.size(), MMDataItem(MMType::NONE));
    const auto image_buckets = group_images_by_shape(images);
    for (const auto& image_bucket : image_buckets) {
      const ImageBatchBucket& bucket = image_bucket.second;
      std::vector<MMDataItem> batch_items;
      if (!image_processor_->process(bucket.images, batch_items)) {
        LOG(ERROR)
            << "Failed to process image batch. The shape(channels, height, "
               "width) is: "
            << bucket.images[0].sizes();
        return false;
      }
      CHECK(batch_items.size() == bucket.indices.size())
          << "image processor returned mismatched batch result count.";
      const size_t bucket_size = bucket.indices.size();
      for (size_t index = 0; index < bucket_size; ++index) {
        const size_t output_index = bucket.indices[index];
        output_items[output_index] = std::move(batch_items[index]);
      }
    }
    return true;
  }

  bool process_videos(const std::vector<torch::Tensor>& videos,
                      const std::vector<VideoMetadata>& video_metadata,
                      std::vector<MMDataItem>& output_items) const {
    CHECK(videos.size() == video_metadata.size());
    output_items.clear();
    output_items.reserve(videos.size());
    const size_t video_size = videos.size();
    for (size_t index = 0; index < video_size; ++index) {
      MMDataItem item(MMType::VIDEO);
      if (!video_processor_->process(
              videos[index], video_metadata[index], item)) {
        LOG(ERROR) << "Failed to process video. The shape(num_frames, "
                      "channels, height, width) is: "
                   << videos[index].sizes();
        return false;
      }
      output_items.push_back(std::move(item));
    }
    return true;
  }

  bool process_audios(const std::vector<torch::Tensor>& audios,
                      const std::vector<AudioMetadata>& audio_metadata,
                      std::vector<MMDataItem>& output_items) const {
    CHECK(audios.size() == audio_metadata.size());
    output_items.clear();
    output_items.reserve(audios.size());
    const size_t audio_size = audios.size();
    for (size_t index = 0; index < audio_size; ++index) {
      MMDataItem item(MMType::AUDIO);
      if (!audio_processor_->process(
              audios[index], audio_metadata[index], item)) {
        return false;
      }
      output_items.push_back(std::move(item));
    }
    return true;
  }

  std::unique_ptr<ImageProcessor> image_processor_;
  std::unique_ptr<VideoProcessor> video_processor_;
  std::unique_ptr<AudioProcessor> audio_processor_;
  std::unique_ptr<PromptProcessor> prompt_processor_;
};

}  // namespace xllm
