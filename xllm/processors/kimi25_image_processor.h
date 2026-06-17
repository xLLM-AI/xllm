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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/image_processor.h"
#include "processors/prompt_processor.h"
#include "processors/video_processor.h"

namespace xllm {

struct VideoChunkMetadata {
  int32_t chunk_id;
  double start_timestamp;
  int32_t num_frames;
  std::vector<int64_t> frame_indices;
  std::string timestamp_text;
};

struct NavitResizeResult {
  int32_t num_tokens = 0;
  int32_t new_width = 0;
  int32_t new_height = 0;
  int32_t pad_width = 0;
  int32_t pad_height = 0;
  int32_t sampled_nframes = 1;
};

struct KimiK25MediaConfig {
  int32_t patch_size = 14;
  int32_t merge_kernel_size = 2;
  int32_t temporal_merge_kernel_size = 4;

  int32_t in_patch_limit = 16384;
  int32_t in_patch_limit_each_frame = -1;
  int32_t patch_limit_on_one_side = 512;
  int32_t in_patch_limit_video = -1;
  int32_t fixed_output_tokens = -1;

  double sample_fps = 2.0;
  int32_t max_num_frames_each_video = 128;
  int32_t min_frames = 4;
  std::string timestamp_mode = "hh:mm:ss.fff";

  std::vector<double> image_mean = {0.5, 0.5, 0.5};
  std::vector<double> image_std = {0.5, 0.5, 0.5};

  bool do_convert_rgb = true;
  bool do_normalize = true;
  bool do_resize = true;
  int32_t resample = 3;
};

class KimiK25ImageProcessor final : public ImageProcessor {
 public:
  explicit KimiK25ImageProcessor(const ModelArgs& args);
  ~KimiK25ImageProcessor() override = default;

  bool process(const std::vector<torch::Tensor>& images,
               std::vector<MMDataItem>& output_items) const override;

 private:
  bool process_image(const std::vector<torch::Tensor>& images,
                     std::vector<torch::Tensor>& pixel_values,
                     std::vector<torch::Tensor>& thw) const;

  bool process_image(torch::Tensor image,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw) const;

 private:
  KimiK25MediaConfig config_;
};

class KimiK25VideoProcessor final : public VideoProcessor {
 public:
  explicit KimiK25VideoProcessor(const ModelArgs& args);
  ~KimiK25VideoProcessor() override = default;

  bool process(const torch::Tensor& origin_video,
               const VideoMetadata& metadata,
               MMDataItem& output_item) const override;

 private:
  std::vector<VideoChunkMetadata> split_video_chunks(
      const VideoMetadata& video_meta) const;
  bool process_video_chunk(torch::Tensor video_chunk,
                           const VideoMetadata& video_meta,
                           torch::Tensor& pixel_values,
                           torch::Tensor& thw) const;

  torch::Tensor sample_frames(const VideoMetadata& metadata) const;
  std::string timestamp_as_str(double timestamp,
                               const std::string& timestamp_mode) const;

 private:
  KimiK25MediaConfig config_;
};

class KimiK25PromptProcessor final : public PromptProcessor {
  enum class TokenType {
    INVALID,
    IMAGE,
    VIDEO,
  };

 public:
  explicit KimiK25PromptProcessor(const ModelArgs& args);
  ~KimiK25PromptProcessor() override = default;

  void process(std::string& prompt, const MMData& mm_data) override;
  void find_mm_spans(const std::vector<int32_t>& token_ids,
                     MMData& mm_data) override;

 private:
  std::vector<int32_t> get_media_token_counts(
      const torch::Tensor& grid_thw) const;
  std::pair<TokenType, size_t> find_media_prompt(const std::string& prompt,
                                                 size_t begin) const;

 private:
  const std::string media_pad_token_ = "<|media_pad|>";
  const std::string media_prompt_suffix_ = "<|media_end|>";
  const std::string image_prompt_prefix_ =
      "<|media_begin|>image<|media_content|>";
  const std::string video_prompt_prefix_ =
      "<|media_begin|>video<|media_content|>";
  const std::string image_prompt_ =
      "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>";
  const std::string video_prompt_ =
      "<|media_begin|>video<|media_content|><|media_pad|><|media_end|>";
  int32_t vision_start_token_id_ = 0;
  int32_t vision_token_id_ = 0;
  int32_t vision_end_token_id_ = 0;
  int32_t merge_size_ = 0;
};

}  // namespace xllm
