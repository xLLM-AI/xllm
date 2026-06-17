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
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_input.h"
#include "processors/video_processor.h"

namespace xllm {

class Glm4VVideoProcessor final : public VideoProcessor {
 public:
  explicit Glm4VVideoProcessor(const ModelArgs& args);

  bool process(const torch::Tensor& origin_video,
               const VideoMetadata& metadata,
               MMDataItem& output_item) const override;

 private:
  bool process_video(const torch::Tensor& origin_video,
                     VideoMetadata& metadata,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw) const;

  torch::Tensor sample_frames(const VideoMetadata& metadata) const;

 private:
  bool do_convert_rgb_ = true;
  bool do_normalize_ = true;
  bool do_rescale_ = true;
  bool do_resize_ = true;

  torch::Tensor video_mean_;
  torch::Tensor video_std_;

  int32_t video_max_pixels_ = 47040000;
  int32_t video_min_pixels_ = 12544;

  int32_t video_merge_size_ = 2;
  int32_t video_patch_size_ = 14;

  int32_t resample_ = 3;
  double rescale_factor_ = 0.00392156862745098;

  int32_t video_temporal_patch_size_ = 2;
  bool do_sample_frame_ = true;
  int32_t min_frames_ = 4;
  int32_t max_frames_ = 768;
};

}  // namespace xllm
