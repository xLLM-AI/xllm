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
#include <torch/torch.h>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"

namespace xllm {

class VideoProcessor {
 public:
  virtual ~VideoProcessor() = default;

  virtual bool process(const torch::Tensor& origin_video,
                       const VideoMetadata& metadata,
                       MMDataItem& output_item) const = 0;
};

class VideoNoneProcessor final : public VideoProcessor {
 public:
  VideoNoneProcessor() = default;
  explicit VideoNoneProcessor(const ModelArgs&) {}

  bool process(const torch::Tensor& origin_video,
               const VideoMetadata& metadata,
               MMDataItem& output_item) const override {
    LOG(ERROR) << "Video processor is not configured.";
    return false;
  }
};

}  // namespace xllm
