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

#include "core/common/macros.h"
#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"

namespace xllm {

class AudioProcessor {
 public:
  virtual ~AudioProcessor() = default;

  virtual bool process(const torch::Tensor& origin_audio,
                       const AudioMetadata& metadata,
                       MMDataItem& output_item) const = 0;
};

class AudioNoneProcessor final : public AudioProcessor {
 public:
  AudioNoneProcessor() = default;
  explicit AudioNoneProcessor(const ModelArgs&) {};
  bool process(const torch::Tensor& origin_audio,
               const AudioMetadata& metadata,
               MMDataItem& output_item) const override {
    LOG(ERROR) << "Audio processor is not configured.";
    return false;
  }
};

}  // namespace xllm
