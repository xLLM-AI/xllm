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

#include "core/framework/model/model_args.h"
#include "processors/audio_processor.h"

namespace xllm {

class Qwen3AudioProcessor final : public AudioProcessor {
 public:
  explicit Qwen3AudioProcessor(const ModelArgs& args);

  bool process(const torch::Tensor& origin_audio,
               const AudioMetadata& metadata,
               MMDataItem& output_item) const override;

 private:
  torch::Tensor extract_log_mel_features(const torch::Tensor& waveform) const;

  int64_t feature_size_ = 0;
  int64_t sampling_rate_ = 0;
  int64_t hop_length_ = 0;
  int64_t chunk_length_ = 0;
  int64_t n_fft_ = 0;
  int64_t window_length_ = 0;
  double dither_ = 0.0;
  bool truncation_ = false;
  bool do_normalize_ = false;
  torch::Tensor mel_filters_;
};

}  // namespace xllm
