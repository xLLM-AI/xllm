/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cstdint>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "processors/audio_processor.h"

namespace xllm {

// Kaldi-style fbank + CMVN frontend; numerically equivalent to
// kaldi-native-fbank's FbankComputer as used by the vLLM FireRedASR2
// extractor.
class JoyaiASRAudioProcessor final : public AudioProcessor {
 public:
  explicit JoyaiASRAudioProcessor(const ModelArgs& args);

  bool process(const torch::Tensor& origin_audio,
               const AudioMetadata& metadata,
               MMDataItem& output_item) const override;

 private:
  int64_t num_mel_bins_ = 80;
  int64_t frame_length_ = 400;   // samples (25ms @16k)
  int64_t frame_shift_ = 160;    // samples (10ms @16k)
  int64_t padded_window_ = 512;  // frame_length rounded up to power of two
  int64_t max_frames_ = 3000;    // 30s @10ms shift
  int64_t downsample_rate_ = 2;
  // Below this the projector yields zero embed rows (derivation in .cpp).
  int64_t min_frames_ = 5;
  int64_t sampling_rate_ = 16000;  // assigned from mm_audio_sampling_rate

  torch::Tensor window_;          // [frame_length_] povey window
  torch::Tensor mel_filterbank_;  // [num_mel_bins_, padded_window_/2]
  torch::Tensor cmvn_means_;      // [num_mel_bins_] or undefined
  torch::Tensor cmvn_inv_std_;    // [num_mel_bins_] or undefined
};

}  // namespace xllm
