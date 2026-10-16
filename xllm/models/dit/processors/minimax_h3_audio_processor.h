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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/framework/model_context/model_context.h"
#include "models/dit/utils/util.h"

namespace xllm {

class MiniMaxH3AudioProcessorImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioProcessorImpl(const ModelContext& context,
                              int64_t channel_count)
      : sampling_rate_(context.get_model_args().sampling_rate()),
        channel_count_(channel_count) {
    CHECK_GT(sampling_rate_, 0);
    CHECK_GT(channel_count_, 0);
    const std::vector<int64_t>& encoder_rates =
        context.get_model_args().encoder_rates();
    CHECK(!encoder_rates.empty())
        << "MiniMax-H3 audio VAE encoder_rates must not be empty";
    hop_length_ = dit::product_checked(encoder_rates);
    CHECK_EQ(sampling_rate_ % hop_length_, 0)
        << "MiniMax-H3 audio sampling rate must be divisible by hop length";
    latents_per_second_ = sampling_rate_ / hop_length_;
  }

  int64_t sampling_rate() const { return sampling_rate_; }

  int64_t audio_latents_for_video(int64_t num_frames, double fps) const {
    return round_frames_to_rate(num_frames, fps, latents_per_second_);
  }

  int64_t max_reference_samples(int64_t num_frames, double fps) const {
    CHECK_GT(num_frames, 0);
    CHECK_GT(fps, 0.0);
    return static_cast<int64_t>(static_cast<double>(num_frames) / fps *
                                static_cast<double>(sampling_rate_));
  }

  // Rounded sample count matching a generated video's duration, used to trim
  // the decoded audio to the video's exact length.
  int64_t samples_for_video(int64_t num_frames, double fps) const {
    return round_frames_to_rate(num_frames, fps, sampling_rate_);
  }

  int64_t reference_audio_latents(int64_t sample_count) const {
    CHECK_GE(sample_count, 0);
    return (sample_count + hop_length_ - 1) / hop_length_;
  }

  torch::Tensor trim_reference_audio(torch::Tensor waveform,
                                     int64_t max_samples) const {
    CHECK_EQ(waveform.dim(), 2) << "MiniMax-H3 reference audio must be [C,S]";
    CHECK_EQ(waveform.size(0), channel_count_);
    CHECK_GE(max_samples, 0);
    return waveform.slice(1, 0, std::min(waveform.size(1), max_samples))
        .contiguous();
  }

 private:
  // Rounded frames/fps * rate conversion shared by the latents-per-second and
  // samples-per-second callers; max_reference_samples intentionally truncates.
  static int64_t round_frames_to_rate(int64_t num_frames,
                                      double fps,
                                      int64_t rate) {
    CHECK_GT(num_frames, 0);
    CHECK_GT(fps, 0.0);
    return static_cast<int64_t>(std::llround(static_cast<double>(num_frames) /
                                             fps * static_cast<double>(rate)));
  }
  int64_t sampling_rate_;
  int64_t latents_per_second_ = 0;
  int64_t channel_count_;
  int64_t hop_length_ = 0;
};
TORCH_MODULE(MiniMaxH3AudioProcessor);

}  // namespace xllm
