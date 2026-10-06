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

#include "processors/qwen3_audio_processor.h"

#include <algorithm>
#include <optional>

#include "core/util/audio_utils.h"
#include "processors/qwen3_audio_common.h"

namespace xllm {

Qwen3AudioProcessor::Qwen3AudioProcessor(const ModelArgs& args)
    : feature_size_(args.mm_audio_feature_size()),
      sampling_rate_(args.mm_audio_sampling_rate()),
      hop_length_(args.mm_audio_hop_length()),
      chunk_length_(args.mm_audio_chunk_length()),
      n_fft_(args.mm_audio_n_fft()),
      window_length_(args.mm_audio_n_window() * 2),
      dither_(args.mm_audio_dither()),
      truncation_(args.mm_audio_truncation()),
      do_normalize_(args.mm_audio_do_normalize()) {
  CHECK_GT(feature_size_, 0);
  CHECK_GT(sampling_rate_, 0);
  CHECK_GT(hop_length_, 0);
  CHECK_GT(chunk_length_, 0);
  CHECK_GT(n_fft_, 0);
  CHECK_GT(window_length_, 0);
  CHECK_EQ(feature_size_, args.mm_audio_num_mel_bins());
  mel_filters_ =
      audio_utils::mel_filter_bank(1 + n_fft_ / 2,
                                   feature_size_,
                                   /*min_frequency=*/0.0,
                                   /*max_frequency=*/sampling_rate_ / 2.0,
                                   sampling_rate_,
                                   /*norm=*/"slaney",
                                   /*mel_scale=*/"slaney");
}

torch::Tensor Qwen3AudioProcessor::extract_log_mel_features(
    const torch::Tensor& waveform) const {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor audio = waveform.to(options);
  if (dither_ != 0.0) {
    audio = audio + torch::randn_like(audio) * dither_;
  }
  const torch::Tensor window = torch::hann_window(n_fft_, true, options);
  const torch::Tensor stft = torch::stft(audio,
                                         n_fft_,
                                         hop_length_,
                                         n_fft_,
                                         window,
                                         /*center=*/true,
                                         /*pad_mode=*/"reflect",
                                         /*normalized=*/false,
                                         /*onesided=*/std::nullopt,
                                         /*return_complex=*/true);
  const torch::Tensor magnitudes = stft.slice(-1, 0, -1).abs().pow(2);
  const torch::Tensor mel_spec = torch::matmul(mel_filters_.t(), magnitudes);
  torch::Tensor log_spec = torch::clamp(mel_spec, 1e-10).log10();
  const torch::Tensor max_value = log_spec.max();
  log_spec = torch::maximum(log_spec, max_value - 8.0);
  return (log_spec + 4.0) / 4.0;
}

bool Qwen3AudioProcessor::process(const torch::Tensor& origin_audio,
                                  const AudioMetadata& metadata,
                                  MMDataItem& output_item) const {
  if (origin_audio.dim() != 1) {
    LOG(ERROR) << "Qwen3 audio processor only supports mono audio, got shape "
               << origin_audio.sizes();
    return false;
  }
  if (metadata.sample_rate > 0 && metadata.sample_rate != sampling_rate_) {
    LOG(ERROR) << "Qwen3 audio processor expects " << sampling_rate_
               << " Hz audio, got " << metadata.sample_rate << " Hz.";
    return false;
  }

  torch::Tensor waveform = origin_audio.to(torch::kCPU, torch::kFloat32);
  const int64_t max_samples = chunk_length_ * sampling_rate_;
  if (truncation_ && waveform.size(0) > max_samples) {
    waveform = waveform.slice(0, 0, max_samples);
  }
  if (waveform.numel() == 0) {
    LOG(ERROR) << "Qwen3 audio processor received empty audio.";
    return false;
  }
  if (do_normalize_) {
    const torch::Tensor variance = waveform.var(false);
    waveform = (waveform - waveform.mean()) / torch::sqrt(variance + 1e-7);
  }

  torch::Tensor features = extract_log_mel_features(waveform).transpose(0, 1);
  const int64_t valid_frames = waveform.size(0) / hop_length_;
  if (valid_frames <= 0) {
    LOG(ERROR) << "Qwen3 audio is shorter than one feature frame.";
    return false;
  }
  features = features.slice(0, 0, std::min(valid_frames, features.size(0)))
                 .contiguous();

  const torch::Tensor feature_origin_lengths =
      torch::tensor({features.size(0)}, torch::dtype(torch::kLong));
  const torch::Tensor feature_lengths = qwen3_audio::get_feature_output_lengths(
      feature_origin_lengths, window_length_);
  output_item = MMDataItem(
      MMType::AUDIO,
      MMDict{{qwen3_audio::kInputFeaturesKey, features},
             {qwen3_audio::kFeatureLengthKey, feature_lengths},
             {qwen3_audio::kFeatureOriginLengthsKey, feature_origin_lengths}},
      metadata);
  return true;
}

}  // namespace xllm
