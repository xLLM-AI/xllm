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

#include "processors/joyai_asr_audio_processor.h"

#include <glog/logging.h>

#include <bit>
#include <cmath>
#include <limits>
#include <vector>

namespace xllm {
namespace {

float mel_scale(float freq) { return 1127.0f * std::log(1.0f + freq / 700.0f); }

constexpr int64_t kConformerSubsampling = 4;

}  // namespace

JoyaiASRAudioProcessor::JoyaiASRAudioProcessor(const ModelArgs& args) {
  sampling_rate_ = args.mm_audio_sampling_rate();
  CHECK_EQ(sampling_rate_, 16000)
      << "JoyaiASRAudioProcessor requires sampling_rate=16000 (preprocessor "
         "config declares "
      << sampling_rate_ << "); resampling is not implemented";
  num_mel_bins_ = args.mm_audio_num_mel_bins();
  frame_length_ = args.mm_audio_frame_length();
  frame_shift_ = args.mm_audio_frame_shift();
  max_frames_ = args.mm_audio_max_frames();
  downsample_rate_ = args.mm_audio_downsample_rate();
  CHECK_GT(num_mel_bins_, 0);
  CHECK_LT(num_mel_bins_, std::numeric_limits<int64_t>::max() - 1);
  CHECK_GT(frame_length_, 1);
  CHECK_LE(frame_length_, std::numeric_limits<int64_t>::max() / 2 + 1);
  CHECK_GT(frame_shift_, 0);
  CHECK_GT(downsample_rate_, 0);
  CHECK_LE(downsample_rate_,
           std::numeric_limits<int64_t>::max() / kConformerSubsampling);
  padded_window_ =
      static_cast<int64_t>(std::bit_ceil(static_cast<uint64_t>(frame_length_)));
  // ceil(frames / 4) >= downsample_rate_ <=> frames >= 4 * ds - 3.
  min_frames_ = kConformerSubsampling * downsample_rate_ - 3;
  CHECK_GE(max_frames_, min_frames_);
  CHECK_LE(max_frames_,
           std::numeric_limits<int64_t>::max() - kConformerSubsampling + 1);
  CHECK_LE(
      max_frames_ - 1,
      (std::numeric_limits<int64_t>::max() - frame_length_) / frame_shift_);

  CHECK_EQ(args.mm_audio_cmvn_means().empty(),
           args.mm_audio_cmvn_inverse_std().empty())
      << "preprocessor_config means and inverse_std_variences must be "
         "provided together";

  // Povey window: pow(0.5 - 0.5 * cos(2 * pi * i / (L - 1)), 0.85).
  window_ = torch::zeros({frame_length_}, torch::kFloat32);
  float* window_data = window_.data_ptr<float>();
  const double a = 2.0 * M_PI / static_cast<double>(frame_length_ - 1);
  for (int64_t i = 0; i < frame_length_; ++i) {
    const double val = 0.5 - 0.5 * std::cos(a * static_cast<double>(i));
    window_data[i] = static_cast<float>(std::pow(val, 0.85));
  }

  // Kaldi mel filterbank: mel-evenly-spaced triangular bands, 20Hz..nyquist.
  const int64_t num_fft_bins = padded_window_ / 2;  // 256 for a 512 window
  const float nyquist = 0.5f * static_cast<float>(sampling_rate_);
  const float low_freq = 20.0f;
  const float high_freq = nyquist;  // high_freq == 0 -> nyquist
  const float mel_low = mel_scale(low_freq);
  const float mel_high = mel_scale(high_freq);
  const float mel_freq_delta =
      (mel_high - mel_low) / static_cast<float>(num_mel_bins_ + 1);
  const float fft_bin_width =
      static_cast<float>(sampling_rate_) / static_cast<float>(padded_window_);

  mel_filterbank_ =
      torch::zeros({num_mel_bins_, num_fft_bins}, torch::kFloat32);
  float* mel_data = mel_filterbank_.data_ptr<float>();
  for (int64_t bin = 0; bin < num_mel_bins_; ++bin) {
    const float left_mel = mel_low + static_cast<float>(bin) * mel_freq_delta;
    const float center_mel =
        mel_low + static_cast<float>(bin + 1) * mel_freq_delta;
    const float right_mel =
        mel_low + static_cast<float>(bin + 2) * mel_freq_delta;
    for (int64_t i = 0; i < num_fft_bins; ++i) {
      const float mel = mel_scale(fft_bin_width * static_cast<float>(i));
      if (mel > left_mel && mel < right_mel) {
        float weight = 0.0f;
        if (mel <= center_mel) {
          weight = (mel - left_mel) / (center_mel - left_mel);
        } else {
          weight = (right_mel - mel) / (right_mel - center_mel);
        }
        mel_data[bin * num_fft_bins + i] = weight;
      }
    }
  }

  const auto to_mel_tensor = [this](const std::vector<double>& values,
                                    const char* what) {
    torch::Tensor tensor =
        torch::from_blob(const_cast<double*>(values.data()),
                         {static_cast<int64_t>(values.size())},
                         torch::kFloat64)
            .to(torch::kFloat32);
    CHECK_EQ(tensor.numel(), num_mel_bins_)
        << "preprocessor_config " << what << " dim != num_mel_bins";
    return tensor;
  };
  if (!args.mm_audio_cmvn_means().empty()) {
    cmvn_means_ = to_mel_tensor(args.mm_audio_cmvn_means(), "means");
    cmvn_inv_std_ = to_mel_tensor(args.mm_audio_cmvn_inverse_std(),
                                  "inverse_std_variences");
  }

  LOG(INFO) << "JoyaiASRAudioProcessor: mel_bins=" << num_mel_bins_
            << " frame_length=" << frame_length_
            << " frame_shift=" << frame_shift_ << " max_frames=" << max_frames_
            << " cmvn=" << (cmvn_means_.defined() ? "on" : "off");
}

bool JoyaiASRAudioProcessor::process(const torch::Tensor& origin_audio,
                                     const AudioMetadata& metadata,
                                     MMDataItem& output_item) const {
  if (!origin_audio.defined() || origin_audio.numel() == 0) {
    LOG(ERROR) << "JoyaiASRAudioProcessor: empty audio input.";
    return false;
  }

  // 16k tripwire: the decoder normally resamples to 16k, so this fires
  // only if its target changed or a path bypassed it.
  if (metadata.sample_rate > 0 && metadata.sample_rate != sampling_rate_) {
    LOG(ERROR) << "JoyaiASRAudioProcessor: audio sample rate "
               << metadata.sample_rate << " != " << sampling_rate_
               << "; resampling is not supported.";
    return false;
  }

  // kaldiio-scale the [-1, 1] float waveform (matches the vLLM extractor).
  torch::Tensor wave = origin_audio.to(torch::kFloat32) * 32768.0f;

  const int64_t num_samples = wave.numel();
  int64_t frames = num_samples < frame_length_
                       ? 0
                       : 1 + (num_samples - frame_length_) / frame_shift_;
  if (frames < min_frames_) {
    LOG(ERROR) << "JoyaiASRAudioProcessor: audio too short (" << frames
               << " frames, " << (frames * frame_shift_ * 1000 / sampling_rate_)
               << "ms; minimum " << min_frames_ << " frames for the "
               << "Conformer subsampling + projector to yield an embedding).";
    return false;
  }
  if (frames > max_frames_) {
    LOG(WARNING) << "JoyaiASRAudioProcessor: audio truncated from " << frames
                 << " to " << max_frames_ << " frames ("
                 << (max_frames_ * frame_shift_ / sampling_rate_) << "s).";
    frames = max_frames_;
  }

  const int64_t kept_samples = (frames - 1) * frame_shift_ + frame_length_;
  torch::Tensor wave_flat = wave.flatten().narrow(0, 0, kept_samples);
  auto frames_tensor = wave_flat.unfold(0, frame_length_, frame_shift_);

  frames_tensor = frames_tensor - frames_tensor.mean(
                                      /*dim=*/1, /*keepdim=*/true);

  // Preemphasis (backward, coeff 0.97): d[i] -= 0.97 * d[i-1], d[0] *= 0.03.
  torch::Tensor preemph = frames_tensor.clone();
  preemph.narrow(/*dim=*/1, /*start=*/1, /*length=*/frame_length_ - 1)
      .sub_(frames_tensor.narrow(/*dim=*/1,
                                 /*start=*/0,
                                 /*length=*/frame_length_ - 1),
            /*alpha=*/0.97f);
  preemph.select(/*dim=*/1, /*index=*/0).mul_(1.0f - 0.97f);

  // Window, zero-pad, and take the power spectrum |rfft|^2.
  preemph.mul_(window_);
  torch::Tensor spectrum = torch::fft::rfft(preemph, padded_window_);
  torch::Tensor power =
      (torch::real(spectrum).square() + torch::imag(spectrum).square())
          .narrow(/*dim=*/1, /*start=*/0, /*length=*/padded_window_ / 2);

  // Mel energies, floored at FLT_EPSILON then logged (kaldi semantics).
  torch::Tensor mel = torch::matmul(power, mel_filterbank_.t());
  mel.clamp_min_(std::numeric_limits<float>::epsilon());
  mel.log_();

  // CMVN: (mel - means) * inverse_std, in place (mel is owned here).
  if (cmvn_means_.defined()) {
    mel.sub_(cmvn_means_).mul_(cmvn_inv_std_);
  }

  // ceil(frames / 4) (vLLM fake_token_lengths) then // downsample_rate_;
  // equals the python encode() row count.
  const int64_t subsampled =
      (frames + kConformerSubsampling - 1) / kConformerSubsampling;
  const int64_t token_num = subsampled / downsample_rate_;

  output_item.add("input_features", mel);
  output_item.add("speech_lengths",
                  torch::tensor({static_cast<int64_t>(frames)}, torch::kInt64));
  output_item.add("audio_token_num", torch::tensor({token_num}, torch::kInt64));
  return true;
}

}  // namespace xllm
