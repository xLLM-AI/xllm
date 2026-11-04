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

#include "core/framework/multimodal/dit_media_encoder.h"

#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <utility>

#include "core/framework/multimodal/mm_codec.h"

namespace xllm {

Status encode_minimax_h3_video_output(const DiTForwardOutput& output,
                                      double fps,
                                      int32_t sample_rate,
                                      DiTEncodedMedia& encoded) {
  if (output.tensors.size() != 1 || output.audio_tensors.size() != 1 ||
      !output.text_output.empty() || !output.encoded_media.empty() ||
      !std::isfinite(fps) || fps <= 0 || sample_rate <= 0) {
    return {StatusCode::UNKNOWN,
            "MiniMax-H3 encoding requires one video/audio pair and positive "
            "fps/sample rate"};
  }
  const torch::Tensor& video = output.tensors.front();
  if (!video.defined() || video.dim() != 5 || video.size(0) != 1 ||
      video.size(1) <= 0 || video.size(2) != 3 || video.size(3) <= 0 ||
      video.size(4) <= 0 ||
      video.size(1) > std::numeric_limits<int32_t>::max() ||
      video.size(3) > std::numeric_limits<int32_t>::max() ||
      video.size(4) > std::numeric_limits<int32_t>::max()) {
    return {StatusCode::UNKNOWN,
            "MiniMax-H3 video must have shape [1,T,3,H,W]"};
  }
  torch::Tensor audio = output.audio_tensors.front();
  if (!audio.defined()) {
    return {StatusCode::UNKNOWN, "MiniMax-H3 audio is undefined"};
  }
  if (audio.dim() == 3 && audio.size(1) == 1) {
    audio = audio.squeeze(1);
  }
  if (audio.dim() == 3 && audio.size(0) == 1) {
    audio = audio.squeeze(0);
  }
  if (audio.dim() != 2 || audio.size(0) <= 0 || audio.size(1) <= 0 ||
      audio.size(0) > std::numeric_limits<int32_t>::max()) {
    return {StatusCode::UNKNOWN,
            "MiniMax-H3 audio must have shape [channels,samples]"};
  }
  DiTEncodedMedia result;
  result.mime_type = "video/mp4";
  result.container = "mp4";
  result.width = static_cast<int32_t>(video.size(4));
  result.height = static_cast<int32_t>(video.size(3));
  result.num_frames = static_cast<int32_t>(video.size(1));
  result.fps = fps;
  result.audio_sample_rate = sample_rate;
  result.audio_channels = static_cast<int32_t>(audio.size(0));
  FFmpegVideoEncoder encoder;
  const torch::Tensor cpu_video =
      video.squeeze(0).to(torch::kCPU, torch::kFloat32).contiguous();
  const torch::Tensor cpu_audio =
      audio.to(torch::kCPU, torch::kFloat32).contiguous();
  if (!encoder.encode(cpu_video,
                      cpu_audio,
                      fps,
                      sample_rate,
                      result.container,
                      result.data) ||
      result.data.empty()) {
    return {StatusCode::UNKNOWN, "MiniMax-H3 video/audio encoding failed"};
  }
  encoded = std::move(result);
  return Status();
}

}  // namespace xllm
