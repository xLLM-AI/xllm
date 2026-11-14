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

#include <torch/types.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

namespace proto {
class DiTForwardOutput;
}  // namespace proto

struct DiTEncodedMedia {
  std::string data;
  std::string mime_type;
  std::string container;
  int32_t width = 0;
  int32_t height = 0;
  int32_t num_frames = 0;
  double fps = 0.0;
  int32_t audio_sample_rate = 0;
  int32_t audio_channels = 0;
};

inline bool valid_dit_encoded_media(const DiTEncodedMedia& media) {
  const bool no_audio =
      media.audio_sample_rate == 0 && media.audio_channels == 0;
  const bool has_audio =
      media.audio_sample_rate > 0 && media.audio_channels > 0;
  return !media.data.empty() && !media.mime_type.empty() &&
         !media.container.empty() && media.width >= 0 && media.height >= 0 &&
         media.num_frames >= 0 && std::isfinite(media.fps) &&
         (no_audio || has_audio);
}

class DiTForwardOutput final {
 public:
  bool to_proto(proto::DiTForwardOutput* proto_output) const;
  bool from_proto(const proto::DiTForwardOutput& proto_output);

  // Generated tensors for image/audio models.
  std::vector<torch::Tensor> tensors;
  // Optional generated audio tensors for video+audio models.
  std::vector<torch::Tensor> audio_tensors;
  // Generated text for text diffusion models such as Cola-DLM.
  std::vector<std::string> text_output;
  // Request-major, candidate-major owned encoded media; exclusive with raw
  // outputs.
  std::vector<DiTEncodedMedia> encoded_media;
};

inline bool valid_dit_output_mode(const DiTForwardOutput& output) {
  return output.encoded_media.empty() ||
         (output.tensors.empty() && output.audio_tensors.empty() &&
          output.text_output.empty());
}

}  // namespace xllm
