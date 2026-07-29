/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "api_service/call.h"
#include "core/framework/multimodal/mm_codec.h"

namespace xllm {
namespace {

// Write a minimal PCM WAV file into `out`.
// samples: float32 mono, range [-1, 1], shape (N,) on CPU.
void encode_wav(const torch::Tensor& samples,
                int32_t sample_rate,
                std::string& out) {
  CHECK(samples.device().is_cpu()) << "encode_wav: tensor must be on CPU";
  CHECK(samples.is_contiguous()) << "encode_wav: tensor must be contiguous";
  CHECK_EQ(samples.scalar_type(), torch::kFloat32)
      << "encode_wav: tensor must be float32";
  CHECK_GT(samples.numel(), 0) << "encode_wav: tensor must not be empty";
  const int32_t num_samples = static_cast<int32_t>(samples.numel());
  const int32_t num_channels = 1;
  const int32_t bits_per_sample = 16;
  const int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
  const int16_t block_align =
      static_cast<int16_t>(num_channels * bits_per_sample / 8);
  const int32_t data_size = num_samples * num_channels * bits_per_sample / 8;
  const int32_t chunk_size = 36 + data_size;

  out.resize(44 + data_size);
  char* p = out.data();

  auto write4 = [&](const char* s) {
    std::memcpy(p, s, 4);
    p += 4;
  };
  auto writeI32 = [&](int32_t v) {
    std::memcpy(p, &v, 4);
    p += 4;
  };
  auto writeI16 = [&](int16_t v) {
    std::memcpy(p, &v, 2);
    p += 2;
  };

  write4("RIFF");
  writeI32(chunk_size);
  write4("WAVE");
  write4("fmt ");
  writeI32(16);  // PCM subchunk size
  writeI16(1);   // AudioFormat = PCM
  writeI16(static_cast<int16_t>(num_channels));
  writeI32(sample_rate);
  writeI32(byte_rate);
  writeI16(block_align);
  writeI16(static_cast<int16_t>(bits_per_sample));
  write4("data");
  writeI32(data_size);

  const float* src = samples.data_ptr<float>();
  // Convert float32 -> int16 and write samples
  int16_t* dst = reinterpret_cast<int16_t*>(p);
  for (int32_t i = 0; i < num_samples; ++i) {
    float v = std::max(-1.0f, std::min(1.0f, src[i]));
    dst[i] = static_cast<int16_t>(v * 32767.0f);
  }
}

}  // namespace

DiTRequest::DiTRequest(const std::string& request_id,
                       const std::string& x_request_id,
                       const std::string& x_request_time,
                       const DiTRequestState& state,
                       const std::string& service_request_id,
                       const std::string& source_xservice_addr)
    : RequestBase(request_id,
                  x_request_id,
                  x_request_time,
                  service_request_id,
                  source_xservice_addr),
      state_(state) {}

bool DiTRequest::finished() const { return true; }

void DiTRequest::log_statistic(double total_latency) {
  LOG(INFO) << "x-request-id: " << x_request_id_ << ", "
            << "x-request-time: " << x_request_time_ << ", "
            << "request_id: " << request_id_ << ", "
            << "total_latency: " << total_latency * 1000 << "ms";
}

void DiTRequest::handle_forward_output(torch::Tensor output) {
  // Pipeline already chunks by batch size along dim 0 before calling here.
  // For image models, also split by num_images_per_prompt.
  // For video models, split by num_images_per_prompt * num_videos_per_prompt.
  // For audio models, num_images_per_prompt defaults to 1 so this is a no-op.
  const int32_t count =
      static_cast<int32_t>(state_.generation_params().num_images_per_prompt *
                           state_.generation_params().num_videos_per_prompt);
  output_.tensors = torch::chunk(output, count);
}

void DiTRequest::handle_forward_text_output(const std::string& text) {
  output_.text_output.push_back(text);
}

const DiTRequestOutput DiTRequest::generate_output() {
  DiTRequestOutput output;
  output.request_id = request_id_;
  output.service_request_id = service_request_id_;
  output.status = Status(StatusCode::OK);
  output.finished = finished();
  output.cancelled = false;

  // Text diffusion models (e.g., Cola-DLM) produce text output directly.
  if (!output_.text_output.empty()) {
    const auto& gen_params = state_.generation_params();
    for (const auto& text : output_.text_output) {
      DiTGenerationOutput result;
      result.seed_is_set = gen_params.seed_is_set;
      if (gen_params.seed_is_set) {
        result.seed = gen_params.seed;
      }
      result.text = text;
      output.outputs.push_back(result);
    }
    return output;
  }

  const bool is_audio =
      !output_.tensors.empty() && output_.tensors[0].dim() <= 2;

  DiTGenerationOutput result;
  result.seed = state_.generation_params().seed;
  if (!is_audio) {
    result.height = state_.generation_params().height;
    result.width = state_.generation_params().width;
  }

  const int32_t count =
      static_cast<int32_t>(state_.generation_params().num_images_per_prompt *
                           state_.generation_params().num_videos_per_prompt);
  OpenCVImageEncoder image_encoder;
  FFmpegVideoEncoder video_encoder;
  for (size_t idx = 0; idx < count; ++idx) {
    torch::Tensor output_tensor =
        output_.tensors[idx].squeeze(0).cpu().to(torch::kFloat32).contiguous();
    if (is_audio) {
      torch::Tensor samples = output_tensor.flatten().contiguous();
      encode_wav(samples,
                 state_.generation_params().audio_sampling_rate,
                 result.audio);
    } else if (output_tensor.dim() == 4 ||
               state_.generation_params().force_video_output) {
      video_encoder.encode(output_tensor,
                           state_.generation_params().video_fps,
                           "mp4",
                           result.image);
      result.num_frames = output_tensor.dim() == 4
                              ? static_cast<int32_t>(output_tensor.size(0))
                              : 0;
      result.video_fps = state_.generation_params().video_fps;
    } else {
      image_encoder.encode(output_tensor, result.image);
    }
    output.outputs.push_back(result);
  }

  return output;
}

}  // namespace xllm
