/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
                       const std::string& source_xservice_addr,
                       RateLimiter* rate_limiter)
    : RequestBase(request_id,
                  x_request_id,
                  x_request_time,
                  service_request_id,
                  source_xservice_addr,
                  rate_limiter),
      state_(state) {}

bool DiTRequest::finished() const { return true; }

void DiTRequest::log_statistic(double total_latency) {
  LOG(INFO) << "x-request-id: " << x_request_id_ << ", "
            << "x-request-time: " << x_request_time_ << ", "
            << "request_id: " << request_id_ << ", "
            << "total_latency: " << total_latency * 1000 << "ms";
}

void DiTRequest::update_connection_status() {
  if (!state_.call().has_value()) {
    return;
  }
  Call* call = state_.call().value();
  if (call == nullptr || !call->is_disconnected()) {
    return;
  }
  set_cancel();
}

void DiTRequest::handle_forward_output(torch::Tensor output) {
  uint32_t output_count = 1;
  switch (state_.request_kind()) {
    case DiTRequestKind::kImage:
      output_count = state_.generation_params().num_images_per_prompt;
      break;
    case DiTRequestKind::kVideo:
      output_count = state_.generation_params().num_videos_per_prompt;
      break;
    case DiTRequestKind::kAudio:
      break;
    case DiTRequestKind::kText:
      LOG(FATAL) << "Text request must not contain tensor output";
  }
  output_.tensors = torch::chunk(output, static_cast<int32_t>(output_count));
}

void DiTRequest::handle_forward_text_output(const std::string& text) {
  output_.text_output.push_back(text);
}

std::vector<DiTGenerationOutput> DiTRequest::generate_image_outputs() const {
  const DiTGenerationParams& params = state_.generation_params();
  CHECK_EQ(output_.tensors.size(), params.num_images_per_prompt);

  std::vector<DiTGenerationOutput> outputs;
  outputs.reserve(output_.tensors.size());
  OpenCVImageEncoder encoder;
  for (size_t index = 0; index < output_.tensors.size(); ++index) {
    torch::Tensor tensor = output_.tensors[index]
                               .squeeze(0)
                               .cpu()
                               .to(torch::kFloat32)
                               .contiguous();
    CHECK_EQ(tensor.dim(), 3);
    DiTGenerationOutput output;
    output.index = index;
    output.seed = params.seed;
    output.seed_is_set = params.seed_is_set;
    output.height = params.height;
    output.width = params.width;
    CHECK(encoder.encode(tensor, output.image));
    outputs.emplace_back(std::move(output));
  }
  return outputs;
}

std::vector<DiTGenerationOutput> DiTRequest::generate_video_outputs() const {
  const DiTGenerationParams& params = state_.generation_params();
  CHECK_EQ(output_.tensors.size(), params.num_videos_per_prompt);

  std::vector<DiTGenerationOutput> outputs;
  outputs.reserve(output_.tensors.size());
  for (size_t index = 0; index < output_.tensors.size(); ++index) {
    torch::Tensor tensor = output_.tensors[index]
                               .squeeze(0)
                               .cpu()
                               .to(torch::kFloat32)
                               .contiguous();
    CHECK_EQ(tensor.dim(), 4);
    DiTGenerationOutput output;
    output.index = index;
    output.seed = params.seed;
    output.seed_is_set = params.seed_is_set;
    output.height = params.height;
    output.width = params.width;
    output.num_frames = static_cast<int32_t>(tensor.size(0));
    output.video_fps = params.video_fps;
    FFmpegVideoEncoder encoder;
    CHECK(encoder.encode(tensor, params.video_fps, "mp4", output.video));
    outputs.emplace_back(std::move(output));
  }
  return outputs;
}

std::vector<DiTGenerationOutput> DiTRequest::generate_audio_outputs() const {
  CHECK_EQ(output_.tensors.size(), 1u);
  const DiTGenerationParams& params = state_.generation_params();
  torch::Tensor samples = output_.tensors[0]
                              .squeeze(0)
                              .cpu()
                              .to(torch::kFloat32)
                              .flatten()
                              .contiguous();
  DiTGenerationOutput output;
  output.index = 0;
  output.seed = params.seed;
  output.seed_is_set = params.seed_is_set;
  encode_wav(samples, params.audio_sampling_rate, output.audio);
  return {std::move(output)};
}

std::vector<DiTGenerationOutput> DiTRequest::generate_text_outputs() const {
  const DiTGenerationParams& params = state_.generation_params();
  std::vector<DiTGenerationOutput> outputs;
  outputs.reserve(output_.text_output.size());
  for (size_t index = 0; index < output_.text_output.size(); ++index) {
    DiTGenerationOutput output;
    output.index = index;
    output.seed = params.seed;
    output.seed_is_set = params.seed_is_set;
    output.text = output_.text_output[index];
    outputs.emplace_back(std::move(output));
  }
  return outputs;
}

const DiTRequestOutput DiTRequest::generate_output() {
  DiTRequestOutput output;
  output.request_id = request_id_;
  output.service_request_id = service_request_id_;
  output.status = Status(StatusCode::OK);
  output.finished = finished();
  output.cancelled = cancelled();

  switch (state_.request_kind()) {
    case DiTRequestKind::kImage:
      output.outputs = generate_image_outputs();
      break;
    case DiTRequestKind::kVideo:
      output.outputs = generate_video_outputs();
      break;
    case DiTRequestKind::kAudio:
      output.outputs = generate_audio_outputs();
      break;
    case DiTRequestKind::kText:
      output.outputs = generate_text_outputs();
      break;
  }
  return output;
}

}  // namespace xllm
