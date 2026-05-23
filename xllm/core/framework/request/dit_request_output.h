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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "core/common/types.h"

namespace xllm {

struct DiTGenerationOutput {
  // the index of the sequence in the request.
  size_t index;

  // the generated image/video as base64-encoded data.
  std::string image;

  // the generated audio as raw WAV bytes (audio models only).
  std::string audio;

  // the generated text (text diffusion models like Cola-DLM).
  std::string text;

  // the height of the generated image/video.
  int32_t height;

  // the width of the generated image/video.
  int32_t width;

  // seed used for generation (only valid when seed_is_set is true).
  int64_t seed = 0;

  // True when the client explicitly requested a seed (omit in API response when
  // false so stochastic runs are not reported as seed=0).
  bool seed_is_set = false;

  // number of video frames
  int32_t num_frames = 0;

  // video fps
  double video_fps = 0.0;
};

struct DiTRequestOutput {
  DiTRequestOutput() = default;

  DiTRequestOutput(Status&& _status, std::string service_request_id = "")
      : status(std::move(_status)), service_request_id(service_request_id) {}

  void log_request_status() const;

  // the id of the request.
  std::string request_id;

  // the id of the request which generated in xllm service.
  std::string service_request_id;

  // the status of the request.
  std::optional<Status> status;

  // the output for each sequence in the request.
  std::vector<DiTGenerationOutput> outputs;

  // whether the request is finished.
  bool finished = false;

  // whether the request is cancelled.
  bool cancelled = false;
};

// callback function for image request output, return true to continue, false to
// stop/cancel
using DiTOutputCallback = std::function<bool(DiTRequestOutput output)>;
// callback function for batch image output, return true to continue, false to
// stop/cancel
using BatchDiTOutputCallback =
    std::function<bool(size_t index, DiTRequestOutput output)>;

}  // namespace xllm
