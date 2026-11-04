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

#include "core/framework/batch/dit_batch.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/batch/dit_forward_input_builder.h"

namespace xllm {
namespace {

bool encoded_media_matches_request_kind(const DiTEncodedMedia& media,
                                        DiTRequestKind kind) {
  if (!valid_dit_encoded_media(media)) {
    return false;
  }
  switch (kind) {
    case DiTRequestKind::kImage:
      return media.mime_type.starts_with("image/") && media.width > 0 &&
             media.height > 0;
    case DiTRequestKind::kVideo:
      return media.mime_type.starts_with("video/") && media.width > 0 &&
             media.height > 0 && media.num_frames > 0 && media.fps > 0;
    case DiTRequestKind::kAudio:
      return media.mime_type.starts_with("audio/") &&
             media.audio_sample_rate > 0 && media.audio_channels > 0;
    case DiTRequestKind::kText:
      return false;
  }
  return false;
}

}  // namespace

void DiTBatch::add(std::shared_ptr<DiTRequest> request) {
  CHECK(request != nullptr);
  request_vec_.emplace_back(std::move(request));
}

DiTForwardInput DiTBatch::prepare_forward_input() {
  return DiTForwardInputBuilder(request_vec_).build_forward_input();
}

void DiTBatch::process_forward_output(const DiTForwardOutput& output) {
  if (!output.encoded_media.empty()) {
    if (!valid_dit_output_mode(output)) {
      throw DiTForwardExecutionError(
          {StatusCode::UNKNOWN, "DiT output mixes encoded and raw payloads"});
    }
    std::vector<size_t> candidate_counts;
    candidate_counts.reserve(request_vec_.size());
    size_t total = 0;
    for (const auto& request : request_vec_) {
      const DiTRequestState& state = request->state();
      size_t count = 1;
      switch (state.request_kind()) {
        case DiTRequestKind::kImage:
          count = state.generation_params().num_images_per_prompt;
          break;
        case DiTRequestKind::kVideo:
          count = state.generation_params().num_videos_per_prompt;
          break;
        case DiTRequestKind::kAudio:
          break;
        case DiTRequestKind::kText:
          throw DiTForwardExecutionError(
              {StatusCode::UNKNOWN,
               "Text request cannot receive encoded media"});
      }
      if (count == 0 || count > output.encoded_media.size() - total) {
        throw DiTForwardExecutionError(
            {StatusCode::UNKNOWN, "DiT encoded candidate count mismatch"});
      }
      for (size_t index = total; index < total + count; ++index) {
        if (!encoded_media_matches_request_kind(output.encoded_media[index],
                                                state.request_kind())) {
          throw DiTForwardExecutionError(
              {StatusCode::UNKNOWN,
               "DiT encoded media does not match request kind"});
        }
      }
      total += count;
      candidate_counts.emplace_back(count);
    }
    if (total != output.encoded_media.size()) {
      throw DiTForwardExecutionError(
          {StatusCode::UNKNOWN, "DiT encoded candidate count mismatch"});
    }
    size_t offset = 0;
    for (size_t index = 0; index < request_vec_.size(); ++index) {
      const size_t count = candidate_counts[index];
      std::vector<DiTEncodedMedia> candidates(
          output.encoded_media.begin() + offset,
          output.encoded_media.begin() + offset + count);
      request_vec_[index]->handle_forward_encoded_output(std::move(candidates));
      offset += count;
    }
    return;
  }
  // Text diffusion models produce text output directly.
  if (!output.text_output.empty()) {
    CHECK(request_vec_.size() == output.text_output.size());
    for (int32_t idx = 0; idx < static_cast<int32_t>(request_vec_.size());
         ++idx) {
      auto& request = request_vec_[idx];
      request->handle_forward_text_output(output.text_output[idx]);
    }
    return;
  }
  CHECK(request_vec_.size() == output.tensors.size());
  CHECK(output.audio_tensors.empty() ||
        output.audio_tensors.size() == output.tensors.size());
  for (int32_t idx = 0; idx < static_cast<int32_t>(request_vec_.size());
       ++idx) {
    auto& request = request_vec_[idx];
    const torch::Tensor audio_output = output.audio_tensors.empty()
                                           ? torch::Tensor()
                                           : output.audio_tensors[idx];
    request->handle_forward_output(output.tensors[idx], audio_output);
  }
}

}  // namespace xllm
