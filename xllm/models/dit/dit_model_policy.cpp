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

#include "models/dit/dit_model_policy.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/common/options.h"
#include "core/framework/config/model_config.h"
#include "core/framework/model_loader/hf_model_config.h"
#include "core/framework/request/dit_request_params.h"
#include "core/util/tensor_helper.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

const char* validate_minimax_h3_t2va_request(const DiTRequestParams& params) {
  const auto& generation = params.generation_params;
  if (params.request_kind != DiTRequestKind::kVideo ||
      !params.input_params.media_sources.entries().empty() ||
      generation.num_videos_per_prompt != 1) {
    return "Python MiniMax-H3 supports T2VA with one video per request";
  }
  if (generation.audio_sampling_rate != 32000 || generation.num_frames <= 0 ||
      !std::isfinite(generation.video_fps) || generation.video_fps <= 0.0) {
    return "Python MiniMax-H3 requires 32 kHz audio, positive frames and fps";
  }
  // The serving checkpoint uses 17-frame clips and rounds up by at most 16.
  const double frames = static_cast<double>(generation.num_frames);
  if (frames / generation.video_fps < 5.0 ||
      (frames + 16.0) / generation.video_fps > 15.0) {
    return "Python MiniMax-H3 duration must be between 5 and 15 seconds";
  }
  const auto source =
      params.input_params.tensor_sources.get_named_tensor("prompt_embed");
  if (!source.has_value()) {
    return "Python MiniMax-H3 requires prompt_embed from the H3 encoder";
  }
  const auto& embedding = source->get().tensor;
  if (embedding.dim() != 2 || embedding.size(0) == 0 ||
      embedding.size(1) != 5120 || !embedding.is_floating_point()) {
    return "Python MiniMax-H3 requires a floating point [L, 5120] prompt_embed";
  }
  const auto* tags = get_tensor_parameter<std::vector<int64_t>>(
      source->get().parameters, "prompt_token_tags");
  if (tags == nullptr ||
      tags->size() != static_cast<size_t>(embedding.size(0)) ||
      !std::all_of(
          tags->begin(), tags->end(), [](int64_t tag) { return tag == 1; })) {
    return "Python MiniMax-H3 T2VA requires L text-only prompt_token_tags";
  }
  return nullptr;
}

}  // namespace

std::string resolve_dit_model_type(const std::string& checkpoint_model_type,
                                   const std::string& model_id) {
  return ModelRegistry::has_dit_model_factory(checkpoint_model_type)
             ? checkpoint_model_type
             : model_id;
}

DiTRequestPolicy resolve_dit_request_policy(const Options& options,
                                            const std::string& model_impl) {
  if (!ModelConfig::is_python_model_impl(model_impl)) {
    return {};
  }
  const std::string model_type = resolve_dit_model_type(
      get_model_type(options.model_path(), options.backend()),
      options.model_id());
  if (model_type == "MiniMaxH3ModularPipeline") {
    return {.validator = validate_minimax_h3_t2va_request,
            .required_batch_size = 1};
  }
  return {};
}

}  // namespace xllm
