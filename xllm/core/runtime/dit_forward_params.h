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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <nlohmann/json.hpp>

#include "framework/request/dit_request_state.h"

namespace xllm {

namespace proto {
class DiTForwardInput;
class DiTForwardOutput;
class PackedForwardInput;
}  // namespace proto

// dit related forward input params
class DiTForwardInput final {
 public:
  bool to_packed_proto(proto::PackedForwardInput* packed_forward_input) const;
  bool from_packed_proto(const proto::PackedForwardInput& packed_forward_input);
  bool to_proto(proto::DiTForwardInput* proto_input) const;
  bool from_proto(const proto::DiTForwardInput& proto_input);

  void debug_print(std::ostream& os = std::cout) const {
    os << "=== DiTForwardInput Debug Info ===" << std::endl;

    // Print basic data types
    os << "batch_size: " << batch_size << std::endl;

    // Print prompts vectors
    os << "prompts: [";
    for (size_t i = 0; i < prompts.size(); ++i) {
      os << "\"" << prompts[i] << "\"";
      if (i < prompts.size() - 1) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    os << "prompts_2: [";
    for (size_t i = 0; i < prompts_2.size(); ++i) {
      os << "\"" << prompts_2[i] << "\"";
      if (i < prompts_2.size() - 1) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    os << "negative_prompts: [";
    for (size_t i = 0; i < negative_prompts.size(); ++i) {
      os << "\"" << negative_prompts[i] << "\"";
      if (i < negative_prompts.size() - 1) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    os << "negative_prompts_2: [";
    for (size_t i = 0; i < negative_prompts_2.size(); ++i) {
      os << "\"" << negative_prompts_2[i] << "\"";
      if (i < negative_prompts_2.size() - 1) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    // Print tensor shapes
    os << "\n--- Tensor Shapes ---" << std::endl;

    os << "media_sources: [";
    for (size_t index = 0; index < media_sources.size(); ++index) {
      const MediaNamedTensor& source = media_sources.at(index);
      os << source.name << ":" << source.tensor.sizes();
      if (index + 1 < media_sources.size()) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    os << "tensor_sources: [";
    for (size_t index = 0; index < tensor_sources.size(); ++index) {
      const NamedTensor& tensor_input = tensor_sources.entries()[index];
      os << tensor_input.name << ":" << tensor_input.tensor.sizes();
      if (index + 1 < tensor_sources.size()) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    // Print generation_params
    os << "\n--- Generation Parameters ---" << std::endl;
    os << "width: " << generation_params.width << std::endl;
    os << "height: " << generation_params.height << std::endl;
    os << "num_inference_steps: " << generation_params.num_inference_steps
       << std::endl;
    os << "true_cfg_scale: " << generation_params.true_cfg_scale << std::endl;
    os << "guidance_scale: " << generation_params.guidance_scale << std::endl;
    os << "num_images_per_prompt: " << generation_params.num_images_per_prompt
       << std::endl;
    os << "seed: " << generation_params.seed << std::endl;
    os << "max_sequence_length: " << generation_params.max_sequence_length
       << std::endl;
    os << "strength: " << generation_params.strength << std::endl;

    os << "===============================" << std::endl;
  }

  DiTForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype = torch::kBFloat16) const {
    DiTForwardInput input = *this;

    input.tensor_sources = tensor_sources.to(device, dtype);
    input.media_sources = media_sources.to(device);
    return input;
  }

  int32_t batch_size = 0;

  // Primary input text description for image generation
  std::vector<std::string> prompts;

  // Secondary prompt for additional details (e.g., color, lighting)
  std::vector<std::string> prompts_2;

  // Negative prompt to exclude low-quality features
  std::vector<std::string> negative_prompts;

  // Secondary negative prompt to exclude additional unwanted features
  std::vector<std::string> negative_prompts_2;

  DiTMediaSources media_sources;

  DiTTensorSources tensor_sources;

  // Transcript of the prompt audio — used for duration estimation only.
  std::string audio_prompt_text;

  // generation params
  DiTGenerationParams generation_params;
};

// dit related forward output params
class DiTForwardOutput final {
 public:
  bool to_proto(proto::DiTForwardOutput* proto_output) const;
  bool from_proto(const proto::DiTForwardOutput& proto_output);

  // generated tensor (for image/audio models)
  std::vector<torch::Tensor> tensors;
  // Optional generated audio tensors for video+audio models.
  std::vector<torch::Tensor> audio_tensors;
  // generated text (for text diffusion models like Cola-DLM)
  std::vector<std::string> text_output;
};

}  // namespace xllm
