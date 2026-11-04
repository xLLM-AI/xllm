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

#include <iostream>
#include <stdexcept>
#include <utility>

#include "core/common/types.h"
#include "core/framework/model/dit_model_output.h"
#include "core/framework/model/dit_model_params.h"

namespace xllm {

class DiTForwardExecutionError final : public std::runtime_error {
 public:
  explicit DiTForwardExecutionError(Status status)
      : std::runtime_error(status.message()), status_(std::move(status)) {}

  const Status& status() const noexcept { return status_; }

 private:
  Status status_;
};

namespace proto {
class DiTForwardInput;
class PackedForwardInput;
}  // namespace proto

// Execution input transported to DiT workers.
class DiTForwardInput final {
 public:
  bool to_packed_proto(proto::PackedForwardInput* packed_forward_input) const;
  bool from_packed_proto(const proto::PackedForwardInput& packed_forward_input);
  bool to_proto(proto::DiTForwardInput* proto_input) const;
  bool from_proto(const proto::DiTForwardInput& proto_input);

  void debug_print(std::ostream& os = std::cout) const {
    os << "=== DiTForwardInput Debug Info ===" << std::endl;

    // Print basic data types
    os << "batch_size: " << model_params.batch_size << std::endl;

    // Print prompts vectors
    os << "prompts: [";
    for (size_t i = 0; i < model_params.prompts.size(); ++i) {
      os << "\"" << model_params.prompts[i] << "\"";
      if (i < model_params.prompts.size() - 1) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    os << "prompts_2: [";
    for (size_t i = 0; i < model_params.prompts_2.size(); ++i) {
      os << "\"" << model_params.prompts_2[i] << "\"";
      if (i < model_params.prompts_2.size() - 1) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    os << "negative_prompts: [";
    for (size_t i = 0; i < model_params.negative_prompts.size(); ++i) {
      os << "\"" << model_params.negative_prompts[i] << "\"";
      if (i < model_params.negative_prompts.size() - 1) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    os << "negative_prompts_2: [";
    for (size_t i = 0; i < model_params.negative_prompts_2.size(); ++i) {
      os << "\"" << model_params.negative_prompts_2[i] << "\"";
      if (i < model_params.negative_prompts_2.size() - 1) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    // Print tensor shapes
    os << "\n--- Tensor Shapes ---" << std::endl;

    os << "media_sources: [";
    for (size_t index = 0; index < model_params.media_sources.size(); ++index) {
      const MediaNamedTensor& source = model_params.media_sources.at(index);
      os << source.name << ":" << source.tensor.sizes();
      if (index + 1 < model_params.media_sources.size()) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    os << "tensor_sources: [";
    for (size_t index = 0; index < model_params.tensor_sources.size();
         ++index) {
      const NamedTensor& tensor_input =
          model_params.tensor_sources.entries()[index];
      os << tensor_input.name << ":" << tensor_input.tensor.sizes();
      if (index + 1 < model_params.tensor_sources.size()) {
        os << ", ";
      }
    }
    os << "]" << std::endl;

    // Print generation_params
    os << "\n--- Generation Parameters ---" << std::endl;
    os << "width: " << model_params.generation_params.width << std::endl;
    os << "height: " << model_params.generation_params.height << std::endl;
    os << "num_inference_steps: "
       << model_params.generation_params.num_inference_steps << std::endl;
    os << "true_cfg_scale: " << model_params.generation_params.true_cfg_scale
       << std::endl;
    os << "guidance_scale: " << model_params.generation_params.guidance_scale
       << std::endl;
    os << "num_images_per_prompt: "
       << model_params.generation_params.num_images_per_prompt << std::endl;
    os << "seed: " << model_params.generation_params.seed << std::endl;
    os << "max_sequence_length: "
       << model_params.generation_params.max_sequence_length << std::endl;
    os << "strength: " << model_params.generation_params.strength << std::endl;

    os << "===============================" << std::endl;
  }

  DiTForwardInput to(const torch::Device& device,
                     torch::ScalarType dtype = torch::kBFloat16) const {
    DiTForwardInput input;
    input.model_params = model_params.to(device, dtype);
    return input;
  }

  DiTModelParams model_params;
};

}  // namespace xllm
