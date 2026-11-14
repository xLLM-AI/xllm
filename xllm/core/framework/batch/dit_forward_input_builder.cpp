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

#include "core/framework/batch/dit_forward_input_builder.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/request/minimax_h3_conditioning.h"

namespace {

bool check_tensors_valid(const std::vector<torch::Tensor>& vec) {
  CHECK(!vec.empty());

  torch::Tensor ref_tensor = vec[0];
  if (!ref_tensor.defined()) {
    return false;
  }

  if (vec.size() == 1) {
    return true;
  }

  const auto ref_shape = ref_tensor.sizes();
  for (size_t i = 1; i < vec.size(); ++i) {
    if (!vec[i].defined()) {
      return false;
    }

    if (vec[i].sizes() != ref_shape) {
      return false;
    }
  }

  return true;
}

torch::Tensor batch_tensors(const std::vector<torch::Tensor>& tensors) {
  CHECK(check_tensors_valid(tensors));
  if (tensors.size() == 1) {
    return tensors.front().unsqueeze(0);
  }
  return torch::stack(tensors);
}

}  // namespace

namespace xllm {

DiTForwardInput DiTForwardInputBuilder::build_forward_input() const {
  CHECK(!requests_.empty());
  if (::xllm::DiTConfig::get_instance().dit_debug_print()) {
    LOG(INFO) << "DiT batch_size=" << requests_.size();
  }
  if (requests_[0]->state().request_kind() == DiTRequestKind::kText) {
    CHECK_EQ(requests_.size(), 1U)
        << "Cola-DLM text generation supports batch_size=1 only.";
  }

  DiTForwardInput input;
  DiTModelParams& model_params = input.model_params;
  model_params.batch_size = requests_.size();
  model_params.generation_params = requests_[0]->state().generation_params();

  const size_t batch_size = requests_.size();
  model_params.prompts.reserve(batch_size);
  model_params.prompts_2.reserve(batch_size);
  model_params.negative_prompts.reserve(batch_size);
  model_params.negative_prompts_2.reserve(batch_size);
  for (const auto& request : requests_) {
    const auto& generation_params = request->state().generation_params();
    CHECK(model_params.generation_params == generation_params)
        << "DiT generation params must be equal in the same batch";

    const auto& input_params = request->state().input_params();
    if (!input_params.prompt.empty()) {
      model_params.prompts.emplace_back(input_params.prompt);
    }

    if (!input_params.prompt_2.empty()) {
      model_params.prompts_2.emplace_back(input_params.prompt_2);
    }

    if (!input_params.negative_prompt.empty()) {
      model_params.negative_prompts.emplace_back(input_params.negative_prompt);
    }

    if (!input_params.negative_prompt_2.empty()) {
      model_params.negative_prompts_2.emplace_back(
          input_params.negative_prompt_2);
    }

    if (!input_params.audio_prompt_text.empty() &&
        model_params.audio_prompt_text.empty()) {
      model_params.audio_prompt_text = input_params.audio_prompt_text;
    }
  }

  if (model_params.prompts.size() != requests_.size()) {
    model_params.prompts.clear();
  }

  if (model_params.prompts_2.size() != requests_.size()) {
    model_params.prompts_2.clear();
  }

  if (model_params.negative_prompts.size() != requests_.size()) {
    model_params.negative_prompts.clear();
  }

  if (model_params.negative_prompts_2.size() != requests_.size()) {
    model_params.negative_prompts_2.clear();
  }

  const DiTMediaSources& first_media_sources =
      requests_[0]->state().input_params().media_sources;
  for (const auto& request : requests_) {
    CHECK(first_media_sources.batch_signature_matches(
        request->state().input_params().media_sources));
  }
  for (size_t index = 0; index < first_media_sources.size(); ++index) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(batch_size);
    for (const auto& request : requests_) {
      tensors.emplace_back(
          request->state().input_params().media_sources.at(index).tensor);
    }
    const MediaNamedTensor& source = first_media_sources.at(index);
    model_params.media_sources.add(source.name,
                                   source.modality,
                                   batch_tensors(tensors),
                                   source.parameters);
  }

  const DiTTensorSources& first_tensor_sources =
      requests_[0]->state().input_params().tensor_sources;
  for (const auto& request : requests_) {
    CHECK(first_tensor_sources.batch_signature_matches(
        request->state().input_params().tensor_sources));
  }
  for (const NamedTensor& tensor_input : first_tensor_sources.entries()) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(batch_size);
    for (const auto& request : requests_) {
      std::optional<torch::Tensor> tensor =
          request->state().input_params().tensor_sources.get(tensor_input.name);
      CHECK(tensor.has_value());
      tensors.emplace_back(*tensor);
    }
    model_params.tensor_sources.add(
        tensor_input.name, batch_tensors(tensors), tensor_input.parameters);
  }

  const auto& config = DiTConfig::get_instance();
  if (config.dit_h3_execution_profile() == kMiniMaxH3ExecutionProfile ||
      config.dit_h3_phase_timing()) {
    if (requests_.size() != 1 || requests_[0]->request_id().empty()) {
      throw std::invalid_argument(
          "H3 observation forward input requires one nonempty server request "
          "ID");
    }
    bool assigned = false;
    for (NamedTensor& tensor : model_params.tensor_sources.entries()) {
      if (tensor.name == "prompt_embed") {
        if (assigned) {
          throw std::invalid_argument(
              "H3 observation forward input contains duplicate prompt_embed");
        }
        // Overwrite any client-supplied value. Never derive identity from the
        // conditioning digest: repeated requests may use identical inputs.
        tensor.parameters.insert_or_assign(
            std::string(kMiniMaxH3InternalRequestId),
            requests_[0]->request_id());
        assigned = true;
      }
    }
    if (!assigned) {
      throw std::invalid_argument(
          "H3 observation forward input is missing prompt_embed");
    }
  }

  return input;
}

}  // namespace xllm
