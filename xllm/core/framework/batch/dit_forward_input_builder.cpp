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
#include <vector>

#include "core/framework/config/dit_config.h"

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
  input.batch_size = requests_.size();
  input.generation_params = requests_[0]->state().generation_params();

  const size_t batch_size = requests_.size();
  input.prompts.reserve(batch_size);
  input.prompts_2.reserve(batch_size);
  input.negative_prompts.reserve(batch_size);
  input.negative_prompts_2.reserve(batch_size);
  for (const auto& request : requests_) {
    const auto& generation_params = request->state().generation_params();
    CHECK(input.generation_params == generation_params)
        << "DiT generation params must be equal in the same batch";

    const auto& input_params = request->state().input_params();
    if (!input_params.prompt.empty()) {
      input.prompts.emplace_back(input_params.prompt);
    }

    if (!input_params.prompt_2.empty()) {
      input.prompts_2.emplace_back(input_params.prompt_2);
    }

    if (!input_params.negative_prompt.empty()) {
      input.negative_prompts.emplace_back(input_params.negative_prompt);
    }

    if (!input_params.negative_prompt_2.empty()) {
      input.negative_prompts_2.emplace_back(input_params.negative_prompt_2);
    }

    if (!input_params.audio_prompt_text.empty() &&
        input.audio_prompt_text.empty()) {
      input.audio_prompt_text = input_params.audio_prompt_text;
    }
  }

  if (input.prompts.size() != requests_.size()) {
    input.prompts.clear();
  }

  if (input.prompts_2.size() != requests_.size()) {
    input.prompts_2.clear();
  }

  const bool has_full_negative_prompts =
      input.negative_prompts.size() == requests_.size();
  if (!has_full_negative_prompts) {
    input.negative_prompts.clear();
  }

  if (input.negative_prompts_2.size() != requests_.size()) {
    input.negative_prompts_2.clear();
  }

  const DiTImageSources& first_image_sources =
      requests_[0]->state().input_params().image_sources;
  for (size_t index = 0; index < first_image_sources.size(); ++index) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(batch_size);
    for (const auto& request : requests_) {
      tensors.emplace_back(
          request->state().input_params().image_sources.at(index).tensor);
    }
    CHECK(check_tensors_valid(tensors));
    input.image_sources.add(first_image_sources.at(index).name,
                            torch::stack(tensors));
  }

  const DiTTensorSources& first_tensor_sources =
      requests_[0]->state().input_params().tensor_sources;
  for (const NamedTensor& tensor_input : first_tensor_sources.entries()) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(batch_size);
    for (const auto& request : requests_) {
      std::optional<torch::Tensor> tensor =
          request->state().input_params().tensor_sources.get(tensor_input.name);
      CHECK(tensor.has_value());
      tensors.emplace_back(*tensor);
    }
    CHECK(check_tensors_valid(tensors));
    input.tensor_sources.add(tensor_input.name, torch::stack(tensors));
  }

  return input;
}

}  // namespace xllm
