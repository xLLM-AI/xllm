/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/request/dit_input_sources.h"

#include <utility>

#include "core/util/tensor_helper.h"

namespace xllm {

void DiTImageSources::add(std::string name, torch::Tensor tensor) {
  if (name.empty()) {
    name = "unknown";
  }
  entries_.emplace_back(
      NamedTensor{.name = std::move(name), .tensor = std::move(tensor)});
}

std::vector<torch::Tensor> DiTImageSources::get(
    const std::vector<std::string>& names) const {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(entries_.size());
  if (!names.empty()) {
    std::vector<bool> matched(entries_.size(), false);
    for (const std::string& name : names) {
      bool found = false;
      for (size_t index = 0; index < entries_.size(); ++index) {
        if (!matched[index] && entries_[index].name == name) {
          tensors.emplace_back(entries_[index].tensor);
          matched[index] = true;
          found = true;
          break;
        }
      }
      if (!found) {
        tensors.clear();
        break;
      }
    }
    if (tensors.size() == names.size()) {
      return tensors;
    }
  }

  const size_t tensor_count =
      names.empty() ? entries_.size() : std::min(names.size(), entries_.size());
  tensors.reserve(tensor_count);
  for (size_t index = 0; index < tensor_count; ++index) {
    tensors.emplace_back(entries_[index].tensor);
  }
  return tensors;
}

NamedTensor& DiTImageSources::at(size_t index) { return entries_.at(index); }

const NamedTensor& DiTImageSources::at(size_t index) const {
  return entries_.at(index);
}

std::vector<NamedTensor>& DiTImageSources::entries() { return entries_; }

const std::vector<NamedTensor>& DiTImageSources::entries() const {
  return entries_;
}

size_t DiTImageSources::size() const { return entries_.size(); }

bool DiTImageSources::empty() const { return entries_.empty(); }

bool DiTImageSources::batch_signature_matches(
    const DiTImageSources& other) const {
  if (entries_.size() != other.entries_.size()) {
    return false;
  }
  for (size_t index = 0; index < entries_.size(); ++index) {
    const NamedTensor& lhs = entries_[index];
    const NamedTensor& rhs = other.entries_[index];
    if (lhs.name != rhs.name ||
        !tensor_batch_signature_matches(lhs.tensor, rhs.tensor)) {
      return false;
    }
  }
  return true;
}

DiTImageSources DiTImageSources::to(const torch::Device& device) const {
  DiTImageSources result;
  result.entries_.reserve(entries_.size());
  for (const NamedTensor& source : entries_) {
    result.add(source.name, source.tensor.to(device, /*dtype=*/torch::kUInt8));
  }
  return result;
}

void DiTTensorSources::add(std::string name, torch::Tensor tensor) {
  entries_.emplace_back(
      NamedTensor{.name = std::move(name), .tensor = std::move(tensor)});
}

bool DiTTensorSources::contains(std::string_view name) const {
  return get(name).has_value();
}

std::optional<torch::Tensor> DiTTensorSources::get(
    std::string_view name) const {
  for (const NamedTensor& input : entries_) {
    if (input.name == name) {
      return input.tensor;
    }
  }
  return std::nullopt;
}

std::vector<NamedTensor>& DiTTensorSources::entries() { return entries_; }

const std::vector<NamedTensor>& DiTTensorSources::entries() const {
  return entries_;
}

size_t DiTTensorSources::size() const { return entries_.size(); }

bool DiTTensorSources::empty() const { return entries_.empty(); }

bool DiTTensorSources::batch_signature_matches(
    const DiTTensorSources& other) const {
  if (entries_.size() != other.entries_.size()) {
    return false;
  }
  for (const NamedTensor& input : entries_) {
    std::optional<torch::Tensor> other_tensor = other.get(input.name);
    if (!other_tensor.has_value() ||
        !tensor_batch_signature_matches(input.tensor, *other_tensor)) {
      return false;
    }
  }
  return true;
}

DiTTensorSources DiTTensorSources::to(const torch::Device& device,
                                      torch::ScalarType dtype) const {
  DiTTensorSources result;
  result.entries_.reserve(entries_.size());
  for (const NamedTensor& input : entries_) {
    const torch::ScalarType target_dtype =
        input.name == "prompt_audio" ? torch::kFloat32 : dtype;
    result.add(input.name, input.tensor.to(device, target_dtype));
  }
  return result;
}

}  // namespace xllm
