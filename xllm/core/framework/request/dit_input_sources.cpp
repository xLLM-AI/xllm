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

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <utility>

#include "tensor.pb.h"

namespace xllm {

std::optional<TensorParameters> proto_to_tensor_parameters(
    const proto::Tensor& proto_tensor) {
  TensorParameters parameters;
  for (const auto& [name, proto_parameter] : proto_tensor.parameters()) {
    if (name == "is_binary" || name == "offset" || name == "len") {
      continue;
    }
    const int32_t value_count =
        static_cast<int32_t>(proto_parameter.has_bool_param()) +
        static_cast<int32_t>(proto_parameter.has_int64_param()) +
        static_cast<int32_t>(proto_parameter.has_string_param()) +
        static_cast<int32_t>(proto_parameter.has_double_param()) +
        static_cast<int32_t>(proto_parameter.has_uint64_param()) +
        static_cast<int32_t>(proto_parameter.int64_tensor_param_size() > 0);
    if (value_count != 1) {
      LOG(ERROR) << "Tensor parameter must contain exactly one value: " << name;
      return std::nullopt;
    }
    if (proto_parameter.has_bool_param()) {
      parameters.emplace(name, proto_parameter.bool_param());
    } else if (proto_parameter.has_int64_param()) {
      parameters.emplace(name, proto_parameter.int64_param());
    } else if (proto_parameter.has_string_param()) {
      parameters.emplace(name, proto_parameter.string_param());
    } else if (proto_parameter.has_double_param()) {
      parameters.emplace(name, proto_parameter.double_param());
    } else if (proto_parameter.has_uint64_param()) {
      parameters.emplace(name, proto_parameter.uint64_param());
    } else {
      parameters.emplace(
          name,
          std::vector<int64_t>(proto_parameter.int64_tensor_param().begin(),
                               proto_parameter.int64_tensor_param().end()));
    }
  }
  return parameters;
}

bool tensor_parameters_to_proto(const TensorParameters& parameters,
                                proto::Tensor* proto_tensor) {
  if (proto_tensor == nullptr) {
    return false;
  }
  for (const auto& [name, value] : parameters) {
    proto::Parameter& proto_parameter =
        (*proto_tensor->mutable_parameters())[name];
    if (const bool* typed_value = std::get_if<bool>(&value)) {
      proto_parameter.set_bool_param(*typed_value);
    } else if (const int64_t* typed_value = std::get_if<int64_t>(&value)) {
      proto_parameter.set_int64_param(*typed_value);
    } else if (const std::string* typed_value =
                   std::get_if<std::string>(&value)) {
      proto_parameter.set_string_param(*typed_value);
    } else if (const double* typed_value = std::get_if<double>(&value)) {
      proto_parameter.set_double_param(*typed_value);
    } else if (const uint64_t* typed_value = std::get_if<uint64_t>(&value)) {
      proto_parameter.set_uint64_param(*typed_value);
    } else if (const std::vector<int64_t>* typed_value =
                   std::get_if<std::vector<int64_t>>(&value)) {
      proto_parameter.mutable_int64_tensor_param()->Reserve(
          static_cast<int32_t>(typed_value->size()));
      for (const int64_t item : *typed_value) {
        proto_parameter.add_int64_tensor_param(item);
      }
    } else {
      LOG(ERROR) << "Unsupported tensor parameter: " << name;
      return false;
    }
  }
  return true;
}

void DiTMediaSources::add(std::string name,
                          std::string modality,
                          torch::Tensor tensor,
                          TensorParameters parameters) {
  entries_.emplace_back(MediaNamedTensor{.name = std::move(name),
                                         .modality = std::move(modality),
                                         .tensor = std::move(tensor),
                                         .parameters = std::move(parameters)});
}

void DiTMediaSources::add_decoded(std::vector<MediaNamedTensor> decoded) {
  entries_.reserve(entries_.size() + decoded.size());
  entries_.insert(entries_.end(),
                  std::make_move_iterator(decoded.begin()),
                  std::make_move_iterator(decoded.end()));
}

std::vector<torch::Tensor> DiTMediaSources::get(
    const std::vector<std::string>& names) const {
  std::vector<torch::Tensor> tensors;
  tensors.reserve(names.empty() ? entries_.size() : names.size());
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
  for (size_t index = 0; index < tensor_count; ++index) {
    tensors.emplace_back(entries_[index].tensor);
  }
  return tensors;
}

bool DiTMediaSources::contains(std::string_view name) const {
  return std::any_of(
      entries_.begin(), entries_.end(), [name](const MediaNamedTensor& source) {
        return source.name == name;
      });
}

MediaNamedTensor& DiTMediaSources::at(size_t index) {
  return entries_.at(index);
}

const MediaNamedTensor& DiTMediaSources::at(size_t index) const {
  return entries_.at(index);
}

std::optional<torch::Tensor> DiTMediaSources::first_tensor() const {
  if (entries_.empty()) {
    return std::nullopt;
  }
  return entries_.front().tensor;
}

std::vector<MediaNamedTensor>& DiTMediaSources::entries() { return entries_; }

const std::vector<MediaNamedTensor>& DiTMediaSources::entries() const {
  return entries_;
}

size_t DiTMediaSources::size() const { return entries_.size(); }

bool DiTMediaSources::batch_signature_matches(
    const DiTMediaSources& other) const {
  if (entries_.size() != other.entries_.size()) {
    return false;
  }
  for (size_t index = 0; index < entries_.size(); ++index) {
    const MediaNamedTensor& lhs = entries_[index];
    const MediaNamedTensor& rhs = other.entries_[index];
    if (lhs.name != rhs.name || lhs.modality != rhs.modality ||
        lhs.parameters != rhs.parameters ||
        !tensor_batch_signature_matches(lhs.tensor, rhs.tensor)) {
      return false;
    }
  }
  return true;
}

DiTMediaSources DiTMediaSources::to(const torch::Device& device) const {
  DiTMediaSources result;
  result.entries_.reserve(entries_.size());
  for (const MediaNamedTensor& source : entries_) {
    const torch::ScalarType dtype =
        source.modality == "audio" ? torch::kFloat32 : torch::kUInt8;
    result.add(source.name,
               source.modality,
               source.tensor.to(device, dtype),
               source.parameters);
  }
  return result;
}

void DiTTensorSources::add(std::string name,
                           torch::Tensor tensor,
                           TensorParameters parameters) {
  entries_.emplace_back(NamedTensor{.name = std::move(name),
                                    .tensor = std::move(tensor),
                                    .parameters = std::move(parameters)});
}

bool DiTTensorSources::contains(std::string_view name) const {
  return get_named_tensor(name).has_value();
}

std::optional<torch::Tensor> DiTTensorSources::get(
    std::string_view name) const {
  const std::optional<NamedTensorConstRef> found = get_named_tensor(name);
  if (!found.has_value()) {
    return std::nullopt;
  }
  return found->get().tensor;
}

std::optional<NamedTensorConstRef> DiTTensorSources::get_named_tensor(
    std::string_view name) const {
  for (const NamedTensor& input : entries_) {
    if (input.name == name) {
      return std::cref(input);
    }
  }
  return std::nullopt;
}

std::vector<NamedTensor>& DiTTensorSources::entries() { return entries_; }

const std::vector<NamedTensor>& DiTTensorSources::entries() const {
  return entries_;
}

size_t DiTTensorSources::size() const { return entries_.size(); }

bool DiTTensorSources::batch_signature_matches(
    const DiTTensorSources& other) const {
  if (entries_.size() != other.entries_.size()) {
    return false;
  }
  for (const NamedTensor& input : entries_) {
    const std::optional<NamedTensorConstRef> other_input =
        other.get_named_tensor(input.name);
    if (!other_input.has_value() ||
        input.parameters != other_input->get().parameters ||
        !tensor_batch_signature_matches(input.tensor,
                                        other_input->get().tensor)) {
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
    result.add(input.name, input.tensor.to(device, dtype), input.parameters);
  }
  return result;
}

}  // namespace xllm
