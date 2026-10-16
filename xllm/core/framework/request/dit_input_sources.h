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

#pragma once

#include <torch/torch.h>

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/util/tensor_helper.h"

namespace xllm::proto {
class Tensor;
}

namespace xllm {

std::optional<TensorParameters> proto_to_tensor_parameters(
    const proto::Tensor& proto_tensor);
bool tensor_parameters_to_proto(const TensorParameters& parameters,
                                proto::Tensor* proto_tensor);

struct NamedTensor {
  std::string name;
  torch::Tensor tensor;
  TensorParameters parameters;
};

using NamedTensorConstRef = std::reference_wrapper<const NamedTensor>;

struct MediaNamedTensor {
  std::string name;
  std::string modality;
  torch::Tensor tensor;
  TensorParameters parameters;
};

class DiTMediaSources final {
 public:
  void add(std::string name,
           std::string modality,
           torch::Tensor tensor,
           TensorParameters parameters = {});

  void add_decoded(std::vector<MediaNamedTensor> decoded);

  std::vector<torch::Tensor> get(
      const std::vector<std::string>& names = {}) const;

  bool contains(std::string_view name) const;

  MediaNamedTensor& at(size_t index);
  const MediaNamedTensor& at(size_t index) const;

  // First media tensor, or no value when there is none.
  std::optional<torch::Tensor> first_tensor() const;

  std::vector<MediaNamedTensor>& entries();
  const std::vector<MediaNamedTensor>& entries() const;

  size_t size() const;

  bool batch_signature_matches(const DiTMediaSources& other) const;
  DiTMediaSources to(const torch::Device& device) const;

 private:
  std::vector<MediaNamedTensor> entries_;
};

class DiTTensorSources final {
 public:
  void add(std::string name,
           torch::Tensor tensor,
           TensorParameters parameters = {});

  bool contains(std::string_view name) const;
  std::optional<torch::Tensor> get(std::string_view name) const;
  std::optional<NamedTensorConstRef> get_named_tensor(
      std::string_view name) const;

  std::vector<NamedTensor>& entries();
  const std::vector<NamedTensor>& entries() const;

  size_t size() const;

  bool batch_signature_matches(const DiTTensorSources& other) const;
  DiTTensorSources to(const torch::Device& device,
                      torch::ScalarType dtype) const;

 private:
  std::vector<NamedTensor> entries_;
};

}  // namespace xllm
