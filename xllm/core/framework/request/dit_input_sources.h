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

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xllm {

struct NamedTensor {
  std::string name;
  torch::Tensor tensor;
};

class DiTImageSources final {
 public:
  void add(std::string name, torch::Tensor tensor);

  std::vector<torch::Tensor> get(
      const std::vector<std::string>& names = {}) const;

  NamedTensor& at(size_t index);
  const NamedTensor& at(size_t index) const;

  std::vector<NamedTensor>& entries();
  const std::vector<NamedTensor>& entries() const;

  size_t size() const;
  bool empty() const;

  bool batch_signature_matches(const DiTImageSources& other) const;
  DiTImageSources to(const torch::Device& device) const;

 private:
  std::vector<NamedTensor> entries_;
};

class DiTTensorSources final {
 public:
  void add(std::string name, torch::Tensor tensor);

  bool contains(std::string_view name) const;
  std::optional<torch::Tensor> get(std::string_view name) const;

  std::vector<NamedTensor>& entries();
  const std::vector<NamedTensor>& entries() const;

  size_t size() const;
  bool empty() const;

  bool batch_signature_matches(const DiTTensorSources& other) const;
  DiTTensorSources to(const torch::Device& device,
                      torch::ScalarType dtype) const;

 private:
  std::vector<NamedTensor> entries_;
};

}  // namespace xllm
