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

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/runtime/dit_forward_params.h"
namespace xllm {

class DiTModel : public torch::nn::Module {
 public:
  ~DiTModel() override = default;

  virtual DiTForwardOutput forward(const DiTForwardInput& input) = 0;
  virtual torch::Device device() const = 0;
  virtual const torch::TensorOptions& options() const = 0;
  virtual void load_model(std::unique_ptr<DiTModelLoader> loader) = 0;
};

template <typename Model>
class DiTModelImpl : public DiTModel {
 public:
  DiTModelImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}
  DiTForwardOutput forward(const DiTForwardInput& input) override {
    return model_->forward(input);
  }
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void load_model(std::unique_ptr<DiTModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

 private:
  Model model_;
  torch::TensorOptions options_;
};
}  // namespace xllm
