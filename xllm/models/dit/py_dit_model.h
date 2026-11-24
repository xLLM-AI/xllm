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

#pragma once

#include <pybind11/pybind11.h>

#include "core/framework/model/dit_model.h"

namespace xllm {

class DiTModelContext;

class __attribute__((visibility("hidden"))) PyDiTModel final : public DiTModel {
 public:
  explicit PyDiTModel(const DiTModelContext& context);
  ~PyDiTModel() override;

  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }

  void load_model(std::unique_ptr<DiTModelLoader> loader) override;
  DiTForwardOutput forward(const DiTModelParams& params) override;

 private:
  torch::TensorOptions options_;
  pybind11::object pipeline_;
};

}  // namespace xllm
