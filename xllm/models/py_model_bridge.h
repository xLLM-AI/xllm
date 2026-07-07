/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <string>

#include "core/framework/state_dict/state_dict.h"

namespace xllm {

// Initializes the embedded CPython interpreter (idempotent, process-wide).
void ensure_python_interpreter();

// pybind11-visible wrapper around StateDict for Python weight loading.
class PyStateDict {
 public:
  explicit PyStateDict(const StateDict* sd) : sd_(sd) {}

  torch::Tensor get_tensor(const std::string& name) const;
  bool has(const std::string& name) const;
  pybind11::list keys() const;

 private:
  const StateDict* sd_;
};

}  // namespace xllm
