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
#include <unordered_set>
#include <vector>

#include "core/common/property_reflect.h"
#include "core/framework/state_dict/state_dict.h"

namespace xllm {

// Initializes the embedded CPython interpreter (idempotent, process-wide).
void ensure_python_interpreter();

// Convert torch dtype to the string form used by Python model config.
std::string dtype_to_string(const torch::TensorOptions& options);

// PropertyVisitor that writes each field into a pybind11 dict.
class __attribute__((visibility("hidden"))) PyDictVisitor final
    : public PropertyVisitor {
 public:
  explicit PyDictVisitor(pybind11::dict& dict) : dict_(dict) {}

  void visit(const std::string& name, bool value) override { set(name, value); }
  void visit(const std::string& name, int32_t value) override {
    set(name, value);
  }
  void visit(const std::string& name, int64_t value) override {
    set(name, value);
  }
  void visit(const std::string& name, float value) override {
    set(name, value);
  }
  void visit(const std::string& name, double value) override {
    set(name, value);
  }
  void visit(const std::string& name, const std::string& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<int32_t>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<int64_t>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<float>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<double>& value) override {
    set(name, value);
  }
  void visit(const std::string& name, const std::vector<bool>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<std::string>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::unordered_set<int32_t>& value) override {
    set(name, value);
  }
  void visit_absent(const std::string& name) override {
    dict_[pybind11::str(name)] = pybind11::none();
  }

 private:
  template <typename T>
  void set(const std::string& name, const T& value) {
    dict_[pybind11::str(name)] = value;
  }

  pybind11::dict& dict_;
};

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
