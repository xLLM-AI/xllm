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

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <optional>

// C++<->Python (pybind11) boundary helpers. Include only from translation
// units that already require the Torch Python type caster.

namespace xllm {

// Releases a pybind object with the GIL held; if the interpreter is already
// finalized (static teardown) it is leaked instead of running an off-GIL
// dec_ref that would abort with "pybind11 PyGILState_Check() failure".
inline void clear_python_object(pybind11::object& object) {
  if (!object) {
    return;
  }
  if (!Py_IsInitialized()) {
    (void)object.release();
    return;
  }
  pybind11::gil_scoped_acquire gil;
  object = pybind11::object();
}

inline pybind11::object optional_tensor(const torch::Tensor& tensor) {
  return tensor.defined() ? pybind11::cast(tensor) : pybind11::none();
}

inline pybind11::object optional_tensor(
    const std::optional<torch::Tensor>& tensor) {
  return tensor.has_value() ? optional_tensor(*tensor) : pybind11::none();
}

inline torch::Tensor tensor_from_python(const pybind11::object& value) {
  return value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
}

}  // namespace xllm
