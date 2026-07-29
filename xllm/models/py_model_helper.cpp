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

// Infrastructure for the embedded Python model executor:
// - Interpreter lifecycle (ensure_python_interpreter)
// - Weight loading (PyStateDict + PYBIND11_EMBEDDED_MODULE)
// - Config serialization (dtype_to_string, PyDictVisitor)

#include "models/py_model_helper.h"

#include <Python.h>
#include <c10/util/Exception.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/extension.h>

#include <cstdlib>
#include <mutex>
#include <string>

#include "core/framework/config/model_config.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kernels/xllm_torch_ops.h"

namespace py = pybind11;

namespace xllm {

namespace {

void prepend_sys_path(const std::string& dir) {
  if (dir.empty()) {
    return;
  }
  py::module_ sys = py::module_::import("sys");
  py::list path = py::reinterpret_borrow<py::list>(sys.attr("path"));
  for (auto item : path) {
    if (py::isinstance<py::str>(item) && item.cast<std::string>() == dir) {
      return;
    }
  }
  path.attr("insert")(0, py::str(dir));
}

}  // namespace

// ---------------------------------------------------------------------------
// dtype_to_string
// ---------------------------------------------------------------------------

std::string dtype_to_string(const torch::TensorOptions& options) {
  switch (c10::typeMetaToScalarType(options.dtype())) {
    case torch::kBFloat16:
      return "bfloat16";
    case torch::kFloat16:
      return "float16";
    case torch::kFloat32:
      return "float32";
    case torch::kFloat64:
      return "float64";
    default:
      return "bfloat16";
  }
}

// ---------------------------------------------------------------------------
// ensure_python_interpreter
// ---------------------------------------------------------------------------

void ensure_python_interpreter() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    ensure_xllm_torch_ops_registered();

    const bool we_initialized = !Py_IsInitialized();
    if (we_initialized) {
#if defined(USE_NPU)
      setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 0);
#endif
      py::initialize_interpreter(/*init_signal_handlers=*/false);
    }

    {
      py::gil_scoped_acquire gil;
      std::string model_path = ModelConfig::get_instance().python_model_path();
      if (model_path.empty()) {
        const char* env = std::getenv("XLLM_PYTHON_MODEL_PATH");
        if (env != nullptr) {
          model_path = env;
        }
      }
      prepend_sys_path(model_path);
#if defined(USE_NPU)
      if (we_initialized) {
        py::module_::import("xllm.python._npu_bootstrap");
      }
#endif
      try {
        py::module_::import("xllm.python");
      } catch (const py::error_already_set& e) {
        LOG(FATAL) << "Failed to import the 'xllm.python' model package for "
                      "the Python model executor. Set --python_model_path (or "
                      "XLLM_PYTHON_MODEL_PATH) to the directory containing the "
                      "'xllm' package. Error: "
                   << e.what();
      }
    }

    if (we_initialized) {
      PyEval_SaveThread();
    }
  });
}

// ---------------------------------------------------------------------------
// PyStateDict
// ---------------------------------------------------------------------------

torch::Tensor PyStateDict::get_tensor(const std::string& name) const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  return sd_->get_tensor(name);
}

bool PyStateDict::has(const std::string& name) const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  return sd_->has(name);
}

py::list PyStateDict::keys() const {
  CHECK(sd_ != nullptr) << "PyStateDict: access after release";
  py::list result;
  for (const auto& [key, _] : *sd_) {
    result.append(py::str(key));
  }
  return result;
}

PYBIND11_EMBEDDED_MODULE(xllm_weight_loader, m) {
  py::class_<PyStateDict>(m, "StateDict")
      .def("get_tensor", &PyStateDict::get_tensor, py::arg("name"))
      .def("has", &PyStateDict::has, py::arg("name"))
      .def("keys", &PyStateDict::keys);
}

}  // namespace xllm
