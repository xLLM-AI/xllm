/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "backend.h"

#include <pybind11/embed.h>

#include <cstdint>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#if defined(USE_MLU)
#include "backends/mlu_backend.h"
#endif

namespace py = pybind11;

namespace xllm::triton_jit {

namespace {

std::once_flag& embedded_init_flag() {
  static std::once_flag flag;
  return flag;
}

const char* kind_str(Kind k) {
  switch (k) {
    case Kind::PTR: {
      return "ptr";
    }
    case Kind::VAL: {
      return "val";
    }
    case Kind::CONST: {
      return "const";
    }
  }
  return "const";
}

const char* hint_str(SpecHint h) {
  switch (h) {
    case SpecHint::DIV16: {
      return "16";
    }
    case SpecHint::EQ1: {
      return "1";
    }
    default: {
      return "0";
    }
  }
}

py::list specs_to_py(const SpecList& specs) {
  py::list out;
  for (const ArgSpec& s : specs) {
    py::dict d;
    d["kind"] = kind_str(s.kind);
    d["type"] = s.type;
    d["hint"] = hint_str(s.hint);
    d["specialize"] = s.specialize;
    d["const_val"] = s.const_val;
    out.append(d);
  }
  return out;
}

py::dict options_to_py(const std::string& json_str) {
  py::dict d;
  if (json_str.empty()) {
    return d;
  }
  nlohmann::json j = nlohmann::json::parse(json_str);
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string& k = it.key();
    if (it->is_boolean()) {
      d[k.c_str()] = it->get<bool>();
    } else if (it->is_number_integer()) {
      d[k.c_str()] = it->get<int64_t>();
    } else if (it->is_number_float()) {
      d[k.c_str()] = it->get<double>();
    } else if (it->is_string()) {
      d[k.c_str()] = it->get<std::string>();
    }
  }
  return d;
}

}  // namespace

void ensure_embedded_interpreter() {
  std::call_once(embedded_init_flag(), []() {
    if (!Py_IsInitialized()) {
      py::initialize_interpreter();
    }
  });
}

std::string TritonBackend::compile(const std::string& path,
                                   const std::string& name,
                                   const SpecList& specs,
                                   const LaunchCfg& cfg,
                                   int32_t dev) {
  ensure_embedded_interpreter();
  py::gil_scoped_acquire gil;
  py::module_ mod = py::module_::import(kTritonCompileModule);
  py::list args_spec = specs_to_py(specs);
  py::dict options = options_to_py(compile_options_json());
  py::object ans = mod.attr("compile")(path,
                                       name,
                                       args_spec,
                                       this->name(),
                                       options,
                                       cfg.num_warps,
                                       cfg.num_stages,
                                       dev);
  return ans.cast<std::string>();
}

// Conservative default search space. It targets the currently implemented MLU
// backend (num_warps ∈ {1,4,8,16,32}; 16/32 usually overflow SRAM, so it sticks
// to 1/4/8). New backends should override with a space matching their launch
// model rather than relying on this.
std::vector<LaunchCfg> TritonBackend::default_autotune_space() const {
  return {{1, 1}, {1, 2}, {1, 3}, {4, 1}, {4, 2}, {8, 1}};
}

#if defined(USE_MLU)
TritonBackend& get_backend() {
  static MluBackend backend;
  return backend;
}
#elif defined(USE_CUDA)
// TODO(triton_jit): implement backends/cuda_backend.{h,cpp} (cuModuleLoad /
// cuLaunchKernel) and enable here.
#error "triton_jit: CUDA backend not yet implemented"
#elif defined(USE_NPU)
// TODO(triton_jit): implement backends/npu_backend.{h,cpp} (ACL) and enable.
#error "triton_jit: NPU backend not yet implemented"
#else
#error "triton_jit: no backend selected (set one of USE_MLU/USE_CUDA/USE_NPU)"
#endif

}  // namespace xllm::triton_jit
