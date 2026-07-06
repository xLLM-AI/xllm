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

#include <dlfcn.h>
#include <pybind11/embed.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#if defined(USE_MLU)
#include "backends/mlu_backend.h"
#endif

namespace py = pybind11;

namespace xllm::triton_jit {

namespace {

bool has_triton_compile_helper(const std::filesystem::path& python_root) {
  std::error_code ec;
  return std::filesystem::exists(python_root / "xllm" / "core" / "triton_jit" /
                                     "scripts" / "triton_compile.py",
                                 ec);
}

std::filesystem::path normalize_path(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::path normalized =
      std::filesystem::weakly_canonical(path, ec);
  if (!ec) {
    return normalized;
  }
  normalized = std::filesystem::absolute(path, ec);
  if (!ec) {
    return normalized;
  }
  return path;
}

std::filesystem::path current_object_path() {
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<const void*>(&current_object_path), &dl_info) ==
          0 ||
      dl_info.dli_fname == nullptr || dl_info.dli_fname[0] == '\0') {
    return {};
  }
  return normalize_path(std::filesystem::path(dl_info.dli_fname));
}

std::string current_python_build_tag() {
  py::module_ sysconfig = py::module_::import("sysconfig");
  std::string version =
      sysconfig.attr("get_python_version")().cast<std::string>();
  version.erase(std::remove(version.begin(), version.end(), '.'),
                version.end());
  return "cpython-" + version;
}

std::optional<std::filesystem::path> find_build_lib_root(
    const std::filesystem::path& build_dir,
    const std::string& python_build_tag,
    bool require_python_build_tag) {
  std::error_code ec;
  if (!std::filesystem::is_directory(build_dir, ec)) {
    return std::nullopt;
  }

  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(build_dir, ec)) {
    if (ec) {
      return std::nullopt;
    }
    if (!entry.is_directory(ec)) {
      continue;
    }

    const std::filesystem::path candidate = entry.path();
    const std::string dirname = candidate.filename().string();
    if (dirname.rfind("lib.", /*pos=*/0) != 0) {
      continue;
    }
    if (require_python_build_tag &&
        dirname.find(python_build_tag) == std::string::npos) {
      continue;
    }
    if (has_triton_compile_helper(candidate)) {
      return normalize_path(candidate);
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path> find_xllm_python_root(
    const std::filesystem::path& object_path) {
  if (object_path.empty()) {
    return std::nullopt;
  }

  const std::string python_build_tag = current_python_build_tag();
  for (std::filesystem::path dir = object_path.parent_path(); !dir.empty();
       dir = dir.parent_path()) {
    if (dir.filename() == "xllm") {
      const std::filesystem::path candidate = dir.parent_path();
      if (has_triton_compile_helper(candidate)) {
        return normalize_path(candidate);
      }
    }

    const std::optional<std::filesystem::path> exact_build_root =
        find_build_lib_root(dir / "build",
                            python_build_tag,
                            /*require_python_build_tag=*/true);
    if (exact_build_root.has_value()) {
      return exact_build_root;
    }

    const std::optional<std::filesystem::path> nested_exact_build_root =
        find_build_lib_root(
            dir, python_build_tag, /*require_python_build_tag=*/true);
    if (nested_exact_build_root.has_value()) {
      return nested_exact_build_root;
    }

    if (has_triton_compile_helper(dir)) {
      return normalize_path(dir);
    }

    const std::optional<std::filesystem::path> any_build_root =
        find_build_lib_root(dir / "build",
                            python_build_tag,
                            /*require_python_build_tag=*/false);
    if (any_build_root.has_value()) {
      return any_build_root;
    }

    if (dir == dir.root_path()) {
      break;
    }
  }
  return std::nullopt;
}

bool python_path_contains(py::handle python_path, const std::string& path) {
  for (py::handle item : python_path) {
    if (py::str(item).cast<std::string>() == path) {
      return true;
    }
  }
  return false;
}

void prepend_sys_path(const std::filesystem::path& python_root) {
  const std::string root = python_root.string();
  py::module_ sys = py::module_::import("sys");
  py::list sys_path = sys.attr("path");
  if (!python_path_contains(sys_path, root)) {
    sys_path.attr("insert")(/*index=*/0, root);
  }
}

void prepend_loaded_package_path(const char* module_name,
                                 const std::filesystem::path& package_path) {
  std::error_code ec;
  if (!std::filesystem::exists(package_path, ec)) {
    return;
  }

  py::module_ sys = py::module_::import("sys");
  py::dict modules = sys.attr("modules");
  if (!modules.contains(module_name)) {
    return;
  }

  py::object module = modules[module_name];
  if (!py::hasattr(module, "__path__")) {
    return;
  }

  const std::string path = package_path.string();
  py::object current_path = module.attr("__path__");
  if (python_path_contains(current_path, path)) {
    return;
  }

  py::list new_path;
  new_path.append(path);
  for (py::handle item : current_path) {
    new_path.append(item);
  }
  module.attr("__path__") = new_path;
}

void prepend_loaded_xllm_package_paths(
    const std::filesystem::path& python_root) {
  prepend_loaded_package_path("xllm", python_root / "xllm");
  prepend_loaded_package_path("xllm.core", python_root / "xllm" / "core");
  prepend_loaded_package_path("xllm.core.triton_jit",
                              python_root / "xllm" / "core" / "triton_jit");
  prepend_loaded_package_path(
      "xllm.core.triton_jit.scripts",
      python_root / "xllm" / "core" / "triton_jit" / "scripts");
  prepend_loaded_package_path("xllm.core.kernels",
                              python_root / "xllm" / "core" / "kernels");
  prepend_loaded_package_path(
      "xllm.core.kernels.mlu",
      python_root / "xllm" / "core" / "kernels" / "mlu");
  prepend_loaded_package_path(
      "xllm.core.kernels.mlu.triton_kernel",
      python_root / "xllm" / "core" / "kernels" / "mlu" / "triton_kernel");
}

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

void prepare_triton_compile_import_path() {
  static std::once_flag import_path_once;
  std::call_once(import_path_once, []() {
    const std::filesystem::path object_path = current_object_path();
    const std::optional<std::filesystem::path> python_root =
        find_xllm_python_root(object_path);
    if (!python_root.has_value()) {
      return;
    }
    prepend_sys_path(*python_root);
    prepend_loaded_xllm_package_paths(*python_root);
  });
}

std::string TritonBackend::compile(const std::string& path,
                                   const std::string& name,
                                   const SpecList& specs,
                                   const LaunchCfg& cfg,
                                   int32_t dev) {
  ensure_embedded_interpreter();
  py::gil_scoped_acquire gil;
  prepare_triton_compile_import_path();
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
