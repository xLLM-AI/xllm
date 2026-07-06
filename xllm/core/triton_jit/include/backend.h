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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "spec.h"
#include "types.h"

namespace xllm::triton_jit {

class CompiledKernel {
 public:
  virtual ~CompiledKernel() = default;

  // Launch on `stream` (opaque, backend-native) with packed runtime args.
  virtual void launch(void* stream, Grid g, const void* const* args) = 0;
};

class TritonBackend {
 public:
  virtual ~TritonBackend() = default;

  virtual const char* name() const = 0;
  virtual int32_t current_device() const = 0;
  virtual std::string compile_options_json() const = 0;

  virtual std::string compile(const std::string& path,
                              const std::string& name,
                              const SpecList& specs,
                              const LaunchCfg& cfg,
                              int32_t dev);

  // Default autotune search space for this backend. Override when a backend's
  // launch model restricts the viable {num_warps, num_stages} set (e.g. MLU's
  // num_warps ∈ {1,4,8,16,32} where 16/32 usually overflow SRAM).
  virtual std::vector<LaunchCfg> default_autotune_space() const;

  virtual std::unique_ptr<CompiledKernel> load_kernel(
      const std::string& cache_dir,
      const std::string& name) const = 0;

  virtual void sync(void* stream) const = 0;
  virtual bool is_capturing(void* stream) const { return false; }
};

inline constexpr const char* kTritonCompileModule =
    "xllm.core.triton_jit.scripts.triton_compile";

// Lazily initialize the embedded Python interpreter on first use. Safe to call
// from any thread; guarded by std::call_once. Callers must still acquire the
// GIL themselves before touching Python state.
void ensure_embedded_interpreter();

// Ensure the embedded interpreter can import the packaged Triton compile
// helper from a build-tree or installed xllm package. Callers must hold the
// GIL.
void prepare_triton_compile_import_path();

TritonBackend& get_backend();

}  // namespace xllm::triton_jit
