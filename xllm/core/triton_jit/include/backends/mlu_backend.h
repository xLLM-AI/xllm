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

#include <cn_api.h>
#include <cnrt.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "backend.h"
#include "core/common/macros.h"

namespace xllm::triton_jit {

struct MluMeta {
  uint32_t shared = 0;
  int32_t num_warps = 1;
  bool promote_shared = false;
  uint32_t arch = 0;  // ISA version; 0 means "unknown / skip check"
};

class MluCompiledKernel final : public CompiledKernel {
 public:
  MluCompiledKernel(std::string dir, std::string name);
  ~MluCompiledKernel() override;

  DISALLOW_COPY_AND_ASSIGN(MluCompiledKernel);

  void launch(void* stream, Grid g, const void* const* args) override;

 private:
  void lazy_load();

  std::string dir_;
  std::string name_;
  std::once_flag load_once_;
  CNmodule module_ = nullptr;
  CNkernel kernel_ = nullptr;
  MluMeta meta_;
  int32_t core_per_cluster_ = 1;
  int32_t cluster_count_ = 1;
};

class MluBackend final : public TritonBackend {
 public:
  const char* name() const override { return "mlu"; }
  int32_t current_device() const override;
  std::string compile_options_json() const override;

  std::unique_ptr<CompiledKernel> load_kernel(
      const std::string& cache_dir,
      const std::string& name) const override;
  void sync(void* stream) const override;
  bool is_capturing(void* stream) const override;
};

}  // namespace xllm::triton_jit
