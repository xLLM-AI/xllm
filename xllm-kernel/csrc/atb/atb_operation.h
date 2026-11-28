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

#include <atb/atb_infer.h>
#include <c10/util/Logging.h>
#include <torch/types.h>
#include <torch_npu/csrc/core/npu/NPUGraphsUtils.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "csrc/atb/atb_context.h"

namespace xllm_kernel {

class AtbOperation final {
 public:
  template <typename Param>
  AtbOperation(aclrtStream stream, const Param& param)
      : context_(get_context(stream)) {
    CHECK((atb::CreateOperation(param, &operation_)) == 0);
  }

  ~AtbOperation() { CHECK(atb::DestroyOperation(operation_) == 0); }

  AtbOperation(const AtbOperation&) = delete;
  AtbOperation& operator=(const AtbOperation&) = delete;

  uint64_t setup(const atb::VariantPack& pack) {
    uint64_t workspace_size = 0;
    CHECK((operation_->Setup(pack, workspace_size, context_->get())) == 0);
    return workspace_size;
  }

  void execute(const atb::VariantPack& pack,
               void* workspace,
               uint64_t workspace_size) {
    CHECK((operation_->Execute(pack,
                               static_cast<uint8_t*>(workspace),
                               workspace_size,
                               context_->get())) == 0);
  }

 private:
  std::shared_ptr<AtbContext> context_;
  atb::Operation* operation_ = nullptr;
};

// Each Param has its own eager cache. The caller supplies a variant key when
// different parameter values create different operations of the same type.
template <typename Param>
void run_atb_operation(const char* name,
                       const Param& param,
                       int32_t variant,
                       atb::VariantPack pack,
                       std::vector<torch::Tensor> tensors) {
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
  bool capturing = static_cast<int32_t>(
                       c10_npu::currentStreamCaptureStatusMayInitCtx()) != 0;
  if (capturing) {
    // Setup and allocation must use the capture thread and its graph pool.
    auto operation = std::make_shared<AtbOperation>(stream, param);
    uint64_t workspace_size = operation->setup(pack);
    torch::Tensor workspace =
        torch::empty({static_cast<int64_t>(workspace_size)},
                     tensors.front().options().dtype(torch::kUInt8));
    auto call = [operation,
                 pack = std::move(pack),
                 tensors = std::move(tensors),
                 workspace,
                 workspace_size]() -> int {
      operation->execute(pack, workspace.data_ptr(), workspace_size);
      return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(name, call);
    return;
  }

  auto call = [param,
               variant,
               stream,
               pack = std::move(pack),
               tensors = std::move(tensors)]() -> int {
    // ATB setup mutates the operation, so eager submissions reuse one only
    // on the same execution thread and stream.
    static thread_local std::unordered_map<
        aclrtStream,
        std::unordered_map<int32_t, std::unique_ptr<AtbOperation>>>
        cache;
    auto& operation = cache[stream][variant];
    if (!operation) {
      operation = std::make_unique<AtbOperation>(stream, param);
    }
    uint64_t workspace_size = operation->setup(pack);
    torch::Tensor workspace;
    if (workspace_size != 0) {
      workspace = at_npu::native::allocate_workspace(workspace_size, stream);
    }
    operation->execute(pack,
                       workspace.defined() ? workspace.data_ptr() : nullptr,
                       workspace_size);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(name, call);
}

}  // namespace xllm_kernel
