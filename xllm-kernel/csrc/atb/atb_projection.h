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
#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "csrc/atb/atb_context.h"

namespace xllm_kernel {

// Keep adapter symbols private: torch_npu exports an incompatible ATB cache.
class AtbProjection final {
 public:
  explicit AtbProjection(aclrtStream stream) : context_(get_context(stream)) {
    atb::infer::LinearParam param;
    param.transposeA = false;
    param.transposeB = false;
    param.hasBias = false;
    param.enAccum = false;
    param.outDataType = ACL_DT_UNDEFINED;
    param.matmulType = atb::infer::LinearParam::MATMUL_EIN_SUM;
    CHECK_EQ(atb::CreateOperation(param, &operation_), 0);
  }

  ~AtbProjection() { CHECK_EQ(atb::DestroyOperation(operation_), 0); }

  AtbProjection(const AtbProjection&) = delete;
  AtbProjection& operator=(const AtbProjection&) = delete;

  uint64_t setup(const atb::VariantPack& pack) {
    uint64_t workspace_size = 0;
    CHECK_EQ(operation_->Setup(pack, workspace_size, context_->get()), 0);
    return workspace_size;
  }

  void execute(const atb::VariantPack& pack,
               void* workspace,
               uint64_t workspace_size) {
    CHECK_EQ(operation_->Execute(pack,
                                 static_cast<uint8_t*>(workspace),
                                 workspace_size,
                                 context_->get()),
             0);
  }

 private:
  std::shared_ptr<AtbContext> context_;
  atb::Operation* operation_ = nullptr;
};

}  // namespace xllm_kernel
