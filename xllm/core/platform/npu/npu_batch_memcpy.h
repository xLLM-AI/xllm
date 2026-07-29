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

#include <acl/acl.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/macros.h"
#include "platform/batch_memcpy.h"
#include "platform/stream.h"

namespace xllm {
namespace npu {

class NPUBatchMemcpy final : public BatchMemcpy {
 public:
  NPUBatchMemcpy() = default;
  ~NPUBatchMemcpy() override = default;

  void init(int32_t device_id) override;

  bool copy_h2d(const std::vector<torch::Tensor>& src_tensors,
                const std::vector<torch::Tensor>& dst_tensors,
                Stream* stream) override;

  bool copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                const std::vector<torch::Tensor>& dst_tensors,
                Stream* stream) override;

 private:
  static constexpr size_t kMaxBatchCopyCount = 4096;

  DISALLOW_COPY_AND_ASSIGN(NPUBatchMemcpy);

  bool copy(const std::vector<torch::Tensor>& src_tensors,
            const std::vector<torch::Tensor>& dst_tensors,
            const aclrtMemcpyBatchAttr& attr,
            Stream* stream);

  bool initialized_ = false;
  int32_t device_id_ = -1;
  aclrtMemcpyBatchAttr h2d_attr_{};
  aclrtMemcpyBatchAttr d2h_attr_{};
};

}  // namespace npu
}  // namespace xllm
