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

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/macros.h"
#include "platform/batch_memcpy.h"

namespace xllm::mlu {

class MLUBatchMemcpy final : public BatchMemcpy {
 public:
  MLUBatchMemcpy() = default;
  ~MLUBatchMemcpy() override = default;

  void init(int32_t device_id) override;

  bool copy_h2d(const std::vector<torch::Tensor>& src_tensors,
                const std::vector<torch::Tensor>& dst_tensors,
                Stream* stream) override;

  bool copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                const std::vector<torch::Tensor>& dst_tensors,
                Stream* stream) override;

 private:
  enum class Direction : int8_t { H2D = 0, D2H = 1 };

  static constexpr size_t kMaxBatchCopyCount = 4096;

  bool copy(const std::vector<torch::Tensor>& src_tensors,
            const std::vector<torch::Tensor>& dst_tensors,
            Stream* stream,
            Direction direction);
  bool valid_inputs(const std::vector<torch::Tensor>& src_tensors,
                    const std::vector<torch::Tensor>& dst_tensors,
                    const Stream* stream,
                    Direction direction) const;

  DISALLOW_COPY_AND_ASSIGN(MLUBatchMemcpy);

  bool initialized_ = false;
  int32_t device_id_ = -1;
};

}  // namespace xllm::mlu
