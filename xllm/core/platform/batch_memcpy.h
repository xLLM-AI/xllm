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

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "platform/stream.h"

namespace xllm {

class Device;

class BatchMemcpy {
 public:
  virtual ~BatchMemcpy() = default;

  virtual void init(int32_t device_id) = 0;

  virtual bool copy_h2d(const std::vector<torch::Tensor>& src_tensors,
                        const std::vector<torch::Tensor>& dst_tensors,
                        Stream* stream) = 0;

  virtual bool copy_d2h(const std::vector<torch::Tensor>& src_tensors,
                        const std::vector<torch::Tensor>& dst_tensors,
                        Stream* stream) = 0;
};

std::unique_ptr<BatchMemcpy> create_batch_memcpy(const Device& device);

}  // namespace xllm
