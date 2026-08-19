/* Copyright 2025-2026 The xLLM Authors.

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

#include "atb_buffer.h"

#include <acl/acl.h>
#include <atb/types.h>
#include <atb_speed/utils/timer.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>

#include "xllm_atb_layers/core/include/atb_speed/utils/statistic.h"
#include "xllm_atb_layers/pytorch/adapter/utils/utils.h"

namespace xllm {

constexpr uint64_t KB_1 = 1024;
constexpr uint64_t MB_1 = 1024 * 1024;
constexpr uint64_t GB_1 = 1024 * 1024 * 1024;
constexpr uint64_t DIM_NUM_2 = 2;

AtbBuffer::AtbBuffer(uint64_t buffer_size, at::Device device)
    : buffer_size_(buffer_size), device_(device) {
  buffer_size_ = buffer_size;

  options_ = at::TensorOptions()
                 .dtype(at::ScalarType::Byte)
                 .layout(torch::kStrided)
                 .requires_grad(false)
                 .device(device_);

  if (buffer_size_ > 0) {
    at_tensor_ = create_attensor(buffer_size_);
  }
}

AtbBuffer::~AtbBuffer() = default;

void* AtbBuffer::get_buffer(uint64_t buffer_size) {
  if (buffer_size <= buffer_size_) {
    return at_tensor_.data_ptr();
  }

  aclrtSynchronizeStream(
      c10_npu::getCurrentNPUStream(device_.index()).stream());

  at_tensor_.reset();
  at_tensor_ = create_attensor(buffer_size);
  buffer_size_ = uint64_t(at_tensor_.numel());

  return at_tensor_.data_ptr();
}

torch::Tensor AtbBuffer::create_attensor(uint64_t buffer_size) const {
  at::Tensor newTensor = at_npu::native::empty_with_format(
      at::IntArrayRef({static_cast<int64_t>(KB_1),
                       static_cast<int64_t>(buffer_size / KB_1 + 1)}),
      options_,
      ACL_FORMAT_ND);

  if (!newTensor.is_contiguous()) {
    newTensor = newTensor.contiguous();
  }

  return newTensor;
}

}  // namespace xllm
