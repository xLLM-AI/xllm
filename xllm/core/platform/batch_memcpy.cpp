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

#include "platform/batch_memcpy.h"

#include "platform/device.h"

#if defined(USE_NPU)
#include "platform/npu/npu_batch_memcpy.h"
#endif

namespace xllm {

std::unique_ptr<BatchMemcpy> create_batch_memcpy(const Device& device) {
#if defined(USE_NPU)
  auto memcpy = std::make_unique<npu::NPUBatchMemcpy>();
  memcpy->init(device.index());
  return memcpy;
#else
  (void)device;
  return nullptr;
#endif
}

}  // namespace xllm
