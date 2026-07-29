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

#include <memory>
#include <vector>

#include "framework/kv_cache/kv_cache_tensor_role.h"

namespace xllm {

// Controls only the physical allocation of tensors selected by the KV cache
// factory. Implementations must preserve the requested role, shape, and dtype.
class KVCacheTensorAllocator {
 public:
  virtual ~KVCacheTensorAllocator() = default;

  virtual torch::Tensor allocate(KVCacheTensorRole role,
                                 const std::vector<int64_t>& shape,
                                 torch::ScalarType dtype,
                                 const torch::Device& device) = 0;
};

std::shared_ptr<KVCacheTensorAllocator> default_kv_tensor_allocator();

#if defined(USE_MLU)
std::shared_ptr<KVCacheTensorAllocator> mlu_mooncake_tensor_allocator();
#endif

}  // namespace xllm
