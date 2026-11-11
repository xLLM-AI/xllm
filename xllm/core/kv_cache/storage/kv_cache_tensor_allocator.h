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

#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/kv_cache/layout/kv_cache_tensor_role.h"

namespace xllm {

// Controls only the physical allocation of tensors selected by the KV cache
// factory. Implementations preserve the role, inner row shape, and dtype. A
// backend using physical index translation may expose extra leading rows.
class KVCacheTensorAllocator {
 public:
  virtual ~KVCacheTensorAllocator() = default;

  virtual void begin_layer(int64_t /*layer_id*/) {}

  virtual torch::Tensor allocate(KVCacheTensorRole role,
                                 const std::vector<int64_t>& shape,
                                 torch::ScalarType dtype,
                                 const torch::Device& device) = 0;
};

std::shared_ptr<KVCacheTensorAllocator> default_kv_tensor_allocator();

}  // namespace xllm
