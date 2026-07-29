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

#include "framework/kv_cache/kv_cache_tensor_allocator.h"

#if defined(USE_MLU)
#include "platform/mlu/mlu_tensor_alloc.h"
#endif

namespace xllm {
namespace {

class DefaultKVCacheTensorAllocator final : public KVCacheTensorAllocator {
 public:
  torch::Tensor allocate(KVCacheTensorRole /*role*/,
                         const std::vector<int64_t>& shape,
                         torch::ScalarType dtype,
                         const torch::Device& device) override {
    return torch::zeros(shape, torch::dtype(dtype).device(device));
  }
};

#if defined(USE_MLU)
class MluMooncakeTensorAllocator final : public KVCacheTensorAllocator {
 public:
  torch::Tensor allocate(KVCacheTensorRole role,
                         const std::vector<int64_t>& shape,
                         torch::ScalarType dtype,
                         const torch::Device& device) override {
    // INDEX_SCALE is currently the only Mooncake role whose registered length
    // can exceed its logical tensor bytes. Extend this branch explicitly if a
    // future role requires RDMA padding.
    if (role == KVCacheTensorRole::INDEX_SCALE) {
      return mlu::alloc_rdma_registerable_zero_tensor(shape, dtype, device);
    }
    return mlu::alloc_zero_tensor(shape, dtype, device);
  }
};
#endif

}  // namespace

std::shared_ptr<KVCacheTensorAllocator> default_kv_tensor_allocator() {
  static std::shared_ptr<KVCacheTensorAllocator> allocator =
      std::make_shared<DefaultKVCacheTensorAllocator>();
  return allocator;
}

#if defined(USE_MLU)
std::shared_ptr<KVCacheTensorAllocator> mlu_mooncake_tensor_allocator() {
  return std::make_shared<MluMooncakeTensorAllocator>();
}
#endif

}  // namespace xllm
