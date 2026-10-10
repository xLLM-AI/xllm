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

#include "core/framework/allocator/kv_cache/paged_kv_cache_tensor_allocator.h"

#include <glog/logging.h>

#include <utility>
#include <vector>

#include "core/framework/allocator/model_memory_manager.h"

namespace xllm {
namespace {

class PagedKVCacheTensorAllocator final : public KVCacheTensorAllocator {
 public:
  PagedKVCacheTensorAllocator(std::string model_id, int64_t num_layers)
      : model_id_(std::move(model_id)), num_layers_(num_layers) {
    CHECK(!model_id_.empty());
    CHECK_GT(num_layers_, 0);
  }

  torch::Tensor allocate(KVCacheTensorRole role,
                         const std::vector<int64_t>& shape,
                         torch::ScalarType dtype,
                         const torch::Device& device) override {
    CHECK(role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE)
        << "Virtual memory KV cache supports only key and value tensors.";
    const bool is_key = role == KVCacheTensorRole::KEY;
    auto& tensors = is_key ? k_tensors_ : v_tensors_;
    size_t& layer_id = is_key ? next_k_layer_ : next_v_layer_;
    if (tensors.empty()) {
      auto& manager = ModelMemoryManager::get_instance();
      tensors =
          is_key
              ? manager.create_k_tensors(model_id_, shape, dtype, num_layers_)
              : manager.create_v_tensors(model_id_, shape, dtype, num_layers_);
    }
    CHECK_LT(layer_id, tensors.size())
        << "Virtual memory KV cache allocator cannot be reused across models.";
    torch::Tensor tensor = tensors[layer_id++];
    CHECK_EQ(tensor.device(), device);
    CHECK_EQ(tensor.scalar_type(), dtype);
    CHECK(tensor.sizes().vec() == shape);
    return tensor;
  }

 private:
  std::string model_id_;
  int64_t num_layers_;
  size_t next_k_layer_ = 0;
  size_t next_v_layer_ = 0;
  std::vector<torch::Tensor> k_tensors_;
  std::vector<torch::Tensor> v_tensors_;
};

}  // namespace

std::shared_ptr<KVCacheTensorAllocator> create_paged_kv_cache_tensor_allocator(
    std::string model_id,
    int64_t num_layers) {
  return std::make_shared<PagedKVCacheTensorAllocator>(std::move(model_id),
                                                       num_layers);
}

}  // namespace xllm
