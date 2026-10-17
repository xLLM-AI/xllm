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

#include "core/kv_cache/storage/virtual_kv_cache_tensor_allocator.h"

#include <glog/logging.h>

#include <utility>
#include <vector>

#include "core/kv_cache/storage/virtual_kv_cache_memory_registry.h"

namespace xllm {
namespace {

class VirtualKVCacheTensorAllocator final : public KVCacheTensorAllocator {
 public:
  VirtualKVCacheTensorAllocator(VirtualKVCacheMemoryRegistry& registry,
                                std::string model_id,
                                int64_t num_layers,
                                std::vector<bool> index_layer_owned)
      : registry_(registry),
        model_id_(std::move(model_id)),
        num_layers_(num_layers),
        index_layer_owned_(std::move(index_layer_owned)) {
    CHECK(!model_id_.empty());
    CHECK_GT(num_layers_, 0);
    CHECK(index_layer_owned_.empty() ||
          index_layer_owned_.size() == static_cast<size_t>(num_layers_))
        << "INDEX cache ownership mask must match num_layers";
  }

  torch::Tensor allocate(KVCacheTensorRole role,
                         const std::vector<int64_t>& shape,
                         torch::ScalarType dtype,
                         const torch::Device& device) override {
    CHECK(role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE ||
          role == KVCacheTensorRole::INDEX)
        << "Virtual memory KV cache supports only KEY, VALUE and INDEX tensors";
    auto& tensors = role == KVCacheTensorRole::KEY     ? k_tensors_
                    : role == KVCacheTensorRole::VALUE ? v_tensors_
                                                       : index_tensors_;
    size_t& layer_id = role == KVCacheTensorRole::KEY     ? next_k_layer_
                       : role == KVCacheTensorRole::VALUE ? next_v_layer_
                                                          : next_index_layer_;
    if (tensors.empty()) {
      if (role == KVCacheTensorRole::KEY) {
        tensors =
            registry_.create_k_tensors(model_id_, shape, dtype, num_layers_);
      } else if (role == KVCacheTensorRole::VALUE) {
        tensors =
            registry_.create_v_tensors(model_id_, shape, dtype, num_layers_);
      } else {
        tensors = registry_.create_index_tensors(
            model_id_, shape, dtype, num_layers_, index_layer_owned_);
      }
    }
    while (layer_id < tensors.size() && !tensors[layer_id].defined()) {
      ++layer_id;
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
  VirtualKVCacheMemoryRegistry& registry_;
  std::string model_id_;
  int64_t num_layers_;
  std::vector<bool> index_layer_owned_;
  size_t next_k_layer_ = 0;
  size_t next_v_layer_ = 0;
  size_t next_index_layer_ = 0;
  std::vector<torch::Tensor> k_tensors_;
  std::vector<torch::Tensor> v_tensors_;
  std::vector<torch::Tensor> index_tensors_;
};

}  // namespace

std::shared_ptr<KVCacheTensorAllocator>
create_virtual_kv_cache_tensor_allocator(VirtualKVCacheMemoryRegistry& registry,
                                         std::string model_id,
                                         int64_t num_layers,
                                         std::vector<bool> index_layer_owned) {
  return std::make_shared<VirtualKVCacheTensorAllocator>(
      registry, std::move(model_id), num_layers, std::move(index_layer_owned));
}

}  // namespace xllm
