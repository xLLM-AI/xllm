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

#include "core/kv_cache/storage/kv_cache_memory_registry.h"

#include <glog/logging.h>

namespace xllm {

void KVCacheMemoryRegistry::init(const torch::Device& device,
                                 size_t page_size) {
  CHECK_GT(page_size, 0);
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    CHECK_EQ(device_, device) << "KV cache registry device mismatch";
    CHECK_EQ(page_size_, page_size) << "KV cache registry page size mismatch";
    return;
  }
  device_ = device;
  page_size_ = page_size;
  initialized_ = true;
}

std::vector<torch::Tensor> KVCacheMemoryRegistry::create_k_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(initialized_) << "KV cache registry is not initialized";
  CHECK(!model_id.empty());
  return regions_[model_id].create_k_tensors(
      dims, dtype, num_layers, device_, page_size_);
}

std::vector<torch::Tensor> KVCacheMemoryRegistry::create_v_tensors(
    const std::string& model_id,
    const std::vector<int64_t>& dims,
    torch::Dtype dtype,
    int64_t num_layers) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(initialized_) << "KV cache registry is not initialized";
  CHECK(!model_id.empty());
  return regions_[model_id].create_v_tensors(
      dims, dtype, num_layers, device_, page_size_);
}

bool KVCacheMemoryRegistry::map(const std::string& model_id,
                                const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = regions_.find(model_id);
  if (it == regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return false;
  }
  return it->second.map(offsets);
}

bool KVCacheMemoryRegistry::unmap(const std::string& model_id,
                                  const std::vector<offset_t>& offsets) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = regions_.find(model_id);
  if (it == regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return false;
  }
  return it->second.unmap(offsets);
}

std::optional<int64_t> KVCacheMemoryRegistry::num_layers(
    const std::string& model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = regions_.find(model_id);
  if (it == regions_.end()) {
    return std::nullopt;
  }
  return it->second.num_layers();
}

std::pair<uint64_t, uint64_t>
KVCacheMemoryRegistry::get_global_offsets_for_block(const std::string& model_id,
                                                    int64_t layer_id,
                                                    int64_t block_id,
                                                    size_t block_size) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = regions_.find(model_id);
  if (it == regions_.end()) {
    LOG(ERROR) << "Model " << model_id << " has no KV cache regions";
    return {UINT64_MAX, UINT64_MAX};
  }
  return it->second.get_global_offsets_for_block(
      layer_id, block_id, block_size);
}

void KVCacheMemoryRegistry::erase(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  regions_.erase(model_id);
}

void KVCacheMemoryRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  regions_.clear();
  initialized_ = false;
  page_size_ = 0;
}

}  // namespace xllm
