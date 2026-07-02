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

#include "framework/kv_cache/kv_cache_impl.h"

#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "util/tensor_helper.h"

namespace xllm {

KVCacheImpl::KVCacheImpl(const KVCacheTensors& tensors)
    : key_cache_(tensors.key_cache),
      value_cache_(tensors.value_cache),
      key_cache_shape_(get_tensor_shape(tensors.key_cache)),
      value_cache_shape_(get_tensor_shape(tensors.value_cache)) {}

KVCacheImpl::KVCacheImpl(const KVCacheShape& kv_cache_shape,
                         const KVCacheCreateOptions& create_options)
    : KVCacheImpl(create_kv_cache_tensors(kv_cache_shape, create_options)) {
  key_cache_shape_ = kv_cache_shape.key_cache_shape();
  if (kv_cache_shape.has_value_cache_shape()) {
    value_cache_shape_ = kv_cache_shape.value_cache_shape();
  }
}

torch::Tensor KVCacheImpl::get_k_cache() const { return key_cache_; }

torch::Tensor KVCacheImpl::get_v_cache() const { return value_cache_; }

std::optional<torch::Tensor> KVCacheImpl::get_k_cache_scale() const {
  return std::nullopt;
}

std::optional<torch::Tensor> KVCacheImpl::get_v_cache_scale() const {
  return std::nullopt;
}

std::optional<torch::Tensor> KVCacheImpl::get_indexer_cache_scale() const {
  return std::nullopt;
}

torch::Tensor KVCacheImpl::get_index_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_conv_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_ssm_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_swa_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_compress_kv_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_score_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_index_kv_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_index_score_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_state() const {
  return torch::Tensor();
}

torch::Tensor KVCacheImpl::get_compress_index_state() const {
  return torch::Tensor();
}

bool KVCacheImpl::empty() const {
  return !key_cache_.defined() || !value_cache_.defined();
}

std::vector<std::vector<int64_t>> KVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(2);
  shapes.emplace_back(key_cache_shape_);
  shapes.emplace_back(value_cache_shape_);
  return shapes;
}

void KVCacheImpl::swap_blocks(torch::Tensor& src_tensor,
                              torch::Tensor& dst_tensor) {
  // batch select keys and values
  auto selected_keys = torch::index_select(key_cache_, 0, src_tensor);
  auto selected_values = torch::index_select(value_cache_, 0, src_tensor);

  // batch copy keys and values to dst indices
  key_cache_.index_copy_(0, dst_tensor, selected_keys);
  value_cache_.index_copy_(0, dst_tensor, selected_values);
}

}  // namespace xllm
