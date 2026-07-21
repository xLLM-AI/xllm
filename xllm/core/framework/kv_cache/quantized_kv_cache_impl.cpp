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

#include "framework/kv_cache/quantized_kv_cache_impl.h"

#include "framework/kv_cache/kv_cache_shape.h"

namespace xllm {

QuantizedKVCacheImpl::QuantizedKVCacheImpl(
    const QuantizedKVCacheTensors& tensors)
    : KVCacheImpl(tensors.kv_cache_tensors),
      key_cache_scale_(tensors.key_cache_scale),
      value_cache_scale_(tensors.value_cache_scale) {}

QuantizedKVCacheImpl::QuantizedKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : QuantizedKVCacheImpl(
          create_quantized_kv_cache_tensors(kv_cache_shape, create_options)) {}

std::optional<torch::Tensor> QuantizedKVCacheImpl::get_k_cache_scale() const {
  if (!key_cache_scale_.defined() || key_cache_scale_.numel() == 0) {
    return std::nullopt;
  }
  return key_cache_scale_;
}

std::optional<torch::Tensor> QuantizedKVCacheImpl::get_v_cache_scale() const {
  if (!value_cache_scale_.defined() || value_cache_scale_.numel() == 0) {
    return std::nullopt;
  }
  return value_cache_scale_;
}

void QuantizedKVCacheImpl::swap_blocks(torch::Tensor& src_tensor,
                                       torch::Tensor& dst_tensor) {
  torch::Tensor selected_keys = torch::index_select(key_cache_, 0, src_tensor);
  key_cache_.index_copy_(0, dst_tensor, selected_keys);
  if (value_cache_.defined() && value_cache_.numel() > 0) {
    torch::Tensor selected_values =
        torch::index_select(value_cache_, 0, src_tensor);
    value_cache_.index_copy_(0, dst_tensor, selected_values);
  }

  // batch copy scale tensors
  if (key_cache_scale_.defined() && key_cache_scale_.numel() > 0) {
    torch::Tensor selected_k_scales =
        torch::index_select(key_cache_scale_, 0, src_tensor);
    key_cache_scale_.index_copy_(0, dst_tensor, selected_k_scales);
  }
  if (value_cache_scale_.defined() && value_cache_scale_.numel() > 0) {
    torch::Tensor selected_v_scales =
        torch::index_select(value_cache_scale_, 0, src_tensor);
    value_cache_scale_.index_copy_(0, dst_tensor, selected_v_scales);
  }
}

}  // namespace xllm
