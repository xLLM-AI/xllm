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

#include "framework/kv_cache/linear_attention_kv_cache_impl.h"

#include "framework/kv_cache/kv_cache_shape.h"
#include "util/tensor_helper.h"

namespace xllm {

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const LinearAttentionKVCacheTensors& tensors)
    : conv_cache_(tensors.conv_cache),
      ssm_cache_(tensors.ssm_cache),
      conv_cache_shape_(get_tensor_shape(tensors.conv_cache)),
      ssm_cache_shape_(get_tensor_shape(tensors.ssm_cache)) {}

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : LinearAttentionKVCacheImpl(
          create_linear_attention_kv_cache_tensors(kv_cache_shape,
                                                   create_options)) {
  conv_cache_shape_ = kv_cache_shape.conv_cache_shape();
  ssm_cache_shape_ = kv_cache_shape.ssm_cache_shape();
}

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    BlockType type,
    int64_t layer_count)
    : KVCacheImpl() {
  CHECK(type == BlockType::SINGLE)
      << "LinearAttentionKVCacheImpl host cache only supports "
         "BlockType::SINGLE.";
  host_page_aligned_regions_.reserve(2);
  if (kv_cache_shape.has_conv_cache_shape()) {
    create_host_tensor(
        build_host_group_tensor_shape(kv_cache_shape.conv_cache_shape(),
                                      create_options.host_blocks_factor(),
                                      layer_count),
        create_options.dtype(),
        &conv_cache_,
        &conv_cache_shape_);
  }
  if (kv_cache_shape.has_ssm_cache_shape()) {
    create_host_tensor(
        build_host_group_tensor_shape(kv_cache_shape.ssm_cache_shape(),
                                      create_options.host_blocks_factor(),
                                      layer_count),
        create_options.ssm_dtype(),
        &ssm_cache_,
        &ssm_cache_shape_);
  }
}

torch::Tensor LinearAttentionKVCacheImpl::get_conv_cache() const {
  return conv_cache_;
}

torch::Tensor LinearAttentionKVCacheImpl::get_ssm_cache() const {
  return ssm_cache_;
}

BlockTypeTensorMap LinearAttentionKVCacheImpl::get_block_type_tensors(
    BlockType type) const {
  BlockTypeTensorMap tensor_map;
  if (type != BlockType::SINGLE) {
    return tensor_map;
  }
  if (conv_cache_.defined() && conv_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::CONV, conv_cache_);
  }
  if (ssm_cache_.defined() && ssm_cache_.numel() > 0) {
    tensor_map.emplace(KVCacheTensorRole::SSM, ssm_cache_);
  }
  return tensor_map;
}

bool LinearAttentionKVCacheImpl::empty() const {
  return !conv_cache_.defined() || !ssm_cache_.defined();
}

std::vector<std::vector<int64_t>> LinearAttentionKVCacheImpl::get_shapes()
    const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(2);
  shapes.emplace_back(conv_cache_shape_);
  shapes.emplace_back(ssm_cache_shape_);
  return shapes;
}

}  // namespace xllm
