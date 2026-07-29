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

#pragma once

#include "framework/kv_cache/kv_cache_impl.h"

namespace xllm {

class LinearAttentionKVCacheImpl final : public KVCacheImpl {
 public:
  explicit LinearAttentionKVCacheImpl(
      const LinearAttentionKVCacheTensors& tensors);
  LinearAttentionKVCacheImpl(const KVCacheShape& kv_cache_shape,
                             const KVCacheCreateOptions& create_options);
  LinearAttentionKVCacheImpl(const KVCacheShape& kv_cache_shape,
                             const KVCacheCreateOptions& create_options,
                             BlockType type,
                             int64_t layer_count);

  torch::Tensor get_conv_cache() const override;
  torch::Tensor get_ssm_cache() const override;

  BlockTypeTensorMap get_block_type_tensors(BlockType type) const override;

  bool empty() const override;

  std::vector<std::vector<int64_t>> get_shapes() const override;

  void swap_blocks(torch::Tensor& src_tensor,
                   torch::Tensor& dst_tensor) override {
    NOT_IMPLEMENTED();
  };

 private:
  torch::Tensor conv_cache_;
  torch::Tensor ssm_cache_;
  std::vector<int64_t> conv_cache_shape_;
  std::vector<int64_t> ssm_cache_shape_;
};

}  // namespace xllm
