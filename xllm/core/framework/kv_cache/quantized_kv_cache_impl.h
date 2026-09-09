/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/kv_cache/kv_cache_impl.h"

namespace xllm {

class QuantizedKVCacheImpl final : public KVCacheImpl {
 public:
  explicit QuantizedKVCacheImpl(const QuantizedKVCacheTensors& tensors);
  QuantizedKVCacheImpl(const KVCacheShape& kv_cache_shape,
                       const KVCacheCreateOptions& create_options);

  std::optional<torch::Tensor> get_k_cache_scale() const override;
  std::optional<torch::Tensor> get_v_cache_scale() const override;

  void swap_blocks(torch::Tensor& src_tensor,
                   torch::Tensor& dst_tensor) override;

 private:
  // scale tensors for quantized KV cache (int8)
  torch::Tensor key_cache_scale_;
  torch::Tensor value_cache_scale_;
};

}  // namespace xllm
