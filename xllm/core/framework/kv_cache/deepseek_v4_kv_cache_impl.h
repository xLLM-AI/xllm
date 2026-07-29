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

#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache_impl.h"

namespace xllm {

class KVCacheShape;

class Dsv4StateCache final {
 public:
  Dsv4StateCache() = default;

  static Dsv4StateCache from_split(torch::Tensor kv, torch::Tensor score);
  static Dsv4StateCache from_packed(torch::Tensor packed,
                                    torch::Tensor fallback_kv,
                                    torch::Tensor fallback_score);

  torch::Tensor kv() const;
  torch::Tensor score() const;
  torch::Tensor packed() const;

  void swap_blocks(const torch::Tensor& src, const torch::Tensor& dst);

 private:
  bool packed_layout_ = false;
  torch::Tensor packed_;
  torch::Tensor kv_;
  torch::Tensor score_;
};

class DeepSeekV4KVCacheImpl final : public KVCacheImpl {
 public:
  explicit DeepSeekV4KVCacheImpl(const DeepSeekV4KVCacheTensors& tensors);
  DeepSeekV4KVCacheImpl(const KVCacheShape& kv_cache_shape,
                        const KVCacheCreateOptions& create_options,
                        BlockType type,
                        int64_t layer_count);

  torch::Tensor get_k_cache() const override;
  torch::Tensor get_index_cache() const override;
  std::optional<torch::Tensor> get_indexer_cache_scale() const override;
  torch::Tensor get_swa_cache() const override;
  torch::Tensor get_compress_kv_state() const override;
  torch::Tensor get_compress_score_state() const override;
  torch::Tensor get_compress_index_kv_state() const override;
  torch::Tensor get_compress_index_score_state() const override;
  torch::Tensor get_compress_state() const override;
  torch::Tensor get_compress_index_state() const override;
  std::vector<KVCacheTensor> get_cache_tensors() const override;

  BlockTypeTensorMap get_block_type_tensors(BlockType type) const override;

  bool empty() const override;

  std::vector<std::vector<int64_t>> get_shapes() const override;

  void swap_blocks(torch::Tensor& src_tensor,
                   torch::Tensor& dst_tensor) override;

 private:
  torch::Tensor key_cache_;
  torch::Tensor index_cache_;
  torch::Tensor indexer_cache_scale_;
  torch::Tensor swa_cache_;
  Dsv4StateCache compress_state_;
  Dsv4StateCache index_state_;
  BlockType compressed_block_type_ = BlockType::KV;
};

DeepSeekV4KVCacheTensors create_dsv4_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    int64_t layer_idx);

std::string dsv4_shape_summary(const DeepSeekV4KVCacheTensors& tensors,
                               int32_t compress_ratio);

}  // namespace xllm
