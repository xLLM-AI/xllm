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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache_utils.h"
namespace xllm {

class KVCacheShape;

class KVCacheImpl {
 public:
  KVCacheImpl() = default;
  explicit KVCacheImpl(const KVCacheTensors& tensors);
  KVCacheImpl(const KVCacheShape& kv_cache_shape,
              const KVCacheCreateOptions& create_options);
  KVCacheImpl(const KVCacheShape& kv_cache_shape,
              const KVCacheCreateOptions& create_options,
              BlockType type,
              int64_t layer_count);

  virtual ~KVCacheImpl() = default;

  virtual torch::Tensor get_k_cache() const;
  virtual torch::Tensor get_v_cache() const;
  virtual std::optional<torch::Tensor> get_k_cache_scale() const;
  virtual std::optional<torch::Tensor> get_v_cache_scale() const;
  virtual std::optional<torch::Tensor> get_indexer_cache_scale() const;
  virtual torch::Tensor get_index_cache() const;
  virtual torch::Tensor get_conv_cache() const;
  virtual torch::Tensor get_ssm_cache() const;
  virtual torch::Tensor get_swa_cache() const;
  virtual torch::Tensor get_compress_kv_state() const;
  virtual torch::Tensor get_compress_score_state() const;
  virtual torch::Tensor get_compress_index_kv_state() const;
  virtual torch::Tensor get_compress_index_score_state() const;
  virtual torch::Tensor get_compress_state() const;
  virtual torch::Tensor get_compress_index_state() const;
  virtual std::vector<KVCacheTensor> get_cache_tensors() const;

  // Host prefix cache extraction by block type. Unlike the device-side
  // build_block_type_tensor_map in the transfer layer, this carries no
  // device-pool discriminator guards: a per-type host cache holds exactly the
  // tensors for that block type.
  virtual BlockTypeTensorMap get_block_type_tensors(BlockType type) const;

  virtual bool empty() const;

  virtual std::vector<std::vector<int64_t>> get_shapes() const;

  virtual void swap_blocks(torch::Tensor& src_tensor,
                           torch::Tensor& dst_tensor);

 protected:
  void create_host_tensor(const std::vector<int64_t>& dims,
                          torch::ScalarType dtype,
                          torch::Tensor* tensor,
                          std::vector<int64_t>* shape);

  std::vector<HostPageAlignedRegion> host_page_aligned_regions_;
  torch::Tensor key_cache_;
  torch::Tensor value_cache_;
  std::vector<int64_t> key_cache_shape_;
  std::vector<int64_t> value_cache_shape_;
};

}  // namespace xllm
