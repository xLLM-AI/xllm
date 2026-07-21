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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/macros.h"
#include "core/common/constants.h"
#include "util/tensor_helper.h"

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache_capacity.h"
#include "framework/kv_cache/kv_cache_tensor_role.h"

namespace xllm {

class KVCacheShape;

struct KVCacheCreateOptions {
  PROPERTY(torch::Device, device) = torch::Device(torch::kCPU);
  // kvcache dtype for key/value cacahe, index cache
  PROPERTY(torch::ScalarType, dtype) = torch::kBFloat16;
  // ssm dtype for linear attention layers
  PROPERTY(torch::ScalarType, ssm_dtype) = torch::kBFloat16;
  PROPERTY(double, host_blocks_factor) = 0.0;
  PROPERTY(int64_t, num_layers) = 0;
  // full attention interval for linear attention layers
  PROPERTY(int64_t, full_attention_interval) = 1;
  // model_id are required for XTensor mode
  PROPERTY(std::string, model_id);
  PROPERTY(std::string, model_type);
  PROPERTY(bool, enable_xtensor) = false;
  // RL deep-sleep mode: build KV cache over a VMM-backed SleepableAllocator
  // region so it can be released/re-acquired by sleep()/wake_up().
  PROPERTY(bool, enable_sleep_mode) = false;
  PROPERTY(bool, enable_linear_attention) = false;
  PROPERTY(bool, enable_lighting_indexer) = false;
  PROPERTY(bool, enable_kv_cache_quant) = false;
  PROPERTY(bool, enable_raw_device_allocator) = false;
#if defined(USE_NPU)
  PROPERTY(bool, enable_kv_cache_huge_page_allocator) = false;
#endif
  PROPERTY(bool, enable_indexer_cache_quant) = false;

  // DeepSeek V4 cache allocation metadata.
  PROPERTY(int64_t, block_size) = 0;
  PROPERTY(int64_t, head_dim) = 0;
  PROPERTY(int64_t, index_head_dim) = 0;
  PROPERTY(int64_t, window_size) = 0;
  PROPERTY(std::vector<int32_t>, compress_ratios);
};

struct KVCacheTensors {
  torch::Tensor key_cache;
  torch::Tensor value_cache;
};

struct IndexedKVCacheTensors {
  KVCacheTensors kv_cache_tensors;
  torch::Tensor index_cache;
  std::optional<torch::Tensor> index_cache_scale;
  std::optional<torch::Tensor> key_cache_scale;
  std::optional<torch::Tensor> value_cache_scale;
};

struct QuantizedKVCacheTensors {
  KVCacheTensors kv_cache_tensors;
  torch::Tensor key_cache_scale;
  torch::Tensor value_cache_scale;
};

struct LinearAttentionKVCacheTensors {
  torch::Tensor conv_cache;
  torch::Tensor ssm_cache;
};

struct KVCacheTensor {
  KVCacheTensorRole role;
  torch::Tensor tensor;
  int32_t group_id = cache_group_id(BlockType::KV);
  bool sequence_scoped = false;
};

using BlockTypeTensorMap = std::map<KVCacheTensorRole::Value, torch::Tensor>;

struct HostPageAlignedRegion {
  void* base_ptr = nullptr;
  size_t total_bytes = 0;

  HostPageAlignedRegion() = default;
  explicit HostPageAlignedRegion(size_t bytes);
  HostPageAlignedRegion(const HostPageAlignedRegion&) = delete;
  HostPageAlignedRegion& operator=(const HostPageAlignedRegion&) = delete;
  HostPageAlignedRegion(HostPageAlignedRegion&& other) noexcept;
  HostPageAlignedRegion& operator=(HostPageAlignedRegion&& other) noexcept;
  ~HostPageAlignedRegion();
};

struct DeepSeekV4KVCacheTensors {
  torch::Tensor key_cache;
  torch::Tensor index_cache;
  torch::Tensor indexer_cache_scale;
  torch::Tensor swa_cache;
  torch::Tensor compress_kv_state;
  torch::Tensor compress_score_state;
  torch::Tensor compress_index_kv_state;
  torch::Tensor compress_index_score_state;
#if defined(USE_MLU)
  torch::Tensor compress_state;
  torch::Tensor compress_index_state;
#endif
  BlockType compressed_block_type = BlockType::KV;
};

// for qwen3.5
bool is_linear_attention_layer(int64_t layer_idx,
                               int64_t full_attention_interval);

// Whether NPU KV cache should use FRACTAL_NZ layout for a model type.
bool use_npu_nz_kv_cache_layout(const std::string& model_type);

KVCacheTensors create_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

IndexedKVCacheTensors create_indexed_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

QuantizedKVCacheTensors create_quantized_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

LinearAttentionKVCacheTensors create_linear_attention_kv_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options);

// Scale a device block count to the host block count using host_blocks_factor
// (clamped to >= 1.0 so the host pool is never smaller than the device pool).
int64_t scale_host_block_count(int64_t block_count, double host_blocks_factor);

// Build a host tensor shape from a per-layer device shape by scaling dim 0
// (block count) by host_blocks_factor.
std::vector<int64_t> build_host_tensor_shape(
    const std::vector<int64_t>& base_shape,
    double host_blocks_factor);

// Build a grouped host tensor shape: scales dim 0 then inserts a layer
// dimension at index 1, yielding [host_blocks, layer_count, ...per_block_dims].
std::vector<int64_t> build_host_group_tensor_shape(
    const std::vector<int64_t>& base_shape,
    double host_blocks_factor,
    int64_t layer_count);

// Allocate a page-aligned, mlock'd (and NPU-registered) host tensor over a
// HostPageAlignedRegion. The region owns the memory; the tensor is a view.
void create_host_page_aligned_tensor(const std::vector<int64_t>& dims,
                                     torch::ScalarType dtype,
                                     torch::Tensor* tensor,
                                     HostPageAlignedRegion* region);

#if defined(USE_NPU)
aclFormat get_npu_kv_cache_format(const std::string& model_type);

// Allocate an NPU tensor from the huge-page device allocator. The returned
// tensor owns the ACL allocation and carries the requested NPU format.
torch::Tensor alloc_npu_huge_page_tensor(const std::vector<int64_t>& dims,
                                         torch::ScalarType dtype,
                                         aclFormat format);
#endif

}  // namespace xllm
