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

#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

namespace xllm {

// Base class for Mooncake-based KV cache transfer.
// Default and XTensor subclasses inherit this class (single inheritance).
class MooncakeKVCacheTransferBase : public KVCacheTransfer {
 public:
  MooncakeKVCacheTransferBase(const int32_t device_id,
                              const uint16_t listen_port,
                              const torch::Device& device,
                              std::unique_ptr<MooncakeTransferEngine> engine);
  ~MooncakeKVCacheTransferBase() override = default;

  void initialize(int32_t device_id) override;

  void get_cache_info(uint64_t& cluster_id, std::string& addr) override;

  bool link_cluster(const uint64_t cluster_id,
                    const std::string& remote_addr,
                    const uint16_t port) override;

  bool unlink_cluster(const uint64_t& cluster_id,
                      const std::string& remote_addr,
                      const uint16_t port,
                      bool force_flag = false) override;

 protected:
  std::string addr_;
  uint64_t cluster_id_;
  uint16_t listen_port_;
  int32_t device_id_;
  torch::Device device_;
  int64_t num_layers_ = 0;
  int64_t size_per_block_ = 0;

  std::unique_ptr<MooncakeTransferEngine> mooncake_te_;
};

class MooncakeKVCacheTransferDefault final
    : public MooncakeKVCacheTransferBase {
 public:
  MooncakeKVCacheTransferDefault(const int32_t device_id,
                                 const uint16_t listen_port,
                                 const torch::Device& device,
                                 const std::string& model_type);
  MooncakeKVCacheTransferDefault(
      const int32_t device_id,
      const uint16_t listen_port,
      const torch::Device& device,
      const std::string& model_type,
      std::unique_ptr<MooncakeTransferEngine> engine);

  void allocate_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                         const int64_t num_layers,
                         const KVCacheShape& kv_cache_shape,
                         torch::ScalarType dtype) override;

  void allocate_kv_cache_spec(std::vector<xllm::KVCache>& kv_caches,
                              const int64_t num_layers,
                              const KVCacheShape& kv_cache_shape,
                              torch::ScalarType dtype) override;

  void register_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                         const KVCacheShape& kv_cache_shape,
                         const torch::ScalarType dtype) override;

  void register_kv_cache_spec(std::vector<xllm::KVCache>& kv_caches,
                              const KVCacheShape& kv_cache_shape,
                              torch::ScalarType dtype) override;

  bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids,
      const std::vector<uint64_t>& dst_linear_state_ids) override;

  void merge_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args) override;

  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) override;

 private:
  // Mooncake assigns buffer ids in registration order. Main KV cache registers
  // first, then the speculative draft KV cache is appended after it.
  struct BufLayout {
    // Number of layers owned by this layout.
    int64_t num_layers = 0;
    // Starting buffer id of this layout in the Mooncake registration table.
    int64_t offset = 0;
    // Registration-order offsets for each layer plus one terminal offset.
    // Shared DSA layers omit indexer buffers, so counts may vary by layer.
    std::vector<int64_t> layer_offsets;
    // Legacy uniform buffers-per-layer view. It is zero when buffer counts
    // vary. Keep this for callers that construct a uniform layout directly.
    int64_t buf_cnt = 0;
    // Total buffers registered by this layout.
    int64_t total_buf_cnt = 0;
    // True after the corresponding KV cache memory has been registered.
    bool registered = false;
  };

  enum class RegisterLengthPolicy : int8_t {
    LOGICAL_BYTES = 0,
    RDMA_REGISTERABLE_BYTES = 1,
  };

  void allocate_kv_cache_impl(std::vector<xllm::KVCache>& kv_caches,
                              int64_t num_layers,
                              const KVCacheShape& kv_cache_shape,
                              torch::ScalarType dtype);

  void add_buf(const torch::Tensor& tensor,
               std::vector<void*>& addrs,
               std::vector<size_t>& lens,
               std::vector<uint64_t>& buf_bytes,
               RegisterLengthPolicy register_length_policy =
                   RegisterLengthPolicy::LOGICAL_BYTES) const;
  std::vector<int64_t> get_buf_ids(const std::vector<int64_t>& layer_ids,
                                   bool is_spec_draft) const;
  std::vector<int64_t> get_buf_ids(const std::vector<int64_t>& layer_ids,
                                   const BufLayout& layout) const;

  // Register per-layer K/V tensor memory.
  void register_kv_cache_impl(const std::vector<xllm::KVCache>& kv_caches);

  bool has_v_cache_ = true;
  BufLayout main_layout_;
  BufLayout spec_layout_;
  std::string model_type_;
};

class MooncakeKVCacheTransferXTensor final
    : public MooncakeKVCacheTransferBase {
 public:
  MooncakeKVCacheTransferXTensor(const int32_t device_id,
                                 const uint16_t listen_port,
                                 const torch::Device& device);

  void set_model_id(const std::string& model_id) { model_id_ = model_id; }

  void allocate_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                         const int64_t num_layers,
                         const KVCacheShape& kv_cache_shape,
                         torch::ScalarType dtype) override;

  void register_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                         const KVCacheShape& kv_cache_shape,
                         const torch::ScalarType dtype) override;

  bool pull_kv_blocks(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids,
      const std::vector<uint64_t>& dst_linear_state_ids) override;

  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) override;

 private:
  void allocate_kv_cache_impl(std::vector<xllm::KVCache>& kv_caches,
                              int64_t num_layers,
                              const KVCacheShape& kv_cache_shape,
                              torch::ScalarType dtype);

  // Register GlobalXTensor memory region.
  void register_kv_cache_impl();

  bool pull_kv_blocks_impl(const std::string& src_addr,
                           const std::vector<uint64_t>& src_blocks,
                           const std::vector<uint64_t>& dst_blocks);

  bool push_kv_blocks_impl(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      int32_t kv_split_rank,
      int32_t kv_split_size);

  std::string model_id_;
};

}  // namespace xllm
