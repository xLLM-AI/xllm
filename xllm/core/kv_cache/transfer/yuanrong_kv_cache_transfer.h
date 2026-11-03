/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/kv_cache/transfer/kv_cache_transfer.h"

namespace xllm {

// DataSystem ("YuanRong") KV transfer backend.
//
// DataSystem is a content-addressed host-object store: prefill publishes each
// finished prompt block under a key derived from the block's token prefix,
// decode fetches the same key from its own prompt. There is no peer-to-peer
// link and no destination address, which is why the address-based fields in
// KVCacheInfo are unused here.
//
// The DataSystem SDK is never linked into this binary. Every call goes through
// a pure C ABI bridge (xllm_yr_ds_*) loaded at runtime, so a build without the
// SDK still produces a working binary that simply refuses to select this
// backend.
class YuanRongKVCacheTransfer final : public KVCacheTransfer {
 public:
  struct BridgeBlob {
    void* data;
    uint64_t size;
  };

  struct BridgeBlobList {
    int32_t device_idx;
    const BridgeBlob* blobs;
    size_t blob_count;
  };

  YuanRongKVCacheTransfer(int32_t device_id,
                          uint16_t listen_port,
                          const torch::Device& device,
                          std::string model_type,
                          std::string model_id,
                          bool publish_source_blocks);

  ~YuanRongKVCacheTransfer() override;

  void initialize(int32_t device_id) override;

  void finalize() override;

  void configure_cache_layout(const ParallelArgs& parallel_args,
                              const ModelArgs& model_args,
                              int32_t block_token_capacity,
                              bool is_spec_draft) override;

  void register_kv_cache(std::vector<xllm::KVCache>& kv_caches,
                         const KVCacheShape& kv_cache_shape,
                         torch::ScalarType dtype) override;

  void get_cache_info(uint64_t& cluster_id, std::string& addr) override;

  bool link_clusters(const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& remote_addrs,
                     const std::vector<uint16_t>& ports) override;

  bool unlink_cluster(const uint64_t& cluster_id,
                      const std::string& remote_addr,
                      const uint16_t port,
                      bool force_flag = true) override;

  bool pull_kv_blocks(const uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const std::vector<KVTransferMapping>& mappings) override;

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft,
      int32_t kv_split_rank,
      int32_t kv_split_size) override;
#endif

 private:
  using BridgeCreateFn =
      void* (*)(const char*, int32_t, int32_t, int32_t, int32_t);
  using BridgeDestroyFn = void (*)(void*);
  using BridgeMSetFn = bool (*)(void*,
                                const char* const*,
                                const BridgeBlobList*,
                                size_t);
  using BridgeMGetFn = bool (*)(void*,
                                const char* const*,
                                const BridgeBlobList*,
                                size_t,
                                int32_t);
  using BridgeLastErrorFn = const char* (*)(void*);

  // Stable identity of one published object: model, parallel topology and the
  // content hash of the block prefix.
  std::string block_key(const std::string& block_hash) const;

  // Byte ranges covering every cache tensor of `block_id` across all layers.
  std::vector<BridgeBlob> make_blob_list(uint64_t block_id) const;

  bool load_bridge();

  const char* bridge_last_error() const;

  bool ensure_initialized(bool publish_source_blocks);

  std::string model_type_;
  std::string model_id_;
  bool publish_source_blocks_ = false;
  uint16_t listen_port_ = 0;
  int32_t device_id_ = 0;
  torch::Device device_{torch::kCPU};

  int32_t dp_rank_ = 0;
  int32_t tp_rank_ = 0;
  int32_t cp_rank_ = 0;
  int32_t kv_split_rank_ = 0;
  bool layout_configured_ = false;
  // Layout identity shared by P and D: model shape plus cache layout family.
  // Distinguishes two models with the same name but different shapes, which a
  // bare model_type would conflate into one object namespace.
  std::string layout_fingerprint_;

  int64_t num_layers_ = 0;
  int64_t block_bytes_ = 0;
  std::string addr_;
  uint64_t cluster_id_ = 0;

  std::vector<xllm::KVCache>* kv_caches_ = nullptr;
  std::unordered_map<uint64_t, std::string> linked_clusters_;

  void* bridge_handle_ = nullptr;
  void* bridge_client_ = nullptr;
  BridgeCreateFn bridge_create_ = nullptr;
  BridgeDestroyFn bridge_destroy_ = nullptr;
  BridgeMSetFn bridge_mset_ = nullptr;
  BridgeMGetFn bridge_mget_ = nullptr;
  BridgeLastErrorFn bridge_last_error_ = nullptr;
};

}  // namespace xllm
