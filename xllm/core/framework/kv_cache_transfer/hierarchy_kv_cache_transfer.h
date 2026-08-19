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
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "common/types.h"
#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_input_params.h"
#include "platform/batch_memcpy.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

namespace xllm {

class KVCacheStore;

using HostGroupedCaches = std::map<BlockType, std::unique_ptr<KVCache>>;

enum class CacheParticipant : int8_t {
  TARGET = 0,
  DRAFT = 1,
};

struct HostCacheComponentSchema {
  CacheParticipant participant = CacheParticipant::TARGET;
  BlockType block_type = BlockType::KV;
  std::string model_identity;
  std::string schema_fingerprint;
  uint32_t tp_rank = 0;
  uint32_t tp_size = 1;
};

class HostCacheSliceProvider {
 public:
  virtual ~HostCacheSliceProvider() = default;

  virtual std::vector<HostCacheComponentSchema> store_components() const = 0;
  virtual std::vector<torch::Tensor> component_storage_tensors(
      CacheParticipant participant,
      BlockType block_type) const = 0;
  virtual std::vector<torch::Tensor> component_tensors(
      CacheParticipant participant,
      BlockType block_type,
      int32_t host_block_id) const = 0;
};

enum class HostPayloadCoverage : int8_t {
  REQUIRED = 0,
  NOT_APPLICABLE = 1,
  REGENERATED = 2,
};

enum class HostEntryState : int8_t {
  EMPTY = 0,
  WRITING = 1,
  READY = 2,
  INVALID = 3,
};

class HierarchyKVCacheTransfer final : public HostCacheSliceProvider {
 public:
  struct LayerBatchRange {
    int64_t begin_layer = 0;
    int64_t end_layer = 0;
  };

  struct CopyPlan {
    std::vector<torch::Tensor> src_tensors;
    std::vector<torch::Tensor> dst_tensors;
  };

  using GroupedCaches = std::map<BlockType, std::vector<KVCache*>>;

  struct Options {
    PROPERTY(uint32_t, tp_rank);
    PROPERTY(uint32_t, tp_size);
    PROPERTY(uint32_t, layers);
    PROPERTY(double, host_blocks_factor) = 0.0;
    PROPERTY(uint32_t, layers_wise_copy_batchs) = 1;
    PROPERTY(bool, enable_mla) = false;
    PROPERTY(bool, enable_kvcache_store) = false;
    PROPERTY(std::string, store_protocol) = "rdma";
    PROPERTY(std::string, store_master_server_address) = "";
    PROPERTY(std::string, store_metadata_server) = "";
    PROPERTY(std::string, store_local_hostname) = "";
    PROPERTY(std::string, store_namespace) = "";
    PROPERTY(uint32_t, store_worker_id) = 0;
  };

  struct HostCacheTensorSpec {
    CacheParticipant participant = CacheParticipant::TARGET;
    BlockType block_type = BlockType::KV;
    int64_t absolute_layer_id = 0;
    int64_t host_layer_slot = 0;
    KVCacheTensorRole::Value role = KVCacheTensorRole::KEY;
    HostPayloadCoverage coverage = HostPayloadCoverage::REQUIRED;
    torch::ScalarType dtype = torch::kFloat32;
    std::vector<int64_t> block_shape;
  };

  struct ParticipantHostCacheSchema {
    std::vector<HostCacheTensorSpec> tensor_specs;
    std::map<BlockType, std::string> component_fingerprints;
  };

  struct ParticipantRegistration {
    CacheParticipant participant = CacheParticipant::TARGET;
    const Stream* actual_compute_stream = nullptr;
    std::vector<KVCache>* device_caches = nullptr;
    KVCacheShape cache_shape;
    KVCacheCreateOptions create_options;
    std::string model_identity;
    uint32_t tp_rank = 0;
    uint32_t tp_size = 1;
  };

  struct HostCacheEntryMetadata {
    uint64_t schema_fingerprint = 0;
    uint64_t generation = 0;
    uint32_t required_participant_mask = 0;
    uint32_t completed_participant_mask = 0;
    HostEntryState state = HostEntryState::EMPTY;
  };

  HierarchyKVCacheTransfer(const Options& options, const torch::Device& device);
  HierarchyKVCacheTransfer(const Options& options,
                           const torch::Device& device,
                           const Stream* compute_stream,
                           std::vector<KVCache>* kv_caches_ptr,
                           const KVCacheShape& kv_cache_shape,
                           const KVCacheCreateOptions& create_options);
  ~HierarchyKVCacheTransfer() override;

  void register_participant(ParticipantRegistration registration);
  bool finalize_registration();

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  uint32_t transfer_kv_blocks(uint64_t batch_id,
                              Slice<BlockTransferInfo>& block_transfer_info);

  std::vector<uint8_t> prefetch_kv_blocks(
      Slice<BlockTransferInfo>& block_transfer_info);

  void set_layer_synchronizer(ModelInputParams& params);
  void set_layer_synchronizer(CacheParticipant participant,
                              ModelInputParams& params);

  std::vector<HostCacheComponentSchema> store_components() const override;
  std::vector<torch::Tensor> component_storage_tensors(
      CacheParticipant participant,
      BlockType block_type) const override;
  std::vector<torch::Tensor> component_tensors(
      CacheParticipant participant,
      BlockType block_type,
      int32_t host_block_id) const override;

 private:
  friend class HierarchyKVCacheTransferTestPeer;

  enum class RegistrationState : int8_t {
    CREATED = 0,
    REGISTERING = 1,
    READY = 2,
  };

  struct ParticipantState {
    ParticipantRegistration registration;
    GroupedCaches device_grouped_caches;
    std::map<BlockType, std::vector<int64_t>> absolute_layer_ids;
    HostGroupedCaches host_grouped_caches;
    std::vector<LayerBatchRange> layer_batch_ranges;
    ParticipantHostCacheSchema schema;
  };

  struct LoadTransaction {
    std::mutex mutex;
    std::map<CacheParticipant, std::shared_ptr<LayerSynchronizer>>
        synchronizers;
    std::map<std::pair<BlockType, int32_t>, uint64_t> entry_generations;
    uint32_t required_participant_mask = 0;
    uint32_t consumed_participant_mask = 0;
    bool aborted = false;

    void abort();
  };

  static uint32_t participant_mask(CacheParticipant participant);
  static uint64_t schema_fingerprint_value(const std::string& fingerprint);

  void initialize_resources();
  void build_participant_state(ParticipantState& state);
  void build_device_block_type_map(ParticipantState& state);
  void create_host_cache(ParticipantState& state);
  void build_and_validate_schema(ParticipantState& state);
  void validate_composite_schema() const;
  void initialize_store();

  CopyPlan build_copy_plan(
      const ParticipantState& state,
      const std::vector<BlockTransferInfo>& block_transfer_info,
      const LayerBatchRange& layer_batch_range) const;

  uint32_t offload(const std::vector<BlockTransferInfo>& block_transfer_info);
  bool offload_to_host(Slice<BlockTransferInfo>& block_transfer_info);
  bool load_from_host(
      CacheParticipant participant,
      const std::shared_ptr<LoadTransaction>& transaction,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  uint32_t required_participant_mask(BlockType block_type) const;
  uint64_t composite_schema_fingerprint(BlockType block_type) const;
  bool participant_requires_type(const ParticipantState& state,
                                 BlockType block_type) const;
  bool begin_entry_write(
      const std::vector<BlockTransferInfo>& block_transfer_info);
  void complete_participant_write(
      CacheParticipant participant,
      const std::vector<BlockTransferInfo>& block_transfer_info);
  void commit_entry_write(
      const std::vector<BlockTransferInfo>& block_transfer_info);
  void abort_entry_write(
      const std::vector<BlockTransferInfo>& block_transfer_info);
  bool snapshot_ready_entries(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      LoadTransaction* transaction) const;
  bool entry_snapshot_matches(
      const std::vector<BlockTransferInfo>& block_transfer_info,
      const LoadTransaction& transaction) const;
  void update_prefetched_entries(Slice<BlockTransferInfo>& block_transfer_info,
                                 const std::vector<uint8_t>& statuses);

 private:
  Options options_;
  Device device_;
  RegistrationState registration_state_ = RegistrationState::CREATED;
  std::map<CacheParticipant, ParticipantState> participant_states_;

  std::unique_ptr<ThreadPool> load_threadpool_;
  moodycamel::BlockingConcurrentQueue<std::unique_ptr<Stream>> copy_stream_;
  std::unique_ptr<BatchMemcpy> batch_memcpy_;
  std::unique_ptr<KVCacheStore> kv_cache_store_;

  mutable std::mutex mutex_;
  std::map<std::pair<BlockType, int32_t>, HostCacheEntryMetadata>
      entry_metadata_;
  std::unordered_map<uint64_t, std::shared_ptr<LoadTransaction>>
      load_transactions_;
};

}  // namespace xllm
