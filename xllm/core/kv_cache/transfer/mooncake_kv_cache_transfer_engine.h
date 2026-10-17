/* Copyright 2026 The xLLM Authors.

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

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "core/framework/transfer/mooncake_transfer_engine.h"
#include "core/kv_cache/transfer/cache_layout.h"
#include "core/kv_cache/transfer/reshard_planner.h"
#include "core/kv_cache/transfer/virtual_kv_cache_transfer_memory_provider.h"
#include "mooncake_transfer_engine.pb.h"

namespace xllm {

enum class CachePeerMode : int8_t {
  ACTIVE = 0,
  PLAN_ONLY = 1,
  ABSENT = 2,
};

class MooncakeKVCacheTransferService;

class MooncakeKVCacheTransferState final {
 public:
  static MooncakeKVCacheTransferState& get_instance();
  static std::shared_ptr<MooncakeKVCacheTransferState> shared_instance();
  ~MooncakeKVCacheTransferState();

  Status set_local_cache_layout(const WorkerCacheLayoutManifest& manifest);
  Status set_virtual_memory_provider(
      std::string model_id,
      std::shared_ptr<VirtualKVCacheTransferMemoryProvider> provider);
  Status get_virtual_offsets(
      const std::string& incarnation_id,
      uint64_t layout_generation,
      const std::vector<uint64_t>& block_ids,
      WorkerCacheLayoutManifest* manifest,
      std::vector<VirtualKVCacheLayerGlobalOffsets>* layer_offsets,
      uint64_t* physical_page_size_bytes) const;
  std::optional<WorkerCacheLayoutManifest> local_cache_layout() const;
  Status set_cache_peer(const WorkerCacheLayoutManifest& peer_manifest,
                        CachePeerMode mode);
  bool has_outgoing_plan(const std::string& remote_addr,
                         CacheNamespace cache_namespace) const;
  bool has_reshard_plan(const std::string& remote_addr) const;
  Status bind_outgoing_regions(const std::string& remote_addr,
                               const std::vector<KVTransferMapping>& mappings,
                               CacheNamespace cache_namespace,
                               int64_t layer_id,
                               std::vector<ByteRegion>* regions) const;
  Status bind_outgoing_regions_explicit(
      const std::string& remote_addr,
      const std::vector<ExplicitResourceMapping>& mappings,
      CacheNamespace cache_namespace,
      int64_t layer_id,
      std::vector<ByteRegion>* regions) const;

 private:
  MooncakeKVCacheTransferState();
  mutable std::mutex mutex_;
  MooncakeTransferEngineCore& transport_;
  struct CachePeerLink {
    std::string destination_incarnation;
    uint64_t destination_layout_generation = 0;
    CachePeerMode mode = CachePeerMode::PLAN_ONLY;
    std::optional<ReshardPlanTemplate> plan;
    bool holds_session = false;
  };
  std::optional<WorkerCacheLayoutManifest> local_cache_layout_;
  std::string virtual_memory_model_id_;
  std::shared_ptr<VirtualKVCacheTransferMemoryProvider>
      virtual_memory_provider_;
  std::unordered_map<std::string, CachePeerLink> cache_peer_links_;
};

// KV-specific adapter over the shared byte transport.
class MooncakeKVCacheTransferEngine : public MooncakeTransferEngine {
 public:
  MooncakeKVCacheTransferEngine(uint16_t listen_port,
                                const torch::Device& device);
  std::string initialize() override;
  bool link_sessions(const std::vector<uint64_t>& cluster_ids,
                     const std::vector<std::string>& remote_addrs);

  Status set_local_cache_layout(const WorkerCacheLayoutManifest& manifest);
  Status set_virtual_memory_provider(
      std::string model_id,
      std::shared_ptr<VirtualKVCacheTransferMemoryProvider> provider);
  virtual bool get_remote_virtual_offsets(
      uint64_t cluster_id,
      const std::string& remote_addr,
      const std::vector<uint64_t>& block_ids,
      WorkerCacheLayoutManifest* manifest,
      std::vector<VirtualKVCacheLayerGlobalOffsets>* layer_offsets);

  bool has_outgoing_plan(const std::string& remote_addr,
                         CacheNamespace cache_namespace) const;
  virtual bool has_reshard_plan(const std::string& remote_addr) const;

  Status bind_outgoing_regions(const std::string& remote_addr,
                               const std::vector<KVTransferMapping>& mappings,
                               CacheNamespace cache_namespace,
                               int64_t layer_id,
                               std::vector<ByteRegion>* regions) const;
  Status bind_outgoing_regions_explicit(
      const std::string& remote_addr,
      const std::vector<ExplicitResourceMapping>& mappings,
      CacheNamespace cache_namespace,
      int64_t layer_id,
      std::vector<ByteRegion>* regions) const;

  bool close_session(uint64_t cluster_id, const std::string& remote_addr);

  // Keep a remote data session alive for the complete READ operation.
  virtual bool move_memory_regions_with_session(
      const std::string& remote_addr,
      const std::vector<ByteRegion>& regions,
      MoveOpcode move_opcode);

 protected:
  virtual bool fetch_cache_layout(uint64_t cluster_id,
                                  const std::string& remote_addr,
                                  WorkerCacheLayoutManifest* manifest);
  virtual bool set_remote_peer(uint64_t cluster_id,
                               const std::string& remote_addr,
                               const WorkerCacheLayoutManifest& manifest,
                               CachePeerMode mode);
  virtual bool open_local_session(const std::string& remote_addr);
  virtual bool close_local_session(const std::string& remote_addr);
  virtual bool fetch_virtual_offsets(
      uint64_t cluster_id,
      const proto::VirtualKVCacheOffsetsRequest& request,
      proto::VirtualKVCacheOffsetsResponse* response);

 private:
  void record_pending_remote_unlink(
      uint64_t cluster_id,
      const std::string& remote_addr,
      const WorkerCacheLayoutManifest& destination_manifest);

  struct LocalCachePeer {
    WorkerCacheLayoutManifest destination_manifest;
    WorkerCacheLayoutManifest source_manifest;
    std::string source_incarnation;
    uint64_t source_layout_generation = 0;
    CachePeerMode mode = CachePeerMode::PLAN_ONLY;
    bool holds_session = false;
  };
  struct PendingRemoteUnlink {
    uint64_t cluster_id = 0;
    WorkerCacheLayoutManifest destination_manifest;
  };
  MooncakeKVCacheTransferState& cache_state_;
  std::mutex peer_operation_mutex_;
  std::mutex peer_mutex_;
  std::unordered_map<std::string, LocalCachePeer> cache_peers_;
  std::unordered_map<std::string, PendingRemoteUnlink> pending_remote_unlinks_;
};

class MooncakeKVCacheTransferService final
    : public proto::MooncakeTransferEngineService {
 public:
  MooncakeKVCacheTransferService();
  explicit MooncakeKVCacheTransferService(
      std::shared_ptr<MooncakeKVCacheTransferState> state);
  void SetCachePeer(google::protobuf::RpcController* controller,
                    const proto::CachePeerRequest* request,
                    proto::Status* response,
                    google::protobuf::Closure* done) override;

  void GetCacheLayoutManifest(google::protobuf::RpcController* controller,
                              const proto::Empty* request,
                              proto::WorkerCacheLayoutManifest* response,
                              google::protobuf::Closure* done) override;
  void GetVirtualKVCacheOffsets(
      google::protobuf::RpcController* controller,
      const proto::VirtualKVCacheOffsetsRequest* request,
      proto::VirtualKVCacheOffsetsResponse* response,
      google::protobuf::Closure* done) override;

 private:
  std::shared_ptr<MooncakeKVCacheTransferState> state_;
};

}  // namespace xllm
