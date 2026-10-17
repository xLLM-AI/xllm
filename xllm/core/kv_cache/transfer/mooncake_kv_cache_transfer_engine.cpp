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

#include "core/kv_cache/transfer/mooncake_kv_cache_transfer_engine.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include <utility>

#include "core/kv_cache/layout/kv_cache_tensor_role.h"
#include "core/util/net.h"

namespace xllm {
namespace {

std::string control_endpoint(uint64_t cluster_id) {
  const auto [host, port] = net::convert_uint64_to_ip_port(cluster_id);
  return host + ":" + std::to_string(port);
}

bool set_remote_cache_peer(MooncakeTransferEngineCore* core,
                           uint64_t cluster_id,
                           const WorkerCacheLayoutManifest& manifest,
                           proto::CachePeerMode mode,
                           const std::string& endpoint) {
  const auto channel =
      core->get_or_create_channel(control_endpoint(cluster_id));
  if (channel == nullptr) {
    LOG(ERROR) << "Create cache peer RPC channel failed for " << endpoint;
    return false;
  }
  proto::MooncakeTransferEngineService_Stub stub(channel.get());

  proto::CachePeerRequest request;
  cache_layout_to_proto(manifest, request.mutable_destination_manifest());
  request.set_mode(mode);
  proto::Status response;
  brpc::Controller controller;
  stub.SetCachePeer(&controller, &request, &response, nullptr);
  if (controller.Failed() || !response.ok()) {
    LOG(ERROR) << "SetCachePeer failed for " << endpoint
               << ", rpc_error=" << controller.ErrorText()
               << ", peer_ok=" << response.ok();
    return false;
  }
  return true;
}

struct VirtualLayerManifest {
  const CacheTensorManifest* key = nullptr;
  const CacheTensorManifest* value = nullptr;
  const CacheTensorManifest* index = nullptr;
};

Status virtual_layer_manifests(const WorkerCacheLayoutManifest& manifest,
                               std::vector<VirtualLayerManifest>* layers) {
  const std::string family_suffix = "_virtual_memory_v2";
  if (manifest.layout_family.size() < family_suffix.size() ||
      manifest.layout_family.compare(
          manifest.layout_family.size() - family_suffix.size(),
          family_suffix.size(),
          family_suffix) != 0 ||
      manifest.tensors.empty()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "virtual offsets require a virtual-memory v2 layout");
  }
  const int32_t group_id = manifest.tensors.front().group_id;
  size_t layer_count = 0;
  for (const CacheTensorManifest& tensor : manifest.tensors) {
    if (tensor.cache_namespace != CacheNamespace::MAIN ||
        !tensor.explicit_resource_offsets || tensor.group_id != group_id ||
        tensor.mooncake_buffer_id != 0 || tensor.storage_offset_bytes != 0 ||
        tensor.physical_rows_per_resource != 1 || !tensor.contiguous ||
        tensor.shard.resource_scope != CacheResourceScope::BLOCK ||
        tensor.layer_id < 0 ||
        static_cast<uint64_t>(tensor.layer_id) >= manifest.tensors.size()) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "virtual offsets require explicit MAIN K/V/INDEX blocks");
    }
    layer_count =
        std::max(layer_count, static_cast<size_t>(tensor.layer_id) + 1);
  }
  layers->resize(layer_count);
  for (const CacheTensorManifest& tensor : manifest.tensors) {
    VirtualLayerManifest& layer = (*layers)[tensor.layer_id];
    const CacheTensorManifest** role = nullptr;
    if (tensor.role == static_cast<int32_t>(KVCacheTensorRole::KEY)) {
      role = &layer.key;
    } else if (tensor.role == static_cast<int32_t>(KVCacheTensorRole::VALUE)) {
      role = &layer.value;
    } else if (tensor.role == static_cast<int32_t>(KVCacheTensorRole::INDEX)) {
      role = &layer.index;
    }
    if (role == nullptr || *role != nullptr) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "virtual layout has unsupported or duplicate roles");
    }
    *role = &tensor;
  }
  for (const VirtualLayerManifest& layer : *layers) {
    if (layer.key == nullptr || layer.value == nullptr) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "virtual layout must own K/V on each layer");
    }
  }
  return Status();
}

bool valid_virtual_geometry(const CacheTensorManifest& tensor,
                            uint64_t page_size_bytes) {
  return tensor.resource_stride_bytes > 0 &&
         tensor.resource_stride_bytes <= page_size_bytes &&
         page_size_bytes % tensor.resource_stride_bytes == 0 &&
         tensor.buffer_bytes >= tensor.resource_stride_bytes;
}

bool valid_virtual_offset(const CacheTensorManifest& tensor,
                          uint64_t page_size_bytes,
                          uint64_t block_id,
                          uint64_t offset) {
  return valid_virtual_geometry(tensor, page_size_bytes) &&
         block_id < tensor.resource_count &&
         offset <= tensor.buffer_bytes - tensor.resource_stride_bytes &&
         offset % tensor.resource_stride_bytes == 0 &&
         offset % page_size_bytes <=
             page_size_bytes - tensor.resource_stride_bytes;
}

Status validate_virtual_region(
    const WorkerCacheLayoutManifest& manifest,
    const VirtualKVCacheTransferMemoryRegion& region) {
  if (region.base_address == nullptr || region.size_bytes == 0 ||
      region.page_size_bytes == 0) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "virtual memory provider has no physical page region");
  }
  for (const CacheTensorManifest& tensor : manifest.tensors) {
    if (tensor.buffer_bytes != region.size_bytes ||
        !valid_virtual_geometry(tensor, region.page_size_bytes)) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "virtual provider region does not match registered layout");
    }
  }
  return Status();
}

}  // namespace

MooncakeKVCacheTransferState::MooncakeKVCacheTransferState()
    : transport_(MooncakeTransferEngineCore::get_instance()) {}

std::shared_ptr<MooncakeKVCacheTransferState>
MooncakeKVCacheTransferState::shared_instance() {
  static const std::shared_ptr<MooncakeKVCacheTransferState> instance = [] {
    auto state = std::shared_ptr<MooncakeKVCacheTransferState>(
        new MooncakeKVCacheTransferState());
    // The RPC server retains state until it has stopped and joined callbacks.
    MooncakeTransferEngineCore::get_instance().register_rpc_service(
        std::make_shared<MooncakeKVCacheTransferService>(state));
    return state;
  }();
  return instance;
}

MooncakeKVCacheTransferState& MooncakeKVCacheTransferState::get_instance() {
  return *shared_instance();
}

MooncakeKVCacheTransferState::~MooncakeKVCacheTransferState() {
  for (const auto& [addr, peer] : cache_peer_links_) {
    if (peer.holds_session) {
      transport_.close_session("", addr);
    }
  }
}

Status MooncakeKVCacheTransferState::set_local_cache_layout(
    const WorkerCacheLayoutManifest& manifest) {
  const Status status = validate_worker_cache_layout(manifest);
  if (!status.ok()) {
    return status;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!cache_peer_links_.empty()) {
    return Status(StatusCode::UNAVAILABLE,
                  "cache layout cannot change while cache peers are linked");
  }
  if (local_cache_layout_.has_value() &&
      local_cache_layout_->incarnation_id == manifest.incarnation_id &&
      manifest.layout_generation <= local_cache_layout_->layout_generation) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "cache layout generation must increase monotonically");
  }
  local_cache_layout_ = manifest;
  virtual_memory_model_id_.clear();
  virtual_memory_provider_.reset();
  return Status();
}

Status MooncakeKVCacheTransferState::set_virtual_memory_provider(
    std::string model_id,
    std::shared_ptr<VirtualKVCacheTransferMemoryProvider> provider) {
  if (model_id.empty() || provider == nullptr) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "virtual offsets require a model and memory provider");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!local_cache_layout_.has_value()) {
    return Status(StatusCode::UNAVAILABLE,
                  "local cache layout is not registered");
  }
  std::vector<VirtualLayerManifest> layers;
  const Status layout_status =
      virtual_layer_manifests(*local_cache_layout_, &layers);
  if (!layout_status.ok()) {
    return layout_status;
  }
  const VirtualKVCacheTransferMemoryRegion region = provider->memory_region();
  const Status region_status =
      validate_virtual_region(*local_cache_layout_, region);
  if (!region_status.ok()) {
    return region_status;
  }
  if (virtual_memory_provider_ != nullptr &&
      (virtual_memory_provider_ != provider ||
       virtual_memory_model_id_ != model_id)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "virtual memory provider requires a new layout generation");
  }
  virtual_memory_model_id_ = std::move(model_id);
  virtual_memory_provider_ = std::move(provider);
  return Status();
}

Status MooncakeKVCacheTransferState::get_virtual_offsets(
    const std::string& incarnation_id,
    uint64_t layout_generation,
    const std::vector<uint64_t>& block_ids,
    WorkerCacheLayoutManifest* manifest,
    std::vector<VirtualKVCacheLayerGlobalOffsets>* layer_offsets,
    uint64_t* physical_page_size_bytes) const {
  if (manifest == nullptr || layer_offsets == nullptr ||
      physical_page_size_bytes == nullptr) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "virtual offsets output pointers must be present");
  }
  *manifest = WorkerCacheLayoutManifest();
  layer_offsets->clear();
  *physical_page_size_bytes = 0;
  if (block_ids.empty()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "virtual offsets require at least one block");
  }
  WorkerCacheLayoutManifest registered_manifest;
  std::shared_ptr<VirtualKVCacheTransferMemoryProvider> provider;
  std::string model_id;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!local_cache_layout_.has_value() ||
        local_cache_layout_->incarnation_id != incarnation_id ||
        local_cache_layout_->layout_generation != layout_generation ||
        virtual_memory_provider_ == nullptr) {
      return Status(StatusCode::UNAVAILABLE,
                    "virtual source layout identity is stale or unavailable");
    }
    registered_manifest = *local_cache_layout_;
    provider = virtual_memory_provider_;
    model_id = virtual_memory_model_id_;
  }
  std::vector<VirtualLayerManifest> layers;
  const Status layout_status =
      virtual_layer_manifests(registered_manifest, &layers);
  if (!layout_status.ok()) {
    return layout_status;
  }
  // Retain the provider and mapping while looking up offsets outside the state
  // lock. Re-registration may proceed, but its obsolete response is rejected.
  const VirtualKVCacheTransferMemoryRegion region = provider->memory_region();
  const Status region_status =
      validate_virtual_region(registered_manifest, region);
  if (!region_status.ok()) {
    return region_status;
  }
  std::vector<VirtualKVCacheLayerGlobalOffsets> offsets(layers.size());
  for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
    const VirtualLayerManifest& layer = layers[layer_id];
    VirtualKVCacheLayerGlobalOffsets& layer_output = offsets[layer_id];
    layer_output.k_offsets.reserve(block_ids.size());
    layer_output.v_offsets.reserve(block_ids.size());
    if (layer.index != nullptr) {
      layer_output.index_offsets.reserve(block_ids.size());
    }
    for (uint64_t block_id : block_ids) {
      if (block_id >= layer.key->resource_count ||
          block_id >= layer.value->resource_count ||
          (layer.index != nullptr && block_id >= layer.index->resource_count)) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "virtual source block is out of registered bounds");
      }
      const auto block = provider->get_global_offsets_for_block(
          model_id, static_cast<int64_t>(layer_id), block_id);
      if (!block.has_value() ||
          block->index_offset.has_value() != (layer.index != nullptr) ||
          !valid_virtual_offset(
              *layer.key, region.page_size_bytes, block_id, block->k_offset) ||
          !valid_virtual_offset(*layer.value,
                                region.page_size_bytes,
                                block_id,
                                block->v_offset) ||
          (layer.index != nullptr &&
           !valid_virtual_offset(*layer.index,
                                 region.page_size_bytes,
                                 block_id,
                                 *block->index_offset))) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "virtual source block is unmapped or has invalid roles");
      }
      layer_output.k_offsets.emplace_back(block->k_offset);
      layer_output.v_offsets.emplace_back(block->v_offset);
      if (layer.index != nullptr) {
        layer_output.index_offsets.emplace_back(*block->index_offset);
      }
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!local_cache_layout_.has_value() ||
      local_cache_layout_->incarnation_id != incarnation_id ||
      local_cache_layout_->layout_generation != layout_generation ||
      virtual_memory_provider_ != provider ||
      virtual_memory_model_id_ != model_id) {
    return Status(StatusCode::UNAVAILABLE,
                  "virtual source layout changed during offset lookup");
  }
  *manifest = std::move(registered_manifest);
  *layer_offsets = std::move(offsets);
  *physical_page_size_bytes = region.page_size_bytes;
  return Status();
}

std::optional<WorkerCacheLayoutManifest>
MooncakeKVCacheTransferState::local_cache_layout() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_cache_layout_;
}

Status MooncakeKVCacheTransferState::set_cache_peer(
    const WorkerCacheLayoutManifest& peer_manifest,
    CachePeerMode mode) {
  const Status manifest_status = validate_worker_cache_layout(peer_manifest);
  if (!manifest_status.ok()) {
    return manifest_status;
  }

  std::optional<WorkerCacheLayoutManifest> local_manifest;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = cache_peer_links_.find(peer_manifest.addr);
    const bool identity_matches =
        existing != cache_peer_links_.end() &&
        existing->second.destination_incarnation ==
            peer_manifest.incarnation_id &&
        existing->second.destination_layout_generation ==
            peer_manifest.layout_generation;
    if (mode == CachePeerMode::ABSENT) {
      if (!identity_matches) {
        return Status();
      }
      if (existing->second.holds_session) {
        transport_.close_session("", peer_manifest.addr);
      }
      cache_peer_links_.erase(existing);
      return Status();
    }
    if (identity_matches) {
      if (existing->second.mode == mode) {
        return Status();
      }
      return Status(StatusCode::INVALID_ARGUMENT,
                    "cache peer mode change requires unlink");
    }
    if (existing != cache_peer_links_.end() &&
        existing->second.destination_incarnation ==
            peer_manifest.incarnation_id &&
        peer_manifest.layout_generation <
            existing->second.destination_layout_generation) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "peer cache layout generation is stale");
    }
    local_manifest = local_cache_layout_;
  }
  if (!local_manifest.has_value()) {
    return Status(StatusCode::UNAVAILABLE,
                  "local cache layout is not registered");
  }

  std::optional<ReshardPlanTemplate> plan;
  if (mode == CachePeerMode::ACTIVE) {
    ReshardPlanTemplate active_plan;
    const Status plan_status = ReshardPlanner().build_outgoing_plan(
        *local_manifest, peer_manifest, &active_plan);
    if (!plan_status.ok()) {
      return plan_status;
    }
    if (active_plan.regions.empty()) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "active cache peer requires a non-empty plan");
    }
    plan = std::move(active_plan);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!local_cache_layout_.has_value() ||
      local_cache_layout_->incarnation_id != local_manifest->incarnation_id ||
      local_cache_layout_->layout_generation !=
          local_manifest->layout_generation) {
    return Status(StatusCode::UNAVAILABLE,
                  "local cache layout changed during plan construction");
  }

  const auto existing = cache_peer_links_.find(peer_manifest.addr);
  if (existing != cache_peer_links_.end() &&
      existing->second.destination_incarnation ==
          peer_manifest.incarnation_id &&
      existing->second.destination_layout_generation ==
          peer_manifest.layout_generation) {
    if (existing->second.mode == mode) {
      return Status();
    }
    return Status(StatusCode::INVALID_ARGUMENT,
                  "cache peer mode change requires unlink");
  }
  if (existing != cache_peer_links_.end() &&
      existing->second.destination_incarnation ==
          peer_manifest.incarnation_id &&
      peer_manifest.layout_generation <
          existing->second.destination_layout_generation) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "peer cache layout generation is stale");
  }

  const bool holds_session = mode == CachePeerMode::ACTIVE;
  if (holds_session && !transport_.open_session("", peer_manifest.addr)) {
    return Status(StatusCode::UNAVAILABLE,
                  "failed to open cache peer data session");
  }

  CachePeerLink link;
  link.destination_incarnation = peer_manifest.incarnation_id;
  link.destination_layout_generation = peer_manifest.layout_generation;
  link.mode = mode;
  link.plan = std::move(plan);
  link.holds_session = holds_session;
  if (existing != cache_peer_links_.end() && existing->second.holds_session) {
    transport_.close_session("", peer_manifest.addr);
  }
  cache_peer_links_[peer_manifest.addr] = std::move(link);
  return Status();
}

bool MooncakeKVCacheTransferState::has_outgoing_plan(
    const std::string& remote_addr,
    CacheNamespace cache_namespace) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto link_it = cache_peer_links_.find(remote_addr);
  if (link_it == cache_peer_links_.end() || !link_it->second.plan.has_value()) {
    return false;
  }
  return std::any_of(link_it->second.plan->regions.begin(),
                     link_it->second.plan->regions.end(),
                     [cache_namespace](const StridedRegionTemplate& region) {
                       return region.cache_namespace == cache_namespace;
                     });
}

bool MooncakeKVCacheTransferState::has_reshard_plan(
    const std::string& remote_addr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_peer_links_.find(remote_addr) != cache_peer_links_.end();
}

Status MooncakeKVCacheTransferState::bind_outgoing_regions(
    const std::string& remote_addr,
    const std::vector<KVTransferMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  std::optional<ReshardPlanTemplate> plan;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto link_it = cache_peer_links_.find(remote_addr);
    if (link_it == cache_peer_links_.end()) {
      return Status(StatusCode::UNAVAILABLE,
                    "cache peer not negotiated: " + remote_addr);
    }
    if (link_it->second.mode == CachePeerMode::PLAN_ONLY) {
      regions->clear();
      return Status();
    }
    plan = link_it->second.plan;
  }
  if (!plan.has_value()) {
    return Status(StatusCode::UNAVAILABLE,
                  "active cache peer has no reshard plan");
  }
  RequestRegionBinder binder;
  return binder.bind(*plan, mappings, cache_namespace, layer_id, regions);
}

Status MooncakeKVCacheTransferState::bind_outgoing_regions_explicit(
    const std::string& remote_addr,
    const std::vector<ExplicitResourceMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  std::optional<ReshardPlanTemplate> plan;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto link_it = cache_peer_links_.find(remote_addr);
    if (link_it == cache_peer_links_.end()) {
      return Status(StatusCode::UNAVAILABLE,
                    "cache peer not negotiated: " + remote_addr);
    }
    if (link_it->second.mode == CachePeerMode::PLAN_ONLY) {
      regions->clear();
      return Status();
    }
    plan = link_it->second.plan;
  }
  if (!plan.has_value()) {
    return Status(StatusCode::UNAVAILABLE,
                  "active cache peer has no reshard plan");
  }
  RequestRegionBinder binder;
  return binder.bind_explicit(
      *plan, mappings, cache_namespace, layer_id, regions);
}

MooncakeKVCacheTransferEngine::MooncakeKVCacheTransferEngine(
    uint16_t listen_port,
    const torch::Device& device)
    : MooncakeTransferEngine(listen_port, device),
      cache_state_(MooncakeKVCacheTransferState::get_instance()) {}

std::string MooncakeKVCacheTransferEngine::initialize() {
  const char* tcp_protocol = std::getenv("MC_TCP_PROTO");
  if (tcp_protocol != nullptr && std::string(tcp_protocol) == "1") {
    LOG(ERROR) << "MC_TCP_PROTO=1 does not provide the remote visibility "
                  "semantics required by KV cache transfer.";
    return "";
  }
  return MooncakeTransferEngine::initialize();
}

bool MooncakeKVCacheTransferEngine::fetch_cache_layout(
    uint64_t cluster_id,
    const std::string& remote_addr,
    WorkerCacheLayoutManifest* manifest) {
  const auto channel =
      MooncakeTransferEngineCore::get_instance().get_or_create_channel(
          control_endpoint(cluster_id));
  if (channel == nullptr) {
    LOG(ERROR) << "Failed to create cache layout RPC channel, cluster_id="
               << cluster_id;
    return false;
  }
  proto::Empty request;
  proto::WorkerCacheLayoutManifest response;
  brpc::Controller controller;
  proto::MooncakeTransferEngineService_Stub stub(channel.get());
  stub.GetCacheLayoutManifest(&controller, &request, &response, nullptr);
  if (controller.Failed()) {
    LOG(ERROR) << "GetCacheLayoutManifest failed: " << controller.ErrorText();
    return false;
  }
  const Status status = cache_layout_from_proto(response, manifest);
  if (!status.ok()) {
    LOG(ERROR) << "Invalid remote cache layout from " << remote_addr << ": "
               << status.message();
    return false;
  }
  if (manifest->addr != remote_addr || manifest->cluster_id != cluster_id) {
    LOG(ERROR) << "Remote cache layout endpoint mismatch, expected="
               << cluster_id << "/" << remote_addr
               << ", manifest=" << manifest->cluster_id << "/"
               << manifest->addr;
    return false;
  }
  return true;
}

bool MooncakeKVCacheTransferEngine::fetch_virtual_offsets(
    uint64_t cluster_id,
    const proto::VirtualKVCacheOffsetsRequest& request,
    proto::VirtualKVCacheOffsetsResponse* response) {
  const auto channel =
      MooncakeTransferEngineCore::get_instance().get_or_create_channel(
          control_endpoint(cluster_id));
  if (channel == nullptr) {
    LOG(ERROR) << "Failed to create virtual offsets RPC channel, cluster_id="
               << cluster_id;
    return false;
  }
  brpc::Controller controller;
  proto::MooncakeTransferEngineService_Stub stub(channel.get());
  stub.GetVirtualKVCacheOffsets(&controller, &request, response, nullptr);
  if (controller.Failed() || !response->ok()) {
    LOG(ERROR) << "GetVirtualKVCacheOffsets failed, cluster_id=" << cluster_id
               << ", rpc_error=" << controller.ErrorText();
    return false;
  }
  return true;
}

bool MooncakeKVCacheTransferEngine::get_remote_virtual_offsets(
    uint64_t cluster_id,
    const std::string& remote_addr,
    const std::vector<uint64_t>& block_ids,
    WorkerCacheLayoutManifest* manifest,
    std::vector<VirtualKVCacheLayerGlobalOffsets>* layer_offsets) {
  if (manifest == nullptr || layer_offsets == nullptr) {
    return false;
  }
  *manifest = WorkerCacheLayoutManifest();
  layer_offsets->clear();
  if (block_ids.empty()) {
    return false;
  }
  std::optional<WorkerCacheLayoutManifest> local_manifest;
  LocalCachePeer peer_snapshot;
  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    local_manifest = cache_state_.local_cache_layout();
    const auto peer = cache_peers_.find(remote_addr);
    if (peer == cache_peers_.end() || !local_manifest.has_value() ||
        peer->second.source_manifest.incarnation_id.empty() ||
        peer->second.source_manifest.addr != remote_addr ||
        peer->second.source_manifest.tensors.empty() ||
        peer->second.source_manifest.cluster_id != cluster_id ||
        peer->second.destination_manifest.incarnation_id !=
            local_manifest->incarnation_id ||
        peer->second.destination_manifest.layout_generation !=
            local_manifest->layout_generation) {
      return false;
    }
    peer_snapshot = peer->second;
  }
  const WorkerCacheLayoutManifest& remote_manifest =
      peer_snapshot.source_manifest;
  std::vector<VirtualLayerManifest> layers;
  const Status layout_status =
      virtual_layer_manifests(remote_manifest, &layers);
  if (!layout_status.ok()) {
    LOG(ERROR) << "Invalid virtual source layout: " << layout_status.message();
    return false;
  }
  proto::VirtualKVCacheOffsetsRequest request;
  request.set_incarnation_id(remote_manifest.incarnation_id);
  request.set_layout_generation(remote_manifest.layout_generation);
  request.mutable_block_ids()->Assign(block_ids.begin(), block_ids.end());
  proto::VirtualKVCacheOffsetsResponse response;
  if (!fetch_virtual_offsets(cluster_id, request, &response) ||
      !response.ok() ||
      response.incarnation_id() != remote_manifest.incarnation_id ||
      response.layout_generation() != remote_manifest.layout_generation ||
      static_cast<size_t>(response.layer_offsets_size()) != layers.size()) {
    return false;
  }
  std::vector<VirtualKVCacheLayerGlobalOffsets> offsets(layers.size());
  std::vector<bool> seen(layers.size(), false);
  for (const proto::VirtualKVCacheLayerOffsets& wire_layer :
       response.layer_offsets()) {
    if (wire_layer.layer_id() < 0 ||
        static_cast<uint64_t>(wire_layer.layer_id()) >= layers.size()) {
      return false;
    }
    const size_t layer_id = static_cast<size_t>(wire_layer.layer_id());
    const VirtualLayerManifest& layer = layers[layer_id];
    if (seen[layer_id] ||
        static_cast<size_t>(wire_layer.k_offsets_size()) != block_ids.size() ||
        static_cast<size_t>(wire_layer.v_offsets_size()) != block_ids.size() ||
        static_cast<size_t>(wire_layer.index_offsets_size()) !=
            (layer.index == nullptr ? 0 : block_ids.size())) {
      return false;
    }
    seen[layer_id] = true;
    VirtualKVCacheLayerGlobalOffsets& layer_output = offsets[layer_id];
    layer_output.k_offsets.reserve(block_ids.size());
    layer_output.v_offsets.reserve(block_ids.size());
    if (layer.index != nullptr) {
      layer_output.index_offsets.reserve(block_ids.size());
    }
    for (size_t index = 0; index < block_ids.size(); ++index) {
      const int32_t wire_index = static_cast<int32_t>(index);
      const uint64_t key_offset = wire_layer.k_offsets(wire_index);
      const uint64_t value_offset = wire_layer.v_offsets(wire_index);
      if (!valid_virtual_offset(*layer.key,
                                response.physical_page_size_bytes(),
                                block_ids[index],
                                key_offset) ||
          !valid_virtual_offset(*layer.value,
                                response.physical_page_size_bytes(),
                                block_ids[index],
                                value_offset) ||
          (layer.index != nullptr &&
           !valid_virtual_offset(*layer.index,
                                 response.physical_page_size_bytes(),
                                 block_ids[index],
                                 wire_layer.index_offsets(wire_index)))) {
        return false;
      }
      layer_output.k_offsets.emplace_back(key_offset);
      layer_output.v_offsets.emplace_back(value_offset);
      if (layer.index != nullptr) {
        layer_output.index_offsets.emplace_back(
            wire_layer.index_offsets(wire_index));
      }
    }
  }
  // An established session is stable under peer_operation_mutex_; publish the
  // offsets without waiting for another link/unlink operation. Only the
  // first PLAN_ONLY query needs serialization while it acquires a reference.
  auto peer_matches_snapshot = [&](const LocalCachePeer& peer) {
    return peer.source_manifest.incarnation_id ==
               remote_manifest.incarnation_id &&
           peer.source_manifest.layout_generation ==
               remote_manifest.layout_generation &&
           peer.source_manifest.cluster_id == cluster_id &&
           peer.destination_manifest.incarnation_id ==
               local_manifest->incarnation_id &&
           peer.destination_manifest.layout_generation ==
               local_manifest->layout_generation &&
           peer.mode == peer_snapshot.mode;
  };
  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const auto peer = cache_peers_.find(remote_addr);
    if (peer == cache_peers_.end() || !peer_matches_snapshot(peer->second)) {
      return false;
    }
    if (peer->second.holds_session) {
      *manifest = peer_snapshot.source_manifest;
      *layer_offsets = std::move(offsets);
      return true;
    }
  }

  std::lock_guard<std::mutex> operation_lock(peer_operation_mutex_);
  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const auto peer = cache_peers_.find(remote_addr);
    if (peer == cache_peers_.end() || !peer_matches_snapshot(peer->second)) {
      return false;
    }
    if (peer->second.holds_session) {
      *manifest = peer_snapshot.source_manifest;
      *layer_offsets = std::move(offsets);
      return true;
    }
  }

  if (!open_local_session(remote_addr)) {
    return false;
  }
  bool retain_session = false;
  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const auto peer = cache_peers_.find(remote_addr);
    if (peer == cache_peers_.end() || !peer_matches_snapshot(peer->second) ||
        peer->second.holds_session) {
      retain_session = false;
    } else {
      peer->second.holds_session = true;
      retain_session = true;
    }
  }
  if (!retain_session) {
    if (!close_local_session(remote_addr)) {
      LOG(ERROR) << "Failed to close temporary cache session for "
                 << remote_addr;
    }
    return false;
  }
  *manifest = peer_snapshot.source_manifest;
  *layer_offsets = std::move(offsets);
  return true;
}

bool MooncakeKVCacheTransferEngine::set_remote_peer(
    uint64_t cluster_id,
    const std::string& remote_addr,
    const WorkerCacheLayoutManifest& manifest,
    CachePeerMode mode) {
  proto::CachePeerMode proto_mode = proto::CACHE_PEER_MODE_ABSENT;
  if (mode == CachePeerMode::ACTIVE) {
    proto_mode = proto::CACHE_PEER_MODE_ACTIVE;
  } else if (mode == CachePeerMode::PLAN_ONLY) {
    proto_mode = proto::CACHE_PEER_MODE_PLAN_ONLY;
  }
  return set_remote_cache_peer(&MooncakeTransferEngineCore::get_instance(),
                               cluster_id,
                               manifest,
                               proto_mode,
                               remote_addr);
}

bool MooncakeKVCacheTransferEngine::open_local_session(
    const std::string& remote_addr) {
  return MooncakeTransferEngineCore::get_instance().open_session("",
                                                                 remote_addr);
}

bool MooncakeKVCacheTransferEngine::close_local_session(
    const std::string& remote_addr) {
  return MooncakeTransferEngineCore::get_instance().close_session("",
                                                                  remote_addr);
}

bool MooncakeKVCacheTransferEngine::link_sessions(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs) {
  std::lock_guard<std::mutex> operation_lock(peer_operation_mutex_);
  if (cluster_ids.size() != remote_addrs.size() || cluster_ids.empty()) {
    LOG(ERROR) << "MoonCake link session endpoint sizes are invalid.";
    return false;
  }
  std::unordered_set<std::string> unique_remote_addrs;
  unique_remote_addrs.reserve(remote_addrs.size());
  for (const std::string& remote_addr : remote_addrs) {
    if (!unique_remote_addrs.emplace(remote_addr).second) {
      LOG(ERROR)
          << "MoonCake link session contains a duplicate remote address: "
          << remote_addr;
      return false;
    }
  }

  const std::optional<WorkerCacheLayoutManifest> local_manifest =
      cache_state_.local_cache_layout();
  if (!local_manifest.has_value()) {
    LOG(ERROR) << "Local cache layout must be registered before linking.";
    return false;
  }

  std::vector<WorkerCacheLayoutManifest> remote_manifests;
  remote_manifests.reserve(cluster_ids.size());
  for (size_t index = 0; index < cluster_ids.size(); ++index) {
    WorkerCacheLayoutManifest remote_manifest;
    if (!fetch_cache_layout(
            cluster_ids[index], remote_addrs[index], &remote_manifest)) {
      return false;
    }
    remote_manifests.emplace_back(std::move(remote_manifest));
  }

  ReshardPlanner planner;
  std::vector<size_t> selected_indices;
  const Status selection = planner.select_sources(
      remote_manifests, *local_manifest, &selected_indices);
  if (!selection.ok()) {
    LOG(ERROR) << "Remote cache layouts cannot cover local destination: "
               << selection.message();
    return false;
  }

  const std::unordered_set<size_t> selected_set(selected_indices.begin(),
                                                selected_indices.end());
  std::vector<size_t> new_indices;
  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const std::optional<WorkerCacheLayoutManifest> current_manifest =
        cache_state_.local_cache_layout();
    if (!current_manifest.has_value() ||
        current_manifest->incarnation_id != local_manifest->incarnation_id ||
        current_manifest->layout_generation !=
            local_manifest->layout_generation) {
      LOG(ERROR) << "Local cache layout changed while linking cache peers.";
      return false;
    }
    new_indices.reserve(remote_manifests.size());
    for (size_t index = 0; index < remote_manifests.size(); ++index) {
      if (pending_remote_unlinks_.find(remote_addrs[index]) !=
          pending_remote_unlinks_.end()) {
        LOG(ERROR) << "Cache peer has a pending remote unlink: "
                   << remote_addrs[index];
        return false;
      }
      const auto peer = cache_peers_.find(remote_addrs[index]);
      if (peer == cache_peers_.end()) {
        new_indices.emplace_back(index);
        continue;
      }
      const CachePeerMode mode = selected_set.find(index) != selected_set.end()
                                     ? CachePeerMode::ACTIVE
                                     : CachePeerMode::PLAN_ONLY;
      if (peer->second.source_incarnation !=
              remote_manifests[index].incarnation_id ||
          peer->second.source_layout_generation !=
              remote_manifests[index].layout_generation ||
          peer->second.source_manifest.cluster_id !=
              remote_manifests[index].cluster_id ||
          peer->second.destination_manifest.incarnation_id !=
              local_manifest->incarnation_id ||
          peer->second.destination_manifest.layout_generation !=
              local_manifest->layout_generation ||
          peer->second.mode != mode) {
        LOG(ERROR)
            << "Cache peer identity or mode changed; unlink required for "
            << remote_addrs[index];
        return false;
      }
    }
  }

  std::vector<size_t> attempted_indices;
  attempted_indices.reserve(new_indices.size());
  std::vector<size_t> opened_indices;
  opened_indices.reserve(selected_indices.size());
  bool linked = true;
  for (size_t index : new_indices) {
    const bool active = selected_set.find(index) != selected_set.end();
    const CachePeerMode mode =
        active ? CachePeerMode::ACTIVE : CachePeerMode::PLAN_ONLY;
    attempted_indices.emplace_back(index);
    if (!set_remote_peer(
            cluster_ids[index], remote_addrs[index], *local_manifest, mode)) {
      linked = false;
      break;
    }
    if (active && !open_local_session(remote_addrs[index])) {
      if (!close_local_session(remote_addrs[index])) {
        LOG(ERROR) << "Failed to close cache session after open failure for "
                   << remote_addrs[index];
      }
      linked = false;
      break;
    }
    if (active) {
      opened_indices.emplace_back(index);
    }
  }

  if (!linked) {
    // A failed RPC response may still have published state on the source.
    for (size_t index : attempted_indices) {
      if (!set_remote_peer(cluster_ids[index],
                           remote_addrs[index],
                           *local_manifest,
                           CachePeerMode::ABSENT)) {
        LOG(ERROR) << "Cache peer rollback failed for " << remote_addrs[index];
        record_pending_remote_unlink(
            cluster_ids[index], remote_addrs[index], *local_manifest);
      }
    }
    for (size_t index : opened_indices) {
      close_local_session(remote_addrs[index]);
    }
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const std::optional<WorkerCacheLayoutManifest> current_manifest =
        cache_state_.local_cache_layout();
    if (!current_manifest.has_value() ||
        current_manifest->incarnation_id != local_manifest->incarnation_id ||
        current_manifest->layout_generation !=
            local_manifest->layout_generation) {
      linked = false;
    } else {
      for (size_t index : new_indices) {
        if (cache_peers_.find(remote_addrs[index]) != cache_peers_.end()) {
          linked = false;
          break;
        }
      }
      if (linked) {
        for (size_t index : new_indices) {
          LocalCachePeer cache_peer;
          cache_peer.destination_manifest = *local_manifest;
          cache_peer.source_manifest = remote_manifests[index];
          cache_peer.source_incarnation =
              remote_manifests[index].incarnation_id;
          cache_peer.source_layout_generation =
              remote_manifests[index].layout_generation;
          cache_peer.holds_session =
              selected_set.find(index) != selected_set.end();
          cache_peer.mode = cache_peer.holds_session ? CachePeerMode::ACTIVE
                                                     : CachePeerMode::PLAN_ONLY;
          cache_peers_[remote_addrs[index]] = std::move(cache_peer);
        }
      }
    }
  }
  if (!linked) {
    for (size_t index : new_indices) {
      if (!set_remote_peer(cluster_ids[index],
                           remote_addrs[index],
                           *local_manifest,
                           CachePeerMode::ABSENT)) {
        LOG(ERROR) << "Cache peer rollback failed for " << remote_addrs[index];
        record_pending_remote_unlink(
            cluster_ids[index], remote_addrs[index], *local_manifest);
      }
    }
    for (size_t index : opened_indices) {
      close_local_session(remote_addrs[index]);
    }
  }
  return linked;
}

Status MooncakeKVCacheTransferEngine::set_local_cache_layout(
    const WorkerCacheLayoutManifest& manifest) {
  std::lock_guard<std::mutex> operation_lock(peer_operation_mutex_);
  std::lock_guard<std::mutex> lock(peer_mutex_);
  if (!cache_peers_.empty()) {
    return Status(StatusCode::UNAVAILABLE,
                  "cache layout cannot change while cache peers are linked");
  }
  return cache_state_.set_local_cache_layout(manifest);
}

Status MooncakeKVCacheTransferEngine::set_virtual_memory_provider(
    std::string model_id,
    std::shared_ptr<VirtualKVCacheTransferMemoryProvider> provider) {
  return cache_state_.set_virtual_memory_provider(std::move(model_id),
                                                  std::move(provider));
}

bool MooncakeKVCacheTransferEngine::has_outgoing_plan(
    const std::string& remote_addr,
    CacheNamespace cache_namespace) const {
  return cache_state_.has_outgoing_plan(remote_addr, cache_namespace);
}

bool MooncakeKVCacheTransferEngine::has_reshard_plan(
    const std::string& remote_addr) const {
  return cache_state_.has_reshard_plan(remote_addr);
}

Status MooncakeKVCacheTransferEngine::bind_outgoing_regions(
    const std::string& remote_addr,
    const std::vector<KVTransferMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  return cache_state_.bind_outgoing_regions(
      remote_addr, mappings, cache_namespace, layer_id, regions);
}

Status MooncakeKVCacheTransferEngine::bind_outgoing_regions_explicit(
    const std::string& remote_addr,
    const std::vector<ExplicitResourceMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  return cache_state_.bind_outgoing_regions_explicit(
      remote_addr, mappings, cache_namespace, layer_id, regions);
}

bool MooncakeKVCacheTransferEngine::close_session(
    const uint64_t cluster_id,
    const std::string& remote_addr) {
  std::lock_guard<std::mutex> operation_lock(peer_operation_mutex_);
  LocalCachePeer peer_snapshot;
  std::optional<PendingRemoteUnlink> pending_unlink;
  bool has_peer = false;
  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const auto pending = pending_remote_unlinks_.find(remote_addr);
    if (pending != pending_remote_unlinks_.end()) {
      pending_unlink = pending->second;
      if (pending_unlink->cluster_id != cluster_id) {
        LOG(ERROR) << "Pending cache peer unlink cluster identity mismatch for "
                   << remote_addr;
        return false;
      }
    }
    const auto peer = cache_peers_.find(remote_addr);
    if (peer == cache_peers_.end() && !pending_unlink.has_value()) {
      return true;
    }
    if (peer != cache_peers_.end()) {
      peer_snapshot = peer->second;
      has_peer = true;
    }
  }
  if (has_peer && peer_snapshot.source_manifest.cluster_id != cluster_id) {
    LOG(ERROR) << "Cache peer cluster identity mismatch for " << remote_addr;
    return false;
  }

  const uint64_t remote_cluster_id =
      pending_unlink.has_value() ? pending_unlink->cluster_id : cluster_id;
  const WorkerCacheLayoutManifest& destination_manifest =
      pending_unlink.has_value() ? pending_unlink->destination_manifest
                                 : peer_snapshot.destination_manifest;
  const bool remote_closed = set_remote_peer(remote_cluster_id,
                                             remote_addr,
                                             destination_manifest,
                                             CachePeerMode::ABSENT);
  if (!remote_closed) {
    LOG(ERROR) << "Remote cache peer unlink failed for " << remote_addr;
    record_pending_remote_unlink(
        remote_cluster_id, remote_addr, destination_manifest);
  } else if (pending_unlink.has_value()) {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    pending_remote_unlinks_.erase(remote_addr);
  }
  const bool local_closed = !has_peer || !peer_snapshot.holds_session ||
                            close_local_session(remote_addr);
  if (!local_closed) {
    LOG(ERROR) << "Local cache session close failed for " << remote_addr;
  }
  if (has_peer && local_closed) {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const auto peer = cache_peers_.find(remote_addr);
    if (peer != cache_peers_.end() &&
        peer->second.source_manifest.cluster_id ==
            peer_snapshot.source_manifest.cluster_id &&
        peer->second.source_manifest.incarnation_id ==
            peer_snapshot.source_manifest.incarnation_id &&
        peer->second.source_manifest.layout_generation ==
            peer_snapshot.source_manifest.layout_generation &&
        peer->second.destination_manifest.incarnation_id ==
            peer_snapshot.destination_manifest.incarnation_id &&
        peer->second.destination_manifest.layout_generation ==
            peer_snapshot.destination_manifest.layout_generation &&
        peer->second.mode == peer_snapshot.mode) {
      cache_peers_.erase(peer);
    }
  }
  return remote_closed && local_closed;
}

void MooncakeKVCacheTransferEngine::record_pending_remote_unlink(
    const uint64_t cluster_id,
    const std::string& remote_addr,
    const WorkerCacheLayoutManifest& destination_manifest) {
  std::lock_guard<std::mutex> lock(peer_mutex_);
  pending_remote_unlinks_[remote_addr] =
      PendingRemoteUnlink{cluster_id, destination_manifest};
}

bool MooncakeKVCacheTransferEngine::move_memory_regions_with_session(
    const std::string& remote_addr,
    const std::vector<ByteRegion>& regions,
    const MoveOpcode move_opcode) {
  std::lock_guard<std::mutex> operation_lock(peer_operation_mutex_);
  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const auto peer = cache_peers_.find(remote_addr);
    if (peer == cache_peers_.end() || !peer->second.holds_session) {
      LOG(ERROR) << "No active cache session for remote address "
                 << remote_addr;
      return false;
    }
  }
  return MooncakeTransferEngine::move_memory_regions(
      remote_addr, regions, move_opcode);
}

MooncakeKVCacheTransferService::MooncakeKVCacheTransferService()
    : state_(MooncakeKVCacheTransferState::shared_instance()) {}

MooncakeKVCacheTransferService::MooncakeKVCacheTransferService(
    std::shared_ptr<MooncakeKVCacheTransferState> state)
    : state_(std::move(state)) {}

void MooncakeKVCacheTransferService::SetCachePeer(
    ::google::protobuf::RpcController* controller,
    const proto::CachePeerRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }
  if (!request->has_destination_manifest() ||
      request->mode() == proto::CACHE_PEER_MODE_UNSPECIFIED) {
    LOG(ERROR) << "SetCachePeer request is incomplete.";
    response->set_ok(false);
    return;
  }

  WorkerCacheLayoutManifest peer_manifest;
  const Status decode_status =
      cache_layout_from_proto(request->destination_manifest(), &peer_manifest);
  if (!decode_status.ok()) {
    LOG(ERROR) << "SetCachePeer received an invalid cache layout: "
               << decode_status.message();
    response->set_ok(false);
    return;
  }

  CachePeerMode mode = CachePeerMode::ABSENT;
  if (request->mode() == proto::CACHE_PEER_MODE_ACTIVE) {
    mode = CachePeerMode::ACTIVE;
  } else if (request->mode() == proto::CACHE_PEER_MODE_PLAN_ONLY) {
    mode = CachePeerMode::PLAN_ONLY;
  } else if (request->mode() != proto::CACHE_PEER_MODE_ABSENT) {
    LOG(ERROR) << "SetCachePeer received an invalid mode.";
    response->set_ok(false);
    return;
  }

  const Status status = state_->set_cache_peer(peer_manifest, mode);
  if (!status.ok()) {
    LOG(ERROR) << "SetCachePeer failed: " << status.message();
  }
  response->set_ok(status.ok());
}

void MooncakeKVCacheTransferService::GetCacheLayoutManifest(
    ::google::protobuf::RpcController* controller,
    const proto::Empty* request,
    proto::WorkerCacheLayoutManifest* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }
  const std::optional<WorkerCacheLayoutManifest> manifest =
      state_->local_cache_layout();
  if (!manifest.has_value()) {
    LOG(ERROR) << "GetCacheLayoutManifest called before cache registration.";
    response->Clear();
    return;
  }
  cache_layout_to_proto(*manifest, response);
}

void MooncakeKVCacheTransferService::GetVirtualKVCacheOffsets(
    ::google::protobuf::RpcController* controller,
    const proto::VirtualKVCacheOffsetsRequest* request,
    proto::VirtualKVCacheOffsetsResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }
  response->Clear();
  const std::vector<uint64_t> block_ids(request->block_ids().begin(),
                                        request->block_ids().end());
  WorkerCacheLayoutManifest manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> layer_offsets;
  uint64_t page_size_bytes = 0;
  const Status status =
      state_->get_virtual_offsets(request->incarnation_id(),
                                  request->layout_generation(),
                                  block_ids,
                                  &manifest,
                                  &layer_offsets,
                                  &page_size_bytes);
  if (!status.ok()) {
    LOG(ERROR) << "GetVirtualKVCacheOffsets failed: " << status.message();
    return;
  }
  response->set_incarnation_id(manifest.incarnation_id);
  response->set_layout_generation(manifest.layout_generation);
  response->set_physical_page_size_bytes(page_size_bytes);
  for (size_t layer_id = 0; layer_id < layer_offsets.size(); ++layer_id) {
    proto::VirtualKVCacheLayerOffsets* output = response->add_layer_offsets();
    output->set_layer_id(static_cast<int64_t>(layer_id));
    const VirtualKVCacheLayerGlobalOffsets& offsets = layer_offsets[layer_id];
    output->mutable_k_offsets()->Assign(offsets.k_offsets.begin(),
                                        offsets.k_offsets.end());
    output->mutable_v_offsets()->Assign(offsets.v_offsets.begin(),
                                        offsets.v_offsets.end());
    output->mutable_index_offsets()->Assign(offsets.index_offsets.begin(),
                                            offsets.index_offsets.end());
  }
  response->set_ok(true);
}

}  // namespace xllm
