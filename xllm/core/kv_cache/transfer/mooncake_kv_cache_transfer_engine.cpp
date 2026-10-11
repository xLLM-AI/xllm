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
  if (cluster_ids.size() != remote_addrs.size() || cluster_ids.empty()) {
    LOG(ERROR) << "MoonCake link session endpoint sizes are invalid.";
    return false;
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

  std::vector<size_t> opened_indices;
  opened_indices.reserve(selected_indices.size());
  const std::unordered_set<size_t> selected_set(selected_indices.begin(),
                                                selected_indices.end());
  bool linked = true;
  for (size_t index = 0; index < remote_manifests.size(); ++index) {
    const bool active = selected_set.find(index) != selected_set.end();
    const CachePeerMode mode =
        active ? CachePeerMode::ACTIVE : CachePeerMode::PLAN_ONLY;
    if (!set_remote_peer(
            cluster_ids[index], remote_addrs[index], *local_manifest, mode)) {
      linked = false;
      break;
    }
    if (active && !open_local_session(remote_addrs[index])) {
      linked = false;
      break;
    }
    if (active) {
      opened_indices.emplace_back(index);
    }
  }

  if (!linked) {
    // A failed RPC response may still have published state on the source.
    for (size_t index = 0; index < remote_manifests.size(); ++index) {
      if (!set_remote_peer(cluster_ids[index],
                           remote_addrs[index],
                           *local_manifest,
                           CachePeerMode::ABSENT)) {
        LOG(ERROR) << "Cache peer rollback failed for " << remote_addrs[index];
      }
    }
    for (size_t index : opened_indices) {
      close_local_session(remote_addrs[index]);
    }
    return false;
  }

  std::lock_guard<std::mutex> lock(peer_mutex_);
  for (size_t index = 0; index < remote_manifests.size(); ++index) {
    LocalCachePeer cache_peer;
    cache_peer.destination_manifest = *local_manifest;
    cache_peer.holds_session = selected_set.find(index) != selected_set.end();
    cache_peers_[remote_addrs[index]] = std::move(cache_peer);
  }
  return true;
}

Status MooncakeKVCacheTransferEngine::set_local_cache_layout(
    const WorkerCacheLayoutManifest& manifest) {
  return cache_state_.set_local_cache_layout(manifest);
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
  std::optional<LocalCachePeer> cache_peer;
  {
    std::lock_guard<std::mutex> lock(peer_mutex_);
    const auto peer_it = cache_peers_.find(remote_addr);
    if (peer_it != cache_peers_.end()) {
      cache_peer = peer_it->second;
    }
  }
  if (cache_peer.has_value()) {
    if (!set_remote_peer(cluster_id,
                         remote_addr,
                         cache_peer->destination_manifest,
                         CachePeerMode::ABSENT)) {
      return false;
    }
    if (cache_peer->holds_session && !close_local_session(remote_addr)) {
      return false;
    }
    std::lock_guard<std::mutex> lock(peer_mutex_);
    cache_peers_.erase(remote_addr);
    return true;
  }

  return true;
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

}  // namespace xllm
