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

#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>

#include "common/metrics.h"
#include "util/net.h"

namespace xllm {

namespace {

bool close_remote_session(MooncakeTransferEngineCore* core,
                          uint64_t cluster_id) {
  proto::MooncakeTransferEngineService_Stub* stub =
      core->get_or_create_stub(cluster_id);
  if (stub == nullptr) {
    LOG(ERROR) << "create_rpc_channel failed for cluster_id=" << cluster_id;
    return false;
  }

  proto::SessionInfo session_info;
  session_info.set_addr(core->addr());
  proto::Status response;
  brpc::Controller cntl;
  stub->CloseSession(&cntl, &session_info, &response, nullptr);
  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "CloseSession failed, " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool multiply_overflows(uint64_t lhs, uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs;
}

bool wait_batch(TransferEngine* engine, BatchID batch_id) {
  while (true) {
    TransferStatus status;
    mooncake::Status result = engine->getBatchTransferStatus(batch_id, status);
    if (!result.ok()) {
      LOG(ERROR) << "getBatchTransferStatus not ok";
      return false;
    }
    if (status.s == TransferStatusEnum::COMPLETED) {
      return true;
    }
    if (status.s == TransferStatusEnum::FAILED) {
      LOG(ERROR) << "getBatchTransferStatus failed";
      return false;
    }
    if (status.s == TransferStatusEnum::TIMEOUT) {
      LOG(ERROR) << "Sync data transfer timeout";
      return false;
    }
  }
}

}  // namespace

// ============================================================================
// MooncakeTransferEngineCore (Singleton)
// ============================================================================

MooncakeTransferEngineCore::~MooncakeTransferEngineCore() {
  for (auto& pair : stub_map_) {
    if (pair.second != nullptr) {
      delete pair.second->channel();
      delete pair.second;
    }
  }
  stub_map_.clear();

  if (initialized_) {
    server_.Stop(0);
    server_.Join();
  }
}

bool MooncakeTransferEngineCore::initialize(uint16_t listen_port,
                                            const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    LOG(INFO) << "MooncakeTransferEngineCore already initialized, reusing";
    return true;
  }

  listen_port_ = listen_port;
  host_ip_ = net::get_local_ip_addr();

  const char* tcp_protocol = std::getenv("MC_TCP_PROTO");
  if (tcp_protocol != nullptr && std::string(tcp_protocol) == "1") {
    LOG(ERROR) << "MC_TCP_PROTO=1 does not provide the remote visibility "
                  "semantics required by KV cache transfer.";
    return false;
  }

  engine_ = std::make_unique<TransferEngine>(true);

  Device dev(device);
  dev.set_device();
  dev.init_device_context();

  std::string hostname = host_ip_ + ":" + std::to_string(listen_port_);

  if (engine_->init("P2PHANDSHAKE", hostname, "", 0)) {
    LOG(ERROR) << "engine init failed, hostname=" << hostname;
    return false;
  }

  LOG(INFO) << "[Mooncake][PDTransferEngine] transfer engine init success, "
            << "handshake_endpoint=" << hostname;

  service_ = std::make_shared<MooncakeTransferEngineService>();
  if (server_.AddService(service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Failed to add service to server";
    return false;
  }

  brpc::ServerOptions options;
  if (server_.Start(listen_port_, &options) != 0) {
    LOG(ERROR) << "Fail to start Brpc rpc server on port " << listen_port_;
    return false;
  }

  rpc_port_ = engine_->getRpcPort();
  addr_ = host_ip_ + ":" + std::to_string(rpc_port_);

  initialized_ = true;
  LOG(INFO) << "[Mooncake][PDTransferEngine] ready, handshake_endpoint="
            << hostname << ", data_endpoint=" << addr_;

  return true;
}

bool MooncakeTransferEngineCore::open_session(const uint64_t cluster_id,
                                              const std::string& remote_addr,
                                              bool increment_existing) {
  std::lock_guard<std::mutex> lock(mutex_);

  LOG(INFO) << "open_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  auto it = handles_.find(remote_addr);
  const bool requires_layout_negotiation =
      cluster_id != 0 && local_cache_layout_.has_value();
  if (it != handles_.end() && !requires_layout_negotiation) {
    if (!increment_existing) {
      LOG(INFO) << "Session already exists for " << remote_addr;
      return true;
    }
    // Reuse the existing session until the last caller releases it.
    it->second.ref_count++;
    LOG(INFO) << "Reusing existing session for " << remote_addr
              << ", ref_count=" << it->second.ref_count;
    return true;
  }

  if (cluster_id != 0) {
    proto::MooncakeTransferEngineService_Stub* stub =
        get_or_create_stub_locked(cluster_id);
    if (stub == nullptr) {
      LOG(ERROR) << "create_rpc_channel failed";
      return false;
    }

    proto::SessionInfo request;
    request.set_addr(addr_);
    if (local_cache_layout_.has_value()) {
      cache_layout_to_proto(*local_cache_layout_,
                            request.mutable_cache_layout_manifest());
    }
    proto::Status response;
    brpc::Controller cntl;
    stub->OpenSession(&cntl, &request, &response, nullptr);
    if (cntl.Failed() || !response.ok()) {
      LOG(ERROR) << "OpenSession failed, rpc_error=" << cntl.ErrorText()
                 << ", peer_ok=" << response.ok();
      return false;
    }

    LOG(INFO) << "OpenSession RPC to " << remote_addr
              << ", local_addr=" << addr_;
  }

  // A weight-transfer session may already exist before KV cache registration.
  // The RPC above must still run after a manifest is published so the peer can
  // replace its outgoing plan for the current incarnation/generation.
  if (it != handles_.end()) {
    ++it->second.ref_count;
    LOG(INFO) << "Reusing negotiated session for " << remote_addr
              << ", ref_count=" << it->second.ref_count;
    return true;
  }

  // Keep a local handle as well as asking the peer to open one. WRITE uses
  // the peer-side handle created by the RPC, while READ needs this local
  // handle to address the peer's registered segment.
  Transport::SegmentHandle handle = engine_->openSegment(remote_addr);
  if (handle == static_cast<Transport::SegmentHandle>(-1)) {
    LOG(ERROR) << "Fail to connect to " << remote_addr;
    return false;
  }

  SessionInfo session_info;
  session_info.handle = handle;
  session_info.ref_count = 1;
  handles_[remote_addr] = session_info;

  LOG(INFO) << "Created new session for " << remote_addr << ", ref_count=1";

  return true;
}

bool MooncakeTransferEngineCore::close_session(const uint64_t cluster_id,
                                               const std::string& remote_addr) {
  std::unique_lock<std::mutex> lock(mutex_);

  LOG(INFO) << "close_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  auto it = handles_.find(remote_addr);
  if (cluster_id != 0) {
    if (it != handles_.end()) {
      it->second.ref_count--;
      LOG(INFO) << "Decremented ref_count for " << remote_addr
                << ", ref_count=" << it->second.ref_count;
      if (it->second.ref_count > 0) {
        return true;
      }
      Transport::SegmentHandle handle = it->second.handle;
      if (handle != static_cast<Transport::SegmentHandle>(-1)) {
        engine_->closeSegment(handle);
      }
      handles_.erase(it);
      outgoing_plans_.erase(remote_addr);
    }
    // close_remote_session() obtains the core mutex through
    // get_or_create_stub(). Release it after updating local state to avoid
    // recursively locking the same non-recursive mutex.
    lock.unlock();
    return close_remote_session(this, cluster_id);
  }

  if (it == handles_.end()) {
    return true;
  }

  it->second.ref_count--;
  LOG(INFO) << "Decremented ref_count for " << remote_addr
            << ", ref_count=" << it->second.ref_count;

  if (it->second.ref_count > 0) {
    return true;
  }

  SegmentHandle handle = it->second.handle;
  if (handle != static_cast<SegmentHandle>(-1)) {
    engine_->closeSegment(handle);
  }
  handles_.erase(it);
  outgoing_plans_.erase(remote_addr);

  LOG(INFO) << "Closed session for " << remote_addr;

  return true;
}

SegmentHandle MooncakeTransferEngineCore::get_handle(
    const std::string& remote_addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    return static_cast<SegmentHandle>(-1);
  }
  return it->second.handle;
}

Status MooncakeTransferEngineCore::set_local_cache_layout(
    const WorkerCacheLayoutManifest& manifest) {
  const Status status = validate_worker_cache_layout(manifest);
  if (!status.ok()) {
    return status;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (local_cache_layout_.has_value() &&
      local_cache_layout_->incarnation_id == manifest.incarnation_id &&
      manifest.layout_generation <= local_cache_layout_->layout_generation) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "cache layout generation must increase monotonically");
  }
  local_cache_layout_ = manifest;
  outgoing_plans_.clear();
  return Status();
}

std::optional<WorkerCacheLayoutManifest>
MooncakeTransferEngineCore::local_cache_layout() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_cache_layout_;
}

Status MooncakeTransferEngineCore::install_peer_cache_layout(
    const WorkerCacheLayoutManifest& peer_manifest) {
  std::optional<WorkerCacheLayoutManifest> local_manifest;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    local_manifest = local_cache_layout_;
  }
  if (!local_manifest.has_value()) {
    return Status(StatusCode::UNAVAILABLE,
                  "local cache layout is not registered");
  }

  ReshardPlanTemplate plan;
  ReshardPlanner planner;
  const Status status =
      planner.build_outgoing_plan(*local_manifest, peer_manifest, &plan);
  if (!status.ok()) {
    return status;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!local_cache_layout_.has_value() ||
      local_cache_layout_->incarnation_id != plan.source_incarnation ||
      local_cache_layout_->layout_generation != plan.source_layout_generation) {
    return Status(StatusCode::UNAVAILABLE,
                  "local cache layout changed during plan construction");
  }
  const auto existing = outgoing_plans_.find(peer_manifest.addr);
  if (existing != outgoing_plans_.end() &&
      existing->second.destination_incarnation ==
          peer_manifest.incarnation_id &&
      peer_manifest.layout_generation <
          existing->second.destination_layout_generation) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "peer cache layout generation is stale");
  }
  outgoing_plans_[peer_manifest.addr] = std::move(plan);
  return Status();
}

bool MooncakeTransferEngineCore::has_outgoing_plan(
    const std::string& remote_addr,
    CacheNamespace cache_namespace) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto plan_it = outgoing_plans_.find(remote_addr);
  if (plan_it == outgoing_plans_.end()) {
    return false;
  }
  return std::any_of(plan_it->second.regions.begin(),
                     plan_it->second.regions.end(),
                     [cache_namespace](const StridedRegionTemplate& region) {
                       return region.cache_namespace == cache_namespace;
                     });
}

bool MooncakeTransferEngineCore::has_reshard_plan(
    const std::string& remote_addr) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return outgoing_plans_.find(remote_addr) != outgoing_plans_.end();
}

Status MooncakeTransferEngineCore::bind_outgoing_regions(
    const std::string& remote_addr,
    const std::vector<KVTransferMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  std::optional<ReshardPlanTemplate> plan;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto plan_it = outgoing_plans_.find(remote_addr);
    if (plan_it == outgoing_plans_.end()) {
      return Status(StatusCode::UNAVAILABLE,
                    "no outgoing reshard plan for " + remote_addr);
    }
    plan = plan_it->second;
  }
  RequestRegionBinder binder;
  return binder.bind(*plan, mappings, cache_namespace, layer_id, regions);
}

Status MooncakeTransferEngineCore::bind_outgoing_regions_explicit(
    const std::string& remote_addr,
    const std::vector<ExplicitResourceMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  std::optional<ReshardPlanTemplate> plan;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto plan_it = outgoing_plans_.find(remote_addr);
    if (plan_it == outgoing_plans_.end()) {
      return Status(StatusCode::UNAVAILABLE,
                    "no outgoing reshard plan for " + remote_addr);
    }
    plan = plan_it->second;
  }
  RequestRegionBinder binder;
  return binder.bind_explicit(
      *plan, mappings, cache_namespace, layer_id, regions);
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngineCore::get_or_create_stub(uint64_t cluster_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return get_or_create_stub_locked(cluster_id);
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngineCore::get_or_create_stub_locked(uint64_t cluster_id) {
  auto it = stub_map_.find(cluster_id);
  if (it == stub_map_.end()) {
    auto [remote_ip, remote_port] = net::convert_uint64_to_ip_port(cluster_id);
    std::string remote_addr = remote_ip + ":" + std::to_string(remote_port);

    brpc::Channel* channel = new brpc::Channel();
    brpc::ChannelOptions options;
    options.timeout_ms = -1;
    std::string load_balancer = "";
    if (channel->Init(remote_addr.c_str(), load_balancer.c_str(), &options) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel for " << remote_addr;
      delete channel;
      return nullptr;
    }

    proto::MooncakeTransferEngineService_Stub* stub =
        new proto::MooncakeTransferEngineService_Stub(channel);
    stub_map_[cluster_id] = stub;
    return stub;
  }

  return it->second;
}

// ============================================================================
// MooncakeTransferEngine
// ============================================================================

MooncakeTransferEngine::MooncakeTransferEngine(const uint16_t listen_port,
                                               const torch::Device& device)
    : listen_port_(listen_port),
      device_(device),
      core_(MooncakeTransferEngineCore::get_instance()) {}

std::string MooncakeTransferEngine::initialize() {
  if (!core_.initialize(listen_port_, device_)) {
    LOG(ERROR) << "Failed to initialize MooncakeTransferEngineCore";
    return "";
  }
  return core_.addr();
}

bool MooncakeTransferEngine::register_memory(std::vector<void*> addrs,
                                             std::vector<size_t> lens,
                                             std::vector<uint64_t> buf_bytes) {
  if (addrs.size() != lens.size() || addrs.size() != buf_bytes.size()) {
    LOG(ERROR) << "register_memory input size mismatch, addrs=" << addrs.size()
               << ", lens=" << lens.size()
               << ", buf_bytes=" << buf_bytes.size();
    return false;
  }

  TransferEngine* engine = core_.engine();
  for (size_t i = 0; i < addrs.size(); ++i) {
    int32_t ret = engine->registerLocalMemory(
        addrs[i], lens[i], kWildcardLocation, true, true);
    if (ret != 0) {
      LOG(ERROR) << "registerLocalMemory failed, buf_id=" << i
                 << ", addr=" << addrs[i] << ", len=" << lens[i]
                 << ", ret=" << ret;
      return false;
    }
  }

  buf_bytes_.insert(buf_bytes_.end(), buf_bytes.begin(), buf_bytes.end());
  LOG(INFO) << "register_memory success, total_buf_num=" << buf_bytes_.size();

  return true;
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngine::create_rpc_channel(uint64_t cluster_id) {
  return core_.get_or_create_stub(cluster_id);
}

bool MooncakeTransferEngine::open_session(const uint64_t cluster_id,
                                          const std::string& remote_addr) {
  if (!core_.open_session(cluster_id, remote_addr)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(session_mutex_);
  ++session_ref_counts_[remote_addr];
  return true;
}

bool MooncakeTransferEngine::close_session(const uint64_t cluster_id,
                                           const std::string& remote_addr) {
  {
    std::lock_guard<std::mutex> lock(session_mutex_);
    const auto it = session_ref_counts_.find(remote_addr);
    if (it == session_ref_counts_.end()) {
      return true;
    }
    if (--it->second == 0) {
      session_ref_counts_.erase(it);
    }
  }
  if (core_.close_session(cluster_id, remote_addr)) {
    return true;
  }
  std::lock_guard<std::mutex> lock(session_mutex_);
  ++session_ref_counts_[remote_addr];
  return false;
}

bool MooncakeTransferEngine::link_sessions(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs) {
  if (cluster_ids.size() != remote_addrs.size() || cluster_ids.empty()) {
    LOG(ERROR) << "MoonCake link session endpoint sizes are invalid.";
    return false;
  }

  const std::optional<WorkerCacheLayoutManifest> local_manifest =
      core_.local_cache_layout();
  if (!local_manifest.has_value()) {
    LOG(ERROR) << "Local cache layout must be registered before linking.";
    return false;
  }

  std::vector<WorkerCacheLayoutManifest> remote_manifests;
  remote_manifests.reserve(cluster_ids.size());
  for (size_t index = 0; index < cluster_ids.size(); ++index) {
    proto::MooncakeTransferEngineService_Stub* stub =
        create_rpc_channel(cluster_ids[index]);
    if (stub == nullptr) {
      LOG(ERROR) << "Failed to create cache layout RPC channel, cluster_id="
                 << cluster_ids[index];
      return false;
    }
    proto::Empty request;
    proto::WorkerCacheLayoutManifest response;
    brpc::Controller controller;
    stub->GetCacheLayoutManifest(&controller, &request, &response, nullptr);
    if (controller.Failed()) {
      LOG(ERROR) << "GetCacheLayoutManifest failed: " << controller.ErrorText();
      return false;
    }
    WorkerCacheLayoutManifest remote_manifest;
    const Status status = cache_layout_from_proto(response, &remote_manifest);
    if (!status.ok()) {
      LOG(ERROR) << "Invalid remote cache layout from " << remote_addrs[index]
                 << ": " << status.message();
      return false;
    }
    if (remote_manifest.addr != remote_addrs[index] ||
        remote_manifest.cluster_id != cluster_ids[index]) {
      LOG(ERROR) << "Remote cache layout endpoint mismatch, expected="
                 << cluster_ids[index] << "/" << remote_addrs[index]
                 << ", manifest=" << remote_manifest.cluster_id << "/"
                 << remote_manifest.addr;
      return false;
    }
    remote_manifests.emplace_back(std::move(remote_manifest));
  }

  ReshardPlanner planner;
  const Status coverage =
      planner.validate_destination_coverage(remote_manifests, *local_manifest);
  if (!coverage.ok()) {
    LOG(ERROR) << "Remote cache layouts cannot cover local destination: "
               << coverage.message();
    return false;
  }

  std::vector<size_t> opened_indices;
  opened_indices.reserve(cluster_ids.size());
  for (size_t index = 0; index < cluster_ids.size(); ++index) {
    if (!planner.source_participates(remote_manifests[index],
                                     *local_manifest)) {
      continue;
    }
    if (!open_session(cluster_ids[index], remote_addrs[index])) {
      for (size_t opened_index : opened_indices) {
        close_session(cluster_ids[opened_index], remote_addrs[opened_index]);
      }
      return false;
    }
    opened_indices.emplace_back(index);
  }
  return true;
}

Status MooncakeTransferEngine::set_local_cache_layout(
    const WorkerCacheLayoutManifest& manifest) {
  return core_.set_local_cache_layout(manifest);
}

bool MooncakeTransferEngine::has_outgoing_plan(
    const std::string& remote_addr,
    CacheNamespace cache_namespace) const {
  return core_.has_outgoing_plan(remote_addr, cache_namespace);
}

bool MooncakeTransferEngine::has_reshard_plan(
    const std::string& remote_addr) const {
  return core_.has_reshard_plan(remote_addr);
}

Status MooncakeTransferEngine::bind_outgoing_regions(
    const std::string& remote_addr,
    const std::vector<KVTransferMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  return core_.bind_outgoing_regions(
      remote_addr, mappings, cache_namespace, layer_id, regions);
}

Status MooncakeTransferEngine::bind_outgoing_regions_explicit(
    const std::string& remote_addr,
    const std::vector<ExplicitResourceMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  return core_.bind_outgoing_regions_explicit(
      remote_addr, mappings, cache_namespace, layer_id, regions);
}

// Merge the source and destination block ids into a single block when both are
// consecutive.
void merge_block_ids(const std::vector<uint64_t>& src_blocks,
                     const std::vector<uint64_t>& dst_blocks,
                     std::vector<uint64_t>& merged_src_blocks,
                     std::vector<uint64_t>& merged_dst_blocks,
                     std::vector<uint64_t>& block_lengths) {
  size_t block_num = src_blocks.size();
  if (block_num == 0) {
    return;
  }

  std::vector<uint64_t> indices(block_num);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.begin(), indices.end(), [&src_blocks](uint64_t i, uint64_t j) {
        return src_blocks[i] < src_blocks[j];
      });

  std::vector<uint64_t> sorted_src_blocks;
  std::vector<uint64_t> sorted_dst_blocks;
  sorted_src_blocks.reserve(block_num);
  sorted_dst_blocks.reserve(block_num);
  for (uint64_t id : indices) {
    sorted_src_blocks.emplace_back(src_blocks[id]);
    sorted_dst_blocks.emplace_back(dst_blocks[id]);
  }

  uint64_t current_src_id = sorted_src_blocks[0];
  uint64_t current_dst_id = sorted_dst_blocks[0];
  uint64_t current_length = 1;
  merged_src_blocks.reserve(block_num);
  merged_dst_blocks.reserve(block_num);
  block_lengths.reserve(block_num);
  for (size_t i = 1; i < sorted_src_blocks.size(); ++i) {
    if (sorted_src_blocks[i] == sorted_src_blocks[i - 1] + 1 &&
        sorted_dst_blocks[i] == sorted_dst_blocks[i - 1] + 1) {
      current_length++;
    } else {
      merged_src_blocks.emplace_back(current_src_id);
      merged_dst_blocks.emplace_back(current_dst_id);
      block_lengths.emplace_back(current_length);
      current_src_id = sorted_src_blocks[i];
      current_dst_id = sorted_dst_blocks[i];
      current_length = 1;
    }
  }
  merged_src_blocks.emplace_back(current_src_id);
  merged_dst_blocks.emplace_back(current_dst_id);
  block_lengths.emplace_back(current_length);
}

bool MooncakeTransferEngine::move_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids,
    MoveOpcode move_opcode) {
  if (src_blocks.size() != dst_blocks.size()) {
    LOG(ERROR) << "src_blocks size must equal dst_blocks size, src="
               << src_blocks.size() << ", dst=" << dst_blocks.size();
    return false;
  }

  std::vector<int64_t> active_buf_ids = buf_ids;
  if (active_buf_ids.empty()) {
    active_buf_ids.resize(buf_bytes_.size());
    std::iota(active_buf_ids.begin(), active_buf_ids.end(), 0);
  }

  std::vector<BufferTransferMapping> mappings;
  mappings.reserve(active_buf_ids.size());
  for (int64_t buf_id : active_buf_ids) {
    BufferTransferMapping mapping;
    mapping.buf_id = buf_id;
    if (move_opcode == MoveOpcode::WRITE) {
      mapping.local_ids = src_blocks;
      mapping.remote_ids = dst_blocks;
    } else {
      mapping.local_ids = dst_blocks;
      mapping.remote_ids = src_blocks;
    }
    mappings.emplace_back(std::move(mapping));
  }
  return move_memory_groups(remote_addr, mappings, move_opcode);
}

bool MooncakeTransferEngine::move_memory_groups(
    const std::string& remote_addr,
    const std::vector<BufferTransferMapping>& mappings,
    MoveOpcode move_opcode) {
  std::vector<ByteRegion> regions;
  for (const BufferTransferMapping& mapping : mappings) {
    if (mapping.local_ids.size() != mapping.remote_ids.size()) {
      LOG(ERROR) << "local_ids size must equal remote_ids size, buf_id="
                 << mapping.buf_id << ", local=" << mapping.local_ids.size()
                 << ", remote=" << mapping.remote_ids.size();
      return false;
    }
    const int64_t buf_id = mapping.buf_id;
    if (buf_id < 0 || static_cast<size_t>(buf_id) >= buf_bytes_.size()) {
      LOG(ERROR) << "buf_id out of range, buf_id=" << buf_id
                 << ", buf_cnt=" << buf_bytes_.size();
      return false;
    }

    const uint64_t block_bytes = buf_bytes_[static_cast<size_t>(buf_id)];
    std::vector<uint64_t> merged_local_ids;
    std::vector<uint64_t> merged_remote_ids;
    std::vector<uint64_t> block_lengths;
    merge_block_ids(mapping.local_ids,
                    mapping.remote_ids,
                    merged_local_ids,
                    merged_remote_ids,
                    block_lengths);
    for (size_t i = 0; i < merged_local_ids.size(); ++i) {
      uint64_t local_block_id = merged_local_ids[i];
      uint64_t remote_block_id = merged_remote_ids[i];
      uint64_t block_length = block_lengths[i];
      if (multiply_overflows(local_block_id, block_bytes) ||
          multiply_overflows(remote_block_id, block_bytes) ||
          multiply_overflows(block_length, block_bytes)) {
        LOG(ERROR) << "MoonCake block wrapper offset overflow, buf_id="
                   << buf_id;
        return false;
      }
      ByteRegion region;
      region.local_buffer_id = static_cast<uint64_t>(buf_id);
      region.local_offset = local_block_id * block_bytes;
      region.remote_buffer_id = static_cast<uint64_t>(buf_id);
      region.remote_offset = remote_block_id * block_bytes;
      region.length = block_length * block_bytes;
      regions.emplace_back(std::move(region));
    }
  }

  return move_memory_regions(remote_addr, regions, move_opcode);
}

bool MooncakeTransferEngine::move_memory_regions(
    const std::string& remote_addr,
    const std::vector<ByteRegion>& regions,
    MoveOpcode move_opcode) {
  if (regions.empty()) {
    return true;
  }

  SegmentHandle remote_handle = core_.get_handle(remote_addr);
  if (remote_handle == static_cast<SegmentHandle>(-1)) {
    LOG(ERROR) << "remote addr does not exist: " << remote_addr;
    return false;
  }

  TransferEngine* engine = core_.engine();
  std::shared_ptr<TransferMetadata::SegmentDesc> remote_segment_desc =
      engine->getMetadata()->getSegmentDescByID(remote_handle);
  if (!remote_segment_desc) {
    LOG(ERROR) << "remote_segment_desc is null";
    return false;
  }
  std::shared_ptr<TransferMetadata::SegmentDesc> local_segment_desc =
      engine->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
  if (!local_segment_desc) {
    LOG(ERROR) << "local_segment_desc is null";
    return false;
  }

  TransferRequest::OpCode opcode = TransferRequest::READ;
  if (move_opcode == MoveOpcode::WRITE) {
    opcode = TransferRequest::WRITE;
  }

  std::vector<TransferRequest> entries;
  entries.reserve(regions.size());
  uint64_t total_bytes = 0;
  for (const ByteRegion& region : regions) {
    if (region.local_buffer_id >= local_segment_desc->buffers.size() ||
        region.remote_buffer_id >= remote_segment_desc->buffers.size()) {
      LOG(ERROR) << "MoonCake region buffer id out of range, local_id="
                 << region.local_buffer_id
                 << ", remote_id=" << region.remote_buffer_id
                 << ", local_count=" << local_segment_desc->buffers.size()
                 << ", remote_count=" << remote_segment_desc->buffers.size();
      return false;
    }
    if (region.length == 0) {
      LOG(ERROR) << "MoonCake region length must be positive.";
      return false;
    }

    const auto& local_buffer =
        local_segment_desc->buffers[region.local_buffer_id];
    const auto& remote_buffer =
        remote_segment_desc->buffers[region.remote_buffer_id];
    if (region.local_offset > local_buffer.length ||
        region.length > local_buffer.length - region.local_offset ||
        region.remote_offset > remote_buffer.length ||
        region.length > remote_buffer.length - region.remote_offset) {
      LOG(ERROR) << "MoonCake byte region is outside registered memory, "
                 << "local_buf=" << region.local_buffer_id
                 << ", local_offset=" << region.local_offset
                 << ", remote_buf=" << region.remote_buffer_id
                 << ", remote_offset=" << region.remote_offset
                 << ", length=" << region.length;
      return false;
    }
    if (region.length > std::numeric_limits<uint64_t>::max() - total_bytes) {
      LOG(ERROR) << "MoonCake transfer byte count overflow";
      return false;
    }
    total_bytes += region.length;

    TransferRequest entry;
    entry.opcode = opcode;
    entry.length = region.length;
    entry.source = reinterpret_cast<void*>(
        reinterpret_cast<char*>(local_buffer.addr) + region.local_offset);
    entry.target_id = remote_handle;
    entry.target_offset = remote_buffer.addr + region.remote_offset;
    entry.advise_retry_cnt = 0;
    entries.emplace_back(std::move(entry));
  }

  constexpr size_t kMaxRegionsPerBatch = 4096;
  Timer transfer_timer;
  for (size_t begin = 0; begin < entries.size(); begin += kMaxRegionsPerBatch) {
    const size_t end = std::min(begin + kMaxRegionsPerBatch, entries.size());
    std::vector<TransferRequest> batch_entries(entries.begin() + begin,
                                               entries.begin() + end);
    const BatchID batch_id = engine->allocateBatchID(batch_entries.size());
    mooncake::Status submit_status =
        engine->submitTransfer(batch_id, batch_entries);
    if (!submit_status.ok()) {
      LOG(ERROR) << "submit byte regions failed";
      COUNTER_INC(mooncake_transfer_failed_total);
      engine->freeBatchID(batch_id);
      return false;
    }
    const bool transfer_success = wait_batch(engine, batch_id);
    const mooncake::Status free_status = engine->freeBatchID(batch_id);
    if (!free_status.ok() || !transfer_success) {
      LOG(ERROR) << "MoonCake byte-region batch failed";
      COUNTER_INC(mooncake_transfer_failed_total);
      return false;
    }
  }

  const int64_t latency_microseconds = static_cast<int64_t>(
      transfer_timer.elapsed_seconds() * static_cast<double>(1000000));
  if (move_opcode == MoveOpcode::READ) {
    COUNTER_INC(mooncake_transfer_completed_total_read);
    COUNTER_ADD(mooncake_transfer_bytes_total_read, total_bytes);
    HISTOGRAM_OBSERVE(mooncake_transfer_latency_microseconds_read,
                      latency_microseconds);
  } else {
    COUNTER_INC(mooncake_transfer_completed_total_write);
    COUNTER_ADD(mooncake_transfer_bytes_total_write, total_bytes);
    HISTOGRAM_OBSERVE(mooncake_transfer_latency_microseconds_write,
                      latency_microseconds);
  }
  return true;
}

bool MooncakeTransferEngine::move_memory_by_global_offsets(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_offsets,
    const std::vector<uint64_t>& dst_offsets,
    size_t transfer_size,
    MoveOpcode move_opcode) {
  if (src_offsets.size() != dst_offsets.size() || transfer_size == 0) {
    LOG(ERROR) << "Invalid XTensor byte-region mapping, source="
               << src_offsets.size() << ", destination=" << dst_offsets.size()
               << ", length=" << transfer_size;
    return false;
  }
  std::vector<ByteRegion> regions;
  regions.reserve(src_offsets.size());
  for (size_t index = 0; index < src_offsets.size(); ++index) {
    ByteRegion region;
    region.local_buffer_id = 0;
    region.remote_buffer_id = 0;
    region.length = static_cast<uint64_t>(transfer_size);
    if (move_opcode == MoveOpcode::WRITE) {
      region.local_offset = src_offsets[index];
      region.remote_offset = dst_offsets[index];
    } else {
      region.local_offset = dst_offsets[index];
      region.remote_offset = src_offsets[index];
    }
    regions.emplace_back(std::move(region));
  }
  return move_memory_regions(remote_addr, regions, move_opcode);
}

bool MooncakeTransferEngine::pull_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids) {
  bool ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, buf_ids, MoveOpcode::READ);
  if (!ret) {
    LOG(ERROR) << "Pull memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

bool MooncakeTransferEngine::push_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids) {
  bool ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, buf_ids, MoveOpcode::WRITE);
  if (!ret) {
    LOG(ERROR) << "Push memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

// ============================================================================
// MooncakeTransferEngineService
// ============================================================================

void MooncakeTransferEngineService::OpenSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (request->addr().empty()) {
    LOG(ERROR) << "OpenSession request missing addr";
    response->set_ok(false);
    return;
  }

  std::string remote_addr(request->addr());
  MooncakeTransferEngineCore& core = MooncakeTransferEngineCore::get_instance();
  if (request->has_cache_layout_manifest()) {
    WorkerCacheLayoutManifest peer_manifest;
    const Status decode_status = cache_layout_from_proto(
        request->cache_layout_manifest(), &peer_manifest);
    if (!decode_status.ok() || peer_manifest.addr != remote_addr) {
      LOG(ERROR) << "OpenSession received an invalid peer cache layout: "
                 << decode_status.message();
      response->set_ok(false);
      return;
    }
    const Status plan_status = core.install_peer_cache_layout(peer_manifest);
    if (!plan_status.ok()) {
      LOG(ERROR) << "OpenSession failed to build outgoing reshard plan: "
                 << plan_status.message();
      response->set_ok(false);
      return;
    }
  }
  const bool result = core.open_session(
      0,
      remote_addr,
      /*increment_existing=*/!request->has_cache_layout_manifest());

  response->set_ok(result);
}

void MooncakeTransferEngineService::GetCacheLayoutManifest(
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
      MooncakeTransferEngineCore::get_instance().local_cache_layout();
  if (!manifest.has_value()) {
    LOG(ERROR) << "GetCacheLayoutManifest called before cache registration.";
    response->Clear();
    return;
  }
  cache_layout_to_proto(*manifest, response);
}

void MooncakeTransferEngineService::CloseSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (request->addr().empty()) {
    LOG(ERROR) << "CloseSession request missing addr";
    response->set_ok(false);
    return;
  }

  std::string remote_addr(request->addr());
  bool result =
      MooncakeTransferEngineCore::get_instance().close_session(0, remote_addr);

  response->set_ok(result);
}

}  // namespace xllm
