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

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include "core/kv_cache/transfer/mooncake_kv_cache_transfer_engine.h"

#if defined(USE_NPU)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/kv_cache/layout/kv_cache_capacity.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/transfer/kv_cache_transfer.h"
#include "core/kv_cache/transfer/virtual_kv_cache_transfer_memory_provider.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"
#include "worker.pb.h"

#define private public
#define protected public
#include "core/kv_cache/transfer/mooncake_kv_cache_transfer.h"
#undef private
#undef protected

namespace xllm {

namespace {

constexpr size_t kScaleBlockBytes = 96 * sizeof(float);

TransferKVInfo make_info(int32_t dst_dp_size,
                         int32_t dst_tp_size,
                         int32_t dst_dp_rank) {
  TransferKVInfo info;
  info.request_id = "req";
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {11, 12};
  mapping.remote_ids = {21, 22};
  info.mappings.emplace_back(std::move(mapping));
  info.dp_rank = dst_dp_rank;
  info.remote_instance_info.dp_size = dst_dp_size;

  int32_t dst_world_size = dst_dp_size * dst_tp_size;
  for (int32_t i = 0; i < dst_world_size; ++i) {
    info.remote_instance_info.cluster_ids.emplace_back(
        static_cast<uint64_t>(100 + i));
    info.remote_instance_info.addrs.emplace_back("addr_" + std::to_string(i));
  }

  return info;
}

ParallelArgs make_args(int32_t rank, int32_t world_size, int32_t dp_size) {
  return ParallelArgs(rank, world_size, dp_size, nullptr);
}

WorkerCacheLayoutManifest make_peer_manifest(const std::string& addr,
                                             const std::string& incarnation,
                                             uint64_t generation) {
  WorkerCacheLayoutManifest manifest;
  manifest.incarnation_id = incarnation;
  manifest.layout_generation = generation;
  manifest.fingerprint = "peer-test-model";
  manifest.backend = "cpu";
  manifest.layout_family = "token_head_dim";
  manifest.cluster_id = 1;
  manifest.addr = addr;
  manifest.listen_port = 20000;

  CacheTensorManifest tensor;
  tensor.cache_namespace = CacheNamespace::MAIN;
  tensor.layer_id = 0;
  tensor.role = static_cast<int32_t>(KVCacheTensorRole::KEY);
  tensor.group_id = cache_group_id(BlockType::KV);
  tensor.mooncake_buffer_id = 0;
  tensor.scalar_type = 0;
  tensor.element_bytes = 1;
  tensor.shape = {2, 1, 1, 1};
  tensor.stride = {1, 1, 1, 1};
  tensor.contiguous = true;
  tensor.resource_count = 2;
  tensor.resource_stride_bytes = 1;
  tensor.buffer_bytes = 2;
  tensor.block_token_capacity = 1;
  tensor.shard.kind = LogicalShardKind::SHARDED;
  tensor.shard.resource_scope = CacheResourceScope::BLOCK;
  LogicalSpan span;
  span.logical_tensor = "key";
  span.bytes_per_region = 1;
  span.repeat_count = 1;
  tensor.shard.spans.emplace_back(std::move(span));
  manifest.tensors.emplace_back(std::move(tensor));
  return manifest;
}

WorkerCacheLayoutManifest make_pcp_manifest(int32_t tp_rank,
                                            int32_t tp_size,
                                            int32_t cp_rank,
                                            int32_t cp_size,
                                            const std::string& addr,
                                            uint64_t cluster_id) {
  WorkerCacheLayoutManifest manifest =
      make_peer_manifest(addr, addr + "-incarnation", 1);
  manifest.cluster_id = cluster_id;
  manifest.coordinates.tp_rank = tp_rank;
  manifest.coordinates.tp_size = tp_size;
  manifest.coordinates.cp_rank = cp_rank;
  manifest.coordinates.cp_size = cp_size;
  manifest.tensors[0].mooncake_buffer_id = static_cast<int64_t>(cluster_id);
  manifest.tensors[0].shard.kind = LogicalShardKind::REPLICATED;
  manifest.tensors[0].shard.spans[0].owner_tp_rank = 0;
  return manifest;
}

class RecordingMooncakeTransferEngine final
    : public MooncakeKVCacheTransferEngine {
 public:
  struct MoveCall {
    std::string remote_addr;
    std::vector<BufferTransferMapping> mappings;
    MoveOpcode opcode;
  };

  struct PeerCall {
    std::string remote_addr;
    CachePeerMode mode;
  };

  struct RegionMoveCall {
    std::string remote_addr;
    std::vector<ByteRegion> regions;
    MoveOpcode opcode;
  };

  struct VirtualOffsetQuery {
    uint64_t cluster_id;
    std::string remote_addr;
    std::vector<uint64_t> block_ids;
  };

  RecordingMooncakeTransferEngine(uint16_t listen_port,
                                  const torch::Device& device)
      : MooncakeKVCacheTransferEngine(listen_port, device) {}

  bool register_memory(std::vector<void*> addrs,
                       std::vector<size_t> lens,
                       std::vector<uint64_t> buf_bytes,
                       std::shared_ptr<void> lifetime = {}) override {
    registered_addrs.emplace_back(std::move(addrs));
    registered_lens.emplace_back(std::move(lens));
    registered_block_bytes.emplace_back(std::move(buf_bytes));
    registered_lifetimes.emplace_back(std::move(lifetime));
    return true;
  }

  bool move_memory_groups(const std::string& remote_addr,
                          const std::vector<BufferTransferMapping>& mappings,
                          MoveOpcode opcode) override {
    move_calls.emplace_back(MoveCall{remote_addr, mappings, opcode});
    return move_result;
  }

  bool move_memory_regions(const std::string& remote_addr,
                           const std::vector<ByteRegion>& regions,
                           MoveOpcode opcode) override {
    region_move_calls.emplace_back(
        RegionMoveCall{remote_addr, regions, opcode});
    return move_result;
  }

  bool move_memory_regions_with_session(const std::string& remote_addr,
                                        const std::vector<ByteRegion>& regions,
                                        MoveOpcode opcode) override {
    return move_memory_regions(remote_addr, regions, opcode);
  }

  bool has_reshard_plan(const std::string& remote_addr) const override {
    return planned_addrs.find(remote_addr) != planned_addrs.end();
  }

  bool fetch_cache_layout(uint64_t cluster_id,
                          const std::string& remote_addr,
                          WorkerCacheLayoutManifest* manifest) override {
    ++fetch_cache_layout_calls;
    const auto it = remote_layouts.find(remote_addr);
    if (it == remote_layouts.end() || it->second.cluster_id != cluster_id) {
      return false;
    }
    *manifest = it->second;
    return true;
  }

  bool get_remote_virtual_offsets(
      uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& block_ids,
      WorkerCacheLayoutManifest* manifest,
      std::vector<VirtualKVCacheLayerGlobalOffsets>* layer_offsets) override {
    virtual_offset_queries.emplace_back(
        VirtualOffsetQuery{src_cluster_id, src_addr, block_ids});
    if (use_native_virtual_offsets) {
      return MooncakeKVCacheTransferEngine::get_remote_virtual_offsets(
          src_cluster_id, src_addr, block_ids, manifest, layer_offsets);
    }
    if (!virtual_offset_query_result) {
      return false;
    }
    *manifest = virtual_offset_manifest;
    *layer_offsets = virtual_layer_offsets;
    return true;
  }

  bool fetch_virtual_offsets(
      uint64_t /*cluster_id*/,
      const proto::VirtualKVCacheOffsetsRequest& request,
      proto::VirtualKVCacheOffsetsResponse* response) override {
    virtual_wire_requests.emplace_back(request);
    *response = virtual_wire_response;
    return virtual_wire_result;
  }

  bool set_remote_peer(uint64_t /*cluster_id*/,
                       const std::string& remote_addr,
                       const WorkerCacheLayoutManifest& /*manifest*/,
                       CachePeerMode mode) override {
    peer_calls.emplace_back(PeerCall{remote_addr, mode});
    if (mode == CachePeerMode::ABSENT && remote_addr == failed_absent_peer) {
      return false;
    }
    return mode == CachePeerMode::ABSENT || remote_addr != failed_peer;
  }

  bool open_local_session(const std::string& remote_addr) override {
    opened_sessions.emplace_back(remote_addr);
    return remote_addr != failed_session_peer;
  }

  bool close_local_session(const std::string& remote_addr) override {
    closed_sessions.emplace_back(remote_addr);
    return remote_addr != failed_close_session_peer;
  }

  bool move_result = true;
  bool use_native_virtual_offsets = false;
  bool virtual_wire_result = true;
  proto::VirtualKVCacheOffsetsResponse virtual_wire_response;
  std::vector<proto::VirtualKVCacheOffsetsRequest> virtual_wire_requests;
  bool virtual_offset_query_result = true;
  WorkerCacheLayoutManifest virtual_offset_manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> virtual_layer_offsets;
  std::vector<VirtualOffsetQuery> virtual_offset_queries;
  std::unordered_set<std::string> planned_addrs;
  std::unordered_map<std::string, WorkerCacheLayoutManifest> remote_layouts;
  size_t fetch_cache_layout_calls = 0;
  std::string failed_peer;
  std::string failed_absent_peer;
  std::string failed_close_session_peer;
  std::string failed_session_peer;
  std::vector<std::vector<void*>> registered_addrs;
  std::vector<std::vector<size_t>> registered_lens;
  std::vector<std::vector<uint64_t>> registered_block_bytes;
  std::vector<std::shared_ptr<void>> registered_lifetimes;
  std::vector<MoveCall> move_calls;
  std::vector<RegionMoveCall> region_move_calls;
  std::vector<PeerCall> peer_calls;
  std::vector<std::string> opened_sessions;
  std::vector<std::string> closed_sessions;
};

struct MemoryProviderOffsetQuery {
  std::string model_id;
  int64_t layer_id;
  uint64_t block_id;
  size_t block_size;
};

struct SharedTransferMemoryState {
  std::vector<uint8_t> backing = std::vector<uint8_t>(256);
  size_t page_size = 256;
  std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> block_offsets;
  std::vector<MemoryProviderOffsetQuery> offset_queries;
  std::unordered_map<uint64_t, VirtualKVCacheBlockGlobalOffsets> role_offsets;
  std::unordered_map<
      int64_t,
      std::unordered_map<uint64_t, VirtualKVCacheBlockGlobalOffsets>>
      layer_role_offsets;
};

class RecordingVirtualKVCacheTransferMemoryProvider final
    : public VirtualKVCacheTransferMemoryProvider {
 public:
  explicit RecordingVirtualKVCacheTransferMemoryProvider(
      std::shared_ptr<SharedTransferMemoryState> state)
      : state_(std::move(state)) {}

  VirtualKVCacheTransferMemoryRegion memory_region() const override {
    return {state_->backing.data(),
            state_->backing.size(),
            state_,
            state_->page_size};
  }

  std::optional<VirtualKVCacheBlockGlobalOffsets> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      uint64_t block_id) const override {
    state_->offset_queries.emplace_back(
        MemoryProviderOffsetQuery{model_id, layer_id, block_id, 0});
    const auto layer_it = state_->layer_role_offsets.find(layer_id);
    if (layer_it != state_->layer_role_offsets.end()) {
      const auto block_it = layer_it->second.find(block_id);
      return block_it != layer_it->second.end()
                 ? std::optional<VirtualKVCacheBlockGlobalOffsets>(
                       block_it->second)
                 : std::nullopt;
    }
    const auto it = state_->role_offsets.find(block_id);
    if (it == state_->role_offsets.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  std::pair<uint64_t, uint64_t> get_global_offsets_for_block(
      const std::string& model_id,
      int64_t layer_id,
      uint64_t block_id,
      size_t block_size) const override {
    state_->offset_queries.emplace_back(
        MemoryProviderOffsetQuery{model_id, layer_id, block_id, block_size});
    const auto offset_it = state_->block_offsets.find(block_id);
    if (offset_it == state_->block_offsets.end()) {
      return {std::numeric_limits<uint64_t>::max(),
              std::numeric_limits<uint64_t>::max()};
    }
    return offset_it->second;
  }

 private:
  std::shared_ptr<SharedTransferMemoryState> state_;
};

void register_virtual_memory_transfer_cache(
    MooncakeVirtualKVCacheTransfer& transfer) {
  proto::KVCacheShape proto_shape;
  proto_shape.add_key_cache_shape(4);
  proto_shape.add_key_cache_shape(2);
  proto_shape.add_value_cache_shape(4);
  proto_shape.add_value_cache_shape(2);
  const KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
  std::vector<KVCache> caches;
  caches.reserve(1);
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}, torch::dtype(torch::kFloat32)),
                     torch::zeros({4, 2}, torch::dtype(torch::kFloat32))});
  transfer.register_kv_cache(caches, shape, torch::kFloat32);
}

TEST(MooncakeVirtualKVCacheTransferTest,
     EachAdapterReceivesBufferMetadataAndLease) {
  auto shared_state = std::make_shared<SharedTransferMemoryState>();
  MooncakeVirtualKVCacheTransfer first_transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(
          shared_state));
  auto first_engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* first_engine_ptr = first_engine.get();
  first_transfer.mooncake_te_ = std::move(first_engine);

  register_virtual_memory_transfer_cache(first_transfer);

  ASSERT_EQ(first_engine_ptr->registered_addrs.size(), 1U);
  EXPECT_EQ(first_engine_ptr->registered_addrs[0],
            std::vector<void*>({shared_state->backing.data()}));
  ASSERT_EQ(first_engine_ptr->registered_lens.size(), 1U);
  EXPECT_EQ(first_engine_ptr->registered_lens[0],
            std::vector<size_t>({shared_state->backing.size()}));
  ASSERT_EQ(first_engine_ptr->registered_block_bytes.size(), 1U);
  EXPECT_EQ(first_engine_ptr->registered_block_bytes[0],
            std::vector<uint64_t>({2 * sizeof(float)}));
  ASSERT_EQ(first_engine_ptr->registered_lifetimes.size(), 1U);
  EXPECT_EQ(first_engine_ptr->registered_lifetimes[0], shared_state);

  MooncakeVirtualKVCacheTransfer second_transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(
          shared_state));
  auto second_engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* second_engine_ptr = second_engine.get();
  second_transfer.mooncake_te_ = std::move(second_engine);

  register_virtual_memory_transfer_cache(second_transfer);

  EXPECT_EQ(second_engine_ptr->registered_addrs,
            first_engine_ptr->registered_addrs);
  EXPECT_EQ(second_engine_ptr->registered_lens,
            first_engine_ptr->registered_lens);
  EXPECT_EQ(second_engine_ptr->registered_block_bytes,
            first_engine_ptr->registered_block_bytes);
  ASSERT_EQ(second_engine_ptr->registered_lifetimes.size(), 1U);
  EXPECT_EQ(second_engine_ptr->registered_lifetimes[0], shared_state);
}

TEST(MooncakeVirtualKVCacheTransferTest,
     RegistersUnequalMlaRolesWithSparseIndexOwnership) {
  constexpr size_t kPageSize = 2 * 1024 * 1024;
  constexpr int64_t kBlockCount = 4;
  constexpr int64_t kBlockTokens = 128;
  const std::vector<bool> index_layers = {true, false, true};
  auto state = std::make_shared<SharedTransferMemoryState>();
  state->page_size = kPageSize;
  state->backing.resize(8 * kPageSize);
  MooncakeVirtualKVCacheTransfer transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(state));
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  transfer.mooncake_te_ = std::move(engine);
  transfer.addr_ = "virtual-mla";
  transfer.set_model_id("glm-test");

  ModelArgs model_args;
  model_args.model_type("glm_moe_dsa")
      .n_layers(3)
      .n_heads(64)
      .n_kv_heads(64)
      .head_dim(128)
      .enable_mla(true)
      .kv_lora_rank(512)
      .qk_rope_head_dim(64)
      .index_n_heads(32)
      .index_head_dim(128);
  KVCacheCapacity capacity;
  capacity.n_blocks(kBlockCount).block_size(kBlockTokens);
  const KVCacheShape shape(capacity, model_args, /*world_size=*/2);
  std::vector<KVCache> caches;
  caches.reserve(index_layers.size());
  for (bool owns_index : index_layers) {
    KVCacheTensors kv_tensors{
        torch::zeros(shape.key_cache_shape(), torch::kBFloat16),
        torch::zeros(shape.value_cache_shape(), torch::kBFloat16)};
    if (owns_index) {
      IndexedKVCacheTensors indexed_tensors;
      indexed_tensors.kv_cache_tensors = std::move(kv_tensors);
      indexed_tensors.index_cache =
          torch::zeros(shape.index_cache_shape(), torch::kBFloat16);
      caches.emplace_back(std::move(indexed_tensors));
    } else {
      caches.emplace_back(std::move(kv_tensors));
    }
  }
  transfer.configure_cache_layout(make_args(/*rank=*/1,
                                            /*world_size=*/2,
                                            /*dp_size=*/1),
                                  model_args,
                                  kBlockTokens,
                                  /*is_spec_draft=*/false);

  transfer.register_kv_cache(caches, shape, torch::kBFloat16);

  ASSERT_EQ(engine_observer->registered_addrs.size(), 1U);
  EXPECT_EQ(engine_observer->registered_addrs[0],
            std::vector<void*>({state->backing.data()}));
  ASSERT_EQ(engine_observer->registered_lens.size(), 1U);
  EXPECT_EQ(engine_observer->registered_lens[0],
            std::vector<size_t>({state->backing.size()}));
  ASSERT_EQ(engine_observer->registered_lifetimes.size(), 1U);
  EXPECT_EQ(engine_observer->registered_lifetimes[0], state);
  const WorkerCacheLayoutManifest& layout = transfer.local_cache_layout_;
#if defined(USE_MLU) || defined(USE_ILU)
  EXPECT_EQ(layout.layout_family, "head_token_dim_virtual_memory_v2");
#else
  EXPECT_EQ(layout.layout_family, "token_head_dim_virtual_memory_v2");
#endif
  EXPECT_EQ(layout.coordinates.tp_rank, 1);
  EXPECT_EQ(layout.coordinates.tp_size, 2);
  ASSERT_EQ(layout.tensors.size(), 8U);
  const std::vector<KVCacheTensorRole> roles = {KVCacheTensorRole::KEY,
                                                KVCacheTensorRole::VALUE,
                                                KVCacheTensorRole::INDEX};
  const std::vector<uint64_t> block_bytes = {128 * 1024, 16 * 1024, 32 * 1024};
  const std::vector<std::vector<int64_t>> shapes = {shape.key_cache_shape(),
                                                    shape.value_cache_shape(),
                                                    shape.index_cache_shape()};
  for (size_t layer = 0; layer < index_layers.size(); ++layer) {
    for (size_t role = 0; role < roles.size(); ++role) {
      SCOPED_TRACE(testing::Message() << "layer=" << layer
                                      << ", role=" << roles[role].to_string());
      const auto tensor = std::find_if(
          layout.tensors.begin(),
          layout.tensors.end(),
          [layer, &roles, role](const CacheTensorManifest& manifest) {
            return manifest.layer_id == static_cast<int64_t>(layer) &&
                   manifest.role ==
                       static_cast<int32_t>(
                           static_cast<KVCacheTensorRole::Value>(roles[role]));
          });
      if (roles[role] == KVCacheTensorRole::INDEX && !index_layers[layer]) {
        EXPECT_EQ(tensor, layout.tensors.end());
        continue;
      }
      ASSERT_NE(tensor, layout.tensors.end());
      EXPECT_EQ(tensor->cache_namespace, CacheNamespace::MAIN);
      EXPECT_EQ(tensor->group_id, cache_group_id(BlockType::KV));
      EXPECT_EQ(tensor->mooncake_buffer_id, 0);
      EXPECT_EQ(tensor->buffer_bytes, state->backing.size());
      EXPECT_EQ(tensor->resource_count, kBlockCount);
      EXPECT_EQ(tensor->resource_stride_bytes, block_bytes[role]);
      EXPECT_EQ(tensor->scalar_type, static_cast<int32_t>(torch::kBFloat16));
      EXPECT_EQ(tensor->element_bytes, 2U);
      EXPECT_EQ(tensor->block_token_capacity, kBlockTokens);
      EXPECT_EQ(tensor->shape, shapes[role]);
      EXPECT_TRUE(tensor->contiguous);
      EXPECT_TRUE(tensor->explicit_resource_offsets);
      EXPECT_EQ(tensor->shard.kind, LogicalShardKind::REPLICATED);
      EXPECT_EQ(tensor->shard.resource_scope, CacheResourceScope::BLOCK);
      ASSERT_EQ(tensor->shard.spans.size(), 1U);
      EXPECT_EQ(tensor->shard.spans[0].bytes_per_region, block_bytes[role]);
      EXPECT_EQ(tensor->shard.spans[0].owner_tp_rank, 0);
    }
  }
  EXPECT_TRUE(state->offset_queries.empty());
}

TEST(MooncakeVirtualKVCacheTransferTest,
     MoveMemoryRegionsWithSessionRequiresActivePeer) {
  MooncakeKVCacheTransferEngine engine(/*listen_port=*/0,
                                       torch::Device(torch::kCPU));
  ByteRegion region;
  region.length = 1;
  EXPECT_FALSE(engine.move_memory_regions_with_session(
      "source", {region}, MooncakeTransferEngine::MoveOpcode::READ));
}

class MooncakeVirtualKVCachePullTest : public testing::Test {
 protected:
  static constexpr uint64_t kSourceClusterId = 17;
  static constexpr uint64_t kPageBytes = 2 * 1024 * 1024;
  static constexpr int64_t kBlockTokens = 128;

  void SetUp() override {
    state_ = std::make_shared<SharedTransferMemoryState>();
    state_->page_size = kPageBytes;
    state_->backing.resize(10 * kPageBytes);
    transfer_ = std::make_unique<MooncakeVirtualKVCacheTransfer>(
        /*device_id=*/0,
        /*listen_port=*/0,
        torch::Device(torch::kCPU),
        std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(
            state_));
    auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
        /*listen_port=*/0, torch::Device(torch::kCPU));
    engine_ = engine.get();
    transfer_->mooncake_te_ = std::move(engine);
    transfer_->set_model_id("glm-pull");
    transfer_->addr_ = "destination";

    ModelArgs model_args;
    model_args.model_type("glm_moe_dsa")
        .n_layers(3)
        .n_heads(64)
        .n_kv_heads(64)
        .head_dim(128)
        .enable_mla(true)
        .kv_lora_rank(512)
        .qk_rope_head_dim(64)
        .index_n_heads(32)
        .index_head_dim(128);
    KVCacheCapacity capacity;
    capacity.n_blocks(4).block_size(kBlockTokens);
    const KVCacheShape shape(capacity, model_args, /*world_size=*/2);
    std::vector<KVCache> caches;
    caches.reserve(3);
    for (int64_t layer_id = 0; layer_id < 3; ++layer_id) {
      KVCacheTensors kv_tensors{
          torch::zeros(shape.key_cache_shape(), torch::kBFloat16),
          torch::zeros(shape.value_cache_shape(), torch::kBFloat16)};
      if (layer_id == 1) {
        caches.emplace_back(std::move(kv_tensors));
        continue;
      }
      IndexedKVCacheTensors indexed_tensors;
      indexed_tensors.kv_cache_tensors = std::move(kv_tensors);
      indexed_tensors.index_cache =
          torch::zeros(shape.index_cache_shape(), torch::kBFloat16);
      caches.emplace_back(std::move(indexed_tensors));
    }
    transfer_->configure_cache_layout(make_args(/*rank=*/1,
                                                /*world_size=*/2,
                                                /*dp_size=*/1),
                                      model_args,
                                      kBlockTokens,
                                      /*is_spec_draft=*/false);
    transfer_->register_kv_cache(caches, shape, torch::kBFloat16);

    mapping_.group_id = cache_group_id(BlockType::KV);
    mapping_.local_ids = {1, 3};
    mapping_.remote_ids = {0, 2};
    engine_->virtual_offset_manifest = transfer_->local_cache_layout_;
    engine_->virtual_offset_manifest.cluster_id = kSourceClusterId;
    engine_->virtual_offset_manifest.addr = "source";
    engine_->virtual_offset_manifest.incarnation_id = "source-incarnation";
    engine_->virtual_layer_offsets.resize(3);
    for (int64_t layer_id = 0; layer_id < 3; ++layer_id) {
      auto& local_layer = state_->layer_role_offsets[layer_id];
      auto& remote_layer =
          engine_->virtual_layer_offsets[static_cast<size_t>(layer_id)];
      remote_layer.k_offsets.reserve(mapping_.remote_ids.size());
      remote_layer.v_offsets.reserve(mapping_.remote_ids.size());
      remote_layer.index_offsets.reserve(mapping_.remote_ids.size());
      for (size_t index = 0; index < mapping_.local_ids.size(); ++index) {
        const uint64_t local_id = mapping_.local_ids[index];
        const uint64_t remote_id = mapping_.remote_ids[index];
        VirtualKVCacheBlockGlobalOffsets local_offsets;
        local_offsets.k_offset =
            static_cast<uint64_t>(layer_id * 3) * kPageBytes +
            local_id * 128 * 1024;
        local_offsets.v_offset =
            static_cast<uint64_t>(layer_id * 3 + 1) * kPageBytes +
            local_id * 16 * 1024;
        remote_layer.k_offsets.emplace_back(
            static_cast<uint64_t>(8 - layer_id * 3) * kPageBytes +
            remote_id * 128 * 1024);
        remote_layer.v_offsets.emplace_back(
            static_cast<uint64_t>(7 - layer_id * 3) * kPageBytes +
            remote_id * 16 * 1024);
        if (layer_id != 1) {
          local_offsets.index_offset =
              static_cast<uint64_t>(layer_id * 3 + 2) * kPageBytes +
              local_id * 32 * 1024;
          remote_layer.index_offsets.emplace_back(
              static_cast<uint64_t>(6 - layer_id * 3) * kPageBytes +
              remote_id * 32 * 1024);
        }
        local_layer.emplace(local_id, std::move(local_offsets));
      }
    }
  }

  bool pull() {
    return transfer_->pull_kv_blocks(kSourceClusterId, "source", {mapping_});
  }

  void expect_no_move() const {
    EXPECT_TRUE(engine_->region_move_calls.empty());
    EXPECT_TRUE(engine_->move_calls.empty());
  }

  void enable_protocol_queries() {
    engine_->use_native_virtual_offsets = true;
    engine_->remote_layouts.emplace("source", engine_->virtual_offset_manifest);
    WorkerCacheLayoutManifest owner = engine_->virtual_offset_manifest;
    owner.addr = "owner-source";
    owner.cluster_id = kSourceClusterId - 1;
    owner.incarnation_id = "owner-source-incarnation";
    owner.coordinates.tp_rank = 0;
    engine_->remote_layouts.emplace("owner-source", std::move(owner));
    ASSERT_TRUE(engine_->link_sessions({kSourceClusterId - 1, kSourceClusterId},
                                       {"owner-source", "source"}));
    auto& response = engine_->virtual_wire_response;
    response.set_ok(true);
    response.set_incarnation_id(
        engine_->virtual_offset_manifest.incarnation_id);
    response.set_layout_generation(
        engine_->virtual_offset_manifest.layout_generation);
    response.set_physical_page_size_bytes(kPageBytes);
    for (size_t layer_id = 0; layer_id < 3; ++layer_id) {
      auto* layer = response.add_layer_offsets();
      layer->set_layer_id(static_cast<int64_t>(layer_id));
      const auto& offsets = engine_->virtual_layer_offsets[layer_id];
      layer->mutable_k_offsets()->Assign(offsets.k_offsets.begin(),
                                         offsets.k_offsets.end());
      layer->mutable_v_offsets()->Assign(offsets.v_offsets.begin(),
                                         offsets.v_offsets.end());
      layer->mutable_index_offsets()->Assign(offsets.index_offsets.begin(),
                                             offsets.index_offsets.end());
    }
  }

  void TearDown() override {
    if (engine_->use_native_virtual_offsets) {
      EXPECT_TRUE(engine_->close_session(kSourceClusterId - 1, "owner-source"));
      EXPECT_TRUE(engine_->close_session(kSourceClusterId, "source"));
    }
  }

  std::shared_ptr<SharedTransferMemoryState> state_;
  std::unique_ptr<MooncakeVirtualKVCacheTransfer> transfer_;
  RecordingMooncakeTransferEngine* engine_ = nullptr;
  KVTransferMapping mapping_;
};

TEST_F(MooncakeVirtualKVCachePullTest,
       ReadsUnequalMlaRolesWithSparseIndexAndIndependentPhysicalOffsets) {
  ASSERT_EQ(transfer_->local_cache_layout_.coordinates.tp_rank, 1);
  for (const CacheTensorManifest& tensor :
       transfer_->local_cache_layout_.tensors) {
    ASSERT_EQ(tensor.shard.kind, LogicalShardKind::REPLICATED);
    ASSERT_EQ(tensor.shard.spans[0].owner_tp_rank, 0);
  }
  ASSERT_TRUE(pull());
  ASSERT_EQ(engine_->virtual_offset_queries.size(), 1U);
  EXPECT_EQ(engine_->virtual_offset_queries[0].cluster_id, kSourceClusterId);
  EXPECT_EQ(engine_->virtual_offset_queries[0].remote_addr, "source");
  EXPECT_EQ(engine_->virtual_offset_queries[0].block_ids, mapping_.remote_ids);

  std::vector<ByteRegion> regions;
  regions.reserve(16);
  for (const auto& call : engine_->region_move_calls) {
    EXPECT_EQ(call.remote_addr, "source");
    EXPECT_EQ(call.opcode, MooncakeTransferEngine::MoveOpcode::READ);
    regions.insert(regions.end(), call.regions.begin(), call.regions.end());
  }
  ASSERT_EQ(regions.size(), 16U);
  for (int64_t layer_id = 0; layer_id < 3; ++layer_id) {
    const auto& remote_layer =
        engine_->virtual_layer_offsets[static_cast<size_t>(layer_id)];
    for (size_t index = 0; index < mapping_.local_ids.size(); ++index) {
      const auto& local =
          state_->layer_role_offsets.at(layer_id).at(mapping_.local_ids[index]);
      const auto expect_region = [&](uint64_t local_offset,
                                     uint64_t remote_offset,
                                     uint64_t length) {
        SCOPED_TRACE(testing::Message() << "layer=" << layer_id << ", block="
                                        << index << ", bytes=" << length);
        EXPECT_NE(local_offset, remote_offset);
        EXPECT_EQ(std::count_if(regions.begin(),
                                regions.end(),
                                [=](const ByteRegion& region) {
                                  return region.local_buffer_id == 0 &&
                                         region.remote_buffer_id == 0 &&
                                         region.local_offset == local_offset &&
                                         region.remote_offset ==
                                             remote_offset &&
                                         region.length == length;
                                }),
                  1);
      };
      expect_region(local.k_offset, remote_layer.k_offsets[index], 128 * 1024);
      expect_region(local.v_offset, remote_layer.v_offsets[index], 16 * 1024);
      if (local.index_offset.has_value()) {
        expect_region(
            *local.index_offset, remote_layer.index_offsets[index], 32 * 1024);
      }
    }
  }
  EXPECT_TRUE(engine_->move_calls.empty());
  ASSERT_EQ(state_->offset_queries.size(), 6U);
  for (const auto& query : state_->offset_queries) {
    EXPECT_EQ(query.model_id, "glm-pull");
    EXPECT_TRUE(query.block_id == 1 || query.block_id == 3);
  }
}

TEST_F(MooncakeVirtualKVCachePullTest,
       MissingModelIdentityRejectsBeforeRemoteQuery) {
  transfer_->set_model_id("");
  EXPECT_FALSE(pull());
  EXPECT_TRUE(engine_->virtual_offset_queries.empty());
  EXPECT_TRUE(state_->offset_queries.empty());
  expect_no_move();
}

#if defined(USE_NPU)
TEST(MooncakeVirtualKVCacheTransferTest,
     FactoryRejectsVirtualMemoryWithoutModelIdentity) {
  auto state = std::make_shared<SharedTransferMemoryState>();
  EXPECT_EQ(KVCacheTransferFactory::create(
                /*transfer_listen_port=*/0,
                torch::Device(torch::kCPU),
                /*model_type=*/"glm_moe_dsa",
                /*model_id=*/"",
                std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(
                    state)),
            nullptr);
}
#endif

TEST_F(MooncakeVirtualKVCachePullTest, FailedRemoteQueryDoesNotReadMemory) {
  engine_->virtual_offset_query_result = false;
  EXPECT_FALSE(pull());
  ASSERT_EQ(engine_->virtual_offset_queries.size(), 1U);
  EXPECT_TRUE(state_->offset_queries.empty());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       MissingLastLayerLocalMappingRejectsBeforeAnyRead) {
  state_->layer_role_offsets[2].erase(mapping_.local_ids.back());
  EXPECT_FALSE(pull());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       MissingLastLayerLocalIndexRejectsBeforeAnyRead) {
  state_->layer_role_offsets[2]
      .at(mapping_.local_ids.back())
      .index_offset.reset();
  EXPECT_FALSE(pull());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       MissingOrExtraRemoteIndexRejectsBeforeAnyRead) {
  const auto original = engine_->virtual_layer_offsets;
  engine_->virtual_layer_offsets[2].index_offsets.clear();
  EXPECT_FALSE(pull());
  expect_no_move();
  engine_->virtual_layer_offsets = original;
  engine_->virtual_layer_offsets[1].index_offsets = {0, 32 * 1024};
  EXPECT_FALSE(pull());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       MalformedRemoteLayerAndRoleCountsRejectBeforeAnyRead) {
  const auto original = engine_->virtual_layer_offsets;
  engine_->virtual_layer_offsets.pop_back();
  EXPECT_FALSE(pull());
  expect_no_move();
  engine_->virtual_layer_offsets = original;
  engine_->virtual_layer_offsets.emplace_back(original.back());
  EXPECT_FALSE(pull());
  expect_no_move();
  engine_->virtual_layer_offsets = original;
  engine_->virtual_layer_offsets[2].v_offsets.pop_back();
  EXPECT_FALSE(pull());
  expect_no_move();
  engine_->virtual_layer_offsets = original;
  engine_->virtual_layer_offsets[2].index_offsets.pop_back();
  EXPECT_FALSE(pull());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       IncompatibleRemoteLayoutRejectsBeforeAnyRead) {
  engine_->virtual_offset_manifest.fingerprint = "different-model";
  EXPECT_FALSE(pull());
  EXPECT_TRUE(state_->offset_queries.empty());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       OutOfBoundsRemotePhysicalOffsetRejectsBeforeAnyRead) {
  engine_->virtual_layer_offsets[2].k_offsets.back() = state_->backing.size();
  EXPECT_FALSE(pull());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       OutOfBoundsLocalPhysicalOffsetRejectsBeforeAnyRead) {
  state_->layer_role_offsets[2].at(mapping_.local_ids.back()).v_offset =
      state_->backing.size();
  EXPECT_FALSE(pull());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest, ReadFailureIsReported) {
  engine_->move_result = false;
  EXPECT_FALSE(pull());
  ASSERT_FALSE(engine_->region_move_calls.empty());
  EXPECT_EQ(engine_->region_move_calls[0].opcode,
            MooncakeTransferEngine::MoveOpcode::READ);
  EXPECT_TRUE(engine_->move_calls.empty());
}

TEST_F(MooncakeVirtualKVCachePullTest,
       MalformedMappingsRejectWithoutReadingMemory) {
  EXPECT_FALSE(transfer_->pull_kv_blocks(kSourceClusterId, "source", {}));
  expect_no_move();
  EXPECT_FALSE(transfer_->pull_kv_blocks(
      kSourceClusterId, "source", {mapping_, mapping_}));
  expect_no_move();
  mapping_.remote_ids.pop_back();
  EXPECT_FALSE(pull());
  expect_no_move();
  mapping_.remote_ids = {0, 2};
  mapping_.group_id = cache_group_id(BlockType::LINEAR);
  EXPECT_FALSE(pull());
  expect_no_move();
  mapping_.group_id = cache_group_id(BlockType::KV);
  mapping_.local_ids = {1, 4};
  EXPECT_FALSE(pull());
  expect_no_move();
  mapping_.local_ids = {1, 3};
  mapping_.remote_ids = {0, 4};
  EXPECT_FALSE(pull());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       AllowsEmptyLinearHelperMappingAlongsideKvMapping) {
  KVTransferMapping empty_linear_mapping;
  empty_linear_mapping.group_id = cache_group_id(BlockType::LINEAR);

  EXPECT_TRUE(transfer_->pull_kv_blocks(
      kSourceClusterId, "source", {mapping_, empty_linear_mapping}));
  ASSERT_EQ(engine_->virtual_offset_queries.size(), 1U);
  EXPECT_EQ(engine_->virtual_offset_queries[0].block_ids, mapping_.remote_ids);
  EXPECT_FALSE(engine_->region_move_calls.empty());

  engine_->region_move_calls.clear();
  engine_->move_calls.clear();
  empty_linear_mapping.local_ids = {0};
  empty_linear_mapping.remote_ids = {0};
  EXPECT_FALSE(transfer_->pull_kv_blocks(
      kSourceClusterId, "source", {mapping_, empty_linear_mapping}));
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       SourceServiceReturnsPerLayerPhysicalOffsetsAndSparseIndex) {
  MooncakeKVCacheTransferService service;
  proto::VirtualKVCacheOffsetsRequest request;
  request.set_incarnation_id(transfer_->local_cache_layout_.incarnation_id);
  request.set_layout_generation(
      transfer_->local_cache_layout_.layout_generation);
  for (uint64_t block_id : mapping_.local_ids) {
    request.add_block_ids(block_id);
  }
  proto::VirtualKVCacheOffsetsResponse response;
  brpc::Controller controller;

  service.GetVirtualKVCacheOffsets(&controller, &request, &response, nullptr);

  ASSERT_TRUE(response.ok());
  EXPECT_EQ(response.incarnation_id(), request.incarnation_id());
  EXPECT_EQ(response.layout_generation(), request.layout_generation());
  EXPECT_EQ(response.physical_page_size_bytes(), kPageBytes);
  ASSERT_EQ(response.layer_offsets_size(), 3);
  for (int32_t layer_id = 0; layer_id < 3; ++layer_id) {
    const auto& offsets = response.layer_offsets(layer_id);
    EXPECT_EQ(offsets.layer_id(), layer_id);
    ASSERT_EQ(offsets.k_offsets_size(), 2);
    ASSERT_EQ(offsets.v_offsets_size(), 2);
    EXPECT_EQ(offsets.index_offsets_size(), layer_id == 1 ? 0 : 2);
    for (int32_t index = 0; index < 2; ++index) {
      const auto& local = state_->layer_role_offsets.at(layer_id).at(
          mapping_.local_ids[static_cast<size_t>(index)]);
      EXPECT_EQ(offsets.k_offsets(index), local.k_offset);
      EXPECT_EQ(offsets.v_offsets(index), local.v_offset);
      if (local.index_offset.has_value()) {
        EXPECT_EQ(offsets.index_offsets(index), *local.index_offset);
      }
    }
  }
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       SourceServiceRejectsStaleIdentityWithoutOffsetPayload) {
  MooncakeKVCacheTransferService service;
  proto::VirtualKVCacheOffsetsRequest request;
  request.set_incarnation_id("stale-incarnation");
  request.set_layout_generation(
      transfer_->local_cache_layout_.layout_generation);
  request.add_block_ids(mapping_.local_ids.front());
  proto::VirtualKVCacheOffsetsResponse response;
  brpc::Controller controller;
  service.GetVirtualKVCacheOffsets(&controller, &request, &response, nullptr);
  EXPECT_FALSE(response.ok());
  EXPECT_TRUE(response.incarnation_id().empty());
  EXPECT_EQ(response.layer_offsets_size(), 0);
  EXPECT_EQ(response.physical_page_size_bytes(), 0U);
  EXPECT_TRUE(state_->offset_queries.empty());

  request.set_incarnation_id(transfer_->local_cache_layout_.incarnation_id);
  request.set_layout_generation(
      transfer_->local_cache_layout_.layout_generation + 1);
  service.GetVirtualKVCacheOffsets(&controller, &request, &response, nullptr);
  EXPECT_FALSE(response.ok());
  EXPECT_TRUE(response.incarnation_id().empty());
  EXPECT_EQ(response.layer_offsets_size(), 0);
  EXPECT_EQ(response.physical_page_size_bytes(), 0U);
  EXPECT_TRUE(state_->offset_queries.empty());
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       SourceServiceRejectsInvalidBlocksWithoutPartialOffsetPayload) {
  MooncakeKVCacheTransferService service;
  proto::VirtualKVCacheOffsetsRequest request;
  request.set_incarnation_id(transfer_->local_cache_layout_.incarnation_id);
  request.set_layout_generation(
      transfer_->local_cache_layout_.layout_generation);
  request.add_block_ids(mapping_.local_ids.front());
  request.add_block_ids(mapping_.local_ids.back());
  state_->layer_role_offsets[2].erase(mapping_.local_ids.back());
  proto::VirtualKVCacheOffsetsResponse response;
  brpc::Controller controller;
  service.GetVirtualKVCacheOffsets(&controller, &request, &response, nullptr);
  EXPECT_FALSE(response.ok());
  EXPECT_TRUE(response.incarnation_id().empty());
  EXPECT_EQ(response.layer_offsets_size(), 0);
  EXPECT_EQ(response.physical_page_size_bytes(), 0U);

  request.clear_block_ids();
  request.add_block_ids(4);
  service.GetVirtualKVCacheOffsets(&controller, &request, &response, nullptr);
  EXPECT_FALSE(response.ok());
  EXPECT_EQ(response.layer_offsets_size(), 0);
  request.clear_block_ids();
  service.GetVirtualKVCacheOffsets(&controller, &request, &response, nullptr);
  EXPECT_FALSE(response.ok());
  EXPECT_EQ(response.layer_offsets_size(), 0);
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       ProtocolQueriesBindSourceIdentityAndRetainPlanOnlySession) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  EXPECT_EQ(engine_->fetch_cache_layout_calls, 2U);
  ASSERT_EQ(engine_->opened_sessions,
            (std::vector<std::string>{"owner-source"}));
  WorkerCacheLayoutManifest manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> offsets;
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  ASSERT_EQ(engine_->virtual_wire_requests.size(), 1U);
  const auto& request = engine_->virtual_wire_requests[0];
  EXPECT_EQ(request.incarnation_id(), manifest.incarnation_id);
  EXPECT_EQ(request.layout_generation(), manifest.layout_generation);
  EXPECT_EQ((std::vector<uint64_t>(request.block_ids().begin(),
                                   request.block_ids().end())),
            mapping_.remote_ids);
  ASSERT_EQ(offsets.size(), 3U);
  for (size_t layer_id = 0; layer_id < offsets.size(); ++layer_id) {
    EXPECT_EQ(offsets[layer_id].k_offsets,
              engine_->virtual_layer_offsets[layer_id].k_offsets);
    EXPECT_EQ(offsets[layer_id].v_offsets,
              engine_->virtual_layer_offsets[layer_id].v_offsets);
    EXPECT_EQ(offsets[layer_id].index_offsets,
              engine_->virtual_layer_offsets[layer_id].index_offsets);
  }
  EXPECT_EQ(engine_->opened_sessions,
            (std::vector<std::string>{"owner-source", "source"}));
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  EXPECT_EQ(engine_->opened_sessions.size(), 2U);
  EXPECT_EQ(engine_->fetch_cache_layout_calls, 2U);
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       FailedLazySessionOpenCanRetryWithoutUnlinkingPeer) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  const size_t linked_peer_calls = engine_->peer_calls.size();
  engine_->failed_session_peer = "source";
  WorkerCacheLayoutManifest manifest = engine_->virtual_offset_manifest;
  auto offsets = engine_->virtual_layer_offsets;

  EXPECT_FALSE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  EXPECT_TRUE(manifest.incarnation_id.empty());
  EXPECT_TRUE(offsets.empty());
  EXPECT_EQ(engine_->opened_sessions,
            (std::vector<std::string>{"owner-source", "source"}));
  EXPECT_TRUE(engine_->closed_sessions.empty());
  EXPECT_EQ(engine_->peer_calls.size(), linked_peer_calls);

  engine_->failed_session_peer.clear();
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  EXPECT_EQ(manifest.incarnation_id,
            engine_->virtual_offset_manifest.incarnation_id);
  EXPECT_EQ(offsets.size(), engine_->virtual_layer_offsets.size());
  EXPECT_EQ(engine_->opened_sessions,
            (std::vector<std::string>{"owner-source", "source", "source"}));
  EXPECT_TRUE(engine_->closed_sessions.empty());
  EXPECT_EQ(engine_->peer_calls.size(), linked_peer_calls);

  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  EXPECT_EQ(engine_->opened_sessions.size(), 3U);
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       RepeatedLinkPreservesPlanOnlyPullSessionUntilSingleUnlink) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  WorkerCacheLayoutManifest manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> offsets;
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  ASSERT_EQ(engine_->opened_sessions,
            (std::vector<std::string>{"owner-source", "source"}));
  const size_t linked_peer_calls = engine_->peer_calls.size();

  ASSERT_TRUE(engine_->link_sessions({kSourceClusterId - 1, kSourceClusterId},
                                     {"owner-source", "source"}));
  EXPECT_EQ(engine_->peer_calls.size(), linked_peer_calls);
  EXPECT_EQ(engine_->opened_sessions.size(), 2U);
  WorkerCacheLayoutManifest changed_destination =
      transfer_->local_cache_layout_;
  ++changed_destination.layout_generation;
  EXPECT_FALSE(engine_->set_local_cache_layout(changed_destination).ok());
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  EXPECT_EQ(engine_->opened_sessions.size(), 2U);

  EXPECT_FALSE(engine_->close_session(kSourceClusterId + 1, "source"));
  EXPECT_TRUE(engine_->closed_sessions.empty());
  ASSERT_TRUE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->closed_sessions, (std::vector<std::string>{"source"}));
  ASSERT_EQ(engine_->peer_calls.size(), linked_peer_calls + 1);
  EXPECT_EQ(engine_->peer_calls.back().remote_addr, "source");
  EXPECT_EQ(engine_->peer_calls.back().mode, CachePeerMode::ABSENT);
  ASSERT_TRUE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->closed_sessions.size(), 1U);
  EXPECT_EQ(engine_->peer_calls.size(), linked_peer_calls + 1);
  EXPECT_FALSE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  EXPECT_TRUE(manifest.incarnation_id.empty());
  EXPECT_TRUE(offsets.empty());
  EXPECT_EQ(engine_->opened_sessions.size(), 2U);
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       CloseSessionErasesLocalStateWhenRemoteUnlinkFails) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  WorkerCacheLayoutManifest manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> offsets;
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));

  engine_->failed_absent_peer = "source";
  EXPECT_FALSE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->closed_sessions, std::vector<std::string>({"source"}));
  EXPECT_FALSE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));

  WorkerCacheLayoutManifest changed_destination =
      transfer_->local_cache_layout_;
  ++changed_destination.layout_generation;
  EXPECT_TRUE(engine_->set_local_cache_layout(changed_destination).ok());

  const size_t peer_calls_after_first_close = engine_->peer_calls.size();
  EXPECT_FALSE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->peer_calls.size(), peer_calls_after_first_close + 1);
  EXPECT_EQ(engine_->closed_sessions, std::vector<std::string>({"source"}));

  engine_->failed_absent_peer.clear();
  EXPECT_TRUE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->closed_sessions, std::vector<std::string>{"source"});
  EXPECT_EQ(engine_->peer_calls.size(), peer_calls_after_first_close + 2);
}

TEST_F(MooncakeVirtualKVCachePullTest,
       CloseSessionRetainsStateWhenLocalCloseFails) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  WorkerCacheLayoutManifest manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> offsets;
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));

  engine_->failed_close_session_peer = "source";
  EXPECT_FALSE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->closed_sessions, std::vector<std::string>({"source"}));

  engine_->failed_close_session_peer.clear();
  EXPECT_TRUE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->closed_sessions,
            std::vector<std::string>({"source", "source"}));
  EXPECT_FALSE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
}

TEST_F(MooncakeVirtualKVCachePullTest,
       RepeatedLinkRejectsChangedSourceBeforeMutatingRetainedSession) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  WorkerCacheLayoutManifest manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> offsets;
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  const size_t linked_peer_calls = engine_->peer_calls.size();
  engine_->remote_layouts.at("source").incarnation_id =
      "new-source-incarnation";

  EXPECT_FALSE(engine_->link_sessions({kSourceClusterId - 1, kSourceClusterId},
                                      {"owner-source", "source"}));
  EXPECT_EQ(engine_->peer_calls.size(), linked_peer_calls);
  EXPECT_EQ(engine_->opened_sessions.size(), 2U);
  EXPECT_TRUE(engine_->closed_sessions.empty());

  engine_->remote_layouts.at("source").incarnation_id =
      engine_->virtual_offset_manifest.incarnation_id;
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  EXPECT_EQ(engine_->opened_sessions.size(), 2U);
  ASSERT_TRUE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->closed_sessions, (std::vector<std::string>{"source"}));
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       NewPeerLinkFailurePreservesExistingPlanOnlyPullSession) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  WorkerCacheLayoutManifest manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> offsets;
  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  WorkerCacheLayoutManifest new_owner =
      engine_->remote_layouts.at("owner-source");
  new_owner.addr = "new-owner-source";
  new_owner.cluster_id = kSourceClusterId + 1;
  new_owner.incarnation_id = "new-owner-incarnation";
  engine_->remote_layouts.emplace("new-owner-source", std::move(new_owner));
  engine_->failed_peer = "new-owner-source";
  const size_t linked_peer_calls = engine_->peer_calls.size();

  EXPECT_FALSE(engine_->link_sessions({kSourceClusterId + 1, kSourceClusterId},
                                      {"new-owner-source", "source"}));
  ASSERT_EQ(engine_->peer_calls.size(), linked_peer_calls + 2);
  EXPECT_EQ(engine_->peer_calls[linked_peer_calls].remote_addr,
            "new-owner-source");
  EXPECT_EQ(engine_->peer_calls[linked_peer_calls].mode, CachePeerMode::ACTIVE);
  EXPECT_EQ(engine_->peer_calls.back().remote_addr, "new-owner-source");
  EXPECT_EQ(engine_->peer_calls.back().mode, CachePeerMode::ABSENT);
  EXPECT_EQ(engine_->opened_sessions.size(), 2U);
  EXPECT_TRUE(engine_->closed_sessions.empty());

  ASSERT_TRUE(engine_->get_remote_virtual_offsets(
      kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
  EXPECT_EQ(engine_->opened_sessions.size(), 2U);
  ASSERT_TRUE(engine_->close_session(kSourceClusterId, "source"));
  EXPECT_EQ(engine_->closed_sessions, (std::vector<std::string>{"source"}));
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       ProtocolRejectsStaleIdentityAndFailedQueryWithoutPublishingOffsets) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  const auto original = engine_->virtual_wire_response;
  const auto expect_rejected = [&]() {
    WorkerCacheLayoutManifest manifest = engine_->virtual_offset_manifest;
    auto offsets = engine_->virtual_layer_offsets;
    EXPECT_FALSE(engine_->get_remote_virtual_offsets(
        kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
    EXPECT_TRUE(manifest.incarnation_id.empty());
    EXPECT_TRUE(offsets.empty());
    EXPECT_EQ(engine_->opened_sessions,
              (std::vector<std::string>{"owner-source"}));
  };
  engine_->virtual_wire_response.set_incarnation_id("stale-incarnation");
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.set_layout_generation(
      original.layout_generation() + 1);
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.set_ok(false);
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_result = false;
  expect_rejected();
  expect_no_move();
}

TEST_F(MooncakeVirtualKVCachePullTest,
       ProtocolRejectsMalformedLayerRolesAndPhysicalGeometry) {
  ASSERT_NO_FATAL_FAILURE(enable_protocol_queries());
  const auto original = engine_->virtual_wire_response;
  const auto expect_rejected = [&]() {
    WorkerCacheLayoutManifest manifest;
    std::vector<VirtualKVCacheLayerGlobalOffsets> offsets;
    EXPECT_FALSE(engine_->get_remote_virtual_offsets(
        kSourceClusterId, "source", mapping_.remote_ids, &manifest, &offsets));
    EXPECT_TRUE(manifest.incarnation_id.empty());
    EXPECT_TRUE(offsets.empty());
    EXPECT_EQ(engine_->opened_sessions,
              (std::vector<std::string>{"owner-source"}));
  };
  engine_->virtual_wire_response.mutable_layer_offsets(2)->set_layer_id(0);
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.mutable_layer_offsets(2)->set_layer_id(-1);
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.mutable_layer_offsets(2)->set_layer_id(3);
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.mutable_layer_offsets()->RemoveLast();
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.mutable_layer_offsets(1)->add_index_offsets(0);
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.mutable_layer_offsets(2)
      ->mutable_index_offsets()
      ->RemoveLast();
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.mutable_layer_offsets(2)
      ->mutable_k_offsets()
      ->RemoveLast();
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.set_physical_page_size_bytes(0);
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.mutable_layer_offsets(2)->set_k_offsets(
      0, state_->backing.size());
  expect_rejected();
  engine_->virtual_wire_response = original;
  engine_->virtual_wire_response.mutable_layer_offsets(2)->set_v_offsets(0, 1);
  expect_rejected();
  expect_no_move();
}

TEST(MooncakeVirtualKVCacheTransferTest,
     PushUsesIndependentLocalAndRemoteRoleOffsets) {
  auto state = std::make_shared<SharedTransferMemoryState>();
  state->role_offsets = {{3, {16, 80, 128}}, {4, {32, 96, 160}}};
  MooncakeVirtualKVCacheTransfer transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(state));
  transfer.set_model_id("glm-test");
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {3, 4};
  mapping.remote_ids = {21, 22};
  VirtualKVCacheLayerGlobalOffsets destination{
      {1000, 2000}, {3000, 4000}, {5000, 6000}};
  std::vector<ExplicitResourceMapping> mappings;
  ASSERT_TRUE(
      transfer.build_explicit_mappings(mapping, destination, 7, &mappings));
  ASSERT_EQ(mappings.size(), 3U);
  EXPECT_EQ(mappings[0].role, static_cast<int32_t>(KVCacheTensorRole::KEY));
  EXPECT_EQ(mappings[0].local_offsets, (std::vector<uint64_t>{16, 32}));
  EXPECT_EQ(mappings[0].remote_offsets, destination.k_offsets);
  EXPECT_EQ(mappings[1].local_offsets, (std::vector<uint64_t>{80, 96}));
  EXPECT_EQ(mappings[1].remote_offsets, destination.v_offsets);
  EXPECT_EQ(mappings[2].role, static_cast<int32_t>(KVCacheTensorRole::INDEX));
  EXPECT_EQ(mappings[2].local_offsets, (std::vector<uint64_t>{128, 160}));
  EXPECT_EQ(mappings[2].remote_offsets, destination.index_offsets);
  for (const auto& item : mappings) {
    EXPECT_EQ(item.local_ids, mapping.local_ids);
    EXPECT_EQ(item.remote_ids, mapping.remote_ids);
  }
  ASSERT_EQ(state->offset_queries.size(), 2U);
  for (const auto& query : state->offset_queries) {
    EXPECT_EQ(query.model_id, "glm-test");
    EXPECT_EQ(query.layer_id, 7);
  }
}

TEST(MooncakeVirtualKVCacheTransferTest,
     PushSupportsSharedIndexerAndRejectsInvalidOffsets) {
  auto state = std::make_shared<SharedTransferMemoryState>();
  state->role_offsets = {{3, {16, 80, std::nullopt}}};
  MooncakeVirtualKVCacheTransfer transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(state));
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {3};
  mapping.remote_ids = {21};
  VirtualKVCacheLayerGlobalOffsets destination{{1000}, {2000}, {}};
  std::vector<ExplicitResourceMapping> mappings;
  ASSERT_TRUE(
      transfer.build_explicit_mappings(mapping, destination, 1, &mappings));
  EXPECT_EQ(mappings.size(), 2U);
  destination.index_offsets = {3000};
  EXPECT_FALSE(
      transfer.build_explicit_mappings(mapping, destination, 1, &mappings));
  state->role_offsets[3].index_offset = 128;
  destination.index_offsets.clear();
  EXPECT_FALSE(
      transfer.build_explicit_mappings(mapping, destination, 1, &mappings));
  destination.index_offsets = {3000, 4000};
  EXPECT_FALSE(
      transfer.build_explicit_mappings(mapping, destination, 1, &mappings));
  destination.index_offsets = {3000};
  state->role_offsets.clear();
  EXPECT_FALSE(
      transfer.build_explicit_mappings(mapping, destination, 1, &mappings));
}

TEST(MooncakeVirtualKVCacheTransferTest, MergePreservesSparseIndexOffsets) {
  auto state = std::make_shared<SharedTransferMemoryState>();
  MooncakeVirtualKVCacheTransfer transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(state));
  TransferKVInfo first = make_info(1, 1, 0);
  first.dst_virtual_kv_cache_layer_global_offsets = {
      {{10, 11}, {20, 21}, {30, 31}}, {{40, 41}, {50, 51}, {}}};
  TransferKVInfo second = first;
  second.dst_virtual_kv_cache_layer_global_offsets[0] = {
      {12, 13}, {22, 23}, {32, 33}};
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged;
  transfer.merge_kv_blocks(merged, {first, second}, make_args(0, 1, 1));
  ASSERT_EQ(merged.size(), 1U);
  const auto& layers =
      merged.begin()->second.dst_virtual_kv_cache_layer_global_offsets;
  ASSERT_EQ(layers.size(), 2U);
  EXPECT_EQ(layers[0].index_offsets, (std::vector<uint64_t>{30, 31, 32, 33}));
  EXPECT_TRUE(layers[1].index_offsets.empty());
}

TEST(MooncakeVirtualKVCacheTransferTest,
     KvSplitFiltersAllRoleOffsetsWithRemoteIds) {
  TransferKVInfo info = make_info(1, 1, 0);
  info.mappings[0].local_ids = {3, 4};
  info.mappings[0].remote_ids = {20, 21, 22};
  info.dst_virtual_kv_cache_layer_global_offsets = {
      {{10, 11, 12}, {20, 21, 22}, {30, 31, 32}},
      {{40, 41, 42}, {50, 51, 52}, {}}};
  const auto filtered = filter_kv_split_infos(1, 2, {info});
  ASSERT_EQ(filtered.size(), 1U);
  EXPECT_EQ(filtered[0].mappings[0].local_ids, (std::vector<uint64_t>{3}));
  EXPECT_EQ(filtered[0].mappings[0].remote_ids, (std::vector<uint64_t>{21}));
  const auto& layers = filtered[0].dst_virtual_kv_cache_layer_global_offsets;
  ASSERT_EQ(layers.size(), 2U);
  EXPECT_EQ(layers[0].k_offsets, (std::vector<uint64_t>{11}));
  EXPECT_EQ(layers[0].v_offsets, (std::vector<uint64_t>{21}));
  EXPECT_EQ(layers[0].index_offsets, (std::vector<uint64_t>{31}));
  EXPECT_TRUE(layers[1].index_offsets.empty());
}

TEST(MooncakeVirtualKVCacheTransferTest, MergeAllowsEmptyKvSplitOffsets) {
  auto state = std::make_shared<SharedTransferMemoryState>();
  MooncakeVirtualKVCacheTransfer transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(state));
  TransferKVInfo first = make_info(1, 1, 0);
  first.dst_virtual_kv_cache_layer_global_offsets = {
      {{10, 11}, {20, 21}, {30, 31}}};
  TransferKVInfo second = first;
  second.dst_virtual_kv_cache_layer_global_offsets[0] = {{}, {}, {}};
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged;
  transfer.merge_kv_blocks(merged, {first, second}, make_args(0, 1, 1));
  ASSERT_EQ(merged.size(), 1U);
  const auto& layers =
      merged.begin()->second.dst_virtual_kv_cache_layer_global_offsets;
  ASSERT_EQ(layers.size(), 1U);
  EXPECT_EQ(layers[0].k_offsets, (std::vector<uint64_t>{10, 11}));
  EXPECT_EQ(layers[0].v_offsets, (std::vector<uint64_t>{20, 21}));
  EXPECT_EQ(layers[0].index_offsets, (std::vector<uint64_t>{30, 31}));

  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> reverse_merged;
  transfer.merge_kv_blocks(reverse_merged, {second, first}, make_args(0, 1, 1));
  ASSERT_EQ(reverse_merged.size(), 1U);
  const auto& reverse_layers =
      reverse_merged.begin()->second.dst_virtual_kv_cache_layer_global_offsets;
  ASSERT_EQ(reverse_layers.size(), 1U);
  EXPECT_EQ(reverse_layers[0].k_offsets, (std::vector<uint64_t>{10, 11}));
  EXPECT_EQ(reverse_layers[0].v_offsets, (std::vector<uint64_t>{20, 21}));
  EXPECT_EQ(reverse_layers[0].index_offsets, (std::vector<uint64_t>{30, 31}));
}

TEST(MooncakeVirtualKVCacheTransferTest,
     MergeAllowsAuxiliaryOnlyInfoWithEmptyKvOffsets) {
  auto state = std::make_shared<SharedTransferMemoryState>();
  MooncakeVirtualKVCacheTransfer transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(state));

  TransferKVInfo auxiliary_only = make_info(1, 1, 0);
  auxiliary_only.mappings[0].local_ids.clear();
  auxiliary_only.mappings[0].remote_ids.clear();
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {31};
  linear_mapping.remote_ids = {41};
  auxiliary_only.mappings.emplace_back(std::move(linear_mapping));
  auxiliary_only.dst_virtual_kv_cache_layer_global_offsets = {{{}, {}, {}}};

  TransferKVInfo kv_info = make_info(1, 1, 0);
  kv_info.mappings[0].local_ids = {13};
  kv_info.mappings[0].remote_ids = {23, 24};
  kv_info.dst_virtual_kv_cache_layer_global_offsets = {
      {{10, 11}, {20, 21}, {30, 31}}};

  const std::vector<TransferKVInfo> filtered = filter_kv_split_infos(
      /*kv_split_rank=*/1, /*kv_split_size=*/2, {auxiliary_only, kv_info});
  ASSERT_EQ(filtered.size(), 2U);
  EXPECT_TRUE(filtered[0]
                  .dst_virtual_kv_cache_layer_global_offsets[0]
                  .k_offsets.empty());
  EXPECT_EQ(filtered[1].dst_virtual_kv_cache_layer_global_offsets[0].k_offsets,
            (std::vector<uint64_t>{11}));

  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged;
  transfer.merge_kv_blocks(merged, filtered, make_args(0, 1, 1));
  ASSERT_EQ(merged.size(), 1U);
  const auto& layers =
      merged.begin()->second.dst_virtual_kv_cache_layer_global_offsets;
  ASSERT_EQ(layers.size(), 1U);
  EXPECT_EQ(layers[0].k_offsets, (std::vector<uint64_t>{11}));
  EXPECT_EQ(layers[0].v_offsets, (std::vector<uint64_t>{21}));
  EXPECT_EQ(layers[0].index_offsets, (std::vector<uint64_t>{31}));

  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> reverse_merged;
  transfer.merge_kv_blocks(
      reverse_merged, {filtered[1], filtered[0]}, make_args(0, 1, 1));
  ASSERT_EQ(reverse_merged.size(), 1U);
  const auto& reverse_layers =
      reverse_merged.begin()->second.dst_virtual_kv_cache_layer_global_offsets;
  ASSERT_EQ(reverse_layers.size(), 1U);
  EXPECT_EQ(reverse_layers[0].k_offsets, (std::vector<uint64_t>{11}));
  EXPECT_EQ(reverse_layers[0].v_offsets, (std::vector<uint64_t>{21}));
  EXPECT_EQ(reverse_layers[0].index_offsets, (std::vector<uint64_t>{31}));
}

TEST(MooncakeVirtualKVCacheTransferTest,
     MergeRejectsMalformedVirtualKvCacheOffsets) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  auto state = std::make_shared<SharedTransferMemoryState>();
  MooncakeVirtualKVCacheTransfer transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      std::make_unique<RecordingVirtualKVCacheTransferMemoryProvider>(state));

  const auto expect_rejected =
      [&transfer](VirtualKVCacheLayerGlobalOffsets offsets) {
        TransferKVInfo info = make_info(1, 1, 0);
        info.dst_virtual_kv_cache_layer_global_offsets = {std::move(offsets)};
        std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged;
        EXPECT_DEATH(
            transfer.merge_kv_blocks(merged, {info}, make_args(0, 1, 1)), "");
      };

  expect_rejected({{10}, {}, {}});
  expect_rejected({{10, 11}, {20}, {}});
  expect_rejected({{}, {}, {30}});
  expect_rejected({{10}, {20}, {30, 31}});

  TransferKVInfo first = make_info(1, 1, 0);
  first.dst_virtual_kv_cache_layer_global_offsets = {{{10}, {20}, {}}};
  TransferKVInfo second = first;
  second.dst_virtual_kv_cache_layer_global_offsets = {{{11}, {21}, {31}}};
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged;
  EXPECT_DEATH(
      transfer.merge_kv_blocks(merged, {first, second}, make_args(0, 1, 1)),
      "");
}

TEST(MooncakeKVCacheTransferEngineTest, LinksAllPcpSourcesWithOneActiveOwner) {
  MooncakeKVCacheTransferState& core =
      MooncakeKVCacheTransferState::get_instance();
  WorkerCacheLayoutManifest destination = make_pcp_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      "destination",
      /*cluster_id=*/1);
  ASSERT_TRUE(core.set_local_cache_layout(destination).ok());

  RecordingMooncakeTransferEngine transfer(/*listen_port=*/0,
                                           torch::Device(torch::kCPU));
  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> remote_addrs;
  for (int32_t cp_rank = 0; cp_rank < 4; ++cp_rank) {
    for (int32_t tp_rank = 0; tp_rank < 2; ++tp_rank) {
      const int32_t rank = cp_rank * 2 + tp_rank;
      const std::string addr = "source_" + std::to_string(rank);
      const uint64_t cluster_id = static_cast<uint64_t>(rank + 10);
      transfer.remote_layouts.emplace(addr,
                                      make_pcp_manifest(tp_rank,
                                                        /*tp_size=*/2,
                                                        cp_rank,
                                                        /*cp_size=*/4,
                                                        addr,
                                                        cluster_id));
      cluster_ids.emplace_back(cluster_id);
      remote_addrs.emplace_back(addr);
    }
  }

  ASSERT_TRUE(transfer.link_sessions(cluster_ids, remote_addrs));
  ASSERT_EQ(transfer.peer_calls.size(), 8U);
  EXPECT_EQ(transfer.peer_calls[0].mode, CachePeerMode::ACTIVE);
  for (size_t index = 1; index < transfer.peer_calls.size(); ++index) {
    EXPECT_EQ(transfer.peer_calls[index].mode, CachePeerMode::PLAN_ONLY);
  }
  EXPECT_EQ(transfer.opened_sessions, std::vector<std::string>({"source_0"}));

  EXPECT_FALSE(transfer.link_sessions({cluster_ids[0], cluster_ids[0]},
                                      {remote_addrs[0], remote_addrs[0]}));
  EXPECT_FALSE(transfer.link_sessions({cluster_ids[0], cluster_ids[0] + 1},
                                      {remote_addrs[0], remote_addrs[0]}));
  EXPECT_EQ(transfer.peer_calls.size(), 8U);

  ASSERT_TRUE(transfer.link_sessions(cluster_ids, remote_addrs));
  EXPECT_EQ(transfer.peer_calls.size(), 8U);
  EXPECT_EQ(transfer.opened_sessions, std::vector<std::string>({"source_0"}));

  for (size_t index = 0; index < remote_addrs.size(); ++index) {
    EXPECT_TRUE(
        transfer.close_session(cluster_ids[index], remote_addrs[index]));
  }
  EXPECT_EQ(transfer.closed_sessions, std::vector<std::string>({"source_0"}));
}

TEST(MooncakeKVCacheTransferEngineTest,
     FailedSessionOpenRollsBackEarlierSessionsAndAllowsRelink) {
  MooncakeKVCacheTransferState& core =
      MooncakeKVCacheTransferState::get_instance();
  WorkerCacheLayoutManifest destination = make_peer_manifest(
      "open-failure-destination", "open-failure-destination-incarnation", 1);
  auto& destination_tensor = destination.tensors[0];
  destination_tensor.shape = {2, 1, 2, 1};
  destination_tensor.stride = {2, 2, 1, 1};
  destination_tensor.resource_stride_bytes = 2;
  destination_tensor.buffer_bytes = 4;
  LogicalSpan second_span = destination_tensor.shard.spans[0];
  second_span.logical_offset_bytes = 1;
  second_span.physical_offset_bytes = 1;
  destination_tensor.shard.spans.emplace_back(std::move(second_span));
  ASSERT_TRUE(core.set_local_cache_layout(destination).ok());

  RecordingMooncakeTransferEngine transfer(/*listen_port=*/0,
                                           torch::Device(torch::kCPU));
  const std::vector<uint64_t> cluster_ids = {10, 11};
  const std::vector<std::string> remote_addrs = {"open-failure-source_0",
                                                 "open-failure-source_1"};
  for (int32_t rank = 0; rank < 2; ++rank) {
    const size_t index = static_cast<size_t>(rank);
    WorkerCacheLayoutManifest source = make_peer_manifest(
        remote_addrs[index], remote_addrs[index] + "-incarnation", 1);
    source.cluster_id = cluster_ids[index];
    source.coordinates.tp_rank = rank;
    source.coordinates.tp_size = 2;
    source.tensors[0].shard.spans[0].logical_offset_bytes =
        static_cast<uint64_t>(rank);
    source.tensors[0].shard.spans[0].owner_tp_rank = rank;
    transfer.remote_layouts.emplace(remote_addrs[index], std::move(source));
  }
  transfer.failed_session_peer = remote_addrs[1];

  EXPECT_FALSE(transfer.link_sessions(cluster_ids, remote_addrs));
  EXPECT_EQ(transfer.opened_sessions, remote_addrs);
  EXPECT_EQ(transfer.closed_sessions,
            (std::vector<std::string>{remote_addrs[1], remote_addrs[0]}));
  ASSERT_EQ(transfer.peer_calls.size(), 4U);
  for (size_t index = 0; index < 2; ++index) {
    EXPECT_EQ(transfer.peer_calls[index].remote_addr, remote_addrs[index]);
    EXPECT_EQ(transfer.peer_calls[index].mode, CachePeerMode::ACTIVE);
    EXPECT_EQ(transfer.peer_calls[index + 2].remote_addr, remote_addrs[index]);
    EXPECT_EQ(transfer.peer_calls[index + 2].mode, CachePeerMode::ABSENT);
  }
  EXPECT_TRUE(transfer.close_session(cluster_ids[0], remote_addrs[0]));
  EXPECT_TRUE(transfer.close_session(cluster_ids[1], remote_addrs[1]));
  EXPECT_EQ(transfer.closed_sessions.size(), 2U);
  EXPECT_EQ(transfer.peer_calls.size(), 4U);

  transfer.failed_session_peer.clear();
  ASSERT_TRUE(transfer.link_sessions(cluster_ids, remote_addrs));
  EXPECT_EQ(
      transfer.opened_sessions,
      (std::vector<std::string>{
          remote_addrs[0], remote_addrs[1], remote_addrs[0], remote_addrs[1]}));
  EXPECT_EQ(transfer.peer_calls.size(), 6U);
  EXPECT_TRUE(transfer.close_session(cluster_ids[0], remote_addrs[0]));
  EXPECT_TRUE(transfer.close_session(cluster_ids[1], remote_addrs[1]));
  EXPECT_EQ(
      transfer.closed_sessions,
      (std::vector<std::string>{
          remote_addrs[1], remote_addrs[0], remote_addrs[0], remote_addrs[1]}));
}

TEST(MooncakeKVCacheTransferEngineTest,
     LinkFailureRollsBackAttemptedPcpSources) {
  MooncakeKVCacheTransferState& core =
      MooncakeKVCacheTransferState::get_instance();
  WorkerCacheLayoutManifest destination = make_pcp_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      "rollback-destination",
      /*cluster_id=*/2);
  ASSERT_TRUE(core.set_local_cache_layout(destination).ok());

  RecordingMooncakeTransferEngine transfer(/*listen_port=*/0,
                                           torch::Device(torch::kCPU));
  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> remote_addrs;
  for (int32_t cp_rank = 0; cp_rank < 4; ++cp_rank) {
    for (int32_t tp_rank = 0; tp_rank < 2; ++tp_rank) {
      const int32_t rank = cp_rank * 2 + tp_rank;
      const std::string addr = "rollback-source_" + std::to_string(rank);
      const uint64_t cluster_id = static_cast<uint64_t>(rank + 20);
      transfer.remote_layouts.emplace(addr,
                                      make_pcp_manifest(tp_rank,
                                                        /*tp_size=*/2,
                                                        cp_rank,
                                                        /*cp_size=*/4,
                                                        addr,
                                                        cluster_id));
      cluster_ids.emplace_back(cluster_id);
      remote_addrs.emplace_back(addr);
    }
  }
  transfer.failed_peer = remote_addrs[3];
  transfer.failed_absent_peer = remote_addrs[3];

  EXPECT_FALSE(transfer.link_sessions(cluster_ids, remote_addrs));
  EXPECT_EQ(
      std::count_if(transfer.peer_calls.begin(),
                    transfer.peer_calls.end(),
                    [](const RecordingMooncakeTransferEngine::PeerCall& call) {
                      return call.mode == CachePeerMode::ABSENT;
                    }),
      4);
  ASSERT_EQ(transfer.peer_calls.size(), 8U);
  for (size_t index = 0; index < 4; ++index) {
    EXPECT_EQ(transfer.peer_calls[index + 4].remote_addr, remote_addrs[index]);
    EXPECT_EQ(transfer.peer_calls[index + 4].mode, CachePeerMode::ABSENT);
  }
  EXPECT_EQ(transfer.closed_sessions,
            std::vector<std::string>({"rollback-source_0"}));

  transfer.failed_peer.clear();
  const size_t peer_calls_after_failed_link = transfer.peer_calls.size();
  EXPECT_FALSE(transfer.link_sessions(cluster_ids, remote_addrs));
  EXPECT_EQ(transfer.peer_calls.size(), peer_calls_after_failed_link);

  transfer.failed_absent_peer.clear();
  EXPECT_TRUE(transfer.close_session(cluster_ids[3], remote_addrs[3]));
  EXPECT_EQ(transfer.peer_calls.back().remote_addr, remote_addrs[3]);
  EXPECT_EQ(transfer.peer_calls.back().mode, CachePeerMode::ABSENT);

  ASSERT_TRUE(transfer.link_sessions(cluster_ids, remote_addrs));
  for (size_t index = 0; index < remote_addrs.size(); ++index) {
    EXPECT_TRUE(
        transfer.close_session(cluster_ids[index], remote_addrs[index]));
  }
}

TEST(MooncakeKVCacheTransferDefaultTest,
     MergeIncludesActiveAndPlanOnlyDestinations) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  engine->planned_addrs = {"addr_1", "addr_3"};
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  const TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                        /*dst_tp_size=*/4,
                                        /*dst_dp_rank=*/0);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/2,
                                               /*dp_size=*/1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);

  ASSERT_EQ(merged_kv_infos.size(), 4U);
  EXPECT_NE(merged_kv_infos.find("100_addr_0"), merged_kv_infos.end());
  EXPECT_NE(merged_kv_infos.find("101_addr_1"), merged_kv_infos.end());
  EXPECT_NE(merged_kv_infos.find("102_addr_2"), merged_kv_infos.end());
  EXPECT_NE(merged_kv_infos.find("103_addr_3"), merged_kv_infos.end());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     MissingNegotiationSurfacesEveryDestinationToPush) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  const TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                        /*dst_tp_size=*/4,
                                        /*dst_dp_rank=*/0);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/2,
                                               /*dp_size=*/1);
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_kv_infos;

  transfer.merge_kv_blocks(merged_kv_infos, {info}, parallel_args);

  EXPECT_EQ(merged_kv_infos.size(), 4U);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     LogicalDcpMappingsPreserveOneToOneRemoteBlocks) {
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.rank_local_mapping = true;
  info.mappings[0].local_ids = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  info.mappings[0].remote_ids = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

  const std::vector<TransferKVInfo> filtered = filter_kv_split_infos(
      /*kv_split_rank=*/3, /*kv_split_size=*/4, {info});

  ASSERT_EQ(filtered.size(), 1U);
  ASSERT_EQ(filtered[0].mappings.size(), 1U);
  EXPECT_EQ(filtered[0].mappings[0].local_ids, info.mappings[0].local_ids);
  EXPECT_EQ(filtered[0].mappings[0].remote_ids, info.mappings[0].remote_ids);
}

#if defined(USE_NPU)
constexpr int32_t kValidatePushCommand = 1;
constexpr int32_t kPreparePullCommand = 2;
constexpr int32_t kStopChildCommand = 3;
constexpr char kPeerCommandFdEnv[] = "XLLM_MOONCAKE_TEST_PEER_COMMAND_FD";
constexpr char kPeerStatusFdEnv[] = "XLLM_MOONCAKE_TEST_PEER_STATUS_FD";
constexpr char kPeerListenPortEnv[] = "XLLM_MOONCAKE_TEST_PEER_LISTEN_PORT";
constexpr char kPeerDeviceIndexEnv[] = "XLLM_MOONCAKE_TEST_PEER_DEVICE_INDEX";
constexpr char kControllerProcessEnv[] =
    "XLLM_MOONCAKE_TEST_CONTROLLER_PROCESS";

bool write_all(int fd, const void* data, size_t size) {
  const char* cursor = static_cast<const char*>(data);
  while (size > 0) {
    const ssize_t written = write(fd, cursor, size);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    cursor += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool read_all(int fd, void* data, size_t size) {
  char* cursor = static_cast<char*>(data);
  while (size > 0) {
    const ssize_t received = read(fd, cursor, size);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      return false;
    }
    cursor += received;
    size -= static_cast<size_t>(received);
  }
  return true;
}

bool write_endpoint(int fd,
                    uint64_t cluster_id,
                    uint16_t listen_port,
                    const std::string& addr) {
  const uint32_t addr_size = static_cast<uint32_t>(addr.size());
  return write_all(fd, &cluster_id, sizeof(cluster_id)) &&
         write_all(fd, &listen_port, sizeof(listen_port)) &&
         write_all(fd, &addr_size, sizeof(addr_size)) &&
         write_all(fd, addr.data(), addr_size);
}

bool read_endpoint(int fd,
                   uint64_t* cluster_id,
                   uint16_t* listen_port,
                   std::string* addr) {
  uint32_t addr_size = 0;
  if (!read_all(fd, cluster_id, sizeof(*cluster_id)) ||
      !read_all(fd, listen_port, sizeof(*listen_port)) ||
      !read_all(fd, &addr_size, sizeof(addr_size)) || addr_size == 0 ||
      addr_size > 1024) {
    return false;
  }
  addr->resize(addr_size);
  return read_all(fd, addr->data(), addr_size);
}

class ChildProcessGuard final {
 public:
  explicit ChildProcessGuard(pid_t pid) : pid_(pid) {}
  ~ChildProcessGuard() {
    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
  }

  void release() { pid_ = -1; }

 private:
  pid_t pid_;
};

class ScopedSigpipeIgnore final {
 public:
  ScopedSigpipeIgnore() : previous_handler_(signal(SIGPIPE, SIG_IGN)) {}
  ~ScopedSigpipeIgnore() {
    if (previous_handler_ != SIG_ERR) {
      signal(SIGPIPE, previous_handler_);
    }
  }

 private:
  using SignalHandler = void (*)(int);
  SignalHandler previous_handler_;
};

class ScopedEnvironmentVariable final {
 public:
  explicit ScopedEnvironmentVariable(const char* name) : name_(name) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      original_value_ = value;
    }
  }

  ~ScopedEnvironmentVariable() {
    if (original_value_.has_value()) {
      setenv(name_.c_str(), original_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  bool set(const char* value) {
    return setenv(name_.c_str(), value, /*overwrite=*/1) == 0;
  }

 private:
  std::string name_;
  std::optional<std::string> original_value_;
};

struct NpuMixedTransferCaches {
  torch::Tensor backing;
  torch::Tensor conv;
  torch::Tensor ssm;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor index;
  torch::Tensor index_scale;
  std::vector<KVCache> caches;
};

NpuMixedTransferCaches register_npu_mixed_transfer_caches(
    MooncakeKVCacheTransferDefault& transfer,
    const torch::Device& device) {
  NpuMixedTransferCaches tensors;
  tensors.backing = torch::zeros({6, 2, 1024, 1, 512},
                                 torch::dtype(torch::kBFloat16).device(device));
  tensors.conv = tensors.backing.index({0}).view({2, 1024, 512});
  tensors.ssm = tensors.backing.index({1}).view({2, 1024, 512});
  tensors.key = tensors.backing.index({2});
  tensors.value = tensors.backing.index({3});
  tensors.index = tensors.backing.index({4});
  tensors.index_scale = tensors.backing.index({5});
  tensors.caches.emplace_back(
      LinearAttentionKVCacheTensors{tensors.conv, tensors.ssm});
  tensors.caches.emplace_back(
      IndexedKVCacheTensors{KVCacheTensors{tensors.key, tensors.value},
                            tensors.index,
                            tensors.index_scale});
  ModelArgs model_args;
  model_args.model_type("test").n_layers(2).n_heads(1).head_dim(512);
  transfer.configure_cache_layout(make_args(0, 1, 1), model_args, 1024, false);
  transfer.register_kv_cache(tensors.caches, KVCacheShape(), torch::kBFloat16);
  return tensors;
}

void fill_mixed_transfer_block(NpuMixedTransferCaches* tensors,
                               int64_t block_id,
                               bool pull_pattern) {
  const double offset = pull_pattern ? 4.0 : 0.0;
  tensors->conv.index({block_id}).fill_(1.25 + offset);
  tensors->ssm.index({block_id}).fill_(-2.5 - offset);
  tensors->key.index({block_id}).fill_(3.5 + offset);
  tensors->value.index({block_id}).fill_(-4.5 - offset);
  tensors->index.index({block_id}).fill_(pull_pattern ? 17 : 42);
  tensors->index_scale.index({block_id}).fill_(pull_pattern ? 0.25 : 0.125);
}

void zero_mixed_transfer_block(NpuMixedTransferCaches* tensors,
                               int64_t block_id) {
  tensors->conv.index({block_id}).zero_();
  tensors->ssm.index({block_id}).zero_();
  tensors->key.index({block_id}).zero_();
  tensors->value.index({block_id}).zero_();
  tensors->index.index({block_id}).zero_();
  tensors->index_scale.index({block_id}).zero_();
}

bool tensor_block_has_value(const torch::Tensor& tensor,
                            int64_t block_id,
                            double value) {
  const torch::Tensor block = tensor.index({block_id}).cpu();
  return torch::equal(block, torch::full_like(block, value));
}

bool mixed_transfer_block_matches(const NpuMixedTransferCaches& tensors,
                                  int64_t block_id,
                                  bool pull_pattern) {
  const double offset = pull_pattern ? 4.0 : 0.0;
  return tensor_block_has_value(tensors.conv, block_id, 1.25 + offset) &&
         tensor_block_has_value(tensors.ssm, block_id, -2.5 - offset) &&
         tensor_block_has_value(tensors.key, block_id, 3.5 + offset) &&
         tensor_block_has_value(tensors.value, block_id, -4.5 - offset) &&
         tensor_block_has_value(
             tensors.index, block_id, pull_pattern ? 17 : 42) &&
         tensor_block_has_value(
             tensors.index_scale, block_id, pull_pattern ? 0.25 : 0.125);
}

int run_npu_round_trip_peer(int command_fd,
                            int status_fd,
                            uint16_t listen_port,
                            int32_t device_index) {
  Device remote_device(device_index);
  remote_device.set_device();
  remote_device.init_device_context();
  const torch::Device remote_torch_device = remote_device.unwrap();
  MooncakeKVCacheTransferDefault remote_transfer(device_index,
                                                 listen_port,
                                                 remote_torch_device,
                                                 /*model_type=*/"test");
  remote_transfer.initialize(device_index);
  NpuMixedTransferCaches remote_caches =
      register_npu_mixed_transfer_caches(remote_transfer, remote_torch_device);

  const auto& layers = remote_transfer.main_layout_.layers;
  const bool layout_matches =
      layers.size() == 2 && layers[0].size() == 2 && layers[1].size() == 4 &&
      layers[0][0].role == KVCacheTensorRole::CONV &&
      layers[0][1].role == KVCacheTensorRole::SSM &&
      layers[0][0].group_id == cache_group_id(BlockType::LINEAR) &&
      layers[0][1].group_id == cache_group_id(BlockType::LINEAR) &&
      layers[1][0].role == KVCacheTensorRole::KEY &&
      layers[1][1].role == KVCacheTensorRole::VALUE &&
      layers[1][2].role == KVCacheTensorRole::INDEX &&
      layers[1][3].role == KVCacheTensorRole::INDEX_SCALE &&
      layers[1][0].group_id == cache_group_id(BlockType::KV) &&
      layers[1][1].group_id == cache_group_id(BlockType::KV) &&
      layers[1][2].group_id == cache_group_id(BlockType::KV) &&
      layers[1][3].group_id == cache_group_id(BlockType::KV);
  if (!layout_matches) {
    return 10;
  }

  uint64_t remote_cluster_id = 0;
  std::string remote_addr;
  remote_transfer.get_cache_info(remote_cluster_id, remote_addr);
  if (remote_addr.empty() ||
      !write_endpoint(status_fd, remote_cluster_id, listen_port, remote_addr)) {
    return 11;
  }

  uint64_t local_cluster_id = 0;
  uint16_t local_port = 0;
  std::string local_addr;
  if (!read_endpoint(command_fd, &local_cluster_id, &local_port, &local_addr) ||
      !remote_transfer.link_clusters(
          {local_cluster_id}, {local_addr}, {local_port})) {
    return 15;
  }
  const uint8_t ready = 1;
  if (!write_all(status_fd, &ready, sizeof(ready))) {
    return 14;
  }

  while (true) {
    int32_t command = 0;
    if (!read_all(command_fd, &command, sizeof(command))) {
      return 12;
    }

    uint8_t success = 0;
    if (command == kValidatePushCommand) {
      remote_device.set_device();
      success = remote_device.synchronize_default_stream() == 0 &&
                        mixed_transfer_block_matches(remote_caches,
                                                     /*block_id=*/1,
                                                     /*pull_pattern=*/false)
                    ? 1
                    : 0;
    } else if (command == kPreparePullCommand) {
      remote_device.set_device();
      fill_mixed_transfer_block(
          &remote_caches, /*block_id=*/1, /*pull_pattern=*/true);
      success = remote_device.synchronize_default_stream() == 0 ? 1 : 0;
    } else if (command == kStopChildCommand) {
      const bool unlinked = remote_transfer.unlink_cluster(
          local_cluster_id, local_addr, local_port, /*force_flag=*/true);
      close(command_fd);
      close(status_fd);
      // Bypass third-party process-global teardown, which can terminate on a
      // still-joinable TransferEngine thread.
      _exit(unlinked ? 0 : 16);
    } else {
      return 13;
    }

    if (!write_all(status_fd, &success, sizeof(success))) {
      return 14;
    }
  }
}
#endif

#if defined(USE_MLU)
KVCacheShape make_indexer_int8_transfer_shape() {
  proto::KVCacheShape proto_shape;
  for (int64_t dim : std::vector<int64_t>{2, 1, 1, 16}) {
    proto_shape.add_key_cache_shape(dim);
    proto_shape.add_value_cache_shape(dim);
  }
  for (int64_t dim : std::vector<int64_t>{2, 96, 1, 8}) {
    proto_shape.add_index_cache_shape(dim);
  }
  for (int64_t dim : std::vector<int64_t>{2, 96, 1}) {
    proto_shape.add_index_cache_scale_shape(dim);
  }
  return KVCacheShape::from_proto(proto_shape);
}

std::vector<KVCache> make_mixed_transfer_caches(const torch::Device& device) {
  std::shared_ptr<KVCacheTensorAllocator> allocator =
      default_kv_tensor_allocator();
  auto make_full_cache = [&allocator, &device]() {
    torch::Tensor key = allocator->allocate(
        KVCacheTensorRole::KEY, {2, 1, 1, 16}, torch::kBFloat16, device);
    torch::Tensor index = allocator->allocate(
        KVCacheTensorRole::INDEX, {2, 96, 1, 8}, torch::kChar, device);
    torch::Tensor index_scale = allocator->allocate(
        KVCacheTensorRole::INDEX_SCALE, {2, 96, 1}, torch::kFloat32, device);
    return KVCache(IndexedKVCacheTensors{
        KVCacheTensors{key, torch::Tensor()}, index, index_scale});
  };
  auto make_shared_cache = [&allocator, &device]() {
    torch::Tensor key = allocator->allocate(
        KVCacheTensorRole::KEY, {2, 1, 1, 16}, torch::kChar, device);
    torch::Tensor key_scale = allocator->allocate(
        KVCacheTensorRole::KEY_SCALE, {2, 1, 1}, torch::kFloat32, device);
    return KVCache(QuantizedKVCacheTensors{
        KVCacheTensors{key, torch::Tensor()}, key_scale, torch::Tensor()});
  };

  std::vector<KVCache> caches;
  caches.reserve(4);
  caches.emplace_back(make_full_cache());
  caches.emplace_back(make_shared_cache());
  caches.emplace_back(make_full_cache());
  caches.emplace_back(make_shared_cache());
  return caches;
}
#endif

}  // namespace

TEST(MooncakeKVCacheTransferServiceTest, SetCachePeerRejectsUnspecifiedMode) {
  MooncakeKVCacheTransferService service;
  proto::CachePeerRequest request;
  proto::Status response;
  brpc::Controller cntl;

  service.SetCachePeer(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeKVCacheTransferServiceTest, PlanOnlyBindsAsSuccessfulNoOp) {
  MooncakeKVCacheTransferState& core =
      MooncakeKVCacheTransferState::get_instance();
  const WorkerCacheLayoutManifest local =
      make_peer_manifest("source", "plan-only-source", 1);
  ASSERT_TRUE(core.set_local_cache_layout(local).ok());
  const WorkerCacheLayoutManifest destination =
      make_peer_manifest("destination", "plan-only-destination", 1);

  MooncakeKVCacheTransferService service;
  proto::CachePeerRequest request;
  cache_layout_to_proto(destination, request.mutable_destination_manifest());
  request.set_mode(proto::CACHE_PEER_MODE_PLAN_ONLY);
  proto::Status response;
  brpc::Controller cntl;
  service.SetCachePeer(&cntl, &request, &response, nullptr);
  ASSERT_TRUE(response.ok());

  MooncakeKVCacheTransferEngine transfer(/*listen_port=*/0,
                                         torch::Device(torch::kCPU));
  std::vector<ByteRegion> regions;
  EXPECT_TRUE(transfer
                  .bind_outgoing_regions(destination.addr,
                                         {},
                                         CacheNamespace::MAIN,
                                         /*layer_id=*/0,
                                         &regions)
                  .ok());
  EXPECT_TRUE(regions.empty());

  std::vector<ByteRegion> explicit_regions;
  EXPECT_TRUE(transfer
                  .bind_outgoing_regions_explicit(destination.addr,
                                                  {},
                                                  CacheNamespace::MAIN,
                                                  /*layer_id=*/0,
                                                  &explicit_regions)
                  .ok());
  EXPECT_TRUE(explicit_regions.empty());

  request.set_mode(proto::CACHE_PEER_MODE_ABSENT);
  service.SetCachePeer(&cntl, &request, &response, nullptr);
  EXPECT_TRUE(response.ok());
}

TEST(MooncakeKVCacheTransferServiceTest, CachePeerTransitionsAreIdempotent) {
  MooncakeKVCacheTransferState& core =
      MooncakeKVCacheTransferState::get_instance();
  const WorkerCacheLayoutManifest local =
      make_peer_manifest("source", "transition-source", 1);
  ASSERT_TRUE(core.set_local_cache_layout(local).ok());
  const WorkerCacheLayoutManifest destination =
      make_peer_manifest("transition-peer", "transition-destination", 1);

  ASSERT_TRUE(core.set_cache_peer(destination, CachePeerMode::PLAN_ONLY).ok());
  EXPECT_TRUE(core.set_cache_peer(destination, CachePeerMode::PLAN_ONLY).ok());
  EXPECT_FALSE(core.set_cache_peer(destination, CachePeerMode::ACTIVE).ok());

  WorkerCacheLayoutManifest mismatched = destination;
  mismatched.incarnation_id = "new-destination";
  EXPECT_TRUE(core.set_cache_peer(mismatched, CachePeerMode::ABSENT).ok());
  EXPECT_TRUE(core.has_reshard_plan(destination.addr));

  WorkerCacheLayoutManifest updated_local = local;
  updated_local.layout_generation = 2;
  EXPECT_FALSE(core.set_local_cache_layout(updated_local).ok());

  EXPECT_TRUE(core.set_cache_peer(destination, CachePeerMode::ABSENT).ok());
  EXPECT_FALSE(core.has_reshard_plan(destination.addr));
  EXPECT_TRUE(core.set_local_cache_layout(updated_local).ok());
}

TEST(MooncakeKVCacheTransferServiceTest,
     ActiveSessionFailureDoesNotPublishPeer) {
  MooncakeKVCacheTransferState& core =
      MooncakeKVCacheTransferState::get_instance();
  const WorkerCacheLayoutManifest local =
      make_peer_manifest("source", "active-failure-source", 1);
  ASSERT_TRUE(core.set_local_cache_layout(local).ok());
  const WorkerCacheLayoutManifest destination =
      make_peer_manifest("unreachable", "active-failure-destination", 1);

  EXPECT_FALSE(core.set_cache_peer(destination, CachePeerMode::ACTIVE).ok());
  EXPECT_FALSE(core.has_reshard_plan(destination.addr));

  MooncakeKVCacheTransferEngine transfer(/*listen_port=*/0,
                                         torch::Device(torch::kCPU));
  std::vector<ByteRegion> regions;
  EXPECT_FALSE(transfer
                   .bind_outgoing_regions(destination.addr,
                                          {},
                                          CacheNamespace::MAIN,
                                          /*layer_id=*/0,
                                          &regions)
                   .ok());
  EXPECT_FALSE(transfer
                   .bind_outgoing_regions_explicit(destination.addr,
                                                   {},
                                                   CacheNamespace::MAIN,
                                                   /*layer_id=*/0,
                                                   &regions)
                   .ok());

  WorkerCacheLayoutManifest updated_local = local;
  updated_local.layout_generation = 2;
  EXPECT_TRUE(core.set_local_cache_layout(updated_local).ok());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PullUsesGroupSpecificMappingsForKvAndLinearBuffers) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  caches.emplace_back(LinearAttentionKVCacheTensors{torch::zeros({1, 3}),
                                                    torch::zeros({1, 5})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {1, 2};
  kv_mapping.remote_ids = {11, 12};
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {7};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {kv_mapping, linear_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const RecordingMooncakeTransferEngine::MoveCall& call =
      engine_observer->move_calls[0];
  EXPECT_EQ(call.remote_addr, "remote");
  EXPECT_EQ(call.opcode, MooncakeTransferEngine::MoveOpcode::READ);
  ASSERT_EQ(call.mappings.size(), 4U);
  EXPECT_EQ(call.mappings[0].buf_id, 0);
  EXPECT_EQ(call.mappings[1].buf_id, 1);
  EXPECT_EQ(call.mappings[0].local_ids, kv_mapping.local_ids);
  EXPECT_EQ(call.mappings[1].remote_ids, kv_mapping.remote_ids);
  EXPECT_EQ(call.mappings[2].buf_id, 2);
  EXPECT_EQ(call.mappings[3].buf_id, 3);
  EXPECT_EQ(call.mappings[2].local_ids, linear_mapping.local_ids);
  EXPECT_EQ(call.mappings[3].remote_ids, linear_mapping.remote_ids);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     RegistersCheckpointedSsmAsLogicalSequenceSlots) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"qwen3_5",
                                          std::move(engine));
  transfer.addr_ = "local";
  ModelArgs model_args;
  model_args.model_type("qwen3_5")
      .n_layers(1)
      .n_heads(4)
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(1)
      .linear_value_head_dim(1);
  transfer.configure_cache_layout(make_args(/*rank=*/0,
                                            /*world_size=*/1,
                                            /*dp_size=*/1),
                                  model_args,
                                  /*block_token_capacity=*/0,
                                  /*is_spec_draft=*/false);

  proto::KVCacheShape proto_shape;
  for (int64_t dim : {2, 1, 6}) {
    proto_shape.add_conv_cache_shape(dim);
  }
  for (int64_t dim : {6, 2, 1, 1}) {
    proto_shape.add_ssm_cache_shape(dim);
  }
  const KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
  std::vector<KVCache> caches;
  caches.emplace_back(LinearAttentionKVCacheTensors{
      torch::zeros({2, 1, 6}), torch::zeros({6, 2, 1, 1})});

  transfer.register_kv_cache(caches, shape, torch::kFloat32);

  ASSERT_EQ(engine_observer->registered_block_bytes.size(), 1U);
  ASSERT_EQ(engine_observer->registered_lifetimes.size(), 1U);
  EXPECT_NE(engine_observer->registered_lifetimes[0], nullptr);
  ASSERT_EQ(engine_observer->registered_block_bytes[0].size(), 2U);
  EXPECT_EQ(engine_observer->registered_block_bytes[0][0], 24U);
  EXPECT_EQ(engine_observer->registered_block_bytes[0][1], 24U);
  ASSERT_EQ(transfer.local_cache_layout_.tensors.size(), 2U);
  const auto ssm_it = std::find_if(
      transfer.local_cache_layout_.tensors.begin(),
      transfer.local_cache_layout_.tensors.end(),
      [](const CacheTensorManifest& tensor) {
        return tensor.role == static_cast<int32_t>(KVCacheTensorRole::SSM);
      });
  ASSERT_NE(ssm_it, transfer.local_cache_layout_.tensors.end());
  EXPECT_EQ(ssm_it->resource_count, 2U);
  EXPECT_EQ(ssm_it->physical_rows_per_resource, 3U);
  EXPECT_EQ(ssm_it->resource_stride_bytes, 24U);
  ASSERT_EQ(ssm_it->shard.spans.size(), 2U);
  EXPECT_EQ(ssm_it->shard.spans[0].repeat_count, 3U);
}

TEST(MooncakeKVCacheTransferDefaultTest, RegistersCompressedIndexAndKpoolTail) {
  for (const int32_t pool_size : {1, 4}) {
    SCOPED_TRACE(pool_size);
    auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
        /*listen_port=*/0, torch::Device(torch::kCPU));
    RecordingMooncakeTransferEngine* engine_observer = engine.get();
    MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                            /*listen_port=*/0,
                                            torch::Device(torch::kCPU),
                                            /*model_type=*/"glm5",
                                            std::move(engine));
    transfer.addr_ = "local";
    ModelArgs args;
    args.n_layers(1)
        .n_heads(1)
        .n_kv_heads(1)
        .head_dim(8)
        .index_n_heads(1)
        .index_head_dim(8)
        .index_kpool(pool_size)
        .index_kpool_compress(true);
    KVCacheCapacity capacity;
    capacity.n_blocks(4).block_size(16).num_linear_state_blocks(2).kpool_layout(
        KPoolCacheLayout::COMPRESSED_WITH_TAIL);
    const KVCacheShape shape(capacity, args, /*world_size=*/1);
    IndexedKVCacheTensors tensors;
    tensors.kv_cache_tensors.key_cache =
        torch::zeros(shape.key_cache_shape(), torch::kBFloat16);
    tensors.kv_cache_tensors.value_cache =
        torch::zeros(shape.value_cache_shape(), torch::kBFloat16);
    tensors.index_cache =
        torch::zeros(shape.index_cache_shape(), torch::kBFloat16);
    tensors.kpool_tail =
        torch::zeros(shape.kpool_tail_shape(), torch::kBFloat16);
    std::vector<KVCache> caches;
    caches.emplace_back(std::move(tensors));
    transfer.configure_cache_layout(make_args(/*rank=*/0,
                                              /*world_size=*/1,
                                              /*dp_size=*/1),
                                    args,
                                    /*block_token_capacity=*/16,
                                    /*is_spec_draft=*/false);
    transfer.register_kv_cache(caches, shape, torch::kBFloat16);

    const auto& manifests = transfer.local_cache_layout_.tensors;
    ASSERT_EQ(manifests.size(), 4U);
    const auto index = std::find_if(
        manifests.begin(), manifests.end(), [](const CacheTensorManifest& m) {
          return m.role == static_cast<int32_t>(KVCacheTensorRole::INDEX);
        });
    const auto tail = std::find_if(
        manifests.begin(), manifests.end(), [](const CacheTensorManifest& m) {
          return m.role == static_cast<int32_t>(KVCacheTensorRole::KPOOL_TAIL);
        });
    ASSERT_NE(index, manifests.end());
    ASSERT_NE(tail, manifests.end());
    EXPECT_EQ(index->group_id, cache_group_id(BlockType::KV));
    EXPECT_EQ(index->resource_count, 4U);
    EXPECT_EQ(index->resource_stride_bytes, 16U / pool_size * 8 * 2);
    EXPECT_EQ(index->block_token_capacity, 16U);
    ASSERT_EQ(index->shard.spans.size(), 1U);
    EXPECT_EQ(index->shard.spans[0].repeat_count, 16U / pool_size);
    EXPECT_EQ(tail->group_id, cache_group_id(BlockType::LINEAR));
    EXPECT_EQ(tail->resource_count, 2U);
    EXPECT_EQ(tail->resource_stride_bytes, 2U * pool_size * 8 * 2);
    EXPECT_EQ(tail->shard.resource_scope, CacheResourceScope::SEQUENCE);

    KVTransferMapping pages;
    pages.group_id = cache_group_id(BlockType::KV);
    pages.local_ids = {1};
    pages.remote_ids = {3};
    KVTransferMapping slots;
    slots.group_id = cache_group_id(BlockType::LINEAR);
    slots.local_ids = {0};
    slots.remote_ids = {1};
    ASSERT_TRUE(transfer.pull_kv_blocks(
        /*src_cluster_id=*/1, "remote", {pages, slots}));
    ASSERT_EQ(engine_observer->move_calls.size(), 1U);
    const auto& mappings = engine_observer->move_calls[0].mappings;
    ASSERT_EQ(mappings.size(), 4U);
    EXPECT_EQ(mappings[index->mooncake_buffer_id].local_ids, pages.local_ids);
    EXPECT_EQ(mappings[index->mooncake_buffer_id].remote_ids, pages.remote_ids);
    EXPECT_EQ(mappings[tail->mooncake_buffer_id].local_ids, slots.local_ids);
    EXPECT_EQ(mappings[tail->mooncake_buffer_id].remote_ids, slots.remote_ids);
  }
}

TEST(MooncakeKVCacheTransferDefaultTest,
     RegistersTp1SpecDraftBesideTp2MainCache) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"qwen3_5",
                                          std::move(engine));
  transfer.addr_ = "local-draft-body-tp1";
  ModelArgs model_args;
  model_args.model_type("qwen3_5")
      .n_layers(1)
      .n_heads(4)
      .n_kv_heads(4)
      .head_dim(1);
  KVCacheCapacity cache_capacity;
  cache_capacity.n_blocks(4).block_size(3);

  transfer.configure_cache_layout(make_args(/*rank=*/0,
                                            /*world_size=*/2,
                                            /*dp_size=*/1),
                                  model_args,
                                  /*block_token_capacity=*/3,
                                  /*is_spec_draft=*/false);
  const KVCacheShape main_shape(cache_capacity, model_args, /*world_size=*/2);
  std::vector<KVCache> main_caches;
  main_caches.emplace_back(
      KVCacheTensors{torch::zeros(main_shape.key_cache_shape()),
                     torch::zeros(main_shape.value_cache_shape())});
  transfer.register_kv_cache(main_caches, main_shape, torch::kFloat32);

  transfer.configure_cache_layout(make_args(/*rank=*/0,
                                            /*world_size=*/1,
                                            /*dp_size=*/1),
                                  model_args,
                                  /*block_token_capacity=*/3,
                                  /*is_spec_draft=*/true);
  const KVCacheShape draft_shape(cache_capacity, model_args, /*world_size=*/1);
  std::vector<KVCache> draft_caches;
  draft_caches.emplace_back(
      KVCacheTensors{torch::zeros(draft_shape.key_cache_shape()),
                     torch::zeros(draft_shape.value_cache_shape())});
  transfer.register_kv_cache_spec(draft_caches, draft_shape, torch::kFloat32);

  EXPECT_EQ(transfer.local_cache_layout_.coordinates.tp_size, 2);
  ASSERT_EQ(transfer.local_cache_layout_.tensors.size(), 4U);
  const CacheTensorManifest& spec_key = transfer.local_cache_layout_.tensors[2];
  EXPECT_EQ(spec_key.cache_namespace, CacheNamespace::SPEC_DRAFT);
  ASSERT_EQ(spec_key.shard.spans.size(), 4U);
  EXPECT_TRUE(std::all_of(
      spec_key.shard.spans.begin(),
      spec_key.shard.spans.end(),
      [](const LogicalSpan& span) { return span.owner_tp_rank == 0; }));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     GroupedPullUsesSwaAndCompressedMappings) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"deepseek_v4",
                                          std::move(engine));
  DeepSeekV4KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({4, 2});
  tensors.swa_cache = torch::zeros({4, 3});
  tensors.compressed_block_type = BlockType::C4;
  std::vector<KVCache> caches;
  caches.emplace_back(tensors);
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping swa_mapping;
  swa_mapping.group_id = cache_group_id(BlockType::SWA);
  swa_mapping.local_ids = {1};
  swa_mapping.remote_ids = {11};
  KVTransferMapping c4_mapping;
  c4_mapping.group_id = cache_group_id(BlockType::C4);
  c4_mapping.local_ids = {2};
  c4_mapping.remote_ids = {12};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {c4_mapping, swa_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const std::vector<MooncakeTransferEngine::BufferTransferMapping>& mappings =
      engine_observer->move_calls[0].mappings;
  ASSERT_EQ(mappings.size(), 2U);
  EXPECT_EQ(mappings[0].buf_id, 0);
  EXPECT_EQ(mappings[0].local_ids, swa_mapping.local_ids);
  EXPECT_EQ(mappings[1].buf_id, 1);
  EXPECT_EQ(mappings[1].remote_ids, c4_mapping.remote_ids);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PullCoversMainAndSpecLayoutsInOneRead) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> main_caches;
  main_caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  std::vector<KVCache> spec_caches;
  spec_caches.emplace_back(LinearAttentionKVCacheTensors{torch::zeros({1, 3}),
                                                         torch::zeros({1, 5})});
  transfer.register_kv_cache(main_caches, KVCacheShape(), torch::kFloat32);
  transfer.register_kv_cache_spec(spec_caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {1};
  kv_mapping.remote_ids = {11};
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {7};

  ASSERT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {kv_mapping, linear_mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  const std::vector<MooncakeTransferEngine::BufferTransferMapping>& mappings =
      engine_observer->move_calls[0].mappings;
  ASSERT_EQ(mappings.size(), 4U);
  EXPECT_EQ(mappings[0].buf_id, 0);
  EXPECT_EQ(mappings[1].buf_id, 1);
  EXPECT_EQ(mappings[2].buf_id, 2);
  EXPECT_EQ(mappings[3].buf_id, 3);
}

TEST(MooncakeKVCacheTransferDefaultTest, PullRejectsInvalidMappings) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping mismatched;
  mismatched.group_id = cache_group_id(BlockType::KV);
  mismatched.local_ids = {1, 2};
  mismatched.remote_ids = {11};
  EXPECT_FALSE(
      transfer.pull_kv_blocks(/*src_cluster_id=*/1, "remote", {mismatched}));

  KVTransferMapping duplicate;
  duplicate.group_id = cache_group_id(BlockType::KV);
  duplicate.local_ids = {1};
  duplicate.remote_ids = {11};
  EXPECT_FALSE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, "remote", {duplicate, duplicate}));

  KVTransferMapping wrong_group;
  wrong_group.group_id = cache_group_id(BlockType::LINEAR);
  wrong_group.local_ids = {0};
  wrong_group.remote_ids = {7};
  EXPECT_FALSE(
      transfer.pull_kv_blocks(/*src_cluster_id=*/1, "remote", {wrong_group}));
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

#if defined(USE_NPU)
TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsDuplicateMappingsBeforeMerge) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings.emplace_back(info.mappings[0]);
  const ParallelArgs parallel_args = make_args(/*rank=*/0,
                                               /*world_size=*/1,
                                               /*dp_size=*/1);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     KvSplitFilterAcceptsCompleteAndPartialFinalCoverage) {
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22, 23};

  std::vector<TransferKVInfo> rank_zero_infos = filter_kv_split_infos(
      /*kv_split_rank=*/0, /*kv_split_size=*/2, {info});
  ASSERT_EQ(rank_zero_infos.size(), 1U);
  ASSERT_EQ(rank_zero_infos[0].mappings.size(), 1U);
  EXPECT_EQ(rank_zero_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{11, 12}));
  EXPECT_EQ(rank_zero_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{21, 23}));

  std::vector<TransferKVInfo> rank_one_infos = filter_kv_split_infos(
      /*kv_split_rank=*/1, /*kv_split_size=*/2, {info});
  ASSERT_EQ(rank_one_infos.size(), 1U);
  ASSERT_EQ(rank_one_infos[0].mappings.size(), 1U);
  EXPECT_EQ(rank_one_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{11}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{22}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     KvSplitFilterRemapsGroupedAttentionCaches) {
  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].group_id = cache_group_id(BlockType::C4);
  info.mappings[0].remote_ids = {21, 22, 23, 24};
  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {31};
  linear_mapping.remote_ids = {41};
  info.mappings.emplace_back(std::move(linear_mapping));

  std::vector<TransferKVInfo> rank_one_infos = filter_kv_split_infos(
      /*kv_split_rank=*/1, /*kv_split_size=*/2, {info});

  ASSERT_EQ(rank_one_infos.size(), 1U);
  ASSERT_EQ(rank_one_infos[0].mappings.size(), 2U);
  EXPECT_EQ(rank_one_infos[0].mappings[0].local_ids,
            (std::vector<uint64_t>{11, 12}));
  EXPECT_EQ(rank_one_infos[0].mappings[0].remote_ids,
            (std::vector<uint64_t>{22, 24}));
  EXPECT_EQ(rank_one_infos[0].mappings[1].local_ids,
            (std::vector<uint64_t>{31}));
  EXPECT_EQ(rank_one_infos[0].mappings[1].remote_ids,
            (std::vector<uint64_t>{41}));
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsIncompleteKvSplitCoverageBeforeFilter) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22};
  ParallelArgs parallel_args = make_args(/*rank=*/0,
                                         /*world_size=*/2,
                                         /*dp_size=*/1);
  parallel_args.kv_split_size(2);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushRejectsExcessKvSplitCoverageBeforeFilter) {
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  TransferKVInfo info = make_info(/*dst_dp_size=*/1,
                                  /*dst_tp_size=*/1,
                                  /*dst_dp_rank=*/0);
  info.mappings[0].remote_ids = {21, 22, 23, 24, 25};
  ParallelArgs parallel_args = make_args(/*rank=*/0,
                                         /*world_size=*/2,
                                         /*dp_size=*/1);
  parallel_args.kv_split_size(2);
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer;

  folly::SemiFuture<bool> future = transfer.push_kv_blocks_async(
      {info}, parallel_args, synchronizer, /*is_spec_draft=*/false);
  EXPECT_FALSE(std::move(future).get());
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

TEST(MooncakeKVCacheTransferDefaultTest,
     DISABLED_NpuLinearIndexerScaleRoundTripPeerProcess) {
  const char* command_fd_env = std::getenv(kPeerCommandFdEnv);
  const char* status_fd_env = std::getenv(kPeerStatusFdEnv);
  const char* listen_port_env = std::getenv(kPeerListenPortEnv);
  const char* device_index_env = std::getenv(kPeerDeviceIndexEnv);
  ASSERT_NE(command_fd_env, nullptr);
  ASSERT_NE(status_fd_env, nullptr);
  ASSERT_NE(listen_port_env, nullptr);
  ASSERT_NE(device_index_env, nullptr);

  const int command_fd = std::atoi(command_fd_env);
  const int status_fd = std::atoi(status_fd_env);
  const int listen_port = std::atoi(listen_port_env);
  const int device_index = std::atoi(device_index_env);
  ASSERT_GE(command_fd, 0);
  ASSERT_GE(status_fd, 0);
  ASSERT_GT(listen_port, 0);
  ASSERT_LE(listen_port, UINT16_MAX);
  ASSERT_GE(device_index, 0);
  EXPECT_EQ(run_npu_round_trip_peer(command_fd,
                                    status_fd,
                                    static_cast<uint16_t>(listen_port),
                                    device_index),
            0);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     NpuLinearIndexerScalePushAndPullRoundTrip) {
  if (std::getenv(kControllerProcessEnv) == nullptr) {
    ScopedEnvironmentVariable controller_process(kControllerProcessEnv);
    ASSERT_TRUE(controller_process.set("1"));

    const pid_t controller_pid = fork();
    ASSERT_GE(controller_pid, 0);
    if (controller_pid == 0) {
      execl("/proc/self/exe",
            "mooncake_kv_cache_transfer_test",
            "--gtest_filter=MooncakeKVCacheTransferDefaultTest."
            "NpuLinearIndexerScalePushAndPullRoundTrip",
            "--gtest_color=no",
            static_cast<char*>(nullptr));
      _exit(127);
    }

    ChildProcessGuard controller_guard(controller_pid);
    int controller_status = 0;
    pid_t waited_pid = -1;
    do {
      waited_pid = waitpid(controller_pid, &controller_status, 0);
    } while (waited_pid < 0 && errno == EINTR);
    if (waited_pid == controller_pid) {
      controller_guard.release();
    }
    ASSERT_EQ(waited_pid, controller_pid);
    ASSERT_TRUE(WIFEXITED(controller_status));
    EXPECT_EQ(WEXITSTATUS(controller_status), 0);
    return;
  }

  const int32_t device_count = Platform::device_count();
  if (device_count < 2) {
    GTEST_SKIP() << "Two NPU devices are required for Mooncake memory "
                    "transfer.";
  }
  const int32_t remote_device_index = device_count > 4 ? 4 : 1;

  const int32_t local_listen_port = net::get_local_free_port();
  int32_t remote_listen_port = net::get_local_free_port();
  while (remote_listen_port == local_listen_port) {
    remote_listen_port = net::get_local_free_port();
  }
  ASSERT_GT(local_listen_port, 0);
  ASSERT_GT(remote_listen_port, 0);

  int parent_to_child[2];
  int child_to_parent[2];
  ASSERT_EQ(pipe(parent_to_child), 0);
  ASSERT_EQ(pipe(child_to_parent), 0);
  ScopedSigpipeIgnore sigpipe_guard;
  ScopedEnvironmentVariable hccl_base_port("HCCL_IF_BASE_PORT");
  ASSERT_TRUE(hccl_base_port.set("35439"));

  ASSERT_EQ(setenv(kPeerCommandFdEnv,
                   std::to_string(parent_to_child[0]).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerStatusFdEnv,
                   std::to_string(child_to_parent[1]).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerListenPortEnv,
                   std::to_string(remote_listen_port).c_str(),
                   /*overwrite=*/1),
            0);
  ASSERT_EQ(setenv(kPeerDeviceIndexEnv,
                   std::to_string(remote_device_index).c_str(),
                   /*overwrite=*/1),
            0);

  const pid_t child_pid = fork();
  ASSERT_GE(child_pid, 0);
  if (child_pid == 0) {
    close(parent_to_child[1]);
    close(child_to_parent[0]);
    execl("/proc/self/exe",
          "mooncake_kv_cache_transfer_test",
          "--gtest_filter=MooncakeKVCacheTransferDefaultTest."
          "DISABLED_NpuLinearIndexerScaleRoundTripPeerProcess",
          "--gtest_also_run_disabled_tests",
          "--gtest_color=no",
          static_cast<char*>(nullptr));
    _exit(127);
  }

  ASSERT_TRUE(hccl_base_port.set("34439"));

  unsetenv(kPeerCommandFdEnv);
  unsetenv(kPeerStatusFdEnv);
  unsetenv(kPeerListenPortEnv);
  unsetenv(kPeerDeviceIndexEnv);

  close(parent_to_child[0]);
  close(child_to_parent[1]);
  ChildProcessGuard child_guard(child_pid);

  Device local_device(/*device_id=*/0);
  local_device.set_device();
  local_device.init_device_context();
  const torch::Device local_torch_device = local_device.unwrap();
  MooncakeKVCacheTransferDefault local_transfer(
      /*device_id=*/0,
      static_cast<uint16_t>(local_listen_port),
      local_torch_device,
      /*model_type=*/"test");
  local_transfer.initialize(/*device_id=*/0);
  NpuMixedTransferCaches local_caches =
      register_npu_mixed_transfer_caches(local_transfer, local_torch_device);

  ASSERT_EQ(local_transfer.main_layout_.layers.size(), 2U);
  ASSERT_EQ(local_transfer.main_layout_.layers[0].size(), 2U);
  ASSERT_EQ(local_transfer.main_layout_.layers[1].size(), 4U);
  EXPECT_EQ(local_transfer.main_layout_.layers[0][0].role,
            KVCacheTensorRole::CONV);
  EXPECT_EQ(local_transfer.main_layout_.layers[0][1].role,
            KVCacheTensorRole::SSM);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][0].role,
            KVCacheTensorRole::KEY);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][1].role,
            KVCacheTensorRole::VALUE);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][2].role,
            KVCacheTensorRole::INDEX);
  EXPECT_EQ(local_transfer.main_layout_.layers[1][3].role,
            KVCacheTensorRole::INDEX_SCALE);
  for (const auto& buffer : local_transfer.main_layout_.layers[0]) {
    EXPECT_EQ(buffer.group_id, cache_group_id(BlockType::LINEAR));
  }
  for (const auto& buffer : local_transfer.main_layout_.layers[1]) {
    EXPECT_EQ(buffer.group_id, cache_group_id(BlockType::KV));
  }

  uint64_t local_cluster_id = 0;
  std::string local_addr;
  local_transfer.get_cache_info(local_cluster_id, local_addr);
  ASSERT_FALSE(local_addr.empty());

  uint64_t remote_cluster_id = 0;
  uint16_t received_remote_port = 0;
  std::string remote_addr;
  ASSERT_TRUE(read_endpoint(child_to_parent[0],
                            &remote_cluster_id,
                            &received_remote_port,
                            &remote_addr));
  ASSERT_EQ(received_remote_port, static_cast<uint16_t>(remote_listen_port));
  ASSERT_FALSE(remote_addr.empty());
  ASSERT_TRUE(write_endpoint(parent_to_child[1],
                             local_cluster_id,
                             static_cast<uint16_t>(local_listen_port),
                             local_addr));
  uint8_t child_success = 0;
  ASSERT_TRUE(
      read_all(child_to_parent[0], &child_success, sizeof(child_success)));
  ASSERT_EQ(child_success, 1);
  ASSERT_TRUE(local_transfer.link_clusters(
      {remote_cluster_id}, {remote_addr}, {received_remote_port}));

  KVTransferMapping linear_mapping;
  linear_mapping.group_id = cache_group_id(BlockType::LINEAR);
  linear_mapping.local_ids = {0};
  linear_mapping.remote_ids = {1};
  KVTransferMapping kv_mapping;
  kv_mapping.group_id = cache_group_id(BlockType::KV);
  kv_mapping.local_ids = {0};
  kv_mapping.remote_ids = {1};

  local_device.set_device();
  fill_mixed_transfer_block(
      &local_caches, /*block_id=*/0, /*pull_pattern=*/false);
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);

  KVCacheTransfer::KVCacheInfo info;
  info.dst_cluster_id = remote_cluster_id;
  info.dst_addr = remote_addr;
  info.mappings = {linear_mapping, kv_mapping};
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_infos;
  merged_infos.emplace(remote_addr, std::move(info));
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer =
      std::make_shared<KVPushSynchronizerImpl>(/*num_layers=*/2);
  ASSERT_TRUE(synchronizer->record_event(/*layer_index=*/0,
                                         /*device_index=*/0));
  ASSERT_TRUE(synchronizer->record_event(/*layer_index=*/1,
                                         /*device_index=*/0));
  ASSERT_TRUE(local_transfer.push_kv_blocks(merged_infos,
                                            synchronizer,
                                            /*is_spec_draft=*/false,
                                            /*kv_split_rank=*/0,
                                            /*kv_split_size=*/1));

  int32_t command = kValidatePushCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  ASSERT_TRUE(
      read_all(child_to_parent[0], &child_success, sizeof(child_success)));
  ASSERT_EQ(child_success, 1);

  command = kPreparePullCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  ASSERT_TRUE(
      read_all(child_to_parent[0], &child_success, sizeof(child_success)));
  ASSERT_EQ(child_success, 1);

  local_device.set_device();
  zero_mixed_transfer_block(&local_caches, /*block_id=*/0);
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);
  ASSERT_TRUE(local_transfer.pull_kv_blocks(
      remote_cluster_id, remote_addr, {linear_mapping, kv_mapping}));
  ASSERT_EQ(local_device.synchronize_default_stream(), 0);
  EXPECT_TRUE(mixed_transfer_block_matches(
      local_caches, /*block_id=*/0, /*pull_pattern=*/true));

  ASSERT_TRUE(local_transfer.unlink_cluster(remote_cluster_id,
                                            remote_addr,
                                            received_remote_port,
                                            /*force_flag=*/true));
  command = kStopChildCommand;
  ASSERT_TRUE(write_all(parent_to_child[1], &command, sizeof(command)));
  close(parent_to_child[1]);
  close(child_to_parent[0]);

  int child_status = 0;
  pid_t waited_pid = -1;
  do {
    waited_pid = waitpid(child_pid, &child_status, 0);
  } while (waited_pid < 0 && errno == EINTR);
  if (waited_pid == child_pid) {
    child_guard.release();
  }
  ASSERT_EQ(waited_pid, child_pid);
  ASSERT_TRUE(WIFEXITED(child_status));
  EXPECT_EQ(WEXITSTATUS(child_status), 0);

  const int32_t exit_status = ::testing::Test::HasFailure() ? 1 : 0;
  _exit(exit_status);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     PushPropagatesSynchronizeLayerFailure) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "NPU device is required for synchronizer failure test.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch::Device(torch::kCPU));
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch::Device(torch::kCPU),
                                          /*model_type=*/"test",
                                          std::move(engine));
  std::vector<KVCache> caches;
  caches.emplace_back(
      KVCacheTensors{torch::zeros({4, 2}), torch::zeros({4, 2})});
  transfer.register_kv_cache(caches, KVCacheShape(), torch::kFloat32);

  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {2};
  KVCacheTransfer::KVCacheInfo info;
  info.dst_cluster_id = 1;
  info.dst_addr = "remote";
  info.mappings.emplace_back(std::move(mapping));
  std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo> merged_infos;
  merged_infos.emplace("remote", std::move(info));
  std::shared_ptr<KVPushSynchronizerImpl> synchronizer =
      std::make_shared<KVPushSynchronizerImpl>(/*num_layers=*/1);
  synchronizer->abort();

  EXPECT_FALSE(transfer.push_kv_blocks(merged_infos,
                                       synchronizer,
                                       /*is_spec_draft=*/false,
                                       /*kv_split_rank=*/0,
                                       /*kv_split_size=*/1));
  EXPECT_TRUE(engine_observer->move_calls.empty());
}

#endif

#if defined(USE_MLU)
TEST(MooncakeKVCacheTransferDefaultTest,
     AddBufUsesLogicalLengthWithoutChangingBlockBytes) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch_device,
      /*model_type=*/"test");
  const torch::Tensor tensor = torch::zeros(
      {2, 96, 1}, torch::dtype(torch::kFloat32).device(torch_device));
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> block_bytes;

  transfer.add_buf(tensor, addrs, lens, block_bytes);

  ASSERT_EQ(addrs.size(), 1U);
  EXPECT_EQ(addrs[0], tensor.data_ptr());
  EXPECT_EQ(lens[0], tensor.nbytes());
  EXPECT_EQ(block_bytes[0], kScaleBlockBytes);
}

TEST(MooncakeKVCacheTransferDefaultTest,
     RegistersMixedLayersFromProtocolRolesInStableOrder) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch_device);
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch_device,
                                          /*model_type=*/"glm_moe_dsa",
                                          std::move(engine));
  std::vector<KVCache> caches = make_mixed_transfer_caches(torch_device);
  const KVCacheShape shape = make_indexer_int8_transfer_shape();

  transfer.register_kv_cache(caches, shape, torch::kBFloat16);

  ASSERT_EQ(engine_observer->registered_addrs.size(), 1U);
  ASSERT_EQ(engine_observer->registered_addrs[0].size(), 10U);
  const std::vector<void*> expected_addrs = {
      caches[0].get_k_cache().data_ptr(),
      caches[0].get_index_cache().data_ptr(),
      caches[0].get_indexer_cache_scale()->data_ptr(),
      caches[1].get_k_cache().data_ptr(),
      caches[1].get_k_cache_scale()->data_ptr(),
      caches[2].get_k_cache().data_ptr(),
      caches[2].get_index_cache().data_ptr(),
      caches[2].get_indexer_cache_scale()->data_ptr(),
      caches[3].get_k_cache().data_ptr(),
      caches[3].get_k_cache_scale()->data_ptr()};
  EXPECT_EQ(engine_observer->registered_addrs[0], expected_addrs);
  EXPECT_EQ(engine_observer->registered_lens[0][2],
            caches[0].get_indexer_cache_scale()->nbytes());
  EXPECT_EQ(engine_observer->registered_block_bytes[0][2],
            caches[0].get_indexer_cache_scale()->nbytes() / 2);
  EXPECT_EQ(engine_observer->registered_lens[0][4],
            caches[1].get_k_cache_scale()->nbytes());
  EXPECT_EQ(engine_observer->registered_block_bytes[0][4],
            caches[1].get_k_cache_scale()->nbytes() / 2);

  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {0};
  EXPECT_TRUE(transfer.pull_kv_blocks(
      /*src_cluster_id=*/1, /*src_addr=*/"remote", {mapping}));
  ASSERT_EQ(engine_observer->move_calls.size(), 1U);
  ASSERT_EQ(engine_observer->move_calls[0].mappings.size(), 10U);
  for (size_t index = 0; index < engine_observer->move_calls[0].mappings.size();
       ++index) {
    EXPECT_EQ(engine_observer->move_calls[0].mappings[index].buf_id,
              static_cast<int64_t>(index));
  }
}

TEST(MooncakeKVCacheTransferDefaultTest,
     SpecRegistrationStartsAfterActualMainBufferCount) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake registration tests.";
  }
  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  auto engine = std::make_unique<RecordingMooncakeTransferEngine>(
      /*listen_port=*/0, torch_device);
  RecordingMooncakeTransferEngine* engine_observer = engine.get();
  MooncakeKVCacheTransferDefault transfer(/*device_id=*/0,
                                          /*listen_port=*/0,
                                          torch_device,
                                          /*model_type=*/"glm_moe_dsa",
                                          std::move(engine));
  std::vector<KVCache> main_caches = make_mixed_transfer_caches(torch_device);
  std::vector<KVCache> draft_source = make_mixed_transfer_caches(torch_device);
  std::vector<KVCache> draft_caches;
  draft_caches.reserve(2);
  draft_caches.emplace_back(std::move(draft_source[1]));
  draft_caches.emplace_back(std::move(draft_source[0]));
  const KVCacheShape shape = make_indexer_int8_transfer_shape();

  transfer.register_kv_cache(main_caches, shape, torch::kBFloat16);
  transfer.register_kv_cache_spec(draft_caches, shape, torch::kBFloat16);

  ASSERT_EQ(engine_observer->registered_addrs.size(), 2U);
  EXPECT_EQ(engine_observer->registered_addrs[0].size(), 10U);
  EXPECT_EQ(engine_observer->registered_addrs[1].size(), 5U);
  ASSERT_EQ(transfer.spec_layout_.layers.size(), 2U);
  ASSERT_EQ(transfer.spec_layout_.layers[0].size(), 2U);
  ASSERT_EQ(transfer.spec_layout_.layers[1].size(), 3U);
  EXPECT_EQ(transfer.spec_layout_.layers[0][0].buf_id, 10);
  EXPECT_EQ(transfer.spec_layout_.layers[0][1].buf_id, 11);
  EXPECT_EQ(transfer.spec_layout_.layers[1][0].buf_id, 12);
  EXPECT_EQ(transfer.spec_layout_.layers[1][1].buf_id, 13);
  EXPECT_EQ(transfer.spec_layout_.layers[1][2].buf_id, 14);
}

TEST(MooncakeKVCacheTransferDefaultTest, AddBufRejectsNonContiguousTensor) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      /*listen_port=*/0,
      torch::Device(torch::kCPU),
      /*model_type=*/"test");
  torch::Tensor tensor = torch::zeros({2, 96, 2}, torch::kFloat32)
                             .transpose(/*dim0=*/1, /*dim1=*/2);
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> block_bytes;

  EXPECT_DEATH(transfer.add_buf(tensor, addrs, lens, block_bytes),
               "contiguous");
}

TEST(MooncakeKVCacheTransferDefaultTest,
     IndexScaleRegistersAndRoundTripsWithKvBlocks) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for Mooncake memory transfer.";
  }

  Device device(/*device_id=*/0);
  device.set_device();
  const torch::Device torch_device = device.unwrap();
  const int32_t listen_port = net::get_local_free_port();
  ASSERT_GT(listen_port, 0);

  MooncakeKVCacheTransferDefault transfer(
      /*device_id=*/0,
      static_cast<uint16_t>(listen_port),
      torch_device,
      /*model_type=*/"deepseek_v32");
  transfer.initialize(/*device_id=*/0);

  const KVCacheShape shape = make_indexer_int8_transfer_shape();
  KVCacheCreateOptions options;
  options.device(torch_device)
      .dtype(torch::kBFloat16)
      .num_layers(1)
      .model_type("deepseek_v32")
      .enable_lighting_indexer(true)
      .enable_indexer_cache_quant(true);
  std::vector<KVCache> caches;
  allocate_kv_caches(caches, shape, options);
  ASSERT_EQ(caches.size(), 1U);

  KVCache& cache = caches[0];
  torch::Tensor key_cache = cache.get_k_cache();
  torch::Tensor value_cache = cache.get_v_cache();
  torch::Tensor index_cache = cache.get_index_cache();
  std::optional<torch::Tensor> index_scale = cache.get_indexer_cache_scale();
  ASSERT_TRUE(index_scale.has_value());
  ASSERT_EQ(index_cache.scalar_type(), torch::kChar);
  ASSERT_EQ(index_scale->scalar_type(), torch::kFloat32);
  EXPECT_EQ(index_scale->nbytes(), 2 * kScaleBlockBytes);
  EXPECT_EQ(index_scale->storage().nbytes(), index_scale->nbytes());

  key_cache.index({0}).fill_(1.25);
  key_cache.index({1}).zero_();
  value_cache.index({0}).fill_(-2.5);
  value_cache.index({1}).zero_();
  index_cache.index({0}).fill_(42);
  index_cache.index({1}).zero_();
  index_scale->index({0}).fill_(0.125);
  index_scale->index({1}).zero_();
  device.synchronize_default_stream();

  transfer.register_kv_cache(caches, shape, torch::kBFloat16);

  ASSERT_EQ(transfer.main_layout_.layers.size(), 1U);
  ASSERT_EQ(transfer.main_layout_.layers[0].size(), 4U);
  for (size_t index = 0; index < transfer.main_layout_.layers[0].size();
       ++index) {
    EXPECT_EQ(transfer.main_layout_.layers[0][index].buf_id,
              static_cast<int64_t>(index));
  }

  uint64_t cluster_id = 0;
  std::string addr;
  transfer.get_cache_info(cluster_id, addr);
  ASSERT_FALSE(addr.empty());
  MooncakeTransferEngine& transfer_engine = *transfer.mooncake_te_;
  ASSERT_TRUE(transfer_engine.open_session(/*control_endpoint=*/"", addr));
  KVTransferMapping mapping;
  mapping.group_id = cache_group_id(BlockType::KV);
  mapping.local_ids = {1};
  mapping.remote_ids = {0};
  const bool transfer_success =
      transfer.pull_kv_blocks(/*src_cluster_id=*/0, addr, {mapping});
  EXPECT_TRUE(transfer_engine.close_session(/*control_endpoint=*/"", addr));
  ASSERT_TRUE(transfer_success);
  device.synchronize_default_stream();

  EXPECT_TRUE(torch::equal(key_cache.index({1}), key_cache.index({0})));
  EXPECT_TRUE(torch::equal(value_cache.index({1}), value_cache.index({0})));
  EXPECT_TRUE(torch::equal(index_cache.index({1}), index_cache.index({0})));
  EXPECT_TRUE(torch::equal(index_scale->index({1}), index_scale->index({0})));
}
#endif

}  // namespace xllm
