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

#include "core/kv_cache/transfer/kv_cache_store.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/kv_cache/transfer/mooncake_store_backend.h"

namespace xllm {

class KVCacheStoreTestPeer final {
 public:
  static void initialize_index(KVCacheStore* store,
                               const KVCacheStoreInitConfig& config,
                               HostCacheStoreIndex store_index) {
    store->config_ = config;
    store->initialize_store_index(std::move(store_index));
  }

  static std::vector<std::string> build_keys(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    std::vector<std::string> keys;
    keys.reserve(requests.size());
    for (const KVCacheStore::PhysicalRequest& request : requests) {
      keys.emplace_back(request.key);
    }
    return keys;
  }

  static std::vector<uint8_t> aggregate(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info,
      const std::vector<uint8_t>& physical_results) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    return KVCacheStore::aggregate_results(
        block_transfer_info.size(), requests, physical_results);
  }

  static size_t physical_request_count(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return store.build_requests(slice).size();
  }

  static size_t unique_key_count(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    return KVCacheStore::group_requests(requests).size();
  }

  static std::vector<uint8_t> put_owners(const KVCacheStore& store) {
    std::vector<uint8_t> owners;
    for (const auto& [block_type, entries] : store.store_index_) {
      for (const KVCacheStore::StoreEntry& entry : entries) {
        owners.emplace_back(entry.is_put_owner);
      }
    }
    return owners;
  }

  static void mark_initialized(KVCacheStore* store) {
    store->is_initialized_ = true;
  }

  static std::optional<std::string> store_device_names(
      const KVCacheStoreInitConfig& config) {
    return KVCacheStore::get_store_device_names(config);
  }

  static std::optional<MooncakeMultiBuffer> multi_buffer(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    if (requests.empty()) {
      return std::nullopt;
    }
    const KVCacheStore::PhysicalRequest& request = requests.front();
    return store.build_multi_buffer(
        *request.entry,
        block_transfer_info[request.logical_index].dst_block_id);
  }
};

class MooncakeStoreBackendTestPeer final {
 public:
  static std::optional<std::vector<MooncakeRegisteredRange>> unique_ranges(
      const std::vector<MooncakeRegisteredRange>& ranges) {
    return MooncakeStoreBackend::unique_ranges(ranges);
  }

  static bool get_succeeded(int64_t expected_bytes,
                            const std::vector<int>& results) {
    return MooncakeStoreBackend::get_succeeded(expected_bytes, results);
  }

  static bool put_succeeded(int result) {
    return MooncakeStoreBackend::put_succeeded(result);
  }
};

namespace {

class ScopedEnvVar final {
 public:
  explicit ScopedEnvVar(const std::string& name) : name_(name) {
    const char* value = std::getenv(name_.c_str());
    if (value != nullptr) {
      old_value_ = value;
    }
  }

  ~ScopedEnvVar() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  bool set(const std::string& value) {
    return setenv(name_.c_str(), value.c_str(), /*overwrite=*/1) == 0;
  }

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

KVCache make_attention_cache(int64_t host_blocks, int64_t width) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return KVCache(
      KVCacheTensors{torch::zeros({host_blocks, 2, width}, options),
                     torch::zeros({host_blocks, 2, width}, options)});
}

KVCache make_linear_cache(int64_t host_blocks, int64_t width) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return KVCache(LinearAttentionKVCacheTensors{
      torch::zeros({host_blocks, width}, options),
      torch::zeros({host_blocks, width}, options)});
}

BlockTransferInfo make_block_info(uint8_t hash_value,
                                  BlockType block_type = BlockType::KV,
                                  int32_t destination_block_id = 0) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash_key;
  hash_key.fill(hash_value);
  return BlockTransferInfo(/*src_id=*/0,
                           destination_block_id,
                           hash_key.data(),
                           TransferType::G2H,
                           block_type);
}

KVCacheStoreInitConfig make_store_config(
    const std::string& model_id = "target-model",
    uint32_t tp_rank = 1,
    uint32_t tp_size = 2) {
  KVCacheStoreInitConfig config;
  config.model_id = model_id;
  config.tp_rank = tp_rank;
  config.tp_size = tp_size;
  config.dp_local_rank = tp_rank;
  config.kv_replica_rank = tp_rank;
  return config;
}

TEST(KVCacheStoreTest, TargetKeyDoesNotDependOnDraftRegistration) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);

  KVCacheStore target_only_store;
  HostCacheStoreIndex target_only_index;
  target_only_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &target_cache});
  KVCacheStoreTestPeer::initialize_index(
      &target_only_store, make_store_config(), std::move(target_only_index));

  KVCacheStore speculative_store;
  HostCacheStoreIndex speculative_index;
  speculative_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &target_cache});
  speculative_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          /*enable_mla=*/false,
                          /*is_target=*/false});
  KVCacheStoreTestPeer::initialize_index(
      &speculative_store, make_store_config(), std::move(speculative_index));

  const std::vector<BlockTransferInfo> block_info = {make_block_info(3)};
  const auto target_only_keys =
      KVCacheStoreTestPeer::build_keys(target_only_store, block_info);
  const auto speculative_keys =
      KVCacheStoreTestPeer::build_keys(speculative_store, block_info);

  ASSERT_EQ(target_only_keys.size(), 1U);
  ASSERT_EQ(speculative_keys.size(), 2U);
  EXPECT_EQ(target_only_keys.front(), speculative_keys[0]);
  EXPECT_NE(speculative_keys[0], speculative_keys[1]);
  EXPECT_EQ(target_only_keys.front().find("xllm-kv-v3:"), 0U);
}

TEST(KVCacheStoreTest, DraftKeyDependsOnTargetDraftAndAlgorithm) {
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(5)};

  const auto build_draft_key = [&draft_cache, &block_info](
                                   const std::string& target_model_id,
                                   const std::string& component) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/1,
                                                          component,
                                                          &draft_cache,
                                                          /*enable_mla=*/false,
                                                          /*is_target=*/false});
    KVCacheStoreTestPeer::initialize_index(
        &store, make_store_config(target_model_id), std::move(index));
    return KVCacheStoreTestPeer::build_keys(store, block_info).front();
  };

  const std::string baseline =
      build_draft_key("target-a", "spec_draft::mtp::draft-a");
  EXPECT_NE(baseline, build_draft_key("target-b", "spec_draft::mtp::draft-a"));
  EXPECT_NE(baseline, build_draft_key("target-a", "spec_draft::mtp::draft-b"));
  EXPECT_NE(baseline,
            build_draft_key("target-a", "spec_draft::dspark::draft-a"));
  EXPECT_EQ(baseline, build_draft_key("target-a", "spec_draft::mtp::draft-a"));
}

TEST(KVCacheStoreTest, SchemaExcludesHostCapacity) {
  KVCache small_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache large_cache = make_attention_cache(/*host_blocks=*/5, /*width=*/8);
  KVCache different_cache =
      make_attention_cache(/*host_blocks=*/2, /*width=*/16);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(7)};

  const auto build_target_key = [&block_info](KVCache* cache) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(
        HostCacheStoreEntry{/*cache_handle=*/0, "main", cache});
    KVCacheStoreTestPeer::initialize_index(
        &store, make_store_config(), std::move(index));
    return KVCacheStoreTestPeer::build_keys(store, block_info).front();
  };

  EXPECT_EQ(build_target_key(&small_cache), build_target_key(&large_cache));
  EXPECT_NE(build_target_key(&small_cache), build_target_key(&different_cache));
}

TEST(KVCacheStoreTest, MixedTargetMlaAndDraftKeyLayouts) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCache linear_cache = make_linear_cache(/*host_blocks=*/2, /*width=*/6);

  const auto build_keys = [&](uint32_t tp_rank,
                              uint32_t tp_size,
                              int32_t split_size,
                              int32_t split_rank) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{
        /*cache_handle=*/0, "main", &target_cache, /*enable_mla=*/true});
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/1,
                                                          "spec_draft::mtp::d",
                                                          &draft_cache,
                                                          /*enable_mla=*/false,
                                                          /*is_target=*/false});
    index[BlockType::LINEAR].emplace_back(HostCacheStoreEntry{
        /*cache_handle=*/0, "main", &linear_cache, /*enable_mla=*/true});
    KVCacheStoreInitConfig config =
        make_store_config("target-model", tp_rank, tp_size);
    config.kv_split_size = split_size;
    config.kv_split_rank = split_rank;
    KVCacheStoreTestPeer::initialize_index(&store, config, std::move(index));
    return KVCacheStoreTestPeer::build_keys(
        store,
        {make_block_info(8, BlockType::KV),
         make_block_info(8, BlockType::LINEAR)});
  };

  const auto keys = build_keys(/*tp_rank=*/3,
                               /*tp_size=*/8,
                               /*split_size=*/2,
                               /*split_rank=*/1);
  ASSERT_EQ(keys.size(), 3U);
  const std::string target_key = keys[0];
  const std::string draft_key = keys[1];
  const std::string linear_key = keys[2];
  EXPECT_NE(target_key.find(":main:mla:0:kv:2:1:"), std::string::npos)
      << target_key;
  EXPECT_NE(draft_key.find(":spec_draft::mtp::d:8:0:3:"), std::string::npos)
      << draft_key;
  EXPECT_EQ(draft_key.find(":kv:"), std::string::npos) << draft_key;
  EXPECT_NE(linear_key.find(":main:8:5:3:"), std::string::npos) << linear_key;
  EXPECT_EQ(linear_key.find(":kv:"), std::string::npos) << linear_key;

  const auto other_split = build_keys(/*tp_rank=*/3,
                                      /*tp_size=*/8,
                                      /*split_size=*/2,
                                      /*split_rank=*/0);
  EXPECT_NE(target_key, other_split[0]);
  const auto other_split_size = build_keys(/*tp_rank=*/3,
                                           /*tp_size=*/8,
                                           /*split_size=*/4,
                                           /*split_rank=*/1);
  EXPECT_NE(target_key, other_split_size[0]);
  EXPECT_EQ(draft_key, other_split[1]);
  EXPECT_EQ(draft_key, other_split_size[1]);
  EXPECT_EQ(linear_key, other_split[2]);
  EXPECT_EQ(linear_key, other_split_size[2]);

  const auto other_rank = build_keys(/*tp_rank=*/7,
                                     /*tp_size=*/8,
                                     /*split_size=*/2,
                                     /*split_rank=*/1);
  EXPECT_EQ(target_key, other_rank[0]);
  const auto single_tp = build_keys(/*tp_rank=*/0,
                                    /*tp_size=*/1,
                                    /*split_size=*/2,
                                    /*split_rank=*/1);
  EXPECT_EQ(target_key, single_tp[0]);
  EXPECT_NE(draft_key, other_rank[1]);
  EXPECT_NE(draft_key, single_tp[1]);
  EXPECT_NE(linear_key, other_rank[2]);
  EXPECT_NE(linear_key, single_tp[2]);
}

TEST(KVCacheStoreTest, WriterOwnershipFollowsTopology) {
  struct Topology {
    uint32_t tp_size;
    uint32_t cp_size;
    int32_t kv_split_size;
  };
  const Topology topologies[] = {{8, 1, 2}, {2, 4, 2}, {8, 4, 2}, {8, 1, 8}};
  struct CacheLayout {
    bool enable_mla;
    bool is_target;
    BlockType block_type;
  };
  const CacheLayout layouts[] = {{true, true, BlockType::KV},
                                 {false, true, BlockType::KV},
                                 {true, false, BlockType::KV},
                                 {false, false, BlockType::KV},
                                 {true, true, BlockType::LINEAR}};
  for (const auto& topology : topologies) {
    for (const auto& layout : layouts) {
      std::map<std::string, uint32_t> writers;
      for (uint32_t rank = 0; rank < 8; ++rank) {
        const uint32_t ranks_per_split = 8 / topology.kv_split_size;
        auto config = make_store_config(
            "target-model", rank % topology.tp_size, topology.tp_size);
        config.dp_local_rank = rank;
        config.kv_split_size = topology.kv_split_size;
        config.kv_split_rank = topology.cp_size == 1
                                   ? rank % topology.kv_split_size
                                   : rank / ranks_per_split;
        config.kv_replica_rank = topology.cp_size == 1
                                     ? rank / topology.kv_split_size
                                     : rank % ranks_per_split;
        KVCache cache =
            layout.block_type == BlockType::LINEAR
                ? make_linear_cache(/*host_blocks=*/2, /*width=*/6)
                : make_attention_cache(/*host_blocks=*/2, /*width=*/8);
        KVCacheStore store;
        HostCacheStoreIndex index;
        index[layout.block_type].emplace_back(HostCacheStoreEntry{
            0, "main", &cache, layout.enable_mla, layout.is_target});
        KVCacheStoreTestPeer::initialize_index(
            &store, config, std::move(index));
        const auto keys = KVCacheStoreTestPeer::build_keys(
            store, {make_block_info(21, layout.block_type)});
        writers[keys.front()] +=
            KVCacheStoreTestPeer::put_owners(store).front();
      }
      for (const auto& [key, count] : writers) {
        EXPECT_EQ(count, 1U) << key << ", cp_size=" << topology.cp_size
                             << ", is_target=" << layout.is_target;
      }
    }
  }
}

TEST(KVCacheStoreTest, WriterRoleControlsRemotePut) {
  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(10, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(11, BlockType::KV, /*destination_block_id=*/1)};
  for (const uint32_t rank : {0U, 5U}) {
    KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{
        /*cache_handle=*/0, "main", &cache, /*enable_mla=*/true});
    auto config = make_store_config("target-model", rank, /*tp_size=*/8);
    config.kv_split_size = 2;
    config.kv_split_rank = rank % 2;
    config.kv_replica_rank = rank / 2;
    KVCacheStoreTestPeer::initialize_index(&store, config, std::move(index));
    KVCacheStoreTestPeer::mark_initialized(&store);
    EXPECT_EQ(store.batch_put(block_info), rank == 0 ? 0U : block_info.size());
  }
}

TEST(KVCacheStoreTest, MixedOwnershipCompletesBatchPut) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache linear_cache = make_linear_cache(/*host_blocks=*/2, /*width=*/6);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &target_cache,
                                                        /*enable_mla=*/true});
  index[BlockType::LINEAR].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &linear_cache,
                          /*enable_mla=*/true});
  auto config = make_store_config("target-model", /*tp_rank=*/3, /*tp_size=*/8);
  config.kv_split_size = 2;
  config.kv_split_rank = 1;
  config.kv_replica_rank = 1;
  KVCacheStoreTestPeer::initialize_index(&store, config, std::move(index));
  KVCacheStoreTestPeer::mark_initialized(&store);
  EXPECT_EQ(KVCacheStoreTestPeer::put_owners(store),
            std::vector<uint8_t>({0, 1}));

  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(10, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(11, BlockType::KV, /*destination_block_id=*/1)};
  EXPECT_EQ(store.batch_put(block_info), block_info.size());
}

TEST(KVCacheStoreTest, SelectsOnlyExplicitRdmaDevices) {
  KVCacheStoreInitConfig config;
  config.protocol = "rdma";
  config.rdma_devices = "mlx5_0,mlx5_1";
  EXPECT_EQ(KVCacheStoreTestPeer::store_device_names(config),
            std::optional<std::string>("mlx5_0,mlx5_1"));

  config.rdma_devices.clear();
  ScopedEnvVar device_names("DEVICE_NAMES");
  ASSERT_TRUE(device_names.set("legacy_hca"));
  EXPECT_EQ(KVCacheStoreTestPeer::store_device_names(config), std::nullopt);
  EXPECT_EQ(config.protocol, "rdma");
}

TEST(KVCacheStoreTest, TcpIgnoresRdmaDevices) {
  KVCacheStoreInitConfig config;
  config.protocol = "tcp";
  config.rdma_devices = "mlx5_0";

  EXPECT_EQ(KVCacheStoreTestPeer::store_device_names(config), std::nullopt);
}

TEST(KVCacheStoreDeathTest, RejectsDuplicateKeyComponentsPerBlockType) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);

  EXPECT_DEATH(
      {
        KVCacheStore store;
        HostCacheStoreIndex index;
        index[BlockType::KV].emplace_back(
            HostCacheStoreEntry{/*cache_handle=*/0, "main", &target_cache});
        index[BlockType::KV].emplace_back(
            HostCacheStoreEntry{/*cache_handle=*/1, "main", &draft_cache});
        KVCacheStoreTestPeer::initialize_index(
            &store, make_store_config(), std::move(index));
      },
      "Duplicate KVCacheStore key component");
}

TEST(KVCacheStoreTest, AggregatesPhysicalResultsPerLogicalBlock) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &target_cache});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          /*enable_mla=*/false,
                          /*is_target=*/false});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(9, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(10, BlockType::KV, /*destination_block_id=*/1)};
  EXPECT_EQ(KVCacheStoreTestPeer::physical_request_count(store, block_info),
            4U);
  EXPECT_EQ(KVCacheStoreTestPeer::aggregate(
                store, block_info, std::vector<uint8_t>{1, 1, 1, 0}),
            std::vector<uint8_t>({1, 0}));
}

TEST(KVCacheStoreTest, DeduplicatesPhysicalKeysAndRespectsBlockTypeEntries) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCache linear_cache = make_linear_cache(/*host_blocks=*/2, /*width=*/6);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &target_cache});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          /*enable_mla=*/false,
                          /*is_target=*/false});
  index[BlockType::LINEAR].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &linear_cache});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> duplicate_kv = {
      make_block_info(11, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(11, BlockType::KV, /*destination_block_id=*/1)};
  EXPECT_EQ(KVCacheStoreTestPeer::physical_request_count(store, duplicate_kv),
            4U);
  EXPECT_EQ(KVCacheStoreTestPeer::unique_key_count(store, duplicate_kv), 2U);

  const std::vector<BlockTransferInfo> linear = {
      make_block_info(12, BlockType::LINEAR)};
  EXPECT_EQ(KVCacheStoreTestPeer::physical_request_count(store, linear), 1U);
}

TEST(KVCacheStoreTest, BuildsMultiBufferForRequestedHostBlock) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/3);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &cache});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(12, BlockType::KV, /*destination_block_id=*/1)};
  const std::optional<MooncakeMultiBuffer> buffer =
      KVCacheStoreTestPeer::multi_buffer(store, block_info);
  ASSERT_TRUE(buffer.has_value());
  ASSERT_EQ(buffer->addresses.size(), 2U);
  ASSERT_EQ(buffer->sizes.size(), 2U);
  EXPECT_EQ(buffer->sizes[0], 6U * sizeof(float));
  EXPECT_EQ(buffer->sizes[1], 6U * sizeof(float));

  const BlockTypeTensorMap tensors =
      cache.get_block_type_tensors(BlockType::KV);
  auto tensor_it = tensors.begin();
  EXPECT_EQ(buffer->addresses[0], tensor_it->second[1].data_ptr());
  ++tensor_it;
  EXPECT_EQ(buffer->addresses[1], tensor_it->second[1].data_ptr());
}

TEST(KVCacheStoreTest, FailsDuplicateGetKeysWithoutStoreAccess) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &cache});
  KVCacheStoreTestPeer::initialize_index(
      &store, make_store_config(), std::move(index));
  KVCacheStoreTestPeer::mark_initialized(&store);
  std::vector<BlockTransferInfo> block_info = {
      make_block_info(13, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(13, BlockType::KV, /*destination_block_id=*/1)};
  Slice<BlockTransferInfo> slice(block_info);

  EXPECT_EQ(store.batch_get_with_status(slice), std::vector<uint8_t>({0, 0}));
}

TEST(MooncakeStoreBackendTest, AcceptsOnlyExactSingleGetResult) {
  EXPECT_TRUE(MooncakeStoreBackendTestPeer::get_succeeded(
      /*expected_bytes=*/32, std::vector<int>{32}));
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::get_succeeded(
      /*expected_bytes=*/32, std::vector<int>{}));
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::get_succeeded(
      /*expected_bytes=*/32, std::vector<int>{-1}));
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::get_succeeded(
      /*expected_bytes=*/32, std::vector<int>{31}));
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::get_succeeded(
      /*expected_bytes=*/32, std::vector<int>{33}));
  EXPECT_TRUE(MooncakeStoreBackendTestPeer::get_succeeded(
      /*expected_bytes=*/32, std::vector<int>{32, 32}));
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::get_succeeded(
      /*expected_bytes=*/static_cast<int64_t>(INT32_MAX) + 1,
      std::vector<int>{INT32_MAX}));
}

TEST(MooncakeStoreBackendTest, TreatsObjectAlreadyExistsAsPutSuccess) {
  EXPECT_TRUE(MooncakeStoreBackendTestPeer::put_succeeded(/*result=*/0));
  EXPECT_TRUE(MooncakeStoreBackendTestPeer::put_succeeded(
      static_cast<int>(mooncake::ErrorCode::OBJECT_ALREADY_EXISTS)));
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::put_succeeded(/*result=*/-1));
}

TEST(MooncakeStoreBackendTest, ValidatesRegistrationRangesBeforeSetup) {
  const auto valid_ranges = MooncakeStoreBackendTestPeer::unique_ranges(
      {{reinterpret_cast<void*>(0x1000), 64},
       {reinterpret_cast<void*>(0x1000), 64},
       {reinterpret_cast<void*>(0x2000), 32}});
  ASSERT_TRUE(valid_ranges.has_value());
  ASSERT_EQ(valid_ranges->size(), 2U);
  EXPECT_EQ((*valid_ranges)[0].address, reinterpret_cast<void*>(0x1000));
  EXPECT_EQ((*valid_ranges)[1].address, reinterpret_cast<void*>(0x2000));

  EXPECT_FALSE(MooncakeStoreBackendTestPeer::unique_ranges(
                   {{reinterpret_cast<void*>(0x1000), 64},
                    {reinterpret_cast<void*>(0x1000), 32}})
                   .has_value());
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::unique_ranges(
                   {{reinterpret_cast<void*>(0x1000), 64},
                    {reinterpret_cast<void*>(0x1020), 64}})
                   .has_value());
  EXPECT_FALSE(MooncakeStoreBackendTestPeer::unique_ranges(
                   {{reinterpret_cast<void*>(UINTPTR_MAX - 7), 8}})
                   .has_value());
}

}  // namespace
}  // namespace xllm
