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

#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/mooncake_store_backend.h"

namespace xllm {

class KVCacheStoreTestPeer final {
 public:
  static void initialize_index(KVCacheStore* store,
                               const KVCacheStoreInitConfig& config,
                               HostCacheStoreIndex store_index) {
    store->config_ = config;
    store->initialize_store_index(std::move(store_index));
  }

  static std::vector<std::pair<std::string, std::string>> build_keys(
      const KVCacheStore& store,
      std::vector<BlockTransferInfo> block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    const std::vector<KVCacheStore::PhysicalRequest> requests =
        store.build_requests(slice);
    std::vector<std::pair<std::string, std::string>> keys;
    keys.reserve(requests.size());
    for (const KVCacheStore::PhysicalRequest& request : requests) {
      keys.emplace_back(request.entry->key_component, request.key);
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
    uint32_t tp_size = 2,
    bool enable_mla = false) {
  KVCacheStoreInitConfig config;
  config.model_id = model_id;
  config.tp_rank = tp_rank;
  config.tp_size = tp_size;
  config.enable_mla = enable_mla;
  return config;
}

std::string key_for_component(
    const std::vector<std::pair<std::string, std::string>>& keys,
    const std::string& component) {
  const auto key =
      std::find_if(keys.begin(), keys.end(), [&component](const auto& entry) {
        return entry.first == component;
      });
  return key == keys.end() ? "" : key->second;
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
  speculative_index[BlockType::KV].emplace_back(HostCacheStoreEntry{
      /*cache_handle=*/1, "spec_draft::mtp::draft-model", &draft_cache});
  KVCacheStoreTestPeer::initialize_index(
      &speculative_store, make_store_config(), std::move(speculative_index));

  const std::vector<BlockTransferInfo> block_info = {make_block_info(3)};
  const auto target_only_keys =
      KVCacheStoreTestPeer::build_keys(target_only_store, block_info);
  const auto speculative_keys =
      KVCacheStoreTestPeer::build_keys(speculative_store, block_info);

  ASSERT_EQ(target_only_keys.size(), 1U);
  ASSERT_EQ(speculative_keys.size(), 2U);
  EXPECT_EQ(target_only_keys.front().second,
            key_for_component(speculative_keys, "main"));
  EXPECT_NE(
      key_for_component(speculative_keys, "main"),
      key_for_component(speculative_keys, "spec_draft::mtp::draft-model"));
  EXPECT_EQ(target_only_keys.front().second.find("xllm-kv-v3:"), 0U);
}

TEST(KVCacheStoreTest, DraftKeyDependsOnTargetDraftAndAlgorithm) {
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(5)};

  const auto build_draft_key = [&draft_cache, &block_info](
                                   const std::string& target_model_id,
                                   const std::string& component) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(
        HostCacheStoreEntry{/*cache_handle=*/1, component, &draft_cache});
    KVCacheStoreTestPeer::initialize_index(
        &store, make_store_config(target_model_id), std::move(index));
    return KVCacheStoreTestPeer::build_keys(store, block_info).front().second;
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
    return KVCacheStoreTestPeer::build_keys(store, block_info).front().second;
  };

  EXPECT_EQ(build_target_key(&small_cache), build_target_key(&large_cache));
  EXPECT_NE(build_target_key(&small_cache), build_target_key(&different_cache));
}

TEST(KVCacheStoreTest, MlaKeyExcludesTensorParallelTopology) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(8)};

  const auto build_key = [&cache, &block_info](uint32_t tp_rank,
                                               uint32_t tp_size,
                                               bool enable_mla) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(
        HostCacheStoreEntry{/*cache_handle=*/0, "main", &cache});
    KVCacheStoreTestPeer::initialize_index(
        &store,
        make_store_config("target-model", tp_rank, tp_size, enable_mla),
        std::move(index));
    return KVCacheStoreTestPeer::build_keys(store, block_info).front().second;
  };

  const std::string mla_key = build_key(/*tp_rank=*/0,
                                        /*tp_size=*/1,
                                        /*enable_mla=*/true);
  EXPECT_EQ(mla_key,
            build_key(/*tp_rank=*/7,
                      /*tp_size=*/8,
                      /*enable_mla=*/true));
  EXPECT_NE(mla_key,
            build_key(/*tp_rank=*/0,
                      /*tp_size=*/1,
                      /*enable_mla=*/false));
  EXPECT_NE(build_key(/*tp_rank=*/0,
                      /*tp_size=*/8,
                      /*enable_mla=*/false),
            build_key(/*tp_rank=*/7,
                      /*tp_size=*/8,
                      /*enable_mla=*/false));
}

TEST(KVCacheStoreTest, MlaNonzeroRankSkipsRemotePut) {
  KVCacheStore store;
  HostCacheStoreIndex index;
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0, "main", &cache});
  KVCacheStoreTestPeer::initialize_index(&store,
                                         make_store_config("target-model",
                                                           /*tp_rank=*/3,
                                                           /*tp_size=*/8,
                                                           /*enable_mla=*/true),
                                         std::move(index));
  KVCacheStoreTestPeer::mark_initialized(&store);
  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(10, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(11, BlockType::KV, /*destination_block_id=*/1)};

  // A null client makes any accidental remote Store access fail loudly.
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
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{
      /*cache_handle=*/1, "spec_draft::mtp::draft-model", &draft_cache});
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
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{
      /*cache_handle=*/1, "spec_draft::mtp::draft-model", &draft_cache});
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
