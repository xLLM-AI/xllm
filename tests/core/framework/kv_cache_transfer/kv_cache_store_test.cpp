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
#include <string>
#include <utility>
#include <vector>

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
};

namespace {

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
    const std::string& model_id = "target-model") {
  KVCacheStoreInitConfig config;
  config.model_id = model_id;
  config.tp_size = 2;
  config.tp_rank = 1;
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

}  // namespace
}  // namespace xllm
