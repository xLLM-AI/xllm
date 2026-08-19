/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/kv_cache_transfer/kv_cache_store.h"

namespace xllm {

class KVCacheStoreTestPeer final {
 public:
  static void set_config(KVCacheStore* store,
                         const KVCacheStoreInitConfig& config) {
    store->config_ = config;
  }

  static void set_components(KVCacheStore* store,
                             std::vector<HostCacheComponentSchema> components) {
    store->components_ = std::move(components);
  }

  static std::string build_component_key(
      const KVCacheStore& store,
      const HostCacheComponentSchema& component,
      const BlockTransferInfo& block_info) {
    return store.build_component_key(component, block_info);
  }

  static size_t required_component_count(const KVCacheStore& store,
                                         BlockType block_type) {
    return store.required_components(block_type).size();
  }
};

namespace {

using namespace std::chrono_literals;

BlockTransferInfo make_block_info(BlockType block_type) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash_key{};
  for (size_t index = 0; index < hash_key.size(); ++index) {
    hash_key[index] = static_cast<uint8_t>(index + 1);
  }
  return BlockTransferInfo(/*src_id=*/3,
                           /*dst_id=*/7,
                           hash_key.data(),
                           TransferType::D2H2G,
                           block_type);
}

HostCacheComponentSchema make_component(CacheParticipant participant,
                                        const std::string& model_identity,
                                        BlockType block_type = BlockType::KV) {
  HostCacheComponentSchema component;
  component.participant = participant;
  component.block_type = block_type;
  component.model_identity = model_identity;
  component.schema_fingerprint = "schema-fingerprint";
  component.tp_rank = 0;
  component.tp_size = 8;
  return component;
}

TEST(KVCacheStoreKeyTest, SeparatesTargetAndDraftComponents) {
  KVCacheStore store;
  KVCacheStoreInitConfig config;
  config.model_id = "deepseek-v4";
  KVCacheStoreTestPeer::set_config(&store, config);
  const BlockTransferInfo block_info = make_block_info(BlockType::C128);
  HostCacheComponentSchema target = make_component(
      CacheParticipant::TARGET, "deepseek-v4-target", BlockType::C128);
  HostCacheComponentSchema draft = target;
  draft.participant = CacheParticipant::DRAFT;

  const std::string target_key =
      KVCacheStoreTestPeer::build_component_key(store, target, block_info);
  const std::string draft_key =
      KVCacheStoreTestPeer::build_component_key(store, draft, block_info);

  EXPECT_EQ(target_key.rfind("xllm-kv-v3:", 0), 0u);
  EXPECT_EQ(draft_key.rfind("xllm-kv-v3:", 0), 0u);
  EXPECT_NE(target_key, draft_key);
}

TEST(KVCacheStoreKeyTest, SeparatesParticipantModelAndSchemaIdentity) {
  KVCacheStore store;
  KVCacheStoreInitConfig config;
  config.model_id = "deepseek-v4";
  KVCacheStoreTestPeer::set_config(&store, config);
  const BlockTransferInfo block_info = make_block_info(BlockType::SWA);
  HostCacheComponentSchema first = make_component(
      CacheParticipant::DRAFT, "draft-revision-a", BlockType::SWA);
  HostCacheComponentSchema second = first;
  second.model_identity = "draft-revision-b";
  HostCacheComponentSchema third = first;
  third.schema_fingerprint = "different-schema";

  const std::string first_key =
      KVCacheStoreTestPeer::build_component_key(store, first, block_info);
  const std::string second_key =
      KVCacheStoreTestPeer::build_component_key(store, second, block_info);
  const std::string third_key =
      KVCacheStoreTestPeer::build_component_key(store, third, block_info);

  EXPECT_NE(first_key, second_key);
  EXPECT_NE(first_key, third_key);
}

TEST(KVCacheStoreKeyTest, ExpandsLogicalTypeToEveryRequiredParticipant) {
  KVCacheStore store;
  std::vector<HostCacheComponentSchema> components;
  components.emplace_back(
      make_component(CacheParticipant::TARGET, "target", BlockType::C4));
  components.emplace_back(
      make_component(CacheParticipant::DRAFT, "draft", BlockType::C4));
  components.emplace_back(
      make_component(CacheParticipant::TARGET, "target", BlockType::SWA));
  KVCacheStoreTestPeer::set_components(&store, std::move(components));

  EXPECT_EQ(
      KVCacheStoreTestPeer::required_component_count(store, BlockType::C4), 2u);
  EXPECT_EQ(
      KVCacheStoreTestPeer::required_component_count(store, BlockType::SWA),
      1u);
  EXPECT_EQ(
      KVCacheStoreTestPeer::required_component_count(store, BlockType::C128),
      0u);
}

TEST(KVTransferCompletionTest, WaitsForEveryTransfer) {
  folly::Promise<bool> first_promise;
  folly::Promise<bool> second_promise;
  KVTransferCompletion completion;
  completion.add(first_promise.getSemiFuture());
  completion.add(second_promise.getSemiFuture());

  std::promise<void> waiter_started;
  std::future<void> started = waiter_started.get_future();
  std::future<bool> result = std::async(std::launch::async, [&]() {
    waiter_started.set_value();
    return completion.wait();
  });

  started.wait();
  first_promise.setValue(true);
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  second_promise.setValue(true);
  EXPECT_TRUE(result.get());
}

TEST(KVTransferCompletionTest, ReportsTransferFailure) {
  folly::Promise<bool> success_promise;
  folly::Promise<bool> failure_promise;
  KVTransferCompletion completion;
  completion.add(success_promise.getSemiFuture());
  completion.add(failure_promise.getSemiFuture());
  success_promise.setValue(true);
  failure_promise.setValue(false);

  EXPECT_FALSE(completion.wait());
}

TEST(KVTransferCompletionTest, RejectsPendingTransferAfterTimeout) {
  EXPECT_DEATH(
      {
        folly::Promise<bool> promise;
        KVTransferCompletion completion(1ms);
        completion.add(promise.getSemiFuture());
        try {
          completion.wait();
        } catch (const folly::FutureTimeout&) {
        }
      },
      "pending KV transfers");
}

TEST(KVTransferCompletionTest, RejectsPendingTransferAtDestruction) {
  EXPECT_DEATH(
      {
        folly::Promise<bool> promise;
        KVTransferCompletion completion;
        completion.add(promise.getSemiFuture());
      },
      "pending KV transfers");
}

TEST(KVTransferTrackerTest, WaitsForEveryTrackedTransfer) {
  KVTransferTracker tracker;
  std::shared_ptr<KVTransferTracker::Completion> first = tracker.track();
  std::shared_ptr<KVTransferTracker::Completion> second = tracker.track();

  std::promise<void> waiter_started;
  std::future<void> started = waiter_started.get_future();
  std::future<void> result = std::async(std::launch::async, [&]() {
    waiter_started.set_value();
    tracker.wait();
  });

  started.wait();
  first.reset();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  second.reset();
  EXPECT_EQ(result.wait_for(1s), std::future_status::ready);
}

TEST(KVTransferTrackerTest, ReportsPendingUntilEveryTransferFinishes) {
  KVTransferTracker tracker;
  EXPECT_FALSE(tracker.has_pending());

  std::shared_ptr<KVTransferTracker::Completion> first = tracker.track();
  std::shared_ptr<KVTransferTracker::Completion> second = tracker.track();
  EXPECT_TRUE(tracker.has_pending());

  first.reset();
  EXPECT_TRUE(tracker.has_pending());

  second.reset();
  EXPECT_FALSE(tracker.has_pending());
}

TEST(KVTransferTrackerTest, DestructionWaitsForTrackedTransfer) {
  auto tracker = std::make_unique<KVTransferTracker>();
  std::shared_ptr<KVTransferTracker::Completion> completion = tracker->track();

  std::promise<void> destruction_started;
  std::future<void> started = destruction_started.get_future();
  std::future<void> result = std::async(
      std::launch::async,
      [tracker = std::move(tracker), &destruction_started]() mutable {
        destruction_started.set_value();
        tracker.reset();
      });

  started.wait();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  completion.reset();
  EXPECT_EQ(result.wait_for(1s), std::future_status::ready);
}

}  // namespace
}  // namespace xllm
