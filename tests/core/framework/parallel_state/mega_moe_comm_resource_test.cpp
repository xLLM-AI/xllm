/* Copyright 2026 The xLLM Authors.

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

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "framework/parallel_state/mega_moe_comm_resource.h"

namespace xllm {
namespace {

MegaMoeCommSpec valid_spec() {
  MegaMoeCommSpec spec;
  spec.group_name = "ep_group";
  spec.hccl_comm = reinterpret_cast<HcclComm>(0x1);
  spec.ep_world_size = 16;
  spec.device_index = 0;
  spec.max_num_tokens_per_rank = 128;
  return spec;
}

TEST(MegaMoeCommResourceTest, AcceptsCompleteEp16Spec) {
  const MegaMoeCommValidation validation =
      validate_mega_moe_comm_spec(valid_spec());

  EXPECT_TRUE(validation.valid);
  EXPECT_EQ(validation.reason, MegaMoeCommRejectReason::NONE);
}

TEST(MegaMoeCommResourceTest, RejectsInvalidConstructionParameters) {
  MegaMoeCommSpec spec = valid_spec();
  spec.group_name.clear();
  EXPECT_EQ(validate_mega_moe_comm_spec(spec).reason,
            MegaMoeCommRejectReason::EMPTY_GROUP);

  spec = valid_spec();
  spec.hccl_comm = nullptr;
  EXPECT_EQ(validate_mega_moe_comm_spec(spec).reason,
            MegaMoeCommRejectReason::NULL_COMM);

  spec = valid_spec();
  spec.ep_world_size = 8;
  EXPECT_EQ(validate_mega_moe_comm_spec(spec).reason,
            MegaMoeCommRejectReason::UNSUPPORTED_EP_WORLD_SIZE);

  spec = valid_spec();
  spec.device_index = -1;
  EXPECT_EQ(validate_mega_moe_comm_spec(spec).reason,
            MegaMoeCommRejectReason::INVALID_DEVICE_INDEX);

  spec = valid_spec();
  spec.max_num_tokens_per_rank = 0;
  EXPECT_EQ(validate_mega_moe_comm_spec(spec).reason,
            MegaMoeCommRejectReason::INVALID_MAX_NUM_TOKENS_PER_RANK);
}

TEST(MegaMoeCommResourceTest, FindsRequiredCann91KfcSymbols) {
  const char* enabled = std::getenv("XLLM_RUN_CANN91_MEGA_MOE_SMOKE");
  if (enabled == nullptr || std::string(enabled) != "1") {
    GTEST_SKIP() << "CANN 9.1 KFC symbol smoke is disabled; set "
                    "XLLM_RUN_CANN91_MEGA_MOE_SMOKE=1 in a CANN 9.1 "
                    "environment.";
  }
  const MegaMoeCommSymbolStatus status = probe_mega_moe_comm_symbols();

  EXPECT_TRUE(status.available) << status.missing_symbol;
  EXPECT_TRUE(status.missing_symbol.empty());
}

TEST(MegaMoeCommResourceTest, AcceptsAccessibleSpansEqualToLocalPayload) {
  const auto validation = validate_mega_moe_buffer_accessible_spans(
      4096, std::vector<uint64_t>(16, 4096));

  EXPECT_TRUE(validation.valid);
  EXPECT_EQ(validation.required_payload_size, 4096);
}

TEST(MegaMoeCommResourceTest,
     AcceptsLocalPayloadAndLargerRemoteIpcAccessibleSpans) {
  constexpr uint64_t kLocalPayloadSize = 512ULL * 1024 * 1024;
  constexpr uint64_t kObservedRemoteIpcOverhead = 1ULL * 1024 * 1024;
  std::vector<uint64_t> accessible_spans(
      16, kLocalPayloadSize + kObservedRemoteIpcOverhead);
  accessible_spans[7] = kLocalPayloadSize;

  const auto validation = validate_mega_moe_buffer_accessible_spans(
      kLocalPayloadSize, accessible_spans);

  EXPECT_TRUE(validation.valid);
  EXPECT_EQ(validation.required_payload_size, kLocalPayloadSize);
}

TEST(MegaMoeCommResourceTest, RejectsZeroLocalHcclPayloadSize) {
  const auto validation = validate_mega_moe_buffer_accessible_spans(
      0, std::vector<uint64_t>(16, 4096));

  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.required_payload_size, 0);
}

TEST(MegaMoeCommResourceTest,
     RejectsRemoteIpcAccessibleSpanSmallerThanLocalPayload) {
  std::vector<uint64_t> accessible_spans(16, 4096);
  accessible_spans[7] = 2048;

  const auto validation = validate_mega_moe_buffer_accessible_spans(
      4096, accessible_spans);

  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.mismatched_rank, 7);
  EXPECT_EQ(validation.required_payload_size, 4096);
  EXPECT_EQ(validation.accessible_span, 2048);
}

TEST(MegaMoeCommResourceTest,
     AcceptsDifferentRemoteIpcSpansThatCoverLocalPayload) {
  std::vector<uint64_t> accessible_spans(16, 4096);
  accessible_spans[2] = 4097;
  accessible_spans[9] = 64ULL * 1024 * 1024;

  const auto validation = validate_mega_moe_buffer_accessible_spans(
      4096, accessible_spans);

  EXPECT_TRUE(validation.valid);
  EXPECT_EQ(validation.required_payload_size, 4096);
}

TEST(MegaMoeCommResourceTest, RejectsZeroRemoteIpcAccessibleSpan) {
  std::vector<uint64_t> accessible_spans(16, 4096);
  accessible_spans[11] = 0;

  const auto validation = validate_mega_moe_buffer_accessible_spans(
      4096, accessible_spans);

  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.mismatched_rank, 11);
  EXPECT_EQ(validation.required_payload_size, 4096);
  EXPECT_EQ(validation.accessible_span, 0);
}

std::shared_ptr<MegaMoeCommResource> fake_resource(int value) {
  auto owner = std::make_shared<int>(value);
  return std::shared_ptr<MegaMoeCommResource>(
      owner, reinterpret_cast<MegaMoeCommResource*>(owner.get()));
}

TEST(MegaMoeCommResourceTest, SharedSlotCreatesOnceForSameCompleteKey) {
  MegaMoeCommResourceSlot slot;
  int create_count = 0;
  const auto factory = [&](const MegaMoeCommSpec&) {
    return fake_resource(++create_count);
  };

  auto first = slot.acquire(valid_spec(), factory);
  auto second = slot.acquire(valid_spec(), factory);

  EXPECT_EQ(create_count, 1);
  EXPECT_EQ(first.get(), second.get());
}

TEST(MegaMoeCommResourceTest, SharedSlotCreatesNewResourceWhenKeyChanges) {
  MegaMoeCommResourceSlot slot;
  int create_count = 0;
  const auto factory = [&](const MegaMoeCommSpec&) {
    return fake_resource(++create_count);
  };
  auto spec = valid_spec();
  auto previous = slot.acquire(spec, factory);

  spec.hccl_comm = reinterpret_cast<HcclComm>(0x2);
  auto changed_comm = slot.acquire(spec, factory);
  EXPECT_NE(previous.get(), changed_comm.get());
  previous = changed_comm;

  spec.group_name = "other_ep_group";
  auto changed_group = slot.acquire(spec, factory);
  EXPECT_NE(previous.get(), changed_group.get());
  previous = changed_group;

  spec.ep_world_size = 32;
  auto changed_world = slot.acquire(spec, factory);
  EXPECT_NE(previous.get(), changed_world.get());
  previous = changed_world;

  spec.device_index = 1;
  auto changed_device = slot.acquire(spec, factory);
  EXPECT_NE(previous.get(), changed_device.get());
  previous = changed_device;

  spec.max_num_tokens_per_rank = 256;
  auto changed_envelope = slot.acquire(spec, factory);
  EXPECT_NE(previous.get(), changed_envelope.get());
  EXPECT_EQ(create_count, 6);
}

TEST(MegaMoeCommResourceTest, SharedSlotDoesNotCacheFactoryFailure) {
  MegaMoeCommResourceSlot slot;
  int create_count = 0;
  EXPECT_THROW(
      slot.acquire(valid_spec(), [&](const MegaMoeCommSpec&) {
        ++create_count;
        throw std::runtime_error("injected create failure");
        return std::shared_ptr<MegaMoeCommResource>();
      }),
      std::runtime_error);

  auto resource = slot.acquire(valid_spec(), [&](const MegaMoeCommSpec&) {
    return fake_resource(++create_count);
  });
  EXPECT_NE(resource, nullptr);
  EXPECT_EQ(create_count, 2);
}

TEST(MegaMoeCommResourceTest, ExplicitResetReleasesBeforeCommunicatorShutdown) {
  std::vector<std::string> events;
  MegaMoeCommResourceSlot slot;
  auto injected = std::shared_ptr<MegaMoeCommResource>(
      reinterpret_cast<MegaMoeCommResource*>(0x1),
      [&](MegaMoeCommResource*) { events.push_back("resource_released"); });
  auto layer_reference = slot.acquire(
      valid_spec(),
      [&](const MegaMoeCommSpec&) { return injected; });

  injected.reset();
  layer_reference.reset();
  EXPECT_TRUE(events.empty());

  // ProcessGroupImpl performs this explicit reset before shutdown/destroy of
  // its HCCL communicator.
  slot.reset();
  events.push_back("communicator_shutdown");

  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0], "resource_released");
  EXPECT_EQ(events[1], "communicator_shutdown");
}

TEST(MegaMoeCommResourceTest, LayerWeakReferenceDoesNotExtendSlotLifetime) {
  MegaMoeCommResourceSlot slot;
  auto acquired = slot.acquire(
      valid_spec(),
      [&](const MegaMoeCommSpec&) { return fake_resource(1); });
  std::weak_ptr<MegaMoeCommResource> layer_reference = acquired;

  EXPECT_EQ(acquired.use_count(), 2);
  acquired.reset();
  EXPECT_FALSE(layer_reference.expired());

  slot.reset_for_teardown();
  EXPECT_TRUE(layer_reference.expired());
}

TEST(MegaMoeCommResourceTest, ActiveForwardLockBlocksTeardownGuard) {
  MegaMoeCommResourceSlot slot;
  auto acquired = slot.acquire(
      valid_spec(),
      [&](const MegaMoeCommSpec&) { return fake_resource(1); });
  std::weak_ptr<MegaMoeCommResource> layer_reference = acquired;
  acquired.reset();
  auto active_forward_lock = layer_reference.lock();
  ASSERT_NE(active_forward_lock, nullptr);

  EXPECT_DEATH(slot.reset_for_teardown(),
               "active MegaMoe forward");

  active_forward_lock.reset();
  slot.reset_for_teardown();
  EXPECT_TRUE(layer_reference.expired());
}

TEST(MegaMoeCommResourceTest, EnforcesSingleOwnerLifecycle) {
  EXPECT_TRUE(std::is_destructible_v<MegaMoeCommResource>);
  EXPECT_FALSE(std::is_copy_constructible_v<MegaMoeCommResource>);
  EXPECT_FALSE(std::is_copy_assignable_v<MegaMoeCommResource>);
  EXPECT_FALSE(std::is_move_constructible_v<MegaMoeCommResource>);
  EXPECT_FALSE(std::is_move_assignable_v<MegaMoeCommResource>);
}

}  // namespace
}  // namespace xllm
