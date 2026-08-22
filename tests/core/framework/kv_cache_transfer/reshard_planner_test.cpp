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

#include "framework/kv_cache_transfer/reshard_planner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache_utils.h"

namespace xllm {

namespace {

constexpr uint64_t kResourceCount = 4;
constexpr uint64_t kTokenCount = 3;

WorkerCacheLayoutManifest make_head_manifest(int32_t tp_rank,
                                             int32_t tp_size,
                                             int64_t global_heads,
                                             uint64_t buffer_id,
                                             const std::string& addr) {
  WorkerCacheLayoutManifest manifest;
  manifest.incarnation_id = addr + "-incarnation";
  manifest.layout_generation = 1;
  manifest.fingerprint = "planner-test-model";
  manifest.backend = "cpu";
  manifest.layout_family = "token_head_dim";
  manifest.cluster_id = static_cast<uint64_t>(tp_rank + 1);
  manifest.addr = addr;
  manifest.listen_port = static_cast<uint16_t>(20000 + tp_rank);
  manifest.coordinates.tp_rank = tp_rank;
  manifest.coordinates.tp_size = tp_size;

  const bool replicated = global_heads < tp_size;
  const int64_t local_heads =
      replicated ? 1 : global_heads / static_cast<int64_t>(tp_size);
  const int32_t replica_count =
      replicated ? tp_size / static_cast<int32_t>(global_heads) : 1;
  const int64_t first_head =
      replicated ? tp_rank / replica_count : tp_rank * local_heads;

  CacheTensorManifest tensor;
  tensor.cache_namespace = CacheNamespace::MAIN;
  tensor.layer_id = 0;
  tensor.role = 0;
  tensor.group_id = 7;
  tensor.mooncake_buffer_id = static_cast<int64_t>(buffer_id);
  tensor.scalar_type = 0;
  tensor.element_bytes = 1;
  tensor.shape = {static_cast<int64_t>(kResourceCount),
                  static_cast<int64_t>(kTokenCount),
                  local_heads,
                  1};
  tensor.stride = {
      static_cast<int64_t>(kTokenCount * local_heads), local_heads, 1, 1};
  tensor.contiguous = true;
  tensor.resource_count = kResourceCount;
  tensor.resource_stride_bytes =
      kTokenCount * static_cast<uint64_t>(local_heads);
  tensor.buffer_bytes = tensor.resource_count * tensor.resource_stride_bytes;
  tensor.block_token_capacity = kTokenCount;
  tensor.shard.kind =
      replicated ? LogicalShardKind::REPLICATED : LogicalShardKind::SHARDED;
  tensor.shard.resource_scope = CacheResourceScope::BLOCK;
  tensor.shard.spans.reserve(static_cast<size_t>(local_heads));
  for (int64_t local_head = 0; local_head < local_heads; ++local_head) {
    const int64_t global_head = first_head + local_head;
    LogicalSpan span;
    span.logical_tensor = "key";
    span.logical_offset_bytes = static_cast<uint64_t>(global_head);
    span.physical_offset_bytes = static_cast<uint64_t>(local_head);
    span.bytes_per_region = 1;
    span.repeat_count = kTokenCount;
    span.logical_stride_bytes = static_cast<uint64_t>(global_heads);
    span.physical_stride_bytes = static_cast<uint64_t>(local_heads);
    span.owner_tp_rank = replicated
                             ? static_cast<int32_t>(global_head * replica_count)
                             : tp_rank;
    tensor.shard.spans.emplace_back(std::move(span));
  }
  manifest.tensors.emplace_back(std::move(tensor));
  return manifest;
}

WorkerCacheLayoutManifest make_ssm_manifest(int32_t tp_rank,
                                            int32_t tp_size,
                                            int64_t global_heads,
                                            uint64_t buffer_id,
                                            const std::string& addr) {
  WorkerCacheLayoutManifest manifest =
      make_head_manifest(tp_rank, tp_size, global_heads, buffer_id, addr);
  const int64_t local_heads = global_heads / tp_size;
  const int64_t first_head = tp_rank * local_heads;
  constexpr uint64_t kCheckpointStride = 3;

  CacheTensorManifest tensor;
  tensor.cache_namespace = CacheNamespace::MAIN;
  tensor.layer_id = 0;
  tensor.role = static_cast<int32_t>(KVCacheTensorRole::SSM);
  tensor.group_id = cache_group_id(BlockType::LINEAR);
  tensor.mooncake_buffer_id = static_cast<int64_t>(buffer_id);
  tensor.scalar_type = 0;
  tensor.element_bytes = 1;
  tensor.shape = {static_cast<int64_t>(kResourceCount * kCheckpointStride),
                  local_heads,
                  1,
                  1};
  tensor.stride = {local_heads, 1, 1, 1};
  tensor.contiguous = true;
  tensor.resource_count = kResourceCount;
  tensor.physical_rows_per_resource = kCheckpointStride;
  tensor.resource_stride_bytes =
      kCheckpointStride * static_cast<uint64_t>(local_heads);
  tensor.buffer_bytes = tensor.resource_count * tensor.resource_stride_bytes;
  tensor.shard.kind = LogicalShardKind::SHARDED;
  tensor.shard.resource_scope = CacheResourceScope::SEQUENCE;
  for (int64_t local_head = 0; local_head < local_heads; ++local_head) {
    LogicalSpan span;
    span.logical_tensor = "ssm";
    span.logical_offset_bytes = static_cast<uint64_t>(first_head + local_head);
    span.physical_offset_bytes = static_cast<uint64_t>(local_head);
    span.bytes_per_region = 1;
    span.repeat_count = kCheckpointStride;
    span.logical_stride_bytes = static_cast<uint64_t>(global_heads);
    span.physical_stride_bytes = static_cast<uint64_t>(local_heads);
    span.owner_tp_rank = tp_rank;
    tensor.shard.spans.emplace_back(std::move(span));
  }
  manifest.tensors = {std::move(tensor)};
  return manifest;
}

WorkerCacheLayoutManifest make_large_head_manifest(int32_t tp_rank,
                                                   int32_t tp_size,
                                                   int64_t global_heads,
                                                   int64_t layer_count,
                                                   uint64_t token_count,
                                                   uint64_t first_buffer_id,
                                                   const std::string& addr) {
  WorkerCacheLayoutManifest manifest =
      make_head_manifest(tp_rank, tp_size, global_heads, first_buffer_id, addr);
  manifest.tensors.clear();
  const int64_t local_heads = global_heads / tp_size;
  const int64_t first_head = tp_rank * local_heads;
  for (int64_t layer_id = 0; layer_id < layer_count; ++layer_id) {
    for (int32_t role = 0; role < 2; ++role) {
      CacheTensorManifest tensor;
      tensor.cache_namespace = CacheNamespace::MAIN;
      tensor.layer_id = layer_id;
      tensor.role = role;
      tensor.group_id = 7;
      tensor.mooncake_buffer_id = static_cast<int64_t>(
          first_buffer_id + static_cast<uint64_t>(layer_id * 2 + role));
      tensor.scalar_type = 0;
      tensor.element_bytes = 1;
      tensor.shape = {static_cast<int64_t>(kResourceCount),
                      static_cast<int64_t>(token_count),
                      local_heads,
                      1};
      tensor.stride = {
          static_cast<int64_t>(token_count) * local_heads, local_heads, 1, 1};
      tensor.contiguous = true;
      tensor.resource_count = kResourceCount;
      tensor.resource_stride_bytes =
          token_count * static_cast<uint64_t>(local_heads);
      tensor.buffer_bytes =
          tensor.resource_count * tensor.resource_stride_bytes;
      tensor.block_token_capacity = token_count;
      tensor.shard.kind = LogicalShardKind::SHARDED;
      tensor.shard.resource_scope = CacheResourceScope::BLOCK;
      for (int64_t local_head = 0; local_head < local_heads; ++local_head) {
        LogicalSpan span;
        span.logical_tensor = role == 0 ? "key" : "value";
        span.logical_offset_bytes =
            static_cast<uint64_t>(first_head + local_head);
        span.physical_offset_bytes = static_cast<uint64_t>(local_head);
        span.bytes_per_region = 1;
        span.repeat_count = token_count;
        span.logical_stride_bytes = static_cast<uint64_t>(global_heads);
        span.physical_stride_bytes = static_cast<uint64_t>(local_heads);
        span.owner_tp_rank = tp_rank;
        tensor.shard.spans.emplace_back(std::move(span));
      }
      manifest.tensors.emplace_back(std::move(tensor));
    }
  }
  return manifest;
}

std::vector<WorkerCacheLayoutManifest> make_instance(
    int32_t tp_size,
    int64_t global_heads,
    uint64_t first_buffer_id,
    const std::string& prefix) {
  std::vector<WorkerCacheLayoutManifest> manifests;
  manifests.reserve(static_cast<size_t>(tp_size));
  for (int32_t tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
    manifests.emplace_back(
        make_head_manifest(tp_rank,
                           tp_size,
                           global_heads,
                           first_buffer_id + tp_rank,
                           prefix + std::to_string(tp_rank)));
  }
  return manifests;
}

std::vector<WorkerCacheLayoutManifest> make_cp_instance(
    int32_t cp_size,
    int32_t kv_split_size,
    int32_t tp_size,
    int64_t global_heads,
    uint64_t first_buffer_id,
    const std::string& prefix) {
  std::vector<WorkerCacheLayoutManifest> manifests;
  manifests.reserve(static_cast<size_t>(cp_size * tp_size));
  for (int32_t cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
    const int32_t kv_split_rank = cp_rank * kv_split_size / cp_size;
    for (int32_t tp_rank = 0; tp_rank < tp_size; ++tp_rank) {
      const int32_t worker_rank = cp_rank * tp_size + tp_rank;
      WorkerCacheLayoutManifest manifest = make_head_manifest(
          tp_rank,
          tp_size,
          global_heads,
          first_buffer_id + static_cast<uint64_t>(worker_rank),
          prefix + std::to_string(worker_rank));
      manifest.cluster_id = static_cast<uint64_t>(worker_rank + 1);
      manifest.listen_port =
          static_cast<uint16_t>(20000 + static_cast<uint16_t>(worker_rank));
      manifest.coordinates.cp_rank = cp_rank;
      manifest.coordinates.cp_size = cp_size;
      manifest.coordinates.kv_split_rank = kv_split_rank;
      manifest.coordinates.kv_split_size = kv_split_size;
      manifests.emplace_back(std::move(manifest));
    }
  }
  return manifests;
}

void append_tp1_spec_tensor(WorkerCacheLayoutManifest* manifest,
                            int64_t global_heads,
                            uint64_t buffer_id) {
  ASSERT_NE(manifest, nullptr);
  WorkerCacheLayoutManifest draft = make_head_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/1,
      global_heads,
      buffer_id,
      manifest->addr + "-draft");
  CacheTensorManifest spec_tensor = std::move(draft.tensors[0]);
  spec_tensor.cache_namespace = CacheNamespace::SPEC_DRAFT;
  manifest->tensors.emplace_back(std::move(spec_tensor));
  manifest->fingerprint.append("|spec:planner-test-model");
}

void fill_source(const WorkerCacheLayoutManifest& manifest,
                 std::vector<uint8_t>* bytes) {
  const CacheTensorManifest& tensor = manifest.tensors[0];
  bytes->assign(static_cast<size_t>(tensor.buffer_bytes), 0);
  for (uint64_t resource = 0; resource < tensor.resource_count; ++resource) {
    for (const LogicalSpan& span : tensor.shard.spans) {
      for (uint64_t repeat = 0; repeat < span.repeat_count; ++repeat) {
        const uint64_t logical =
            span.logical_offset_bytes + repeat * span.logical_stride_bytes;
        const uint64_t physical = resource * tensor.resource_stride_bytes +
                                  span.physical_offset_bytes +
                                  repeat * span.physical_stride_bytes;
        (*bytes)[static_cast<size_t>(physical)] =
            static_cast<uint8_t>(resource * 31 + logical + 1);
      }
    }
  }
}

void expect_destination_resource(const WorkerCacheLayoutManifest& manifest,
                                 const std::vector<uint8_t>& bytes,
                                 uint64_t destination_resource,
                                 uint64_t source_resource) {
  const CacheTensorManifest& tensor = manifest.tensors[0];
  for (const LogicalSpan& span : tensor.shard.spans) {
    for (uint64_t repeat = 0; repeat < span.repeat_count; ++repeat) {
      const uint64_t logical =
          span.logical_offset_bytes + repeat * span.logical_stride_bytes;
      const uint64_t physical =
          destination_resource * tensor.resource_stride_bytes +
          span.physical_offset_bytes + repeat * span.physical_stride_bytes;
      EXPECT_EQ(bytes[static_cast<size_t>(physical)],
                static_cast<uint8_t>(source_resource * 31 + logical + 1));
    }
  }
}

void run_copy_case(int32_t source_tp,
                   int32_t destination_tp,
                   int64_t global_heads) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(source_tp, global_heads, /*first_buffer_id=*/3, "source");
  std::vector<WorkerCacheLayoutManifest> destinations = make_instance(
      destination_tp, global_heads, /*first_buffer_id=*/17, "destination");
  ReshardPlanner planner;

  std::vector<std::vector<uint8_t>> source_bytes(sources.size());
  for (size_t source_rank = 0; source_rank < sources.size(); ++source_rank) {
    fill_source(sources[source_rank], &source_bytes[source_rank]);
  }
  std::vector<std::vector<uint8_t>> destination_bytes(destinations.size());
  for (size_t destination_rank = 0; destination_rank < destinations.size();
       ++destination_rank) {
    const WorkerCacheLayoutManifest& destination =
        destinations[destination_rank];
    ASSERT_TRUE(
        planner.validate_destination_coverage(sources, destination).ok());
    destination_bytes[destination_rank].assign(
        static_cast<size_t>(destination.tensors[0].buffer_bytes), 0);

    for (size_t source_rank = 0; source_rank < sources.size(); ++source_rank) {
      ReshardPlanTemplate plan;
      ASSERT_TRUE(
          planner.build_outgoing_plan(sources[source_rank], destination, &plan)
              .ok());
      KVTransferMapping mapping;
      mapping.group_id = 7;
      mapping.local_ids = {1, 3};
      mapping.remote_ids = {2, 0};
      RequestRegionBinder binder;
      std::vector<ByteRegion> regions;
      ASSERT_TRUE(binder
                      .bind(plan,
                            {mapping},
                            CacheNamespace::MAIN,
                            /*layer_id=*/0,
                            &regions)
                      .ok());
      for (const ByteRegion& region : regions) {
        EXPECT_EQ(region.local_buffer_id,
                  static_cast<uint64_t>(
                      sources[source_rank].tensors[0].mooncake_buffer_id));
        EXPECT_EQ(
            region.remote_buffer_id,
            static_cast<uint64_t>(destination.tensors[0].mooncake_buffer_id));
        std::memcpy(
            destination_bytes[destination_rank].data() + region.remote_offset,
            source_bytes[source_rank].data() + region.local_offset,
            static_cast<size_t>(region.length));
      }
    }
    expect_destination_resource(destination,
                                destination_bytes[destination_rank],
                                /*destination_resource=*/2,
                                /*source_resource=*/1);
    expect_destination_resource(destination,
                                destination_bytes[destination_rank],
                                /*destination_resource=*/0,
                                /*source_resource=*/3);
  }
}

bool is_legal_head_tp(int64_t global_heads, int32_t tp_size) {
  return global_heads >= tp_size ? global_heads % tp_size == 0
                                 : tp_size % global_heads == 0;
}

TEST(ReshardPlannerTest, EnumeratesLegalHeadAndTpPairs) {
  for (int64_t global_heads = 1; global_heads <= 8; ++global_heads) {
    for (int32_t source_tp = 1; source_tp <= 8; ++source_tp) {
      if (!is_legal_head_tp(global_heads, source_tp)) {
        continue;
      }
      for (int32_t destination_tp = 1; destination_tp <= 8; ++destination_tp) {
        if (!is_legal_head_tp(global_heads, destination_tp)) {
          continue;
        }
        SCOPED_TRACE("heads=" + std::to_string(global_heads) +
                     ", source_tp=" + std::to_string(source_tp) +
                     ", destination_tp=" + std::to_string(destination_tp));
        run_copy_case(source_tp, destination_tp, global_heads);
      }
    }
  }
}

TEST(ReshardPlannerTest, NonIntegerTpTwoToThree) {
  run_copy_case(/*source_tp=*/2, /*destination_tp=*/3, /*global_heads=*/6);
}

TEST(ReshardPlannerTest, ExpandsTpOneToFour) {
  run_copy_case(/*source_tp=*/1, /*destination_tp=*/4, /*global_heads=*/4);
}

TEST(ReshardPlannerTest, ShrinksTpFourToOne) {
  run_copy_case(/*source_tp=*/4, /*destination_tp=*/1, /*global_heads=*/8);
}

TEST(ReshardPlannerTest, HomogeneousTpUsesTheSamePlanner) {
  run_copy_case(/*source_tp=*/2, /*destination_tp=*/2, /*global_heads=*/6);
}

TEST(ReshardPlannerTest, NonIntegerTpThreeToTwo) {
  run_copy_case(/*source_tp=*/3, /*destination_tp=*/2, /*global_heads=*/6);
}

TEST(ReshardPlannerTest, ShrinkTpFourToTwo) {
  run_copy_case(/*source_tp=*/4, /*destination_tp=*/2, /*global_heads=*/8);
}

TEST(ReshardPlannerTest, CheckpointedSsmBindsLogicalSlotsAcrossTp) {
  std::vector<WorkerCacheLayoutManifest> sources;
  for (int32_t rank = 0; rank < 2; ++rank) {
    sources.emplace_back(make_ssm_manifest(
        rank, /*tp_size=*/2, /*global_heads=*/4, 3 + rank, "source"));
  }
  const WorkerCacheLayoutManifest destination = make_ssm_manifest(
      /*tp_rank=*/3, /*tp_size=*/4, /*global_heads=*/4, 17, "destination");
  ReshardPlanner planner;
  ASSERT_TRUE(planner.validate_destination_coverage(sources, destination).ok());

  std::vector<std::vector<uint8_t>> source_bytes(sources.size());
  for (size_t rank = 0; rank < sources.size(); ++rank) {
    fill_source(sources[rank], &source_bytes[rank]);
  }
  std::vector<uint8_t> destination_bytes(
      static_cast<size_t>(destination.tensors[0].buffer_bytes), 0);
  for (size_t rank = 0; rank < sources.size(); ++rank) {
    ReshardPlanTemplate plan;
    ASSERT_TRUE(
        planner.build_outgoing_plan(sources[rank], destination, &plan).ok());
    KVTransferMapping mapping;
    mapping.group_id = cache_group_id(BlockType::LINEAR);
    mapping.local_ids = {1};
    mapping.remote_ids = {1};
    std::vector<ByteRegion> regions;
    ASSERT_TRUE(RequestRegionBinder()
                    .bind(plan,
                          {mapping},
                          CacheNamespace::MAIN,
                          /*layer_id=*/0,
                          &regions)
                    .ok());
    for (const ByteRegion& region : regions) {
      EXPECT_GE(region.local_offset,
                sources[rank].tensors[0].resource_stride_bytes);
      EXPECT_GE(region.remote_offset,
                destination.tensors[0].resource_stride_bytes);
      std::memcpy(destination_bytes.data() + region.remote_offset,
                  source_bytes[rank].data() + region.local_offset,
                  static_cast<size_t>(region.length));
    }
  }
  expect_destination_resource(destination,
                              destination_bytes,
                              /*destination_resource=*/1,
                              /*source_resource=*/1);
}

TEST(ReshardPlannerTest, HandlesProductionScaleNonMlaLayout) {
  constexpr int64_t kLayerCount = 80;
  constexpr uint64_t kBlockTokenCount = 128;
  const std::vector<WorkerCacheLayoutManifest> sources = {
      make_large_head_manifest(/*tp_rank=*/0,
                               /*tp_size=*/1,
                               /*global_heads=*/8,
                               kLayerCount,
                               kBlockTokenCount,
                               /*first_buffer_id=*/3,
                               "source")};
  const WorkerCacheLayoutManifest destination = make_large_head_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/2,
      /*global_heads=*/8,
      kLayerCount,
      kBlockTokenCount,
      /*first_buffer_id=*/1000,
      "destination");
  ReshardPlanner planner;

  EXPECT_TRUE(planner.validate_destination_coverage(sources, destination).ok());
  ReshardPlanTemplate plan;
  EXPECT_TRUE(planner.build_outgoing_plan(sources[0], destination, &plan).ok());
  EXPECT_FALSE(plan.regions.empty());
}

TEST(ReshardPlannerTest, MqaReplicationUsesOneStaticSourceWriter) {
  const std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/2, /*global_heads=*/1, 3, "source");
  const std::vector<WorkerCacheLayoutManifest> destinations =
      make_instance(/*tp_size=*/4, /*global_heads=*/1, 17, "destination");
  ReshardPlanner planner;
  for (const WorkerCacheLayoutManifest& destination : destinations) {
    ASSERT_TRUE(
        planner.validate_destination_coverage(sources, destination).ok());
    ReshardPlanTemplate owner_plan;
    ReshardPlanTemplate replica_plan;
    ASSERT_TRUE(
        planner.build_outgoing_plan(sources[0], destination, &owner_plan).ok());
    ASSERT_TRUE(
        planner.build_outgoing_plan(sources[1], destination, &replica_plan)
            .ok());
    EXPECT_FALSE(owner_plan.regions.empty());
    EXPECT_TRUE(replica_plan.regions.empty());
  }
}

TEST(ReshardPlannerTest, GqaReplicationUsesContiguousSourceWriters) {
  constexpr int32_t kTpSize = 4;
  constexpr int64_t kGlobalHeads = 2;
  constexpr int32_t kReplicaCount =
      static_cast<int32_t>(kTpSize / kGlobalHeads);
  const std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(kTpSize, kGlobalHeads, 3, "source");
  const std::vector<WorkerCacheLayoutManifest> destinations =
      make_instance(kTpSize, kGlobalHeads, 17, "destination");
  ReshardPlanner planner;

  for (const WorkerCacheLayoutManifest& destination : destinations) {
    const int32_t expected_source_rank =
        destination.coordinates.tp_rank / kReplicaCount * kReplicaCount;
    ASSERT_TRUE(
        planner.validate_destination_coverage(sources, destination).ok());
    for (const WorkerCacheLayoutManifest& source : sources) {
      ReshardPlanTemplate plan;
      ASSERT_TRUE(planner.build_outgoing_plan(source, destination, &plan).ok());
      EXPECT_EQ(plan.regions.empty(),
                source.coordinates.tp_rank != expected_source_rank);
    }
  }
}

TEST(ReshardPlannerTest, RejectsMultipleStaticWriters) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/2, /*global_heads=*/1, 3, "source");
  sources[1].tensors[0].shard.spans[0].owner_tp_rank = 1;
  const WorkerCacheLayoutManifest destination =
      make_head_manifest(/*tp_rank=*/0,
                         /*tp_size=*/1,
                         /*global_heads=*/1,
                         /*buffer_id=*/17,
                         "destination");
  ReshardPlanner planner;
  const Status status =
      planner.validate_destination_coverage(sources, destination);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("multiple writers"), std::string::npos);
}

TEST(ReshardPlannerTest, RejectsDestinationWithoutSourceWriter) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/1, /*global_heads=*/1, 3, "source");
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 17, "destination");
  destination.tensors[0].shard.spans[0].logical_tensor = "missing_key";

  const Status status =
      ReshardPlanner().validate_destination_coverage(sources, destination);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("no source writer"), std::string::npos);
}

TEST(ReshardPlannerTest, RejectsIncompatibleMatchingTensor) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/1, /*global_heads=*/1, 3, "source");
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 17, "destination");
  destination.tensors[0].scalar_type = 1;

  const Status status =
      ReshardPlanner().validate_destination_coverage(sources, destination);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("incompatible"), std::string::npos);
}

TEST(ReshardPlannerTest, RejectsCpPartitionMismatch) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/1, /*global_heads=*/1, 3, "source");
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 17, "destination");
  destination.coordinates.cp_size = 2;
  const Status status =
      ReshardPlanner().validate_destination_coverage(sources, destination);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("CP/KV-split"), std::string::npos);
}

TEST(ReshardPlannerTest, SelectsMatchingCpPartitionFromCompleteSourceSet) {
  WorkerCacheLayoutManifest cp_zero = make_head_manifest(0, 1, 1, 3, "source0");
  cp_zero.coordinates.cp_size = 2;
  cp_zero.coordinates.kv_split_size = 2;
  WorkerCacheLayoutManifest cp_one = make_head_manifest(0, 1, 1, 4, "source1");
  cp_one.coordinates.cp_size = 2;
  cp_one.coordinates.cp_rank = 1;
  cp_one.coordinates.kv_split_size = 2;
  cp_one.coordinates.kv_split_rank = 1;
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 17, "destination");
  destination.coordinates.cp_size = 2;
  destination.coordinates.cp_rank = 1;
  destination.coordinates.kv_split_size = 2;
  destination.coordinates.kv_split_rank = 1;

  const Status status = ReshardPlanner().validate_destination_coverage(
      {cp_zero, cp_one}, destination);

  EXPECT_TRUE(status.ok()) << status.message();
}

TEST(ReshardPlannerTest, CollapsesPrefillCpAndKvSplitIntoDecodePartition) {
  std::vector<WorkerCacheLayoutManifest> sources = make_cp_instance(
      /*cp_size=*/2,
      /*kv_split_size=*/2,
      /*tp_size=*/2,
      /*global_heads=*/4,
      /*first_buffer_id=*/3,
      "source");
  const WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 4, 17, "destination");
  ReshardPlanner planner;

  ASSERT_TRUE(planner.validate_destination_coverage(sources, destination).ok());
  for (const WorkerCacheLayoutManifest& source : sources) {
    EXPECT_TRUE(planner.source_participates(source, destination));
  }

  std::vector<std::vector<uint8_t>> source_bytes(sources.size());
  std::vector<uint8_t> destination_bytes(
      static_cast<size_t>(destination.tensors[0].buffer_bytes), 0);
  for (size_t source_rank = 0; source_rank < sources.size(); ++source_rank) {
    const WorkerCacheLayoutManifest& source = sources[source_rank];
    fill_source(source, &source_bytes[source_rank]);
    ReshardPlanTemplate plan;
    ASSERT_TRUE(planner.build_outgoing_plan(source, destination, &plan).ok());

    KVTransferMapping mapping;
    mapping.group_id = 7;
    mapping.local_ids = {1};
    mapping.remote_ids = {
        static_cast<uint64_t>(2 + source.coordinates.kv_split_rank)};
    std::vector<ByteRegion> regions;
    ASSERT_TRUE(RequestRegionBinder()
                    .bind(plan,
                          {mapping},
                          CacheNamespace::MAIN,
                          /*layer_id=*/0,
                          &regions)
                    .ok());
    for (const ByteRegion& region : regions) {
      std::memcpy(destination_bytes.data() + region.remote_offset,
                  source_bytes[source_rank].data() + region.local_offset,
                  static_cast<size_t>(region.length));
    }
  }

  expect_destination_resource(destination,
                              destination_bytes,
                              /*destination_resource=*/2,
                              /*source_resource=*/1);
  expect_destination_resource(destination,
                              destination_bytes,
                              /*destination_resource=*/3,
                              /*source_resource=*/1);
}

TEST(ReshardPlannerTest, SupportsTp1SpecDraftBesideShardedMainCache) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/2, /*global_heads=*/4, 3, "source");
  std::vector<WorkerCacheLayoutManifest> destinations =
      make_instance(/*tp_size=*/2, /*global_heads=*/4, 17, "destination");
  for (WorkerCacheLayoutManifest& source : sources) {
    append_tp1_spec_tensor(&source, /*global_heads=*/4, /*buffer_id=*/30);
  }
  for (WorkerCacheLayoutManifest& destination : destinations) {
    append_tp1_spec_tensor(&destination, /*global_heads=*/4, /*buffer_id=*/40);
  }
  ReshardPlanner planner;

  ASSERT_TRUE(
      planner.validate_destination_coverage(sources, destinations[1]).ok());
  ReshardPlanTemplate source_zero_plan;
  ReshardPlanTemplate source_one_plan;
  ASSERT_TRUE(
      planner
          .build_outgoing_plan(sources[0], destinations[1], &source_zero_plan)
          .ok());
  ASSERT_TRUE(
      planner.build_outgoing_plan(sources[1], destinations[1], &source_one_plan)
          .ok());
  EXPECT_TRUE(std::any_of(source_zero_plan.regions.begin(),
                          source_zero_plan.regions.end(),
                          [](const StridedRegionTemplate& region) {
                            return region.cache_namespace ==
                                   CacheNamespace::SPEC_DRAFT;
                          }));
  EXPECT_FALSE(std::any_of(source_one_plan.regions.begin(),
                           source_one_plan.regions.end(),
                           [](const StridedRegionTemplate& region) {
                             return region.cache_namespace ==
                                    CacheNamespace::SPEC_DRAFT;
                           }));
  EXPECT_TRUE(std::any_of(source_one_plan.regions.begin(),
                          source_one_plan.regions.end(),
                          [](const StridedRegionTemplate& region) {
                            return region.cache_namespace ==
                                   CacheNamespace::MAIN;
                          }));
}

TEST(ReshardPlannerTest, RejectsIncompleteSourceWorkerSet) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/2, /*global_heads=*/2, 3, "source");
  sources.pop_back();
  const WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 2, 17, "destination");

  const Status status =
      ReshardPlanner().validate_destination_coverage(sources, destination);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("incomplete"), std::string::npos);
}

TEST(ReshardPlannerTest, RejectsDuplicateSourceRank) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/2, /*global_heads=*/2, 3, "source");
  sources[1].coordinates.tp_rank = 0;
  const WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 2, 17, "destination");

  const Status status =
      ReshardPlanner().validate_destination_coverage(sources, destination);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("duplicate rank"), std::string::npos);
}

TEST(ReshardPlannerTest, RejectsSourceGenerationMismatch) {
  std::vector<WorkerCacheLayoutManifest> sources =
      make_instance(/*tp_size=*/2, /*global_heads=*/2, 3, "source");
  sources[1].layout_generation = 2;
  const WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 2, 17, "destination");

  const Status status =
      ReshardPlanner().validate_destination_coverage(sources, destination);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("disagree"), std::string::npos);
}

TEST(ReshardPlannerTest, KeepsCacheNamespacesIsolated) {
  WorkerCacheLayoutManifest source = make_head_manifest(0, 1, 1, 3, "source");
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 17, "destination");
  source.tensors[0].cache_namespace = CacheNamespace::SPEC_DRAFT;
  destination.tensors[0].cache_namespace = CacheNamespace::SPEC_DRAFT;
  ReshardPlanTemplate plan;
  ASSERT_TRUE(
      ReshardPlanner().build_outgoing_plan(source, destination, &plan).ok());
  KVTransferMapping mapping;
  mapping.group_id = 7;
  mapping.local_ids = {1};
  mapping.remote_ids = {2};
  RequestRegionBinder binder;
  std::vector<ByteRegion> regions;

  ASSERT_TRUE(binder
                  .bind(plan,
                        {mapping},
                        CacheNamespace::MAIN,
                        /*layer_id=*/0,
                        &regions)
                  .ok());
  EXPECT_TRUE(regions.empty());
  ASSERT_TRUE(binder
                  .bind(plan,
                        {mapping},
                        CacheNamespace::SPEC_DRAFT,
                        /*layer_id=*/0,
                        &regions)
                  .ok());
  EXPECT_FALSE(regions.empty());
}

TEST(ReshardPlannerTest, SupportsSequenceScopedResources) {
  WorkerCacheLayoutManifest source = make_head_manifest(0, 1, 1, 3, "source");
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 17, "destination");
  source.tensors[0].shard.resource_scope = CacheResourceScope::SEQUENCE;
  destination.tensors[0].shard.resource_scope = CacheResourceScope::SEQUENCE;
  ReshardPlanTemplate plan;

  ASSERT_TRUE(
      ReshardPlanner().build_outgoing_plan(source, destination, &plan).ok());
  KVTransferMapping mapping;
  mapping.group_id = 7;
  mapping.local_ids = {3};
  mapping.remote_ids = {1};
  std::vector<ByteRegion> regions;
  ASSERT_TRUE(RequestRegionBinder()
                  .bind(plan,
                        {mapping},
                        CacheNamespace::MAIN,
                        /*layer_id=*/0,
                        &regions)
                  .ok());
  ASSERT_EQ(regions.size(), 1U);
  EXPECT_EQ(regions[0].local_offset, 3 * kTokenCount);
  EXPECT_EQ(regions[0].remote_offset, kTokenCount);
  EXPECT_EQ(regions[0].length, kTokenCount);
}

TEST(ReshardPlannerTest, SupportsCompositeDescriptors) {
  WorkerCacheLayoutManifest source = make_head_manifest(0, 1, 1, 3, "source");
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 17, "destination");
  for (WorkerCacheLayoutManifest* manifest : {&source, &destination}) {
    LogicalShardDescriptor& shard = manifest->tensors[0].shard;
    shard.kind = LogicalShardKind::COMPOSITE;
    shard.spans.clear();
    LogicalSpan prefix;
    prefix.logical_tensor = "composite_prefix";
    prefix.bytes_per_region = 1;
    shard.spans.emplace_back(prefix);
    LogicalSpan suffix;
    suffix.logical_tensor = "composite_suffix";
    suffix.physical_offset_bytes = 1;
    suffix.bytes_per_region = kTokenCount - 1;
    shard.spans.emplace_back(suffix);
  }
  ReshardPlanTemplate plan;
  ASSERT_TRUE(
      ReshardPlanner().build_outgoing_plan(source, destination, &plan).ok());
  KVTransferMapping mapping;
  mapping.group_id = 7;
  mapping.local_ids = {0};
  mapping.remote_ids = {0};
  std::vector<ByteRegion> regions;

  ASSERT_TRUE(RequestRegionBinder()
                  .bind(plan,
                        {mapping},
                        CacheNamespace::MAIN,
                        /*layer_id=*/0,
                        &regions)
                  .ok());
  ASSERT_EQ(regions.size(), 2U);
  EXPECT_EQ(regions[0].length + regions[1].length, kTokenCount);
}

TEST(RequestRegionBinderTest, BindsExplicitXTensorResourceOffsets) {
  WorkerCacheLayoutManifest source = make_head_manifest(0, 1, 1, 0, "source");
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 0, "destination");
  source.tensors[0].explicit_resource_offsets = true;
  source.tensors[0].buffer_bytes = 1024;
  destination.tensors[0].explicit_resource_offsets = true;
  destination.tensors[0].buffer_bytes = 1024;
  ReshardPlanTemplate plan;
  ASSERT_TRUE(
      ReshardPlanner().build_outgoing_plan(source, destination, &plan).ok());
  ExplicitResourceMapping mapping;
  mapping.group_id = 7;
  mapping.role = source.tensors[0].role;
  mapping.local_ids = {1};
  mapping.remote_ids = {2};
  mapping.local_offsets = {100};
  mapping.remote_offsets = {500};
  std::vector<ByteRegion> regions;

  ASSERT_TRUE(RequestRegionBinder()
                  .bind_explicit(plan,
                                 {mapping},
                                 CacheNamespace::MAIN,
                                 /*layer_id=*/0,
                                 &regions)
                  .ok());
  ASSERT_EQ(regions.size(), 1U);
  EXPECT_EQ(regions[0].local_buffer_id, 0U);
  EXPECT_EQ(regions[0].remote_buffer_id, 0U);
  EXPECT_EQ(regions[0].local_offset, 100U);
  EXPECT_EQ(regions[0].remote_offset, 500U);
  EXPECT_EQ(regions[0].length, kTokenCount);
}

TEST(RequestRegionBinderTest, RejectsExplicitOffsetOutsideManifestBuffer) {
  WorkerCacheLayoutManifest source = make_head_manifest(0, 1, 1, 0, "source");
  WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 0, "destination");
  source.tensors[0].explicit_resource_offsets = true;
  source.tensors[0].buffer_bytes = 1024;
  destination.tensors[0].explicit_resource_offsets = true;
  destination.tensors[0].buffer_bytes = 1024;
  ReshardPlanTemplate plan;
  ASSERT_TRUE(
      ReshardPlanner().build_outgoing_plan(source, destination, &plan).ok());
  ExplicitResourceMapping mapping;
  mapping.group_id = 7;
  mapping.role = source.tensors[0].role;
  mapping.local_ids = {0};
  mapping.remote_ids = {0};
  mapping.local_offsets = {1023};
  mapping.remote_offsets = {0};
  std::vector<ByteRegion> regions;

  const Status status = RequestRegionBinder().bind_explicit(
      plan, {mapping}, CacheNamespace::MAIN, /*layer_id=*/0, &regions);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(regions.empty());
}

TEST(CacheLayoutTest, ManifestProtoRoundTripPreservesLayoutIdentityAndSpans) {
  WorkerCacheLayoutManifest original = make_head_manifest(0, 1, 1, 3, "source");
  original.incarnation_id = "restart-safe-incarnation";
  original.layout_generation = 9;
  original.tensors[0].storage_offset_bytes = 7;
  original.tensors[0].buffer_bytes += 7;
  original.tensors[0].explicit_resource_offsets = true;
  proto::WorkerCacheLayoutManifest wire;
  cache_layout_to_proto(original, &wire);
  WorkerCacheLayoutManifest decoded;

  const Status status = cache_layout_from_proto(wire, &decoded);

  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(decoded.schema_version, original.schema_version);
  EXPECT_EQ(decoded.incarnation_id, original.incarnation_id);
  EXPECT_EQ(decoded.layout_generation, original.layout_generation);
  ASSERT_EQ(decoded.tensors.size(), 1U);
  const CacheTensorManifest& tensor = decoded.tensors[0];
  EXPECT_EQ(tensor.storage_offset_bytes, 7U);
  EXPECT_TRUE(tensor.contiguous);
  EXPECT_TRUE(tensor.explicit_resource_offsets);
  EXPECT_EQ(tensor.physical_rows_per_resource, 1U);
  EXPECT_EQ(tensor.shape, original.tensors[0].shape);
  EXPECT_EQ(tensor.stride, original.tensors[0].stride);
  ASSERT_EQ(tensor.shard.spans.size(), 1U);
  EXPECT_EQ(tensor.shard.spans[0].logical_tensor, "key");
  EXPECT_EQ(tensor.shard.spans[0].repeat_count, kTokenCount);
}

TEST(CacheLayoutTest, ManifestProtoRoundTripPreservesSsmResourceGeometry) {
  const WorkerCacheLayoutManifest original = make_ssm_manifest(
      /*tp_rank=*/0, /*tp_size=*/1, /*global_heads=*/4, 3, "source");
  proto::WorkerCacheLayoutManifest wire;
  cache_layout_to_proto(original, &wire);
  WorkerCacheLayoutManifest decoded;

  const Status status = cache_layout_from_proto(wire, &decoded);

  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_EQ(decoded.tensors.size(), 1U);
  EXPECT_EQ(decoded.tensors[0].resource_count, kResourceCount);
  EXPECT_EQ(decoded.tensors[0].physical_rows_per_resource, 3U);
  EXPECT_EQ(decoded.tensors[0].resource_stride_bytes,
            original.tensors[0].resource_stride_bytes);
}

TEST(CacheLayoutTest, RejectsDescriptorWithPhysicalGap) {
  WorkerCacheLayoutManifest manifest = make_head_manifest(0, 1, 1, 3, "source");
  manifest.tensors[0].shard.spans[0].repeat_count = kTokenCount - 1;

  const Status status = validate_worker_cache_layout(manifest);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("physical bytes"), std::string::npos);
}

TEST(CacheLayoutTest, RejectsDescriptorWithPhysicalOverlap) {
  WorkerCacheLayoutManifest manifest = make_head_manifest(0, 1, 2, 3, "source");
  manifest.tensors[0].shard.spans[1].physical_offset_bytes = 0;

  const Status status = validate_worker_cache_layout(manifest);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("overlapping"), std::string::npos);
}

TEST(CacheLayoutTest, RejectsDescriptorOffsetOverflow) {
  WorkerCacheLayoutManifest manifest = make_head_manifest(0, 1, 1, 3, "source");
  LogicalSpan& span = manifest.tensors[0].shard.spans[0];
  span.logical_offset_bytes = std::numeric_limits<uint64_t>::max();

  const Status status = validate_worker_cache_layout(manifest);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("overflows"), std::string::npos);
}

TEST(CacheLayoutTest, RejectsNonContiguousStrideMetadata) {
  WorkerCacheLayoutManifest manifest = make_head_manifest(0, 1, 1, 3, "source");
  manifest.tensors[0].stride[1] += 1;

  const Status status = validate_worker_cache_layout(manifest);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("contiguous"), std::string::npos);
}

TEST(CacheLayoutTest, RejectsInvalidPhysicalRowsPerResource) {
  WorkerCacheLayoutManifest manifest = make_head_manifest(0, 1, 1, 3, "source");
  manifest.tensors[0].physical_rows_per_resource = 0;

  const Status status = validate_worker_cache_layout(manifest);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("incomplete"), std::string::npos);
}

TEST(CacheLayoutTest, RejectsDuplicateTensorKey) {
  WorkerCacheLayoutManifest manifest = make_head_manifest(0, 1, 1, 3, "source");
  manifest.tensors.emplace_back(manifest.tensors[0]);

  const Status status = validate_worker_cache_layout(manifest);

  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("duplicate tensor key"), std::string::npos);
}

TEST(RequestRegionBinderTest, RejectsOutOfRangeNonContiguousResourceIds) {
  const WorkerCacheLayoutManifest source =
      make_head_manifest(0, 1, 1, 3, "source");
  const WorkerCacheLayoutManifest destination =
      make_head_manifest(0, 1, 1, 17, "destination");
  ReshardPlanTemplate plan;
  ASSERT_TRUE(
      ReshardPlanner().build_outgoing_plan(source, destination, &plan).ok());
  KVTransferMapping mapping;
  mapping.group_id = 7;
  mapping.local_ids = {kResourceCount};
  mapping.remote_ids = {0};
  std::vector<ByteRegion> regions;
  const Status status = RequestRegionBinder().bind(
      plan, {mapping}, CacheNamespace::MAIN, 0, &regions);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(regions.empty());
}

}  // namespace

}  // namespace xllm
