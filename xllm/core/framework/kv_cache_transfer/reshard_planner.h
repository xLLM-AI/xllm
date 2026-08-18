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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"
#include "framework/kv_cache_transfer/cache_layout.h"

namespace xllm {

struct ByteRegion {
  uint64_t local_buffer_id = 0;
  uint64_t local_offset = 0;
  uint64_t remote_buffer_id = 0;
  uint64_t remote_offset = 0;
  uint64_t length = 0;
};

struct StridedRegionTemplate {
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  int64_t layer_id = 0;
  int32_t role = 0;
  int32_t group_id = 0;
  std::string logical_tensor;
  uint64_t local_buffer_id = 0;
  uint64_t remote_buffer_id = 0;
  uint64_t local_offset_in_resource = 0;
  uint64_t remote_offset_in_resource = 0;
  uint64_t bytes_per_region = 0;
  uint64_t repeat_count = 1;
  uint64_t local_stride = 0;
  uint64_t remote_stride = 0;
  uint64_t local_resource_count = 0;
  uint64_t remote_resource_count = 0;
  uint64_t local_resource_stride = 0;
  uint64_t remote_resource_stride = 0;
  uint64_t local_buffer_bytes = 0;
  uint64_t remote_buffer_bytes = 0;
  bool local_explicit_resource_offsets = false;
  bool remote_explicit_resource_offsets = false;
};

// Runtime resource binding for page-mapped buffers such as GlobalXTensor.
// IDs preserve the logical cache-resource identity used for manifest bounds;
// offsets are byte bases within the registered MoonCake buffer.
struct ExplicitResourceMapping {
  int32_t group_id = 0;
  int32_t role = 0;
  std::vector<uint64_t> local_ids;
  std::vector<uint64_t> remote_ids;
  std::vector<uint64_t> local_offsets;
  std::vector<uint64_t> remote_offsets;
};

struct ReshardPlanTemplate {
  std::string source_incarnation;
  uint64_t source_layout_generation = 0;
  std::string destination_incarnation;
  uint64_t destination_layout_generation = 0;
  ParallelCoordinates source_coordinates;
  ParallelCoordinates destination_coordinates;
  std::string destination_addr;
  std::vector<StridedRegionTemplate> regions;
};

class ReshardPlanner final {
 public:
  bool source_participates(const WorkerCacheLayoutManifest& source,
                           const WorkerCacheLayoutManifest& destination) const;

  Status validate_destination_coverage(
      const std::vector<WorkerCacheLayoutManifest>& sources,
      const WorkerCacheLayoutManifest& destination) const;

  Status build_outgoing_plan(const WorkerCacheLayoutManifest& source,
                             const WorkerCacheLayoutManifest& destination,
                             ReshardPlanTemplate* plan) const;
};

class RequestRegionBinder final {
 public:
  Status bind(const ReshardPlanTemplate& plan,
              const std::vector<KVTransferMapping>& mappings,
              CacheNamespace cache_namespace,
              int64_t layer_id,
              std::vector<ByteRegion>* regions) const;

  Status bind_explicit(const ReshardPlanTemplate& plan,
                       const std::vector<ExplicitResourceMapping>& mappings,
                       CacheNamespace cache_namespace,
                       int64_t layer_id,
                       std::vector<ByteRegion>* regions) const;
};

}  // namespace xllm
