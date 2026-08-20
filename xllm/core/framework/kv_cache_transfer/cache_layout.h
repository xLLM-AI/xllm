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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/types.h"
#include "framework/kv_cache/logical_cache_layout.h"
#include "mooncake_transfer_engine.pb.h"

namespace xllm {

inline constexpr uint32_t kCacheLayoutSchemaVersion = 1;

enum class CacheNamespace : int8_t {
  MAIN = 0,
  SPEC_DRAFT = 1,
};

struct ParallelCoordinates {
  int32_t dp_rank = 0;
  int32_t dp_size = 1;
  int32_t tp_rank = 0;
  int32_t tp_size = 1;
  int32_t cp_rank = 0;
  int32_t cp_size = 1;
  int32_t kv_split_rank = 0;
  int32_t kv_split_size = 1;
};

struct CacheTensorManifest {
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  int64_t layer_id = 0;
  int32_t role = 0;
  int32_t group_id = 0;
  int64_t mooncake_buffer_id = 0;
  int32_t scalar_type = 0;
  uint64_t element_bytes = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  uint64_t storage_offset_bytes = 0;
  bool contiguous = false;
  uint64_t resource_count = 0;
  // Number of physical tensor rows owned by one logical cache resource.
  // This is greater than one for checkpointed speculative SSM state.
  uint64_t physical_rows_per_resource = 1;
  uint64_t resource_stride_bytes = 0;
  uint64_t buffer_bytes = 0;
  uint64_t block_token_capacity = 0;
  // XTensor resources are page-mapped at runtime and therefore bind an
  // explicit GlobalXTensor byte offset instead of resource_id * stride.
  bool explicit_resource_offsets = false;
  LogicalShardDescriptor shard;
};

struct WorkerCacheLayoutManifest {
  uint32_t schema_version = kCacheLayoutSchemaVersion;
  std::string incarnation_id;
  uint64_t layout_generation = 0;
  std::string fingerprint;
  std::string backend;
  std::string layout_family;
  uint64_t cluster_id = 0;
  std::string addr;
  uint16_t listen_port = 0;
  ParallelCoordinates coordinates;
  std::vector<CacheTensorManifest> tensors;
};

struct CacheRegistrationContext {
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  ParallelCoordinates coordinates;
  CacheTensorLayoutContext tensor_layout;
  std::string fingerprint;
  std::string backend;
  std::string layout_family;
  uint64_t cluster_id = 0;
  std::string addr;
  uint16_t listen_port = 0;
};

Status validate_worker_cache_layout(const WorkerCacheLayoutManifest& manifest);

void cache_layout_to_proto(const WorkerCacheLayoutManifest& manifest,
                           proto::WorkerCacheLayoutManifest* proto_manifest);

Status cache_layout_from_proto(
    const proto::WorkerCacheLayoutManifest& proto_manifest,
    WorkerCacheLayoutManifest* manifest);

}  // namespace xllm
