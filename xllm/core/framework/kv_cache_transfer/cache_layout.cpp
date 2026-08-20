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

#include "framework/kv_cache_transfer/cache_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {

namespace {

Status invalid(const std::string& message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

bool add_overflows(uint64_t lhs, uint64_t rhs) {
  return rhs > std::numeric_limits<uint64_t>::max() - lhs;
}

bool multiply_overflows(uint64_t lhs, uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs;
}

Status validate_coordinates(const ParallelCoordinates& coordinates) {
  if (coordinates.dp_size <= 0 || coordinates.tp_size <= 0 ||
      coordinates.cp_size <= 0 || coordinates.kv_split_size <= 0) {
    return invalid("parallel sizes must be positive");
  }
  if (coordinates.dp_rank < 0 || coordinates.dp_rank >= coordinates.dp_size ||
      coordinates.tp_rank < 0 || coordinates.tp_rank >= coordinates.tp_size ||
      coordinates.cp_rank < 0 || coordinates.cp_rank >= coordinates.cp_size ||
      coordinates.kv_split_rank < 0 ||
      coordinates.kv_split_rank >= coordinates.kv_split_size) {
    return invalid("parallel rank is outside its declared size");
  }
  return Status();
}

Status validate_span(const LogicalSpan& span,
                     const CacheTensorManifest& tensor) {
  if (span.logical_tensor.empty() || span.bytes_per_region == 0 ||
      span.repeat_count == 0) {
    return invalid("logical span identity, length, and repeat must be set");
  }
  if (span.owner_tp_rank < 0) {
    return invalid("logical span owner TP rank must be non-negative");
  }

  const uint64_t repeat_delta = span.repeat_count - 1;
  if (multiply_overflows(repeat_delta, span.physical_stride_bytes)) {
    return invalid("logical span physical stride overflows");
  }
  const uint64_t physical_tail = repeat_delta * span.physical_stride_bytes;
  if (add_overflows(span.physical_offset_bytes, physical_tail) ||
      add_overflows(span.physical_offset_bytes + physical_tail,
                    span.bytes_per_region) ||
      span.physical_offset_bytes + physical_tail + span.bytes_per_region >
          tensor.resource_stride_bytes) {
    return invalid("logical span exceeds its cache resource");
  }

  if (multiply_overflows(repeat_delta, span.logical_stride_bytes)) {
    return invalid("logical span canonical stride overflows");
  }
  const uint64_t logical_tail = repeat_delta * span.logical_stride_bytes;
  if (add_overflows(span.logical_offset_bytes, logical_tail) ||
      add_overflows(span.logical_offset_bytes + logical_tail,
                    span.bytes_per_region)) {
    return invalid("logical span canonical range overflows");
  }
  return Status();
}

Status validate_physical_coverage(const CacheTensorManifest& tensor) {
  constexpr uint64_t kMaxExpandedRegions = 1U << 20;
  uint64_t region_count = 0;
  uint64_t mapped_bytes = 0;
  for (const LogicalSpan& span : tensor.shard.spans) {
    if (span.repeat_count > kMaxExpandedRegions - region_count ||
        multiply_overflows(span.repeat_count, span.bytes_per_region) ||
        add_overflows(mapped_bytes,
                      span.repeat_count * span.bytes_per_region)) {
      return invalid("logical shard physical coverage is too large");
    }
    region_count += span.repeat_count;
    mapped_bytes += span.repeat_count * span.bytes_per_region;
  }
  if (mapped_bytes != tensor.resource_stride_bytes) {
    return invalid("logical shard does not describe all physical bytes");
  }

  std::vector<std::pair<uint64_t, uint64_t>> intervals;
  intervals.reserve(static_cast<size_t>(region_count));
  for (const LogicalSpan& span : tensor.shard.spans) {
    for (uint64_t repeat = 0; repeat < span.repeat_count; ++repeat) {
      const uint64_t begin =
          span.physical_offset_bytes + repeat * span.physical_stride_bytes;
      intervals.emplace_back(begin, begin + span.bytes_per_region);
    }
  }
  std::sort(intervals.begin(), intervals.end());

  uint64_t cursor = 0;
  for (const auto& [begin, end] : intervals) {
    if (begin != cursor) {
      return invalid(
          begin < cursor
              ? "logical shard maps overlapping physical bytes"
              : "logical shard does not describe all physical bytes");
    }
    cursor = end;
  }
  if (cursor != tensor.resource_stride_bytes) {
    return invalid("logical shard does not describe all physical bytes");
  }
  return Status();
}

Status validate_contiguous_shape(const CacheTensorManifest& tensor) {
  if (tensor.physical_rows_per_resource == 0 ||
      multiply_overflows(tensor.resource_count,
                         tensor.physical_rows_per_resource) ||
      static_cast<uint64_t>(tensor.shape.front()) !=
          tensor.resource_count * tensor.physical_rows_per_resource) {
    return invalid("cache tensor resource geometry differs from shape[0]");
  }

  uint64_t expected_stride = 1;
  for (size_t reverse_index = tensor.shape.size(); reverse_index > 0;
       --reverse_index) {
    const size_t index = reverse_index - 1;
    if (tensor.shape[index] != 1 &&
        static_cast<uint64_t>(tensor.stride[index]) != expected_stride) {
      return invalid("cache tensor strides are not contiguous");
    }
    if (multiply_overflows(expected_stride,
                           static_cast<uint64_t>(tensor.shape[index]))) {
      return invalid("cache tensor element count overflows");
    }
    expected_stride *= static_cast<uint64_t>(tensor.shape[index]);
  }
  if (multiply_overflows(expected_stride, tensor.element_bytes)) {
    return invalid("cache tensor byte size overflows");
  }
  const uint64_t tensor_bytes = expected_stride * tensor.element_bytes;
  if (multiply_overflows(tensor.resource_count, tensor.resource_stride_bytes) ||
      tensor.resource_count * tensor.resource_stride_bytes != tensor_bytes) {
    return invalid("cache tensor resource geometry differs from its shape");
  }
  if (multiply_overflows(static_cast<uint64_t>(tensor.stride.front()),
                         tensor.element_bytes) ||
      multiply_overflows(
          static_cast<uint64_t>(tensor.stride.front()) * tensor.element_bytes,
          tensor.physical_rows_per_resource) ||
      static_cast<uint64_t>(tensor.stride.front()) * tensor.element_bytes *
              tensor.physical_rows_per_resource !=
          tensor.resource_stride_bytes) {
    return invalid("cache tensor resource stride differs from physical rows");
  }
  return Status();
}

void descriptor_to_proto(const LogicalShardDescriptor& descriptor,
                         proto::LogicalShardDescriptor* proto_descriptor) {
  proto_descriptor->set_kind(
      static_cast<proto::LogicalShardKind>(descriptor.kind));
  proto_descriptor->set_resource_scope(
      static_cast<proto::CacheResourceScope>(descriptor.resource_scope));
  proto_descriptor->mutable_spans()->Reserve(
      static_cast<int32_t>(descriptor.spans.size()));
  for (const LogicalSpan& span : descriptor.spans) {
    proto::LogicalSpan* proto_span = proto_descriptor->add_spans();
    proto_span->set_logical_tensor(span.logical_tensor);
    proto_span->set_logical_offset_bytes(span.logical_offset_bytes);
    proto_span->set_physical_offset_bytes(span.physical_offset_bytes);
    proto_span->set_bytes_per_region(span.bytes_per_region);
    proto_span->set_repeat_count(span.repeat_count);
    proto_span->set_logical_stride_bytes(span.logical_stride_bytes);
    proto_span->set_physical_stride_bytes(span.physical_stride_bytes);
    proto_span->set_owner_tp_rank(span.owner_tp_rank);
  }
}

Status descriptor_from_proto(
    const proto::LogicalShardDescriptor& proto_descriptor,
    LogicalShardDescriptor* descriptor) {
  if (descriptor == nullptr) {
    return invalid("logical shard output must not be null");
  }
  if (!proto::LogicalShardKind_IsValid(proto_descriptor.kind()) ||
      !proto::CacheResourceScope_IsValid(proto_descriptor.resource_scope())) {
    return invalid("logical shard contains an unknown enum value");
  }

  descriptor->kind = static_cast<LogicalShardKind>(proto_descriptor.kind());
  descriptor->resource_scope =
      static_cast<CacheResourceScope>(proto_descriptor.resource_scope());
  descriptor->spans.clear();
  descriptor->spans.reserve(static_cast<size_t>(proto_descriptor.spans_size()));
  for (const proto::LogicalSpan& proto_span : proto_descriptor.spans()) {
    LogicalSpan span;
    span.logical_tensor = proto_span.logical_tensor();
    span.logical_offset_bytes = proto_span.logical_offset_bytes();
    span.physical_offset_bytes = proto_span.physical_offset_bytes();
    span.bytes_per_region = proto_span.bytes_per_region();
    span.repeat_count = proto_span.repeat_count();
    span.logical_stride_bytes = proto_span.logical_stride_bytes();
    span.physical_stride_bytes = proto_span.physical_stride_bytes();
    span.owner_tp_rank = proto_span.owner_tp_rank();
    descriptor->spans.emplace_back(std::move(span));
  }
  return Status();
}

}  // namespace

Status validate_worker_cache_layout(const WorkerCacheLayoutManifest& manifest) {
  if (manifest.schema_version != kCacheLayoutSchemaVersion) {
    return invalid("unsupported cache layout schema version");
  }
  if (manifest.incarnation_id.empty() || manifest.layout_generation == 0 ||
      manifest.fingerprint.empty() || manifest.backend.empty() ||
      manifest.layout_family.empty() || manifest.addr.empty()) {
    return invalid(
        "cache layout identity and compatibility fields are required");
  }
  const Status coordinate_status = validate_coordinates(manifest.coordinates);
  if (!coordinate_status.ok()) {
    return coordinate_status;
  }
  if (manifest.tensors.empty()) {
    return invalid("cache layout must contain at least one tensor");
  }

  std::unordered_map<int64_t, uint64_t> buffer_sizes;
  buffer_sizes.reserve(manifest.tensors.size());
  std::set<std::tuple<int32_t, int64_t, int32_t, int32_t>> tensor_keys;
  for (const CacheTensorManifest& tensor : manifest.tensors) {
    if (tensor.cache_namespace < CacheNamespace::MAIN ||
        tensor.cache_namespace > CacheNamespace::SPEC_DRAFT ||
        tensor.layer_id < 0 || tensor.mooncake_buffer_id < 0 ||
        tensor.element_bytes == 0 || tensor.shape.empty() ||
        tensor.shape.size() != tensor.stride.size() || !tensor.contiguous ||
        tensor.resource_count == 0 || tensor.physical_rows_per_resource == 0 ||
        tensor.resource_stride_bytes == 0 || tensor.buffer_bytes == 0 ||
        tensor.shard.spans.empty()) {
      return invalid("cache tensor manifest is incomplete");
    }
    if (!tensor_keys
             .emplace(static_cast<int32_t>(tensor.cache_namespace),
                      tensor.layer_id,
                      tensor.role,
                      tensor.group_id)
             .second) {
      return invalid("cache tensor manifest contains a duplicate tensor key");
    }
    if (std::any_of(tensor.shape.begin(),
                    tensor.shape.end(),
                    [](int64_t dim) { return dim <= 0; }) ||
        std::any_of(tensor.stride.begin(),
                    tensor.stride.end(),
                    [](int64_t stride) { return stride <= 0; })) {
      return invalid("cache tensor shape and stride must be positive");
    }
    const Status contiguous_status = validate_contiguous_shape(tensor);
    if (!contiguous_status.ok()) {
      return contiguous_status;
    }
    const auto [buffer_it, inserted] =
        buffer_sizes.emplace(tensor.mooncake_buffer_id, tensor.buffer_bytes);
    if (!inserted && buffer_it->second != tensor.buffer_bytes) {
      return invalid("cache tensor records disagree on shared buffer size");
    }
    if (multiply_overflows(tensor.resource_count,
                           tensor.resource_stride_bytes) ||
        add_overflows(tensor.storage_offset_bytes,
                      tensor.resource_count * tensor.resource_stride_bytes) ||
        tensor.storage_offset_bytes +
                tensor.resource_count * tensor.resource_stride_bytes >
            tensor.buffer_bytes) {
      return invalid("cache tensor resources exceed its registered buffer");
    }
    if (tensor.shard.kind < LogicalShardKind::REPLICATED ||
        tensor.shard.kind > LogicalShardKind::COMPOSITE ||
        tensor.shard.resource_scope < CacheResourceScope::BLOCK ||
        tensor.shard.resource_scope > CacheResourceScope::SEQUENCE) {
      return invalid("cache tensor contains an unknown shard enum value");
    }
    for (const LogicalSpan& span : tensor.shard.spans) {
      if (span.owner_tp_rank >= manifest.coordinates.tp_size) {
        return invalid("logical span owner is outside source TP size");
      }
      const Status span_status = validate_span(span, tensor);
      if (!span_status.ok()) {
        return span_status;
      }
    }
    const Status coverage_status = validate_physical_coverage(tensor);
    if (!coverage_status.ok()) {
      return coverage_status;
    }
  }
  return Status();
}

void cache_layout_to_proto(const WorkerCacheLayoutManifest& manifest,
                           proto::WorkerCacheLayoutManifest* proto_manifest) {
  CHECK(proto_manifest != nullptr);
  proto_manifest->Clear();
  proto_manifest->set_schema_version(manifest.schema_version);
  proto_manifest->set_incarnation_id(manifest.incarnation_id);
  proto_manifest->set_layout_generation(manifest.layout_generation);
  proto_manifest->set_fingerprint(manifest.fingerprint);
  proto_manifest->set_backend(manifest.backend);
  proto_manifest->set_layout_family(manifest.layout_family);
  proto_manifest->set_cluster_id(manifest.cluster_id);
  proto_manifest->set_addr(manifest.addr);
  proto_manifest->set_listen_port(manifest.listen_port);

  proto::ParallelCoordinates* coordinates =
      proto_manifest->mutable_coordinates();
  coordinates->set_dp_rank(manifest.coordinates.dp_rank);
  coordinates->set_dp_size(manifest.coordinates.dp_size);
  coordinates->set_tp_rank(manifest.coordinates.tp_rank);
  coordinates->set_tp_size(manifest.coordinates.tp_size);
  coordinates->set_cp_rank(manifest.coordinates.cp_rank);
  coordinates->set_cp_size(manifest.coordinates.cp_size);
  coordinates->set_kv_split_rank(manifest.coordinates.kv_split_rank);
  coordinates->set_kv_split_size(manifest.coordinates.kv_split_size);

  proto_manifest->mutable_tensors()->Reserve(
      static_cast<int32_t>(manifest.tensors.size()));
  for (const CacheTensorManifest& tensor : manifest.tensors) {
    proto::CacheTensorManifest* proto_tensor = proto_manifest->add_tensors();
    proto_tensor->set_cache_namespace(
        static_cast<proto::CacheNamespace>(tensor.cache_namespace));
    proto_tensor->set_layer_id(tensor.layer_id);
    proto_tensor->set_role(tensor.role);
    proto_tensor->set_group_id(tensor.group_id);
    proto_tensor->set_mooncake_buffer_id(tensor.mooncake_buffer_id);
    proto_tensor->set_scalar_type(tensor.scalar_type);
    proto_tensor->set_element_bytes(tensor.element_bytes);
    for (int64_t dim : tensor.shape) {
      proto_tensor->add_shape(dim);
    }
    for (int64_t stride : tensor.stride) {
      proto_tensor->add_stride(stride);
    }
    proto_tensor->set_storage_offset_bytes(tensor.storage_offset_bytes);
    proto_tensor->set_contiguous(tensor.contiguous);
    proto_tensor->set_resource_count(tensor.resource_count);
    proto_tensor->set_physical_rows_per_resource(
        tensor.physical_rows_per_resource);
    proto_tensor->set_resource_stride_bytes(tensor.resource_stride_bytes);
    proto_tensor->set_buffer_bytes(tensor.buffer_bytes);
    proto_tensor->set_block_token_capacity(tensor.block_token_capacity);
    proto_tensor->set_explicit_resource_offsets(
        tensor.explicit_resource_offsets);
    descriptor_to_proto(tensor.shard, proto_tensor->mutable_shard());
  }
}

Status cache_layout_from_proto(
    const proto::WorkerCacheLayoutManifest& proto_manifest,
    WorkerCacheLayoutManifest* manifest) {
  if (manifest == nullptr) {
    return invalid("cache layout output must not be null");
  }
  manifest->schema_version = proto_manifest.schema_version();
  manifest->incarnation_id = proto_manifest.incarnation_id();
  manifest->layout_generation = proto_manifest.layout_generation();
  manifest->fingerprint = proto_manifest.fingerprint();
  manifest->backend = proto_manifest.backend();
  manifest->layout_family = proto_manifest.layout_family();
  manifest->cluster_id = proto_manifest.cluster_id();
  manifest->addr = proto_manifest.addr();
  if (proto_manifest.listen_port() >
      static_cast<uint32_t>(std::numeric_limits<uint16_t>::max())) {
    return invalid("cache layout listen port exceeds uint16 range");
  }
  manifest->listen_port = static_cast<uint16_t>(proto_manifest.listen_port());

  const proto::ParallelCoordinates& proto_coordinates =
      proto_manifest.coordinates();
  manifest->coordinates.dp_rank = proto_coordinates.dp_rank();
  manifest->coordinates.dp_size = proto_coordinates.dp_size();
  manifest->coordinates.tp_rank = proto_coordinates.tp_rank();
  manifest->coordinates.tp_size = proto_coordinates.tp_size();
  manifest->coordinates.cp_rank = proto_coordinates.cp_rank();
  manifest->coordinates.cp_size = proto_coordinates.cp_size();
  manifest->coordinates.kv_split_rank = proto_coordinates.kv_split_rank();
  manifest->coordinates.kv_split_size = proto_coordinates.kv_split_size();

  manifest->tensors.clear();
  manifest->tensors.reserve(static_cast<size_t>(proto_manifest.tensors_size()));
  for (const proto::CacheTensorManifest& proto_tensor :
       proto_manifest.tensors()) {
    if (!proto::CacheNamespace_IsValid(proto_tensor.cache_namespace())) {
      return invalid("cache tensor contains an unknown namespace");
    }
    CacheTensorManifest tensor;
    tensor.cache_namespace =
        static_cast<CacheNamespace>(proto_tensor.cache_namespace());
    tensor.layer_id = proto_tensor.layer_id();
    tensor.role = proto_tensor.role();
    tensor.group_id = proto_tensor.group_id();
    tensor.mooncake_buffer_id = proto_tensor.mooncake_buffer_id();
    tensor.scalar_type = proto_tensor.scalar_type();
    tensor.element_bytes = proto_tensor.element_bytes();
    tensor.shape.assign(proto_tensor.shape().begin(),
                        proto_tensor.shape().end());
    tensor.stride.assign(proto_tensor.stride().begin(),
                         proto_tensor.stride().end());
    tensor.storage_offset_bytes = proto_tensor.storage_offset_bytes();
    tensor.contiguous = proto_tensor.contiguous();
    tensor.resource_count = proto_tensor.resource_count();
    tensor.physical_rows_per_resource =
        std::max<uint64_t>(proto_tensor.physical_rows_per_resource(), 1);
    tensor.resource_stride_bytes = proto_tensor.resource_stride_bytes();
    tensor.buffer_bytes = proto_tensor.buffer_bytes();
    tensor.block_token_capacity = proto_tensor.block_token_capacity();
    tensor.explicit_resource_offsets = proto_tensor.explicit_resource_offsets();
    const Status descriptor_status =
        descriptor_from_proto(proto_tensor.shard(), &tensor.shard);
    if (!descriptor_status.ok()) {
      return descriptor_status;
    }
    manifest->tensors.emplace_back(std::move(tensor));
  }
  return validate_worker_cache_layout(*manifest);
}

}  // namespace xllm
