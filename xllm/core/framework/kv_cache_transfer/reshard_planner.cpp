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

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {

namespace {

struct AtomicLogicalRegion {
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  int64_t layer_id = 0;
  int32_t role = 0;
  int32_t group_id = 0;
  std::string logical_tensor;
  uint64_t logical_offset = 0;
  uint64_t physical_offset = 0;
  uint64_t length = 0;
  int32_t owner_tp_rank = 0;
  const CacheTensorManifest* tensor = nullptr;
  const WorkerCacheLayoutManifest* worker = nullptr;
};

struct PlannedAtomicRegion {
  uint64_t logical_offset = 0;
  StridedRegionTemplate region;
};

struct BoundRegion {
  ByteRegion region;
  std::string logical_tensor;
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  int64_t layer_id = 0;
  int32_t group_id = 0;
  uint64_t local_resource_id = 0;
  uint64_t remote_resource_id = 0;
};

using LogicalTensorKey =
    std::tuple<CacheNamespace, int64_t, int32_t, int32_t, std::string>;
using RegionGroup = std::vector<const AtomicLogicalRegion*>;
using RegionGroups = std::map<LogicalTensorKey, RegionGroup>;

Status invalid(const std::string& message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

bool add_overflows(uint64_t lhs, uint64_t rhs) {
  return rhs > std::numeric_limits<uint64_t>::max() - lhs;
}

bool multiply_overflows(uint64_t lhs, uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs;
}

bool same_partition(const ParallelCoordinates& source,
                    const ParallelCoordinates& destination) {
  return source.cp_size == destination.cp_size &&
         source.cp_rank == destination.cp_rank &&
         source.kv_split_size == destination.kv_split_size &&
         source.kv_split_rank == destination.kv_split_rank;
}

bool same_partition_sizes(const ParallelCoordinates& source,
                          const ParallelCoordinates& destination) {
  return source.cp_size == destination.cp_size &&
         source.kv_split_size == destination.kv_split_size;
}

bool collapses_prefill_partition(const ParallelCoordinates& source,
                                 const ParallelCoordinates& destination) {
  return !same_partition_sizes(source, destination) &&
         destination.cp_size == 1 && destination.cp_rank == 0 &&
         destination.kv_split_size == 1 && destination.kv_split_rank == 0;
}

bool supports_partition_pair(const ParallelCoordinates& source,
                             const ParallelCoordinates& destination) {
  return same_partition(source, destination) ||
         collapses_prefill_partition(source, destination);
}

Status validate_compatibility(const WorkerCacheLayoutManifest& source,
                              const WorkerCacheLayoutManifest& destination) {
  const Status source_status = validate_worker_cache_layout(source);
  if (!source_status.ok()) {
    return invalid("invalid source layout: " + source_status.message());
  }
  const Status destination_status = validate_worker_cache_layout(destination);
  if (!destination_status.ok()) {
    return invalid("invalid destination layout: " +
                   destination_status.message());
  }
  if (source.schema_version != destination.schema_version ||
      source.fingerprint != destination.fingerprint ||
      source.backend != destination.backend ||
      source.layout_family != destination.layout_family) {
    return invalid("source and destination cache layouts are incompatible");
  }
  if (!supports_partition_pair(source.coordinates, destination.coordinates)) {
    return invalid(
        "source and destination CP/KV-split partitions are unsupported");
  }
  return Status();
}

void expand_manifest(const WorkerCacheLayoutManifest& manifest,
                     bool only_static_owner,
                     std::vector<AtomicLogicalRegion>* regions) {
  for (const CacheTensorManifest& tensor : manifest.tensors) {
    for (const LogicalSpan& span : tensor.shard.spans) {
      if (only_static_owner &&
          manifest.coordinates.tp_rank != span.owner_tp_rank) {
        continue;
      }
      for (uint64_t repeat = 0; repeat < span.repeat_count; ++repeat) {
        AtomicLogicalRegion region;
        region.cache_namespace = tensor.cache_namespace;
        region.layer_id = tensor.layer_id;
        region.role = tensor.role;
        region.group_id = tensor.group_id;
        region.logical_tensor = span.logical_tensor;
        region.logical_offset =
            span.logical_offset_bytes + repeat * span.logical_stride_bytes;
        region.physical_offset =
            span.physical_offset_bytes + repeat * span.physical_stride_bytes;
        region.length = span.bytes_per_region;
        region.owner_tp_rank = span.owner_tp_rank;
        region.tensor = &tensor;
        region.worker = &manifest;
        regions->emplace_back(std::move(region));
      }
    }
  }
}

LogicalTensorKey logical_tensor_key(const AtomicLogicalRegion& region) {
  return std::make_tuple(region.cache_namespace,
                         region.layer_id,
                         region.role,
                         region.group_id,
                         region.logical_tensor);
}

uint64_t logical_end(const AtomicLogicalRegion& region) {
  return region.logical_offset + region.length;
}

RegionGroups group_regions(const std::vector<AtomicLogicalRegion>& regions) {
  RegionGroups groups;
  for (const AtomicLogicalRegion& region : regions) {
    groups[logical_tensor_key(region)].emplace_back(&region);
  }
  for (auto& [key, group] : groups) {
    static_cast<void>(key);
    std::sort(
        group.begin(),
        group.end(),
        [](const AtomicLogicalRegion* lhs, const AtomicLogicalRegion* rhs) {
          return std::tie(
                     lhs->logical_offset, lhs->length, lhs->physical_offset) <
                 std::tie(
                     rhs->logical_offset, rhs->length, rhs->physical_offset);
        });
  }
  return groups;
}

template <typename Visitor>
Status visit_overlaps(const RegionGroup& left,
                      const RegionGroup& right,
                      Visitor visitor) {
  size_t right_begin = 0;
  for (const AtomicLogicalRegion* left_region : left) {
    while (right_begin < right.size() &&
           logical_end(*right[right_begin]) <= left_region->logical_offset) {
      ++right_begin;
    }
    for (size_t right_index = right_begin; right_index < right.size();
         ++right_index) {
      const AtomicLogicalRegion* right_region = right[right_index];
      if (right_region->logical_offset >= logical_end(*left_region)) {
        break;
      }
      if (logical_end(*right_region) <= left_region->logical_offset) {
        continue;
      }
      const Status status = visitor(*left_region, *right_region);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status();
}

Status validate_tensor_pair(const AtomicLogicalRegion& source,
                            const AtomicLogicalRegion& destination) {
  if (source.tensor->scalar_type != destination.tensor->scalar_type ||
      source.tensor->element_bytes != destination.tensor->element_bytes ||
      source.tensor->block_token_capacity !=
          destination.tensor->block_token_capacity ||
      source.tensor->shard.resource_scope !=
          destination.tensor->shard.resource_scope ||
      source.tensor->explicit_resource_offsets !=
          destination.tensor->explicit_resource_offsets) {
    return invalid(
        "matching logical tensors have incompatible dtype or resource "
        "semantics");
  }
  return Status();
}

Status validate_coverage_for_sources(
    const std::vector<AtomicLogicalRegion>& source_regions,
    const std::vector<AtomicLogicalRegion>& destination_regions) {
  const RegionGroups source_groups = group_regions(source_regions);
  const RegionGroups destination_groups = group_regions(destination_regions);
  for (const auto& [key, destinations] : destination_groups) {
    const auto source_it = source_groups.find(key);
    if (source_it == source_groups.end()) {
      return invalid("destination logical bytes have no source writer");
    }
    const RegionGroup& sources = source_it->second;
    const Status tensor_status =
        visit_overlaps(sources,
                       destinations,
                       [](const AtomicLogicalRegion& source,
                          const AtomicLogicalRegion& destination) {
                         return validate_tensor_pair(source, destination);
                       });
    if (!tensor_status.ok()) {
      return tensor_status;
    }

    std::vector<std::pair<uint64_t, int32_t>> events;
    events.reserve(sources.size() * 2);
    for (const AtomicLogicalRegion* source : sources) {
      events.emplace_back(source->logical_offset, 1);
      events.emplace_back(logical_end(*source), -1);
    }
    std::sort(events.begin(), events.end());

    struct CoveragePoint {
      uint64_t position = 0;
      int32_t writers = 0;
    };
    std::vector<CoveragePoint> coverage;
    coverage.reserve(events.size());
    int32_t writers = 0;
    size_t event_index = 0;
    while (event_index < events.size()) {
      const uint64_t position = events[event_index].first;
      while (event_index < events.size() &&
             events[event_index].first == position) {
        writers += events[event_index].second;
        ++event_index;
      }
      coverage.emplace_back(CoveragePoint{position, writers});
    }

    for (const AtomicLogicalRegion* destination : destinations) {
      const uint64_t destination_end = logical_end(*destination);
      auto next_point =
          std::upper_bound(coverage.begin(),
                           coverage.end(),
                           destination->logical_offset,
                           [](uint64_t position, const CoveragePoint& point) {
                             return position < point.position;
                           });
      writers =
          next_point == coverage.begin() ? 0 : std::prev(next_point)->writers;
      uint64_t cursor = destination->logical_offset;
      while (cursor < destination_end) {
        const uint64_t next_position =
            next_point == coverage.end()
                ? destination_end
                : std::min(destination_end, next_point->position);
        if (next_position > cursor && writers != 1) {
          return invalid(
              writers == 0 ? "destination logical bytes have no source writer"
                           : "destination logical bytes have multiple writers");
        }
        cursor = next_position;
        if (cursor < destination_end && next_point != coverage.end()) {
          writers = next_point->writers;
          ++next_point;
        }
      }
    }
  }
  return Status();
}

Status validate_source_instance(
    const std::vector<WorkerCacheLayoutManifest>& sources,
    const WorkerCacheLayoutManifest& destination) {
  if (sources.empty()) {
    return invalid("source layout set is empty");
  }
  const WorkerCacheLayoutManifest& reference = sources.front();
  const Status reference_status = validate_worker_cache_layout(reference);
  if (!reference_status.ok()) {
    return invalid("invalid source layout: " + reference_status.message());
  }
  if (reference.schema_version != destination.schema_version ||
      reference.fingerprint != destination.fingerprint ||
      reference.backend != destination.backend ||
      reference.layout_family != destination.layout_family) {
    return invalid("source and destination instance layouts are incompatible");
  }
  if (!same_partition_sizes(reference.coordinates, destination.coordinates) &&
      !collapses_prefill_partition(reference.coordinates,
                                   destination.coordinates)) {
    return invalid(
        "source CP/KV-split partitions can only collapse into a "
        "CP1/KV-split1 destination");
  }

  const ParallelCoordinates& expected = reference.coordinates;
  if (multiply_overflows(static_cast<uint64_t>(expected.dp_size),
                         static_cast<uint64_t>(expected.cp_size)) ||
      multiply_overflows(static_cast<uint64_t>(expected.dp_size) *
                             static_cast<uint64_t>(expected.cp_size),
                         static_cast<uint64_t>(expected.tp_size))) {
    return invalid("source parallel topology size overflows");
  }
  const uint64_t expected_worker_count =
      static_cast<uint64_t>(expected.dp_size) *
      static_cast<uint64_t>(expected.cp_size) *
      static_cast<uint64_t>(expected.tp_size);
  if (expected_worker_count != static_cast<uint64_t>(sources.size())) {
    return invalid("source worker manifest set is incomplete");
  }

  std::set<std::tuple<int32_t, int32_t, int32_t>> rank_tuples;
  for (const WorkerCacheLayoutManifest& source : sources) {
    const Status source_status = validate_worker_cache_layout(source);
    if (!source_status.ok()) {
      return invalid("invalid source layout: " + source_status.message());
    }
    const ParallelCoordinates& coordinates = source.coordinates;
    if (source.schema_version != reference.schema_version ||
        source.layout_generation != reference.layout_generation ||
        source.fingerprint != reference.fingerprint ||
        source.backend != reference.backend ||
        source.layout_family != reference.layout_family ||
        coordinates.dp_size != expected.dp_size ||
        coordinates.tp_size != expected.tp_size ||
        coordinates.cp_size != expected.cp_size ||
        coordinates.kv_split_size != expected.kv_split_size) {
      return invalid("source worker manifests disagree on instance layout");
    }
    if (!rank_tuples
             .emplace(
                 coordinates.dp_rank, coordinates.cp_rank, coordinates.tp_rank)
             .second) {
      return invalid("source worker manifest set contains a duplicate rank");
    }
  }
  return Status();
}

bool same_template_key(const StridedRegionTemplate& lhs,
                       const StridedRegionTemplate& rhs) {
  return lhs.cache_namespace == rhs.cache_namespace &&
         lhs.layer_id == rhs.layer_id && lhs.role == rhs.role &&
         lhs.group_id == rhs.group_id &&
         lhs.logical_tensor == rhs.logical_tensor &&
         lhs.local_buffer_id == rhs.local_buffer_id &&
         lhs.remote_buffer_id == rhs.remote_buffer_id &&
         lhs.bytes_per_region == rhs.bytes_per_region &&
         lhs.local_resource_count == rhs.local_resource_count &&
         lhs.remote_resource_count == rhs.remote_resource_count &&
         lhs.local_resource_stride == rhs.local_resource_stride &&
         lhs.remote_resource_stride == rhs.remote_resource_stride &&
         lhs.local_buffer_bytes == rhs.local_buffer_bytes &&
         lhs.remote_buffer_bytes == rhs.remote_buffer_bytes &&
         lhs.local_explicit_resource_offsets ==
             rhs.local_explicit_resource_offsets &&
         lhs.remote_explicit_resource_offsets ==
             rhs.remote_explicit_resource_offsets;
}

void compact_planned_regions(std::vector<PlannedAtomicRegion>* candidates,
                             std::vector<StridedRegionTemplate>* output) {
  std::sort(candidates->begin(),
            candidates->end(),
            [](const PlannedAtomicRegion& lhs, const PlannedAtomicRegion& rhs) {
              const StridedRegionTemplate& left = lhs.region;
              const StridedRegionTemplate& right = rhs.region;
              return std::tie(left.cache_namespace,
                              left.layer_id,
                              left.role,
                              left.group_id,
                              left.logical_tensor,
                              left.local_buffer_id,
                              left.remote_buffer_id,
                              lhs.logical_offset) <
                     std::tie(right.cache_namespace,
                              right.layer_id,
                              right.role,
                              right.group_id,
                              right.logical_tensor,
                              right.local_buffer_id,
                              right.remote_buffer_id,
                              rhs.logical_offset);
            });

  std::vector<uint64_t> last_logical_offsets;
  last_logical_offsets.reserve(candidates->size());
  for (const PlannedAtomicRegion& candidate : *candidates) {
    if (output->empty() ||
        !same_template_key(output->back(), candidate.region)) {
      output->emplace_back(candidate.region);
      last_logical_offsets.emplace_back(candidate.logical_offset);
      continue;
    }

    StridedRegionTemplate& previous = output->back();
    const uint64_t previous_logical = last_logical_offsets.back();
    if (candidate.logical_offset <= previous_logical ||
        candidate.region.local_offset_in_resource <=
            previous.local_offset_in_resource ||
        candidate.region.remote_offset_in_resource <=
            previous.remote_offset_in_resource) {
      output->emplace_back(candidate.region);
      last_logical_offsets.emplace_back(candidate.logical_offset);
      continue;
    }

    const uint64_t local_stride =
        candidate.region.local_offset_in_resource -
        (previous.local_offset_in_resource +
         (previous.repeat_count - 1) * previous.local_stride);
    const uint64_t remote_stride =
        candidate.region.remote_offset_in_resource -
        (previous.remote_offset_in_resource +
         (previous.repeat_count - 1) * previous.remote_stride);
    const uint64_t logical_stride = candidate.logical_offset - previous_logical;
    const bool can_start_repeat = previous.repeat_count == 1;
    const bool same_repeat = previous.repeat_count > 1 &&
                             local_stride == previous.local_stride &&
                             remote_stride == previous.remote_stride;
    if ((!can_start_repeat && !same_repeat) || logical_stride == 0) {
      output->emplace_back(candidate.region);
      last_logical_offsets.emplace_back(candidate.logical_offset);
      continue;
    }
    if (can_start_repeat) {
      previous.local_stride = local_stride;
      previous.remote_stride = remote_stride;
    }
    ++previous.repeat_count;
    last_logical_offsets.back() = candidate.logical_offset;
  }
}

bool same_bound_domain(const BoundRegion& lhs, const BoundRegion& rhs) {
  return lhs.logical_tensor == rhs.logical_tensor &&
         lhs.cache_namespace == rhs.cache_namespace &&
         lhs.layer_id == rhs.layer_id && lhs.group_id == rhs.group_id &&
         lhs.local_resource_id == rhs.local_resource_id &&
         lhs.remote_resource_id == rhs.remote_resource_id;
}

struct ResourceMappingView {
  const std::vector<uint64_t>* local_ids = nullptr;
  const std::vector<uint64_t>* remote_ids = nullptr;
  const std::vector<uint64_t>* local_offsets = nullptr;
  const std::vector<uint64_t>* remote_offsets = nullptr;
};

uint64_t explicit_mapping_key(int32_t group_id, int32_t role) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(group_id)) << 32) |
         static_cast<uint32_t>(role);
}

Status bind_regions(
    const ReshardPlanTemplate& plan,
    const std::unordered_map<int32_t, ResourceMappingView>* regular_mappings,
    const std::unordered_map<uint64_t, ResourceMappingView>* explicit_mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) {
  if (regions == nullptr) {
    return invalid("bound region output must not be null");
  }
  regions->clear();

  std::vector<BoundRegion> bound;
  for (const StridedRegionTemplate& region_template : plan.regions) {
    if (region_template.cache_namespace != cache_namespace ||
        (layer_id >= 0 && region_template.layer_id != layer_id)) {
      continue;
    }

    const bool needs_explicit_mapping =
        region_template.local_explicit_resource_offsets ||
        region_template.remote_explicit_resource_offsets;
    const ResourceMappingView* mapping = nullptr;
    if (needs_explicit_mapping) {
      if (explicit_mappings == nullptr) {
        return invalid(
            "page-mapped cache plan requires explicit resource offsets");
      }
      const auto mapping_it = explicit_mappings->find(
          explicit_mapping_key(region_template.group_id, region_template.role));
      if (mapping_it == explicit_mappings->end()) {
        return invalid(
            "request is missing an explicit cache tensor mapping required by "
            "its plan");
      }
      mapping = &mapping_it->second;
    } else {
      if (regular_mappings == nullptr) {
        return invalid("strided cache plan requires logical resource IDs");
      }
      const auto mapping_it = regular_mappings->find(region_template.group_id);
      if (mapping_it == regular_mappings->end()) {
        return invalid("request is missing a cache group required by its plan");
      }
      mapping = &mapping_it->second;
    }

    const size_t resource_count = mapping->local_ids->size();
    for (size_t resource_index = 0; resource_index < resource_count;
         ++resource_index) {
      const uint64_t local_id = (*mapping->local_ids)[resource_index];
      const uint64_t remote_id = (*mapping->remote_ids)[resource_index];
      if (local_id >= region_template.local_resource_count ||
          remote_id >= region_template.remote_resource_count) {
        return invalid("request cache resource id is outside its manifest");
      }

      uint64_t local_resource_base = 0;
      if (region_template.local_explicit_resource_offsets) {
        local_resource_base = (*mapping->local_offsets)[resource_index];
      } else {
        if (multiply_overflows(local_id,
                               region_template.local_resource_stride)) {
          return invalid("request local cache resource base overflows");
        }
        local_resource_base = local_id * region_template.local_resource_stride;
      }
      uint64_t remote_resource_base = 0;
      if (region_template.remote_explicit_resource_offsets) {
        remote_resource_base = (*mapping->remote_offsets)[resource_index];
      } else {
        if (multiply_overflows(remote_id,
                               region_template.remote_resource_stride)) {
          return invalid("request remote cache resource base overflows");
        }
        remote_resource_base =
            remote_id * region_template.remote_resource_stride;
      }

      for (uint64_t repeat = 0; repeat < region_template.repeat_count;
           ++repeat) {
        if (multiply_overflows(repeat, region_template.local_stride) ||
            multiply_overflows(repeat, region_template.remote_stride)) {
          return invalid("strided request region overflows");
        }
        const uint64_t local_repeat = repeat * region_template.local_stride;
        const uint64_t remote_repeat = repeat * region_template.remote_stride;
        if (add_overflows(local_resource_base,
                          region_template.local_offset_in_resource) ||
            add_overflows(
                local_resource_base + region_template.local_offset_in_resource,
                local_repeat) ||
            add_overflows(remote_resource_base,
                          region_template.remote_offset_in_resource) ||
            add_overflows(remote_resource_base +
                              region_template.remote_offset_in_resource,
                          remote_repeat)) {
          return invalid("bound request region offset overflows");
        }
        BoundRegion item;
        item.region.local_buffer_id = region_template.local_buffer_id;
        item.region.local_offset = local_resource_base +
                                   region_template.local_offset_in_resource +
                                   local_repeat;
        item.region.remote_buffer_id = region_template.remote_buffer_id;
        item.region.remote_offset = remote_resource_base +
                                    region_template.remote_offset_in_resource +
                                    remote_repeat;
        item.region.length = region_template.bytes_per_region;
        if (item.region.local_offset > region_template.local_buffer_bytes ||
            item.region.length >
                region_template.local_buffer_bytes - item.region.local_offset ||
            item.region.remote_offset > region_template.remote_buffer_bytes ||
            item.region.length > region_template.remote_buffer_bytes -
                                     item.region.remote_offset) {
          return invalid("bound request region exceeds its manifest buffer");
        }
        item.logical_tensor = region_template.logical_tensor;
        item.cache_namespace = region_template.cache_namespace;
        item.layer_id = region_template.layer_id;
        item.group_id = region_template.group_id;
        item.local_resource_id = local_id;
        item.remote_resource_id = remote_id;
        bound.emplace_back(std::move(item));
      }
    }
  }

  std::sort(bound.begin(),
            bound.end(),
            [](const BoundRegion& lhs, const BoundRegion& rhs) {
              return std::tie(lhs.cache_namespace,
                              lhs.layer_id,
                              lhs.group_id,
                              lhs.logical_tensor,
                              lhs.local_resource_id,
                              lhs.remote_resource_id,
                              lhs.region.local_buffer_id,
                              lhs.region.remote_buffer_id,
                              lhs.region.local_offset,
                              lhs.region.remote_offset) <
                     std::tie(rhs.cache_namespace,
                              rhs.layer_id,
                              rhs.group_id,
                              rhs.logical_tensor,
                              rhs.local_resource_id,
                              rhs.remote_resource_id,
                              rhs.region.local_buffer_id,
                              rhs.region.remote_buffer_id,
                              rhs.region.local_offset,
                              rhs.region.remote_offset);
            });

  std::optional<BoundRegion> previous_domain;
  for (const BoundRegion& item : bound) {
    if (regions->empty()) {
      regions->emplace_back(item.region);
      previous_domain = item;
      continue;
    }
    ByteRegion& previous = regions->back();
    const bool adjacent =
        previous.local_buffer_id == item.region.local_buffer_id &&
        previous.remote_buffer_id == item.region.remote_buffer_id &&
        !add_overflows(previous.local_offset, previous.length) &&
        !add_overflows(previous.remote_offset, previous.length) &&
        previous.local_offset + previous.length == item.region.local_offset &&
        previous.remote_offset + previous.length == item.region.remote_offset;
    if (!previous_domain.has_value() ||
        !same_bound_domain(*previous_domain, item) || !adjacent ||
        add_overflows(previous.length, item.region.length)) {
      regions->emplace_back(item.region);
      previous_domain = item;
      continue;
    }
    previous.length += item.region.length;
    previous_domain = item;
  }
  return Status();
}

}  // namespace

bool ReshardPlanner::source_participates(
    const WorkerCacheLayoutManifest& source,
    const WorkerCacheLayoutManifest& destination) const {
  return supports_partition_pair(source.coordinates, destination.coordinates);
}

Status ReshardPlanner::validate_destination_coverage(
    const std::vector<WorkerCacheLayoutManifest>& sources,
    const WorkerCacheLayoutManifest& destination) const {
  const Status destination_status = validate_worker_cache_layout(destination);
  if (!destination_status.ok()) {
    return invalid("invalid destination layout: " +
                   destination_status.message());
  }
  if (sources.empty()) {
    return invalid("source layout set is empty");
  }
  const Status source_instance_status =
      validate_source_instance(sources, destination);
  if (!source_instance_status.ok()) {
    return source_instance_status;
  }

  std::vector<AtomicLogicalRegion> destination_regions;
  expand_manifest(
      destination, /*only_static_owner=*/false, &destination_regions);
  using PartitionKey = std::pair<int32_t, int32_t>;
  std::map<PartitionKey, std::vector<AtomicLogicalRegion>> by_partition;
  const bool collapse_partitions = !same_partition_sizes(
      sources.front().coordinates, destination.coordinates);
  for (const WorkerCacheLayoutManifest& source : sources) {
    if (!source_participates(source, destination)) {
      continue;
    }
    const Status compatibility = validate_compatibility(source, destination);
    if (!compatibility.ok()) {
      return compatibility;
    }
    const int32_t partition_rank =
        collapse_partitions ? source.coordinates.cp_rank : 0;
    std::vector<AtomicLogicalRegion>& regions =
        by_partition[PartitionKey{source.coordinates.dp_rank, partition_rank}];
    expand_manifest(source, /*only_static_owner=*/true, &regions);
  }

  if (by_partition.empty()) {
    return invalid("no source workers belong to the destination partition");
  }
  const int32_t partitions_per_dp =
      collapse_partitions ? sources.front().coordinates.cp_size : 1;
  const size_t expected_partition_count =
      static_cast<size_t>(sources.front().coordinates.dp_size) *
      static_cast<size_t>(partitions_per_dp);
  if (by_partition.size() != expected_partition_count) {
    return invalid(
        "source DP manifest set is incomplete for destination partition");
  }

  for (const auto& [partition, source_regions] : by_partition) {
    const Status coverage =
        validate_coverage_for_sources(source_regions, destination_regions);
    if (!coverage.ok()) {
      return invalid("source DP rank " + std::to_string(partition.first) +
                     ", CP rank " + std::to_string(partition.second) +
                     " cannot cover destination: " + coverage.message());
    }
  }
  return Status();
}

Status ReshardPlanner::build_outgoing_plan(
    const WorkerCacheLayoutManifest& source,
    const WorkerCacheLayoutManifest& destination,
    ReshardPlanTemplate* plan) const {
  if (plan == nullptr) {
    return invalid("reshard plan output must not be null");
  }
  const Status compatibility = validate_compatibility(source, destination);
  if (!compatibility.ok()) {
    return compatibility;
  }

  plan->source_incarnation = source.incarnation_id;
  plan->source_layout_generation = source.layout_generation;
  plan->destination_incarnation = destination.incarnation_id;
  plan->destination_layout_generation = destination.layout_generation;
  plan->source_coordinates = source.coordinates;
  plan->destination_coordinates = destination.coordinates;
  plan->destination_addr = destination.addr;
  plan->regions.clear();

  std::vector<AtomicLogicalRegion> source_regions;
  std::vector<AtomicLogicalRegion> destination_regions;
  expand_manifest(source, /*only_static_owner=*/true, &source_regions);
  expand_manifest(
      destination, /*only_static_owner=*/false, &destination_regions);
  const RegionGroups source_groups = group_regions(source_regions);
  const RegionGroups destination_groups = group_regions(destination_regions);
  std::vector<PlannedAtomicRegion> candidates;
  for (const auto& [key, sources] : source_groups) {
    const auto destination_it = destination_groups.find(key);
    if (destination_it == destination_groups.end()) {
      continue;
    }
    const Status overlap_status = visit_overlaps(
        sources,
        destination_it->second,
        [&candidates](const AtomicLogicalRegion& source_region,
                      const AtomicLogicalRegion& destination_region) {
          const Status tensor_status =
              validate_tensor_pair(source_region, destination_region);
          if (!tensor_status.ok()) {
            return tensor_status;
          }

          const uint64_t logical_begin = std::max(
              source_region.logical_offset, destination_region.logical_offset);
          const uint64_t overlap_end = std::min(
              logical_end(source_region), logical_end(destination_region));
          PlannedAtomicRegion candidate;
          candidate.logical_offset = logical_begin;
          StridedRegionTemplate& region = candidate.region;
          region.cache_namespace = source_region.cache_namespace;
          region.layer_id = source_region.layer_id;
          region.role = source_region.role;
          region.group_id = source_region.group_id;
          region.logical_tensor = source_region.logical_tensor;
          region.local_buffer_id =
              static_cast<uint64_t>(source_region.tensor->mooncake_buffer_id);
          region.remote_buffer_id = static_cast<uint64_t>(
              destination_region.tensor->mooncake_buffer_id);
          region.local_offset_in_resource =
              source_region.tensor->storage_offset_bytes +
              source_region.physical_offset + logical_begin -
              source_region.logical_offset;
          region.remote_offset_in_resource =
              destination_region.tensor->storage_offset_bytes +
              destination_region.physical_offset + logical_begin -
              destination_region.logical_offset;
          region.bytes_per_region = overlap_end - logical_begin;
          region.local_resource_count = source_region.tensor->resource_count;
          region.remote_resource_count =
              destination_region.tensor->resource_count;
          region.local_resource_stride =
              source_region.tensor->resource_stride_bytes;
          region.remote_resource_stride =
              destination_region.tensor->resource_stride_bytes;
          region.local_buffer_bytes = source_region.tensor->buffer_bytes;
          region.remote_buffer_bytes = destination_region.tensor->buffer_bytes;
          region.local_explicit_resource_offsets =
              source_region.tensor->explicit_resource_offsets;
          region.remote_explicit_resource_offsets =
              destination_region.tensor->explicit_resource_offsets;
          candidates.emplace_back(std::move(candidate));
          return Status();
        });
    if (!overlap_status.ok()) {
      return overlap_status;
    }
  }
  compact_planned_regions(&candidates, &plan->regions);
  return Status();
}

Status RequestRegionBinder::bind(const ReshardPlanTemplate& plan,
                                 const std::vector<KVTransferMapping>& mappings,
                                 CacheNamespace cache_namespace,
                                 int64_t layer_id,
                                 std::vector<ByteRegion>* regions) const {
  std::unordered_map<int32_t, ResourceMappingView> mappings_by_group;
  mappings_by_group.reserve(mappings.size());
  for (const KVTransferMapping& mapping : mappings) {
    if (mapping.local_ids.size() != mapping.remote_ids.size()) {
      return invalid("KV transfer mapping source/destination sizes differ");
    }
    ResourceMappingView view;
    view.local_ids = &mapping.local_ids;
    view.remote_ids = &mapping.remote_ids;
    if (!mappings_by_group.emplace(mapping.group_id, view).second) {
      return invalid("KV transfer mapping contains a duplicate group id");
    }
  }
  return bind_regions(plan,
                      &mappings_by_group,
                      /*explicit_mappings=*/nullptr,
                      cache_namespace,
                      layer_id,
                      regions);
}

Status RequestRegionBinder::bind_explicit(
    const ReshardPlanTemplate& plan,
    const std::vector<ExplicitResourceMapping>& mappings,
    CacheNamespace cache_namespace,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) const {
  std::unordered_map<uint64_t, ResourceMappingView> mappings_by_tensor;
  mappings_by_tensor.reserve(mappings.size());
  for (const ExplicitResourceMapping& mapping : mappings) {
    const size_t count = mapping.local_ids.size();
    if (mapping.remote_ids.size() != count ||
        mapping.local_offsets.size() != count ||
        mapping.remote_offsets.size() != count) {
      return invalid("explicit cache resource mapping sizes differ");
    }
    ResourceMappingView view;
    view.local_ids = &mapping.local_ids;
    view.remote_ids = &mapping.remote_ids;
    view.local_offsets = &mapping.local_offsets;
    view.remote_offsets = &mapping.remote_offsets;
    if (!mappings_by_tensor
             .emplace(explicit_mapping_key(mapping.group_id, mapping.role),
                      view)
             .second) {
      return invalid(
          "explicit cache resource mapping contains a duplicate tensor");
    }
  }
  return bind_regions(plan,
                      /*regular_mappings=*/nullptr,
                      &mappings_by_tensor,
                      cache_namespace,
                      layer_id,
                      regions);
}

}  // namespace xllm
