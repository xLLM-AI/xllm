/* Copyright 2026 The xLLM Authors.

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

#include "core/kv_cache/transfer/reshard_planner.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/kv_cache/layout/kv_cache_tensor_role.h"

namespace xllm {

namespace {

struct AtomicLogicalRegion {
  std::string_view logical_tensor;
  uint64_t logical_offset = 0;
  uint64_t physical_offset = 0;
  uint64_t length = 0;
  const CacheTensorManifest* tensor = nullptr;
};

struct PlannedAtomicRegion {
  uint64_t logical_offset = 0;
  StridedRegionTemplate region;
};

struct BoundRegion {
  ByteRegion region;
  std::string_view logical_tensor;
  int64_t layer_id = 0;
  int32_t group_id = 0;
  uint64_t local_resource_id = 0;
  uint64_t remote_resource_id = 0;
};

using LogicalTensorKey =
    std::tuple<uint32_t, int64_t, int32_t, int32_t, std::string_view>;
using RegionGroup = std::vector<const AtomicLogicalRegion*>;
using RegionGroups = std::map<LogicalTensorKey, RegionGroup>;
using CoverageKey = std::pair<int32_t, int32_t>;

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

bool kv_split_spans_cp_and_tp(const ParallelCoordinates& coordinates) {
  return static_cast<int64_t>(coordinates.kv_split_size) ==
         static_cast<int64_t>(coordinates.cp_size) * coordinates.tp_size;
}

bool supports_kv_split_topology(const ParallelCoordinates& coordinates) {
  return coordinates.cp_size % coordinates.kv_split_size == 0 ||
         (coordinates.cp_size == 1 &&
          coordinates.tp_size % coordinates.kv_split_size == 0) ||
         kv_split_spans_cp_and_tp(coordinates);
}

bool supports_partition_layout(const ParallelCoordinates& source,
                               const ParallelCoordinates& destination) {
  if (same_partition_sizes(source, destination)) {
    return true;
  }
  if (source.cp_size == 1 && source.kv_split_size > 1) {
    return false;
  }
  return destination.cp_size == 1 && destination.cp_rank == 0 &&
         (destination.kv_split_size == 1 ||
          (supports_kv_split_topology(source) &&
           source.kv_split_size == destination.kv_split_size));
}

bool supports_partition_pair(const ParallelCoordinates& source,
                             const ParallelCoordinates& destination) {
  if (same_partition_sizes(source, destination)) {
    return same_partition(source, destination);
  }
  // D can collapse P's CP ranks into TP while retaining the same DCP shard.
  return supports_partition_layout(source, destination) &&
         (destination.kv_split_size == 1 ||
          source.kv_split_rank == destination.kv_split_rank);
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

bool same_logical_span(const LogicalSpan& lhs, const LogicalSpan& rhs) {
  return lhs.logical_tensor == rhs.logical_tensor &&
         lhs.logical_offset_bytes == rhs.logical_offset_bytes &&
         lhs.physical_offset_bytes == rhs.physical_offset_bytes &&
         lhs.bytes_per_region == rhs.bytes_per_region &&
         lhs.repeat_count == rhs.repeat_count &&
         lhs.logical_stride_bytes == rhs.logical_stride_bytes &&
         lhs.physical_stride_bytes == rhs.physical_stride_bytes &&
         lhs.owner_tp_rank == rhs.owner_tp_rank;
}

Status validate_native_pull_tensor(const CacheTensorManifest& tensor) {
  const bool supported_role = tensor.role == KVCacheTensorRole::KEY ||
                              tensor.role == KVCacheTensorRole::VALUE ||
                              tensor.role == KVCacheTensorRole::INDEX;
  if (tensor.cache_pool_id != kPrimaryCachePoolId || !supported_role ||
      !tensor.explicit_resource_offsets || !tensor.contiguous ||
      tensor.physical_rows_per_resource != 1 ||
      tensor.block_token_capacity == 0 ||
      tensor.shard.resource_scope != CacheResourceScope::BLOCK) {
    return invalid(
        "native PULL requires explicit block tensors from the primary pool "
        "with "
        "K/V/INDEX roles");
  }
  return Status();
}

Status validate_native_pull_tensor_pair(const CacheTensorManifest& local,
                                        const CacheTensorManifest& remote) {
  if (local.group_id != remote.group_id ||
      local.scalar_type != remote.scalar_type ||
      local.element_bytes != remote.element_bytes ||
      local.block_token_capacity != remote.block_token_capacity ||
      local.resource_stride_bytes != remote.resource_stride_bytes ||
      local.stride != remote.stride ||
      local.shape.size() != remote.shape.size() ||
      !std::equal(local.shape.begin() + 1,
                  local.shape.end(),
                  remote.shape.begin() + 1) ||
      local.shard.kind != remote.shard.kind ||
      local.shard.spans.size() != remote.shard.spans.size() ||
      !std::equal(local.shard.spans.begin(),
                  local.shard.spans.end(),
                  remote.shard.spans.begin(),
                  same_logical_span)) {
    return invalid("native PULL cache tensor geometry or shard differs");
  }
  return Status();
}

using SpanOwnerKey =
    std::tuple<uint32_t, int64_t, int32_t, int32_t, std::string_view, uint64_t>;

SpanOwnerKey span_owner_key(const CacheTensorManifest& tensor,
                            const LogicalSpan& span) {
  return {tensor.cache_pool_id,
          tensor.layer_id,
          tensor.role,
          tensor.group_id,
          span.logical_tensor,
          span.logical_offset_bytes};
}

using SpanOwnerMap = std::map<SpanOwnerKey, int32_t>;

class SpanWriters final {
 public:
  static SpanWriters include_all_spans() { return SpanWriters(); }

  static SpanWriters skip_span_ordinals(
      std::span<const uint64_t> excluded_span_ordinals) {
    SpanWriters writers;
    writers.excluded_span_ordinals_ = excluded_span_ordinals;
    return writers;
  }

  static SpanWriters use_declared_owners() {
    SpanWriters writers;
    writers.uses_declared_owners_ = true;
    return writers;
  }

  static SpanWriters resolve_candidate_owners(
      const std::vector<WorkerCacheLayoutManifest>& sources,
      const std::vector<size_t>& candidates) {
    struct SpanHolders {
      std::optional<int32_t> declared_owner;
      bool has_conflicting_owners = false;
      bool has_declared_owner = false;
      int32_t lowest_tp_rank = -1;
    };
    std::map<SpanOwnerKey, SpanHolders> holders;
    for (size_t index : candidates) {
      const int32_t tp_rank = sources[index].coordinates.tp_rank;
      for (const CacheTensorManifest& tensor : sources[index].tensors) {
        for (const LogicalSpan& span : tensor.shard.spans) {
          SpanHolders& span_holders = holders[span_owner_key(tensor, span)];
          span_holders.lowest_tp_rank =
              span_holders.lowest_tp_rank < 0
                  ? tp_rank
                  : std::min(span_holders.lowest_tp_rank, tp_rank);
          span_holders.has_declared_owner |= (span.owner_tp_rank == tp_rank);
          if (span.owner_tp_rank < 0) {
            span_holders.has_conflicting_owners = true;
          } else if (!span_holders.declared_owner.has_value()) {
            span_holders.declared_owner = span.owner_tp_rank;
          } else if (*span_holders.declared_owner != span.owner_tp_rank) {
            span_holders.has_conflicting_owners = true;
          }
        }
      }
    }
    SpanWriters writers;
    writers.uses_declared_owners_ = true;
    for (const auto& [key, span_holders] : holders) {
      if (!span_holders.has_conflicting_owners) {
        writers.resolved_owner_ranks_.emplace(
            key,
            span_holders.has_declared_owner ? *span_holders.declared_owner
                                            : span_holders.lowest_tp_rank);
      }
    }
    return writers;
  }

  bool writes_span(const CacheTensorManifest& tensor,
                   const LogicalSpan& span,
                   int32_t tp_rank,
                   uint64_t ordinal) const {
    if (!uses_declared_owners_) {
      return !std::binary_search(excluded_span_ordinals_.begin(),
                                 excluded_span_ordinals_.end(),
                                 ordinal);
    }
    const auto owner = resolved_owner_ranks_.find(span_owner_key(tensor, span));
    return tp_rank == (owner == resolved_owner_ranks_.end() ? span.owner_tp_rank
                                                            : owner->second);
  }

 private:
  bool uses_declared_owners_ = false;
  SpanOwnerMap resolved_owner_ranks_;
  std::span<const uint64_t> excluded_span_ordinals_;
};

SpanWriters source_writers(const ParallelCoordinates& coordinates) {
  return kv_split_spans_cp_and_tp(coordinates)
             ? SpanWriters::include_all_spans()
             : SpanWriters::use_declared_owners();
}

template <typename Visit>
uint64_t visit_manifest_spans(const WorkerCacheLayoutManifest& manifest,
                              Visit visit) {
  uint64_t ordinal = 0;
  for (const CacheTensorManifest& tensor : manifest.tensors) {
    for (const LogicalSpan& span : tensor.shard.spans) {
      visit(tensor, span, ordinal++);
    }
  }
  return ordinal;
}

uint64_t expand_manifest(const WorkerCacheLayoutManifest& manifest,
                         const SpanWriters& writers,
                         std::vector<AtomicLogicalRegion>* regions) {
  return visit_manifest_spans(
      manifest,
      [&](const CacheTensorManifest& tensor,
          const LogicalSpan& span,
          uint64_t ordinal) {
        if (!writers.writes_span(
                tensor, span, manifest.coordinates.tp_rank, ordinal)) {
          return;
        }
        for (uint64_t repeat = 0; repeat < span.repeat_count; ++repeat) {
          AtomicLogicalRegion region;
          region.logical_tensor = span.logical_tensor;
          region.logical_offset =
              span.logical_offset_bytes + repeat * span.logical_stride_bytes;
          region.physical_offset =
              span.physical_offset_bytes + repeat * span.physical_stride_bytes;
          region.length = span.bytes_per_region;
          region.tensor = &tensor;
          regions->emplace_back(std::move(region));
        }
      });
}

LogicalTensorKey logical_tensor_key(const AtomicLogicalRegion& region) {
  return std::make_tuple(region.tensor->cache_pool_id,
                         region.tensor->layer_id,
                         region.tensor->role,
                         region.tensor->group_id,
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
  for (auto& logical_group : groups) {
    RegionGroup& group = logical_group.second;
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
  if (source.tensor->role ==
      static_cast<int32_t>(KVCacheTensorRole::KPOOL_TAIL)) {
    constexpr size_t kKpoolRank = 4;
    constexpr int64_t kKpoolPlanes = 2;
    const CacheTensorManifest& source_tensor = *source.tensor;
    const CacheTensorManifest& destination_tensor = *destination.tensor;
    // Circular rows use position % tail capacity, so neither growth nor
    // truncation preserves their meaning without remapping token positions.
    if (source_tensor.shape.size() != kKpoolRank ||
        destination_tensor.shape.size() != kKpoolRank ||
        source_tensor.shape[1] != kKpoolPlanes ||
        !std::equal(source_tensor.shape.begin() + 1,
                    source_tensor.shape.end(),
                    destination_tensor.shape.begin() + 1) ||
        source_tensor.stride != destination_tensor.stride ||
        source_tensor.physical_rows_per_resource != 1 ||
        destination_tensor.physical_rows_per_resource != 1) {
      return invalid("KPool circular-tail layouts differ");
    }
  }
  return Status();
}

Status validate_coverage_for_sources(
    const std::vector<AtomicLogicalRegion>& source_regions,
    const RegionGroups& destination_groups) {
  const RegionGroups source_groups = group_regions(source_regions);
  for (const auto& [key, destinations] : destination_groups) {
    const auto source_it = source_groups.find(key);
    if (source_it == source_groups.end()) {
      return invalid("destination logical bytes have no source writer");
    }
    const RegionGroup& sources = source_it->second;
    const int32_t tensor_role = std::get<2>(key);
    const bool is_conv =
        tensor_role == static_cast<int32_t>(KVCacheTensorRole::CONV);
    const bool is_linear_state_tensor =
        is_conv || tensor_role == static_cast<int32_t>(KVCacheTensorRole::SSM);
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
    bool compact_linear_source = is_linear_state_tensor;
    int64_t max_source_state_len = std::numeric_limits<int64_t>::min();
    for (const AtomicLogicalRegion* source : sources) {
      if (compact_linear_source) {
        compact_linear_source =
            source->tensor->physical_rows_per_resource == 1 &&
            (!is_conv || source->tensor->shape.size() > 1);
        if (compact_linear_source && is_conv) {
          // The NPU CONV layout is [blocks, state_len, dim].
          max_source_state_len =
              std::max(max_source_state_len, source->tensor->shape[1]);
        }
      }
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
    const uint64_t source_coverage_end = coverage.back().position;

    for (const AtomicLogicalRegion* destination : destinations) {
      const uint64_t destination_end = logical_end(*destination);
      const bool expanded_shape =
          compact_linear_source &&
          (is_conv ? destination->tensor->shape.size() > 1 &&
                         destination->tensor->shape[1] > max_source_state_len
                   : destination->tensor->physical_rows_per_resource > 1);
      const auto coverage_end = coverage.end();
      auto next_point =
          std::upper_bound(coverage.begin(),
                           coverage_end,
                           destination->logical_offset,
                           [](uint64_t position, const CoveragePoint& point) {
                             return position < point.position;
                           });
      writers =
          next_point == coverage.begin() ? 0 : std::prev(next_point)->writers;
      uint64_t cursor = destination->logical_offset;
      while (cursor < destination_end) {
        const uint64_t next_position =
            next_point == coverage_end
                ? destination_end
                : std::min(destination_end, next_point->position);
        if (writers != 1) {
          const bool expanded_linear_destination =
              expanded_shape && writers == 0 && cursor >= source_coverage_end;
          if (!expanded_linear_destination) {
            return invalid(
                writers == 0
                    ? "destination logical bytes have no source writer"
                    : "destination logical bytes have multiple writers");
          }
        }
        cursor = next_position;
        if (cursor < destination_end) {
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
  if (!supports_partition_layout(reference.coordinates,
                                 destination.coordinates)) {
    return invalid(
        "source and destination CP/KV-split partitions are unsupported");
  }

  const ParallelCoordinates& expected = reference.coordinates;
  if (!supports_kv_split_topology(expected)) {
    return invalid(
        "source kv_split_size must divide cp_size, divide tp_size when "
        "cp_size is 1, or equal cp_size * tp_size");
  }
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

bool has_logical_overlap(const WorkerCacheLayoutManifest& source,
                         const RegionGroups& destination_groups,
                         const SpanWriters& writers) {
  std::vector<AtomicLogicalRegion> source_regions;
  expand_manifest(source, writers, &source_regions);
  const RegionGroups source_groups = group_regions(source_regions);
  for (const auto& [key, sources] : source_groups) {
    const auto destination_it = destination_groups.find(key);
    if (destination_it == destination_groups.end()) {
      continue;
    }
    const RegionGroup& destinations = destination_it->second;
    auto source_it = sources.begin();
    auto target_it = destinations.begin();
    while (source_it != sources.end() && target_it != destinations.end()) {
      if (logical_end(**source_it) <= (*target_it)->logical_offset) {
        ++source_it;
      } else if (logical_end(**target_it) <= (*source_it)->logical_offset) {
        ++target_it;
      } else {
        return true;
      }
    }
  }
  return false;
}

Status select_collapsed_writers(
    const std::vector<WorkerCacheLayoutManifest>& sources,
    const WorkerCacheLayoutManifest& destination,
    const RegionGroups& destination_groups,
    std::vector<size_t>* writers,
    std::set<CoverageKey>* required_groups) {
  using CpWorkers = std::map<int32_t, std::vector<size_t>>;
  std::map<CoverageKey, CpWorkers> cp_groups;
  std::map<std::pair<int32_t, int32_t>, int32_t> cp_kv_ranks;
  const bool spans_cp_and_tp =
      kv_split_spans_cp_and_tp(sources.front().coordinates);
  const SpanWriters span_writers = source_writers(sources.front().coordinates);
  for (size_t index = 0; index < sources.size(); ++index) {
    const ParallelCoordinates& coordinates = sources[index].coordinates;
    if (!spans_cp_and_tp) {
      const std::pair<int32_t, int32_t> cp_key = {coordinates.dp_rank,
                                                  coordinates.cp_rank};
      const auto [kv_it, inserted] =
          cp_kv_ranks.emplace(cp_key, coordinates.kv_split_rank);
      if (!inserted && kv_it->second != coordinates.kv_split_rank) {
        return invalid("source CP partition disagrees on KV-split rank");
      }
    }
    cp_groups[{coordinates.dp_rank, coordinates.kv_split_rank}]
             [coordinates.cp_rank]
                 .emplace_back(index);
  }

  const size_t expected_cp_count =
      spans_cp_and_tp
          ? 1
          : static_cast<size_t>(sources.front().coordinates.cp_size /
                                sources.front().coordinates.kv_split_size);
  for (const auto& [group, cp_workers] : cp_groups) {
    if (destination.coordinates.kv_split_size > 1 &&
        group.second != destination.coordinates.kv_split_rank) {
      continue;
    }
    if (cp_workers.size() != expected_cp_count) {
      return invalid("CP replica group has an unexpected partition count");
    }
    required_groups->emplace(group);
    // Lowest CP rank is the deterministic writer; replicas are PLAN_ONLY.
    for (size_t index : cp_workers.begin()->second) {
      if (has_logical_overlap(
              sources[index], destination_groups, span_writers)) {
        writers->emplace_back(index);
      }
    }
  }
  return Status();
}

Status validate_writer_coverage(
    const std::map<CoverageKey, std::vector<AtomicLogicalRegion>>&
        writer_regions,
    const std::set<CoverageKey>& required_groups,
    const RegionGroups& destination_groups) {
  for (const CoverageKey& group : required_groups) {
    const auto writers_it = writer_regions.find(group);
    if (writers_it == writer_regions.end()) {
      return invalid("source DP rank " + std::to_string(group.first) +
                     ", KV-split rank " + std::to_string(group.second) +
                     " cannot cover destination: destination logical bytes "
                     "have no source writer");
    }
    const Status coverage =
        validate_coverage_for_sources(writers_it->second, destination_groups);
    if (!coverage.ok()) {
      return invalid("source DP rank " + std::to_string(group.first) +
                     ", KV-split rank " + std::to_string(group.second) +
                     " cannot cover destination: " + coverage.message());
    }
  }
  return Status();
}

bool same_template_key(const StridedRegionTemplate& lhs,
                       const StridedRegionTemplate& rhs) {
  return lhs.cache_pool_id == rhs.cache_pool_id &&
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
              return std::tie(left.cache_pool_id,
                              left.layer_id,
                              left.role,
                              left.group_id,
                              left.logical_tensor,
                              left.local_buffer_id,
                              left.remote_buffer_id,
                              lhs.logical_offset) <
                     std::tie(right.cache_pool_id,
                              right.layer_id,
                              right.role,
                              right.group_id,
                              right.logical_tensor,
                              right.local_buffer_id,
                              right.remote_buffer_id,
                              rhs.logical_offset);
            });

  uint64_t last_logical_offset = 0;
  for (PlannedAtomicRegion& candidate : *candidates) {
    const uint64_t previous_logical =
        std::exchange(last_logical_offset, candidate.logical_offset);
    if (output->empty() ||
        !same_template_key(output->back(), candidate.region)) {
      output->emplace_back(std::move(candidate.region));
      continue;
    }

    StridedRegionTemplate& previous = output->back();
    if (candidate.logical_offset <= previous_logical ||
        candidate.region.local_offset_in_resource <=
            previous.local_offset_in_resource ||
        candidate.region.remote_offset_in_resource <=
            previous.remote_offset_in_resource) {
      output->emplace_back(std::move(candidate.region));
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
    const bool can_start_repeat = previous.repeat_count == 1;
    if (!can_start_repeat && (local_stride != previous.local_stride ||
                              remote_stride != previous.remote_stride)) {
      output->emplace_back(std::move(candidate.region));
      continue;
    }
    if (can_start_repeat) {
      previous.local_stride = local_stride;
      previous.remote_stride = remote_stride;
    }
    ++previous.repeat_count;
  }
}

bool same_bound_domain(const BoundRegion& lhs, const BoundRegion& rhs) {
  return lhs.logical_tensor == rhs.logical_tensor &&
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
    const std::vector<StridedRegionTemplate>& plan,
    const std::unordered_map<int32_t, ResourceMappingView>* regular_mappings,
    const std::unordered_map<uint64_t, ResourceMappingView>* explicit_mappings,
    uint32_t cache_pool_id,
    int64_t layer_id,
    std::vector<ByteRegion>* regions) {
  if (regions == nullptr) {
    return invalid("bound region output must not be null");
  }
  regions->clear();

  std::vector<BoundRegion> bound;
  for (const StridedRegionTemplate& region_template : plan) {
    if (region_template.cache_pool_id != cache_pool_id ||
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
              return std::tie(lhs.layer_id,
                              lhs.group_id,
                              lhs.logical_tensor,
                              lhs.local_resource_id,
                              lhs.remote_resource_id,
                              lhs.region.local_buffer_id,
                              lhs.region.remote_buffer_id,
                              lhs.region.local_offset,
                              lhs.region.remote_offset) <
                     std::tie(rhs.layer_id,
                              rhs.group_id,
                              rhs.logical_tensor,
                              rhs.local_resource_id,
                              rhs.remote_resource_id,
                              rhs.region.local_buffer_id,
                              rhs.region.remote_buffer_id,
                              rhs.region.local_offset,
                              rhs.region.remote_offset);
            });

  if (bound.empty()) {
    return Status();
  }
  regions->emplace_back(bound.front().region);
  for (size_t index = 1; index < bound.size(); ++index) {
    const BoundRegion& item = bound[index];
    ByteRegion& previous = regions->back();
    if (same_bound_domain(bound[index - 1], item) &&
        previous.local_buffer_id == item.region.local_buffer_id &&
        previous.remote_buffer_id == item.region.remote_buffer_id &&
        previous.local_offset + previous.length == item.region.local_offset &&
        previous.remote_offset + previous.length == item.region.remote_offset) {
      previous.length += item.region.length;
    } else {
      regions->emplace_back(item.region);
    }
  }
  return Status();
}

}  // namespace

Status ReshardPlanner::select_sources(
    const std::vector<WorkerCacheLayoutManifest>& sources,
    const WorkerCacheLayoutManifest& destination,
    std::vector<size_t>* selected_indices,
    std::vector<std::vector<uint64_t>>* excluded_spans) const {
  if (selected_indices == nullptr) {
    return invalid("selected source index output must not be null");
  }
  selected_indices->clear();
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
      destination, SpanWriters::include_all_spans(), &destination_regions);
  const RegionGroups destination_groups = group_regions(destination_regions);
  const bool collapse_partitions = !same_partition_sizes(
      sources.front().coordinates, destination.coordinates);
  SpanWriters homogeneous_writers;
  std::vector<size_t> writers;
  std::set<CoverageKey> required_groups;
  if (collapse_partitions) {
    const Status selection = select_collapsed_writers(
        sources, destination, destination_groups, &writers, &required_groups);
    if (!selection.ok()) {
      return selection;
    }
  } else {
    std::vector<size_t> homogeneous_source_indices;
    for (size_t index = 0; index < sources.size(); ++index) {
      const ParallelCoordinates& coordinates = sources[index].coordinates;
      if (!same_partition(coordinates, destination.coordinates)) {
        continue;
      }
      required_groups.emplace(coordinates.dp_rank, 0);
      homogeneous_source_indices.emplace_back(index);
    }
    homogeneous_writers = SpanWriters::resolve_candidate_owners(
        sources, homogeneous_source_indices);
    for (size_t index : homogeneous_source_indices) {
      if (has_logical_overlap(
              sources[index], destination_groups, homogeneous_writers)) {
        writers.emplace_back(index);
      }
    }
  }

  const SpanWriters collapsed_writers =
      source_writers(sources.front().coordinates);
  std::map<CoverageKey, std::vector<AtomicLogicalRegion>> writer_regions;
  for (size_t index : writers) {
    const WorkerCacheLayoutManifest& source = sources[index];
    const Status compatibility = validate_compatibility(source, destination);
    if (!compatibility.ok()) {
      return compatibility;
    }
    const int32_t kv_partition =
        collapse_partitions ? source.coordinates.kv_split_rank : 0;
    expand_manifest(
        source,
        collapse_partitions ? collapsed_writers : homogeneous_writers,
        &writer_regions[{source.coordinates.dp_rank, kv_partition}]);
  }

  const Status coverage = validate_writer_coverage(
      writer_regions, required_groups, destination_groups);
  if (!coverage.ok()) {
    return coverage;
  }
  if (excluded_spans != nullptr) {
    excluded_spans->assign(sources.size(), {});
    for (size_t index : writers) {
      const WorkerCacheLayoutManifest& source = sources[index];
      const SpanWriters& span_writers =
          collapse_partitions ? collapsed_writers : homogeneous_writers;
      visit_manifest_spans(
          source,
          [&](const CacheTensorManifest& tensor,
              const LogicalSpan& span,
              uint64_t ordinal) {
            if (!span_writers.writes_span(
                    tensor, span, source.coordinates.tp_rank, ordinal)) {
              (*excluded_spans)[index].emplace_back(ordinal);
            }
          });
    }
  }
  selected_indices->swap(writers);
  return Status();
}

Status ReshardPlanner::build_outgoing_plan(
    const WorkerCacheLayoutManifest& source,
    const WorkerCacheLayoutManifest& destination,
    std::vector<StridedRegionTemplate>* plan,
    std::optional<std::span<const uint64_t>> excluded_source_spans) const {
  if (plan == nullptr) {
    return invalid("reshard plan output must not be null");
  }
  const Status compatibility = validate_compatibility(source, destination);
  if (!compatibility.ok()) {
    return compatibility;
  }

  plan->clear();

  std::vector<AtomicLogicalRegion> source_regions;
  std::vector<AtomicLogicalRegion> destination_regions;
  const SpanWriters writers =
      excluded_source_spans
          ? SpanWriters::skip_span_ordinals(*excluded_source_spans)
          : source_writers(source.coordinates);
  if (excluded_source_spans &&
      std::adjacent_find(excluded_source_spans->begin(),
                         excluded_source_spans->end(),
                         std::greater_equal<uint64_t>()) !=
          excluded_source_spans->end()) {
    return invalid("excluded source spans must be strictly increasing");
  }
  const uint64_t num_spans = expand_manifest(source, writers, &source_regions);
  if (excluded_source_spans && !excluded_source_spans->empty() &&
      excluded_source_spans->back() >= num_spans) {
    return invalid("excluded source span is outside layout");
  }
  expand_manifest(
      destination, SpanWriters::include_all_spans(), &destination_regions);
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
          region.cache_pool_id = source_region.tensor->cache_pool_id;
          region.layer_id = source_region.tensor->layer_id;
          region.role = source_region.tensor->role;
          region.group_id = source_region.tensor->group_id;
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
  compact_planned_regions(&candidates, plan);
  return Status();
}

Status ReshardPlanner::build_native_pull_plan(
    const WorkerCacheLayoutManifest& local_destination,
    const WorkerCacheLayoutManifest& remote_source,
    std::vector<StridedRegionTemplate>* plan) const {
  if (plan == nullptr) {
    return invalid("native PULL plan output must not be null");
  }
  const Status compatibility =
      validate_compatibility(remote_source, local_destination);
  if (!compatibility.ok()) {
    return compatibility;
  }
  const ParallelCoordinates& local = local_destination.coordinates;
  const ParallelCoordinates& remote = remote_source.coordinates;
  if (local.dp_size != remote.dp_size || local.tp_size != remote.tp_size ||
      local.tp_rank != remote.tp_rank || !same_partition(local, remote)) {
    return invalid(
        "native PULL requires matching DP/TP sizes and TP/CP/KV-split "
        "ranks");
  }
  if (local_destination.tensors.size() != remote_source.tensors.size()) {
    return invalid("native PULL source and destination tensor roles differ");
  }

  std::unordered_map<int64_t, std::array<const CacheTensorManifest*, 3>>
      remote_tensors;
  remote_tensors.reserve(remote_source.tensors.size());
  for (const CacheTensorManifest& tensor : remote_source.tensors) {
    const Status tensor_status = validate_native_pull_tensor(tensor);
    if (!tensor_status.ok()) {
      return tensor_status;
    }
    const CacheTensorManifest*& slot =
        remote_tensors[tensor.layer_id][static_cast<size_t>(tensor.role)];
    if (slot != nullptr) {
      return invalid(
          "native PULL requires a single cache group per layer role");
    }
    slot = &tensor;
  }
  for (const auto& layer : remote_tensors) {
    const auto& roles = layer.second;
    if (roles[KVCacheTensorRole::KEY] == nullptr ||
        roles[KVCacheTensorRole::VALUE] == nullptr) {
      return invalid("native PULL requires both K and V tensors in each layer");
    }
  }

  std::vector<StridedRegionTemplate> candidate;
  candidate.reserve(local_destination.tensors.size());
  for (const CacheTensorManifest& local_tensor : local_destination.tensors) {
    const Status tensor_status = validate_native_pull_tensor(local_tensor);
    if (!tensor_status.ok()) {
      return tensor_status;
    }
    const auto remote_it = remote_tensors.find(local_tensor.layer_id);
    if (remote_it == remote_tensors.end() ||
        remote_it->second[static_cast<size_t>(local_tensor.role)] == nullptr) {
      return invalid("native PULL source and destination tensor roles differ");
    }
    const CacheTensorManifest& remote_tensor =
        *remote_it->second[static_cast<size_t>(local_tensor.role)];
    const Status pair_status =
        validate_native_pull_tensor_pair(local_tensor, remote_tensor);
    if (!pair_status.ok()) {
      return pair_status;
    }

    StridedRegionTemplate region;
    region.cache_pool_id = local_tensor.cache_pool_id;
    region.layer_id = local_tensor.layer_id;
    region.role = local_tensor.role;
    region.group_id = local_tensor.group_id;
    region.logical_tensor =
        KVCacheTensorRole(static_cast<KVCacheTensorRole::Value>(region.role))
            .to_string();
    region.local_buffer_id =
        static_cast<uint64_t>(local_tensor.mooncake_buffer_id);
    region.remote_buffer_id =
        static_cast<uint64_t>(remote_tensor.mooncake_buffer_id);
    region.local_offset_in_resource = local_tensor.storage_offset_bytes;
    region.remote_offset_in_resource = remote_tensor.storage_offset_bytes;
    region.bytes_per_region = local_tensor.resource_stride_bytes;
    region.local_resource_count = local_tensor.resource_count;
    region.remote_resource_count = remote_tensor.resource_count;
    region.local_resource_stride = local_tensor.resource_stride_bytes;
    region.remote_resource_stride = remote_tensor.resource_stride_bytes;
    region.local_buffer_bytes = local_tensor.buffer_bytes;
    region.remote_buffer_bytes = remote_tensor.buffer_bytes;
    region.local_explicit_resource_offsets = true;
    region.remote_explicit_resource_offsets = true;
    candidate.emplace_back(std::move(region));
  }
  *plan = std::move(candidate);
  return Status();
}

Status RequestRegionBinder::bind(const std::vector<StridedRegionTemplate>& plan,
                                 const std::vector<KVTransferMapping>& mappings,
                                 uint32_t cache_pool_id,
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
                      cache_pool_id,
                      layer_id,
                      regions);
}

Status RequestRegionBinder::bind_explicit(
    const std::vector<StridedRegionTemplate>& plan,
    const std::vector<ExplicitResourceMapping>& mappings,
    uint32_t cache_pool_id,
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
                      cache_pool_id,
                      layer_id,
                      regions);
}

}  // namespace xllm
