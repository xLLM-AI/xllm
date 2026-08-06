/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/kv_cache_transfer/spec_kv_cache_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <numeric>
#include <optional>

#include "common/macros.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "util/timer.h"

namespace xllm {

namespace {

constexpr int64_t kSupportedHeterogeneousSourceShardCount = 2;
constexpr int64_t kKeyValueShardDimension = 2;

std::optional<int32_t> get_remote_tp_size(
    const std::vector<TransferKVInfo>& transfer_kv_infos) {
  for (const TransferKVInfo& info : transfer_kv_infos) {
    const int32_t remote_dp_size = info.remote_instance_info.dp_size;
    const size_t remote_world_size =
        info.remote_instance_info.cluster_ids.size();
    if (remote_dp_size <= 0 || remote_world_size == 0 ||
        remote_world_size % static_cast<size_t>(remote_dp_size) != 0) {
      continue;
    }
    return static_cast<int32_t>(remote_world_size /
                                static_cast<size_t>(remote_dp_size));
  }
  return std::nullopt;
}

const KVTransferMapping* find_mapping(
    const std::vector<KVTransferMapping>& mappings,
    int32_t group_id) {
  const auto it = std::find_if(mappings.begin(),
                               mappings.end(),
                               [group_id](const KVTransferMapping& mapping) {
                                 return mapping.group_id == group_id;
                               });
  return it == mappings.end() ? nullptr : &*it;
}

void append_mappings(std::vector<KVTransferMapping>& destination,
                     const std::vector<KVTransferMapping>& source) {
  for (const KVTransferMapping& source_mapping : source) {
    auto destination_it =
        std::find_if(destination.begin(),
                     destination.end(),
                     [&source_mapping](const KVTransferMapping& mapping) {
                       return mapping.group_id == source_mapping.group_id;
                     });
    if (destination_it == destination.end()) {
      destination.emplace_back(source_mapping);
      continue;
    }
    destination_it->local_ids.insert(destination_it->local_ids.end(),
                                     source_mapping.local_ids.begin(),
                                     source_mapping.local_ids.end());
    destination_it->remote_ids.insert(destination_it->remote_ids.end(),
                                      source_mapping.remote_ids.begin(),
                                      source_mapping.remote_ids.end());
  }
}

void merge_heterogeneous_kv_blocks(
    std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo>& merged,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    int32_t source_shard_rank) {
  for (const TransferKVInfo& info : transfer_kv_infos) {
    const int32_t dst_dp_size = info.remote_instance_info.dp_size;
    const size_t dst_world_size = info.remote_instance_info.cluster_ids.size();
    CHECK_GT(dst_dp_size, 0);
    CHECK_EQ(dst_world_size % static_cast<size_t>(dst_dp_size), 0);
    const int32_t dst_tp_size =
        static_cast<int32_t>(dst_world_size / dst_dp_size);
    CHECK_GT(dst_tp_size, 0);
    CHECK_GE(info.dp_rank, 0);
    CHECK_LT(info.dp_rank, dst_dp_size);

    // Every Prefill head shard must reach a Decode worker. The generic
    // merge_kv_blocks() route is one-to-one in TP rank and drops source ranks
    // whose rank is >= dst_tp_size (for TP2 -> TP1, rank 1 is dropped). Map
    // source shards modulo the destination TP group instead; staging row
    // ranges keep multiple source shards disjoint on the selected worker.
    const int32_t dst_rank =
        info.dp_rank * dst_tp_size + source_shard_rank % dst_tp_size;
    CHECK_LT(static_cast<size_t>(dst_rank), dst_world_size);
    const uint64_t dst_cluster_id =
        info.remote_instance_info.cluster_ids[dst_rank];
    const std::string& dst_addr = info.remote_instance_info.addrs[dst_rank];
    const std::string key = std::to_string(dst_cluster_id) + "_" + dst_addr;
    auto& kv_info = merged[key];
    kv_info.dst_cluster_id = dst_cluster_id;
    kv_info.dst_addr = dst_addr;
    append_mappings(kv_info.mappings, info.mappings);
  }
}

bool is_linear_state_cache(KVCacheTensorRole role) {
  return role == KVCacheTensorRole::CONV || role == KVCacheTensorRole::SSM;
}

int64_t sharded_dimension(KVCacheTensorRole role, const torch::Tensor& tensor) {
  if (role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE) {
    return kKeyValueShardDimension;
  }
  if (role == KVCacheTensorRole::CONV) {
    return tensor.dim() - 1;
  }
  if (role == KVCacheTensorRole::SSM) {
    return 1;
  }
  return -1;
}

int64_t get_checkpoint_stride(
    const std::vector<RegisteredCache>& layer_caches) {
  const RegisteredCache* conv_cache = nullptr;
  const RegisteredCache* ssm_cache = nullptr;
  for (const RegisteredCache& cache : layer_caches) {
    if (!cache.tensor.defined()) {
      continue;
    }
    if (cache.role == KVCacheTensorRole::CONV) {
      conv_cache = &cache;
    } else if (cache.role == KVCacheTensorRole::SSM) {
      ssm_cache = &cache;
    }
  }
  if (conv_cache == nullptr || ssm_cache == nullptr) {
    return 1;
  }
  CHECK_GT(conv_cache->tensor.size(0), 0);
  CHECK_EQ(ssm_cache->tensor.size(0) % conv_cache->tensor.size(0), 0);
  return ssm_cache->tensor.size(0) / conv_cache->tensor.size(0);
}

std::vector<uint64_t> make_compact_ids(size_t count) {
  std::vector<uint64_t> ids(count);
  std::iota(ids.begin(), ids.end(), 0);
  return ids;
}

std::vector<uint64_t> expand_checkpoint_ids(
    const std::vector<uint64_t>& logical_ids,
    int64_t checkpoint_stride) {
  std::vector<uint64_t> ids;
  ids.reserve(logical_ids.size() * static_cast<size_t>(checkpoint_stride));
  for (uint64_t logical_id : logical_ids) {
    for (int64_t checkpoint = 0; checkpoint < checkpoint_stride; ++checkpoint) {
      ids.push_back(logical_id * static_cast<uint64_t>(checkpoint_stride) +
                    static_cast<uint64_t>(checkpoint));
    }
  }
  return ids;
}

std::vector<torch::Tensor> make_staging_shard_views(
    const torch::Tensor& staging_tensor,
    int64_t source_shard_count,
    int64_t row_count) {
  CHECK_GT(source_shard_count, 0);
  CHECK_EQ(staging_tensor.size(0) % source_shard_count, 0);
  const int64_t rows_per_shard = staging_tensor.size(0) / source_shard_count;
  CHECK_LE(row_count, rows_per_shard);
  std::vector<torch::Tensor> shard_tensors;
  shard_tensors.reserve(static_cast<size_t>(source_shard_count));
  for (int64_t shard = 0; shard < source_shard_count; ++shard) {
    shard_tensors.emplace_back(
        staging_tensor.narrow(0, shard * rows_per_shard, row_count));
  }
  return shard_tensors;
}

torch::Tensor merge_shard_tensors(
    KVCacheTensorRole role,
    const torch::Tensor& staging_tensor,
    const std::vector<RegisteredCache>& layer_staging_caches,
    const std::vector<torch::Tensor>& shard_tensors,
    int64_t shard_dim,
    int64_t layer_id) {
  if (role != KVCacheTensorRole::CONV) {
    return torch::cat(shard_tensors, shard_dim);
  }

  int64_t local_v_width = -1;
  for (const RegisteredCache& candidate : layer_staging_caches) {
    if (candidate.role == KVCacheTensorRole::SSM &&
        candidate.tensor.defined()) {
      local_v_width = candidate.tensor.size(1) * candidate.tensor.size(3);
      break;
    }
  }
  CHECK_GT(local_v_width, 0)
      << "CONV cache requires a matching SSM cache at layer " << layer_id;
  const int64_t local_conv_width = staging_tensor.size(shard_dim);
  CHECK_EQ((local_conv_width - local_v_width) % 2, 0)
      << "Invalid Q/K/V CONV cache layout at layer " << layer_id;
  const int64_t local_qk_width = (local_conv_width - local_v_width) / 2;
  std::vector<torch::Tensor> q_shards;
  std::vector<torch::Tensor> k_shards;
  std::vector<torch::Tensor> v_shards;
  q_shards.reserve(shard_tensors.size());
  k_shards.reserve(shard_tensors.size());
  v_shards.reserve(shard_tensors.size());
  for (const torch::Tensor& shard : shard_tensors) {
    std::vector<torch::Tensor> qkv = torch::split_with_sizes(
        shard, {local_qk_width, local_qk_width, local_v_width}, shard_dim);
    q_shards.emplace_back(qkv[0]);
    k_shards.emplace_back(qkv[1]);
    v_shards.emplace_back(qkv[2]);
  }
  return torch::cat({torch::cat(q_shards, shard_dim),
                     torch::cat(k_shards, shard_dim),
                     torch::cat(v_shards, shard_dim)},
                    shard_dim);
}

void copy_merged_cache_rows(const RegisteredCache& registered_cache,
                            const std::vector<uint64_t>& final_ids,
                            const torch::Tensor& merged) {
  std::vector<int64_t> signed_final_ids(final_ids.begin(), final_ids.end());
  torch::Tensor final_indices =
      torch::tensor(signed_final_ids,
                    torch::TensorOptions().dtype(torch::kLong))
          .to(registered_cache.tensor.device(), /*non_blocking=*/false);
  registered_cache.tensor.index_copy_(0, final_indices, merged);
}

torch::Tensor make_page_aligned_staging_tensor(
    std::vector<int64_t>& shape,
    const torch::TensorOptions& options,
    int64_t element_size) {
  constexpr int64_t kHcclPageSize = 2 * 1024 * 1024;
  int64_t row_elements = 1;
  for (size_t dim = 1; dim < shape.size(); ++dim) {
    row_elements *= shape[dim];
  }
  const int64_t row_bytes = row_elements * element_size;
  const int64_t rows_per_alignment =
      kHcclPageSize / std::gcd(kHcclPageSize, row_bytes);
  shape[0] = ((shape[0] + rows_per_alignment - 1) / rows_per_alignment) *
             rows_per_alignment;

  const int64_t aligned_numel = std::accumulate(
      shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  const int64_t padding_elements = kHcclPageSize / element_size;
  torch::Tensor backing =
      torch::empty({aligned_numel + padding_elements}, options);
  const uintptr_t base = reinterpret_cast<uintptr_t>(backing.data_ptr());
  const uintptr_t aligned = (base + kHcclPageSize - 1) & ~(kHcclPageSize - 1);
  const int64_t offset_elements =
      static_cast<int64_t>(aligned - base) / element_size;
  torch::Tensor stage =
      backing.narrow(0, offset_elements, aligned_numel).view(shape);
  CHECK_EQ(reinterpret_cast<uintptr_t>(stage.data_ptr()) % kHcclPageSize, 0);
  CHECK_EQ(stage.numel() * element_size % kHcclPageSize, 0);
  return stage;
}

}  // namespace

SpecKVCacheTransfer::SpecKVCacheTransfer(const uint16_t listen_port,
                                         const InstanceRole& instance_role,
                                         bool enable_lighting_indexer,
                                         bool enable_mla,
                                         bool draft_body_uses_tp1)
    : LlmDataDistTransfer(listen_port, instance_role, enable_lighting_indexer) {
  enable_mla_ = enable_mla;
  draft_body_uses_tp1_ = draft_body_uses_tp1;
  heterogeneous_pd_enabled_ =
      DisaggPDConfig::get_instance().enable_heterogeneous_pd();
  parallel_shard_pull_ =
      DisaggPDConfig::get_instance().enable_pd_parallel_shard_pull();
  if (heterogeneous_pd_enabled_ && parallel_shard_pull_) {
    shard_pull_threadpool_ = std::make_unique<ThreadPool>(
        /*num_threads=*/1,
        /*cpu_binding=*/false,
        /*pool_name=*/"SpecKVCacheTransfer.shard_pull");
  }
}

void SpecKVCacheTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  register_kv_cache_internal(kv_caches, layer_registered_caches_);
}

void SpecKVCacheTransfer::register_kv_cache_spec(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  register_kv_cache_internal(kv_caches, spec_layer_registered_caches_);
  if (!heterogeneous_pd_enabled_) {
    return;
  }
  // Register matching staging cache IDs on both Prefill and Decode before the
  // first DataDist link. Prefill pushes each local TP shard into a disjoint row
  // range; Decode merges those already-local rows into its TP1 cache.
  const bool source_is_sharded = role_ == LlmRole::kPrompt;
  register_hetero_staging_caches(layer_registered_caches_,
                                 hetero_staging_registered_caches_,
                                 kSupportedHeterogeneousSourceShardCount,
                                 source_is_sharded);
  if (!draft_body_uses_tp1_) {
    register_hetero_staging_caches(spec_layer_registered_caches_,
                                   spec_hetero_staging_registered_caches_,
                                   kSupportedHeterogeneousSourceShardCount,
                                   source_is_sharded);
  }
}

bool SpecKVCacheTransfer::pull_and_merge_sharded_caches(
    const LayerRegisteredCaches& layer_registered_caches,
    const LayerRegisteredCaches& staging_registered_caches,
    const std::vector<uint64_t>& src_cluster_ids,
    const std::vector<KVTransferMapping>& mappings,
    bool sequence_scoped) {
  if (src_cluster_ids.size() !=
          static_cast<size_t>(kSupportedHeterogeneousSourceShardCount) ||
      layer_registered_caches.size() != staging_registered_caches.size()) {
    LOG(ERROR) << "Invalid heterogeneous KV pull metadata: src_tp="
               << src_cluster_ids.size()
               << ", mapping_count=" << mappings.size();
    return false;
  }
  for (const KVTransferMapping& mapping : mappings) {
    if (mapping.remote_ids.size() != mapping.local_ids.size()) {
      LOG(ERROR) << "Invalid heterogeneous KV mapping size: group_id="
                 << mapping.group_id << ", remote=" << mapping.remote_ids.size()
                 << ", local=" << mapping.local_ids.size();
      return false;
    }
  }

  const int64_t shard_count = static_cast<int64_t>(src_cluster_ids.size());
  Timer breakdown_total_timer;
  double pull_seconds = 0.0;
  double pull_wall_seconds = 0.0;
  double merge_seconds = 0.0;
  double conv_pull_seconds = 0.0;
  double conv_merge_seconds = 0.0;
  double ssm_pull_seconds = 0.0;
  double ssm_merge_seconds = 0.0;
  std::vector<double> shard_pull_seconds(src_cluster_ids.size(), 0.0);
  size_t pull_calls = 0;
  size_t merge_calls = 0;
  bool success = true;
  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(layer_registered_caches.size());
       ++layer_id) {
    const auto& layer_caches = layer_registered_caches[layer_id];
    const auto& layer_staging_caches = staging_registered_caches[layer_id];
    if (layer_caches.size() != layer_staging_caches.size()) {
      LOG(ERROR) << "Heterogeneous KV staging layout mismatch at layer "
                 << layer_id;
      return false;
    }
    const int64_t checkpoint_stride = get_checkpoint_stride(layer_caches);

    for (size_t cache_index = 0; cache_index < layer_caches.size();
         ++cache_index) {
      const RegisteredCache& registered_cache = layer_caches[cache_index];
      const RegisteredCache& stage_cache = layer_staging_caches[cache_index];
      if (registered_cache.sequence_scoped != sequence_scoped) {
        continue;
      }
      const KVTransferMapping* mapping =
          find_mapping(mappings, registered_cache.group_id);
      if (mapping == nullptr) {
        LOG(ERROR) << "Missing heterogeneous KV mapping, layer=" << layer_id
                   << ", role=" << registered_cache.role.to_string()
                   << ", group_id=" << registered_cache.group_id;
        return false;
      }
      const int64_t shard_dim =
          sharded_dimension(registered_cache.role, registered_cache.tensor);
      if (shard_dim < 0) {
        LOG(ERROR) << "Unsupported heterogeneous KV tensor role: "
                   << registered_cache.role.to_string();
        return false;
      }
      CHECK_EQ(registered_cache.tensor.size(shard_dim) % shard_count, 0)
          << "Cache shard dimension is not divisible by source shard count, "
          << "layer=" << layer_id
          << ", role=" << registered_cache.role.to_string()
          << ", shape=" << registered_cache.tensor.sizes();

      std::vector<uint64_t> remote_ids = mapping->remote_ids;
      std::vector<uint64_t> final_ids = mapping->local_ids;
      if (registered_cache.role == KVCacheTensorRole::SSM) {
        remote_ids = expand_checkpoint_ids(remote_ids, checkpoint_stride);
        final_ids = expand_checkpoint_ids(final_ids, checkpoint_stride);
      }
      if (remote_ids.empty()) {
        continue;
      }

      if (stage_cache.tensor.size(0) <
          static_cast<int64_t>(remote_ids.size())) {
        LOG(ERROR) << "Heterogeneous KV transfer exceeds staging capacity, "
                   << "layer=" << layer_id
                   << ", role=" << registered_cache.role.to_string()
                   << ", requested=" << remote_ids.size()
                   << ", capacity=" << stage_cache.tensor.size(0);
        return false;
      }
      CHECK_EQ(stage_cache.tensor.size(0) % shard_count, 0);
      const int64_t rows_per_shard = stage_cache.tensor.size(0) / shard_count;
      CHECK_LE(static_cast<int64_t>(remote_ids.size()), rows_per_shard);

      std::vector<llm_datadist::Status> pull_rets(src_cluster_ids.size(),
                                                  LLM_SUCCESS);
      std::vector<double> current_pull_seconds(src_cluster_ids.size(), 0.0);
      auto pull_one_shard = [&](size_t source_shard) {
        const uint64_t src_cluster_id = src_cluster_ids[source_shard];
        std::vector<uint64_t> staging_ids = make_compact_ids(remote_ids.size());
        const uint64_t staging_offset =
            static_cast<uint64_t>(source_shard * rows_per_shard);
        for (uint64_t& staging_id : staging_ids) {
          staging_id += staging_offset;
        }
        CacheIndex src_cache_index{src_cluster_id,
                                   registered_cache.cache.cache_id};
        KvCacheExtParam ext_param{};
        ext_param.src_layer_range = {0, 0};
        ext_param.dst_layer_range = {0, 0};
        ext_param.tensor_num_per_layer = 1;
        Timer pull_timer;
        pull_rets[source_shard] =
            llm_data_dist_->PullKvBlocks(src_cache_index,
                                         stage_cache.cache,
                                         remote_ids,
                                         staging_ids,
                                         ext_param);
        current_pull_seconds[source_shard] = pull_timer.elapsed_seconds();
      };

      Timer pull_group_timer;
      if (parallel_shard_pull_) {
        CHECK(shard_pull_threadpool_ != nullptr);
        TaskGroup shard_pull_group(1);
        shard_pull_threadpool_->schedule(
            shard_pull_group.wrap([&]() { pull_one_shard(1); }));
        std::exception_ptr request_thread_exception;
        try {
          pull_one_shard(0);
        } catch (...) {
          request_thread_exception = std::current_exception();
        }
        shard_pull_group.wait();
        if (request_thread_exception) {
          std::rethrow_exception(request_thread_exception);
        }
      } else {
        for (size_t source_shard = 0; source_shard < src_cluster_ids.size();
             ++source_shard) {
          pull_one_shard(source_shard);
        }
      }
      pull_wall_seconds += pull_group_timer.elapsed_seconds();

      for (size_t source_shard = 0; source_shard < src_cluster_ids.size();
           ++source_shard) {
        const uint64_t src_cluster_id = src_cluster_ids[source_shard];
        pull_seconds += current_pull_seconds[source_shard];
        shard_pull_seconds[source_shard] += current_pull_seconds[source_shard];
        ++pull_calls;
        if (registered_cache.role == KVCacheTensorRole::CONV) {
          conv_pull_seconds += current_pull_seconds[source_shard];
        } else if (registered_cache.role == KVCacheTensorRole::SSM) {
          ssm_pull_seconds += current_pull_seconds[source_shard];
        }
        if (pull_rets[source_shard] != LLM_SUCCESS) {
          LOG(ERROR) << "Heterogeneous PullKvBlocks failed, layer=" << layer_id
                     << ", role=" << registered_cache.role.to_string()
                     << ", src_cluster_id=" << src_cluster_id
                     << ", src_cache_id=" << registered_cache.cache.cache_id
                     << ", ret=" << std::hex << pull_rets[source_shard];
          success = false;
          break;
        }
      }

      if (success) {
        Timer merge_timer;
        const std::vector<torch::Tensor> shard_tensors =
            make_staging_shard_views(stage_cache.tensor,
                                     shard_count,
                                     static_cast<int64_t>(remote_ids.size()));
        const torch::Tensor merged = merge_shard_tensors(registered_cache.role,
                                                         stage_cache.tensor,
                                                         layer_staging_caches,
                                                         shard_tensors,
                                                         shard_dim,
                                                         layer_id);
        copy_merged_cache_rows(registered_cache, final_ids, merged);
        const double current_merge_seconds = merge_timer.elapsed_seconds();
        merge_seconds += current_merge_seconds;
        ++merge_calls;
        if (registered_cache.role == KVCacheTensorRole::CONV) {
          conv_merge_seconds += current_merge_seconds;
        } else if (registered_cache.role == KVCacheTensorRole::SSM) {
          ssm_merge_seconds += current_merge_seconds;
        }
      }
      if (!success) {
        return false;
      }
    }
  }
  VLOG(1) << "Heterogeneous pull-merge breakdown"
          << " sequence_scoped=" << sequence_scoped
          << " parallel_shard_pull=" << parallel_shard_pull_
          << " pull_calls=" << pull_calls << " merge_calls=" << merge_calls
          << " shard0_pull_ms=" << shard_pull_seconds[0] * 1000.0
          << " shard1_pull_ms=" << shard_pull_seconds[1] * 1000.0
          << " conv_pull_ms=" << conv_pull_seconds * 1000.0
          << " ssm_pull_ms=" << ssm_pull_seconds * 1000.0
          << " pull_work_ms=" << pull_seconds * 1000.0
          << " pull_ms=" << pull_wall_seconds * 1000.0
          << " conv_merge_ms=" << conv_merge_seconds * 1000.0
          << " ssm_merge_ms=" << ssm_merge_seconds * 1000.0
          << " merge_ms=" << merge_seconds * 1000.0 << " other_ms="
          << (breakdown_total_timer.elapsed_seconds() - pull_wall_seconds -
              merge_seconds) *
                 1000.0
          << " total_ms=" << breakdown_total_timer.elapsed_seconds() * 1000.0;
  return true;
}

bool SpecKVCacheTransfer::merge_pre_pushed_sharded_caches(
    const LayerRegisteredCaches& layer_registered_caches,
    const LayerRegisteredCaches& staging_registered_caches,
    const std::vector<KVTransferMapping>& mappings,
    int64_t source_shard_count,
    bool sequence_scoped) {
  if (source_shard_count != kSupportedHeterogeneousSourceShardCount ||
      layer_registered_caches.size() != staging_registered_caches.size()) {
    LOG(ERROR) << "Invalid pre-pushed heterogeneous KV layout: source_tp="
               << source_shard_count;
    return false;
  }

  for (size_t layer_id = 0; layer_id < layer_registered_caches.size();
       ++layer_id) {
    const auto& layer_caches = layer_registered_caches[layer_id];
    const auto& layer_staging_caches = staging_registered_caches[layer_id];
    if (layer_caches.size() != layer_staging_caches.size()) {
      LOG(ERROR) << "Pre-pushed KV staging layout mismatch at layer "
                 << layer_id;
      return false;
    }
    const int64_t checkpoint_stride = get_checkpoint_stride(layer_caches);

    for (size_t cache_index = 0; cache_index < layer_caches.size();
         ++cache_index) {
      const RegisteredCache& registered_cache = layer_caches[cache_index];
      const RegisteredCache& stage_cache = layer_staging_caches[cache_index];
      if (registered_cache.sequence_scoped != sequence_scoped) {
        continue;
      }
      const KVTransferMapping* mapping =
          find_mapping(mappings, registered_cache.group_id);
      if (mapping == nullptr) {
        LOG(ERROR) << "Missing pre-pushed heterogeneous KV mapping, layer="
                   << layer_id << ", role=" << registered_cache.role.to_string()
                   << ", group_id=" << registered_cache.group_id;
        return false;
      }
      const int64_t shard_dim =
          sharded_dimension(registered_cache.role, registered_cache.tensor);
      CHECK_GE(shard_dim, 0);
      std::vector<uint64_t> final_ids = mapping->local_ids;
      if (registered_cache.role == KVCacheTensorRole::SSM) {
        final_ids = expand_checkpoint_ids(final_ids, checkpoint_stride);
      }
      if (final_ids.empty()) {
        continue;
      }

      // Staging is a compact per-request scratch buffer, not a mirror of
      // Decode's block allocator. Both sides preserve request block order, so
      // each shard occupies a contiguous ordinal range.
      const std::vector<torch::Tensor> shard_tensors =
          make_staging_shard_views(stage_cache.tensor,
                                   source_shard_count,
                                   static_cast<int64_t>(final_ids.size()));
      const torch::Tensor merged = merge_shard_tensors(registered_cache.role,
                                                       stage_cache.tensor,
                                                       layer_staging_caches,
                                                       shard_tensors,
                                                       shard_dim,
                                                       layer_id);
      copy_merged_cache_rows(registered_cache, final_ids, merged);
    }
  }
  return true;
}

bool SpecKVCacheTransfer::push_layer_registered_caches_to_staging(
    const LayerRegisteredCaches& layer_registered_caches,
    const LayerRegisteredCaches& staging_registered_caches,
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    int64_t source_shard_rank,
    int64_t source_shard_count) {
  CHECK_GE(source_shard_rank, 0);
  CHECK_LT(source_shard_rank, source_shard_count);
  CHECK_EQ(layer_registered_caches.size(), staging_registered_caches.size());
  std::vector<std::string> keys;
  keys.reserve(merged_kv_infos.size());
  for (const auto& pair : merged_kv_infos) {
    keys.push_back(pair.first);
  }
  std::sort(keys.begin(), keys.end());

  Timer total_timer;
  double layer_wait_seconds = 0.0;
  double push_seconds = 0.0;
  bool success = true;
  for (size_t layer_id = 0; layer_id < layer_registered_caches.size();
       ++layer_id) {
    VLOG(5) << "Heterogeneous staged push waiting for layer=" << layer_id
            << ", source_shard=" << source_shard_rank;
    Timer layer_wait_timer;
    layer_synchronizer->synchronize_layer(layer_id);
    layer_wait_seconds += layer_wait_timer.elapsed_seconds();
    VLOG(5) << "Heterogeneous staged push layer ready: layer=" << layer_id
            << ", source_shard=" << source_shard_rank;
    const auto& layer_caches = layer_registered_caches[layer_id];
    const auto& layer_staging_caches = staging_registered_caches[layer_id];
    CHECK_EQ(layer_caches.size(), layer_staging_caches.size());

    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      for (size_t cache_index = 0; cache_index < layer_caches.size();
           ++cache_index) {
        const RegisteredCache& source_cache = layer_caches[cache_index];
        const RegisteredCache& stage_cache = layer_staging_caches[cache_index];
        // Decode restores CONV/SSM with a synchronous PULL because consuming
        // their pre-pushed staging rows is not correct on the heterogeneous
        // Qwen3.5 path. Avoid sending the same large recurrent state twice;
        // heterogeneous staging PUSH is only useful for target KEY/VALUE.
        if (source_cache.sequence_scoped) {
          continue;
        }
        const KVTransferMapping* mapping =
            find_mapping(kv_info.mappings, source_cache.group_id);
        if (mapping == nullptr) {
          LOG(ERROR) << "Missing heterogeneous staging mapping, layer="
                     << layer_id << ", role=" << source_cache.role.to_string()
                     << ", group_id=" << source_cache.group_id;
          success = false;
          continue;
        }
        const std::vector<uint64_t>& src_ids = mapping->local_ids;
        std::vector<uint64_t> dst_ids = mapping->remote_ids;
        if (src_ids.empty() || dst_ids.empty()) {
          continue;
        }
        CHECK_EQ(stage_cache.tensor.size(0) % source_shard_count, 0);
        const int64_t rows_per_shard =
            stage_cache.tensor.size(0) / source_shard_count;
        CHECK_LE(static_cast<int64_t>(dst_ids.size()), rows_per_shard);
        // Use compact request-local staging rows.  dst_ids are Decode's real
        // allocator ids and may exceed the bounded staging capacity.
        for (size_t ordinal = 0; ordinal < dst_ids.size(); ++ordinal) {
          dst_ids[ordinal] = static_cast<uint64_t>(
              source_shard_rank * rows_per_shard + ordinal);
        }
        CacheIndex destination{kv_info.dst_cluster_id,
                               stage_cache.cache.cache_id};
        KvCacheExtParam ext_param{};
        ext_param.src_layer_range = {0, 0};
        ext_param.dst_layer_range = {0, 0};
        ext_param.tensor_num_per_layer = 1;
        VLOG(5) << "Heterogeneous staged push begin: layer=" << layer_id
                << ", role=" << source_cache.role.to_string()
                << ", source_shard=" << source_shard_rank
                << ", source_cache_id=" << source_cache.cache.cache_id
                << ", destination_cache_id=" << stage_cache.cache.cache_id;
        Timer push_timer;
        const auto ret = llm_data_dist_->PushKvBlocks(
            source_cache.cache, destination, src_ids, dst_ids, ext_param);
        push_seconds += push_timer.elapsed_seconds();
        VLOG(5) << "Heterogeneous staged push end: layer=" << layer_id
                << ", role=" << source_cache.role.to_string()
                << ", source_shard=" << source_shard_rank
                << ", ret=" << std::hex << ret;
        if (ret != LLM_SUCCESS) {
          LOG(ERROR) << "Heterogeneous staged PushKvBlocks failed, layer="
                     << layer_id << ", role=" << source_cache.role.to_string()
                     << ", source_shard=" << source_shard_rank
                     << ", destination_cache_id=" << stage_cache.cache.cache_id
                     << ", ret=" << std::hex << ret;
          success = false;
        }
      }
    }
  }
  VLOG(1) << "Heterogeneous staging push source_shard=" << source_shard_rank
          << ", request_count=" << keys.size()
          << ", layer_wait_ms=" << layer_wait_seconds * 1000.0
          << ", push_ms=" << push_seconds * 1000.0
          << ", total_ms=" << total_timer.elapsed_seconds() * 1000.0;
  return success;
}

void SpecKVCacheTransfer::register_hetero_staging_caches(
    const LayerRegisteredCaches& source_registered_caches,
    LayerRegisteredCaches& staging_registered_caches,
    int64_t source_shard_count,
    bool source_is_sharded) {
  CHECK_EQ(source_shard_count, kSupportedHeterogeneousSourceShardCount)
      << "Only Prefill TP2 to Decode TP1 staging is supported.";
  const int64_t block_size = KVCacheConfig::get_instance().block_size();
  const int64_t max_tokens_per_batch =
      SchedulerConfig::get_instance().max_tokens_per_batch();
  const int64_t max_seqs_per_batch =
      SchedulerConfig::get_instance().max_seqs_per_batch();
  CHECK_GT(block_size, 0);
  CHECK_GT(max_tokens_per_batch, 0);
  CHECK_GT(max_seqs_per_batch, 0);
  // Each scheduled sequence may contribute a partially filled final block.
  const int64_t kv_rows_per_shard =
      (max_tokens_per_batch + block_size - 1) / block_size + max_seqs_per_batch;
  staging_registered_caches.clear();
  staging_registered_caches.resize(source_registered_caches.size());
  for (size_t layer_id = 0; layer_id < source_registered_caches.size();
       ++layer_id) {
    const int64_t checkpoint_stride =
        get_checkpoint_stride(source_registered_caches[layer_id]);
    for (const RegisteredCache& source_cache :
         source_registered_caches[layer_id]) {
      std::vector<int64_t> shape = source_cache.tensor.sizes().vec();
      const int64_t shard_dim =
          sharded_dimension(source_cache.role, source_cache.tensor);
      CHECK_GE(shard_dim, 0);
      if (!source_is_sharded) {
        if (shape[shard_dim] % source_shard_count == 0) {
          shape[shard_dim] /= source_shard_count;
        } else {
          // Homogeneous Decode tensors may already be a single local shard.
          // Such staging is registered for cache-id symmetry but is unused by
          // the homogeneous transfer path.
          CHECK_EQ(shape[shard_dim], 1)
              << "Unsupported Decode staging shard shape.";
        }
      }
      if (source_cache.role == KVCacheTensorRole::KEY ||
          source_cache.role == KVCacheTensorRole::VALUE) {
        shape[0] = std::min(shape[0], kv_rows_per_shard) * source_shard_count;
      } else if (source_cache.role == KVCacheTensorRole::SSM) {
        shape[0] = std::min(shape[0], checkpoint_stride) * source_shard_count;
      } else {
        shape[0] = source_shard_count;
      }
      torch::Tensor stage_tensor =
          make_page_aligned_staging_tensor(shape,
                                           source_cache.tensor.options(),
                                           source_cache.tensor.element_size());
      staging_registered_caches[layer_id].push_back(
          register_cache_tensor(layer_id,
                                KVCacheTensor{source_cache.role,
                                              stage_tensor,
                                              source_cache.group_id,
                                              source_cache.sequence_scoped}));
    }
  }
}

bool SpecKVCacheTransfer::pull_replicated_spec_kv_blocks(
    uint64_t src_cluster_id,
    const std::vector<KVTransferMapping>& mappings) {
  bool success = true;
  for (size_t layer_id = 0; layer_id < spec_layer_registered_caches_.size();
       ++layer_id) {
    for (const RegisteredCache& cache :
         spec_layer_registered_caches_[layer_id]) {
      const KVTransferMapping* mapping = find_mapping(mappings, cache.group_id);
      if (mapping == nullptr ||
          mapping->remote_ids.size() != mapping->local_ids.size()) {
        LOG(ERROR) << "Invalid replicated draft KV mapping, layer=" << layer_id
                   << ", role=" << cache.role.to_string()
                   << ", group_id=" << cache.group_id;
        success = false;
        continue;
      }
      CacheIndex source{src_cluster_id, cache.cache.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range = {0, 0};
      ext_param.dst_layer_range = {0, 0};
      ext_param.tensor_num_per_layer = 1;
      const auto ret = llm_data_dist_->PullKvBlocks(source,
                                                    cache.cache,
                                                    mapping->remote_ids,
                                                    mapping->local_ids,
                                                    ext_param);
      if (ret != LLM_SUCCESS) {
        LOG(ERROR) << "Pull replicated TP1 draft KV failed, layer=" << layer_id
                   << ", role=" << cache.role.to_string()
                   << ", ret=" << std::hex << ret;
        success = false;
      }
    }
  }
  return success;
}

void SpecKVCacheTransfer::register_kv_cache_internal(
    std::vector<xllm::KVCache>& kv_caches,
    LayerRegisteredCaches& layer_registered_caches) {
  register_layer_registered_caches(kv_caches, layer_registered_caches);
}

void SpecKVCacheTransfer::free_kv_cache() {
  layer_registered_caches_.clear();
  spec_layer_registered_caches_.clear();
  hetero_staging_registered_caches_.clear();
  spec_hetero_staging_registered_caches_.clear();
}

bool SpecKVCacheTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  const bool base_success =
      LlmDataDistTransfer::pull_kv_blocks(src_cluster_id, src_addr, mappings);
  bool spec_success = true;
  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(spec_layer_registered_caches_.size());
       ++layer_id) {
    const auto& registered_caches = spec_layer_registered_caches_[layer_id];
    for (const RegisteredCache& registered_cache : registered_caches) {
      const auto mapping_it =
          std::find_if(mappings.begin(),
                       mappings.end(),
                       [&registered_cache](const KVTransferMapping& mapping) {
                         return mapping.group_id == registered_cache.group_id;
                       });
      if (mapping_it == mappings.end()) {
        LOG(ERROR) << "Missing spec KV cache transfer mapping, layer="
                   << layer_id << ", role=" << registered_cache.role.to_string()
                   << ", group_id=" << registered_cache.group_id;
        spec_success = false;
        continue;
      }
      if (mapping_it->local_ids.size() != mapping_it->remote_ids.size()) {
        LOG(ERROR) << "Spec KV cache mapping size mismatch, layer=" << layer_id
                   << ", role=" << registered_cache.role.to_string()
                   << ", group_id=" << registered_cache.group_id
                   << ", local=" << mapping_it->local_ids.size()
                   << ", remote=" << mapping_it->remote_ids.size();
        spec_success = false;
        continue;
      }
      if (mapping_it->local_ids.empty()) {
        continue;
      }
      CacheIndex cache_index{src_cluster_id, registered_cache.cache.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range = {0, 0};
      ext_param.dst_layer_range = {0, 0};
      ext_param.tensor_num_per_layer = 1;
      auto ret = llm_data_dist_->PullKvBlocks(cache_index,
                                              registered_cache.cache,
                                              mapping_it->remote_ids,
                                              mapping_it->local_ids,
                                              ext_param);
      if (ret != LLM_SUCCESS) {
        LOG(ERROR) << "Pull spec KvBlocks failed, layer = " << layer_id
                   << ", ret = " << std::hex << ret;
        spec_success = false;
      }
    }
  }
  return base_success && spec_success;
}

bool SpecKVCacheTransfer::pull_hetero_kv_blocks(
    const std::vector<uint64_t>& src_cluster_ids,
    const std::vector<std::string>& src_addrs,
    const std::vector<KVTransferMapping>& mappings) {
  if (!heterogeneous_pd_enabled_) {
    LOG(ERROR) << "Heterogeneous KV restore requested while "
                  "enable_heterogeneous_pd is false.";
    return false;
  }
  if (!validate_transfer_mappings(
          mappings, /*request_id=*/"heterogeneous PULL", /*kv_split_size=*/1)) {
    return false;
  }
  (void)src_addrs;
  // All heterogeneous cache roles share staging tensors. Serialize the full
  // Linear State -> Target KV -> Draft KV restore transaction so another
  // FirstGeneration request cannot overwrite staging between phases.
  std::lock_guard<std::mutex> restore_lock(hetero_restore_mutex_);
  Timer phase_timer;
  // DataDist PUSH does not expose a Decode-side completion primitive for the
  // large recurrent state tensors. Keep CONV/SSM on the established
  // synchronous PULL path while KEY/VALUE are pre-pushed into staging.
  const bool linear_success =
      pull_and_merge_sharded_caches(layer_registered_caches_,
                                    hetero_staging_registered_caches_,
                                    src_cluster_ids,
                                    mappings,
                                    /*sequence_scoped=*/true);
  if (!linear_success) {
    return false;
  }
  const double linear_seconds = phase_timer.elapsed_seconds();
  phase_timer.reset();
  const bool target_success =
      merge_pre_pushed_sharded_caches(layer_registered_caches_,
                                      hetero_staging_registered_caches_,
                                      mappings,
                                      kSupportedHeterogeneousSourceShardCount,
                                      /*sequence_scoped=*/false);
  if (!target_success) {
    return false;
  }
  const double target_merge_seconds = phase_timer.elapsed_seconds();
  phase_timer.reset();
  // Keep the one-layer MTP draft cache on the established synchronous pull
  // path while validating the new layer-overlapped target-cache push.  The
  // draft pull is small (~1 ms), and this isolates whether its newly-added
  // layer event observes the cache before the MTP prefill write is complete.
  const bool draft_success =
      draft_body_uses_tp1_
          ? pull_replicated_spec_kv_blocks(src_cluster_ids.front(), mappings)
          : pull_and_merge_sharded_caches(
                spec_layer_registered_caches_,
                spec_hetero_staging_registered_caches_,
                src_cluster_ids,
                mappings,
                /*sequence_scoped=*/false);
  if (draft_success) {
    const KVTransferMapping* kv_mapping =
        find_mapping(mappings, cache_group_id(BlockType::KV));
    const KVTransferMapping* linear_mapping =
        find_mapping(mappings, cache_group_id(BlockType::LINEAR));
    const double draft_seconds = phase_timer.elapsed_seconds();
    VLOG(1) << "Merged heterogeneous TP KV cache (target KV pre-pushed, "
               "linear state and draft pulled): source_shards="
            << kSupportedHeterogeneousSourceShardCount << ", blocks="
            << (kv_mapping == nullptr ? 0 : kv_mapping->local_ids.size())
            << ", linear_states="
            << (linear_mapping == nullptr ? 0
                                          : linear_mapping->local_ids.size())
            << ", linear_ms=" << linear_seconds * 1000.0
            << ", target_merge_ms=" << target_merge_seconds * 1000.0
            << ", draft_ms=" << draft_seconds * 1000.0 << ", total_ms="
            << (linear_seconds + target_merge_seconds + draft_seconds) * 1000.0;
  }
  return draft_success;
}

bool SpecKVCacheTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  if (is_spec_draft) {
    return push_kv_blocks_spec(
        merged_kv_infos, layer_synchronizer, kv_split_rank, kv_split_size);
  } else {
    return push_kv_blocks_internal(merged_kv_infos,
                                   layer_synchronizer,
                                   layer_registered_caches_,
                                   kv_split_rank,
                                   kv_split_size);
  }
}

bool SpecKVCacheTransfer::push_kv_blocks_spec(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  return push_kv_blocks_internal(merged_kv_infos,
                                 layer_synchronizer,
                                 spec_layer_registered_caches_,
                                 kv_split_rank,
                                 kv_split_size);
}

bool SpecKVCacheTransfer::push_kv_blocks_internal(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    const LayerRegisteredCaches& layer_registered_caches,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  return push_layer_registered_caches(layer_registered_caches,
                                      merged_kv_infos,
                                      layer_synchronizer,
                                      kv_split_rank,
                                      kv_split_size);
}

bool SpecKVCacheTransfer::push_kv_blocks_to_hetero_staging(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int64_t source_shard_rank,
    int64_t source_shard_count) {
  const LayerRegisteredCaches& source_caches =
      is_spec_draft ? spec_layer_registered_caches_ : layer_registered_caches_;
  const LayerRegisteredCaches& staging_caches =
      is_spec_draft ? spec_hetero_staging_registered_caches_
                    : hetero_staging_registered_caches_;
  return push_layer_registered_caches_to_staging(source_caches,
                                                 staging_caches,
                                                 merged_kv_infos,
                                                 layer_synchronizer,
                                                 source_shard_rank,
                                                 source_shard_count);
}

folly::SemiFuture<bool> SpecKVCacheTransfer::push_kv_blocks_async(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args,
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer,
    bool is_spec_draft) {
  const int32_t local_dp_size = parallel_args.dp_size();
  const int32_t kv_split_size = parallel_args.kv_split_size_effective();
  const std::optional<int32_t> remote_tp_size =
      get_remote_tp_size(transfer_kv_infos);
  bool heterogeneous_non_mla = false;
  int32_t local_tp_size = 1;
  if (heterogeneous_pd_enabled_ && !enable_mla_ && local_dp_size > 0 &&
      kv_split_size > 0 && remote_tp_size.has_value()) {
    local_tp_size = parallel_args.world_size() / local_dp_size / kv_split_size;
    if (local_tp_size != remote_tp_size.value()) {
      heterogeneous_non_mla = true;
      VLOG(1) << "Push non-MLA heterogeneous KV shards to decode staging: "
              << "prefill_tp_size=" << local_tp_size
              << ", decode_tp_size=" << remote_tp_size.value()
              << ", is_spec_draft=" << is_spec_draft
              << "; decode will only perform a local merge.";
    }
  }
  const int64_t source_shard_rank =
      heterogeneous_non_mla ? parallel_args.rank() % local_tp_size : 0;

  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  if (!validate_transfer_mappings(transfer_kv_infos, kv_split_size)) {
    promise.setValue(false);
    return future;
  }
  // In heterogeneous non-MLA mode Decode intentionally restores the draft
  // cache from the source shards with a synchronous PULL.  Pushing the same
  // one-layer draft cache into staging is therefore redundant: no Decode
  // path consumes spec_hetero_staging_registered_caches_.  The source cache
  // remains alive until the synchronous FirstGeneration RPC returns, so
  // skipping this PUSH does not shorten its lifetime for the later PULL.
  if (heterogeneous_non_mla && is_spec_draft) {
    VLOG(5) << "Skip redundant heterogeneous MTP draft staging PUSH; "
               "Decode restores draft KV from source shards.";
    promise.setValue(true);
    return future;
  }
  threadpool_.schedule([this,
                        transfer_kv_infos,
                        &parallel_args,
                        layer_synchronizer,
                        is_spec_draft,
                        heterogeneous_non_mla,
                        local_tp_size,
                        source_shard_rank,
                        promise = std::move(promise)]() mutable {
    std::unordered_map<std::string, KVCacheInfo> merged_kv_infos;
    std::vector<TransferKVInfo> filtered_kv_infos;
    const std::vector<TransferKVInfo>* kv_infos = &transfer_kv_infos;
    // When the KV cache is actually sharded across ranks
    // (kv_split_size_effective > 1), filter remote_blocks_ids down to this
    // rank's slice. When kv_split_size==1 each rank holds the full replica and
    // we keep the legacy 1:1 remote_blocks_ids mapping.
    const int32_t effective_kv_split_size =
        parallel_args.kv_split_size_effective();
    if (!validate_transfer_mappings(*kv_infos, effective_kv_split_size)) {
      promise.setValue(false);
      return;
    }
    if (effective_kv_split_size > 1) {
      filtered_kv_infos = filter_kv_split_infos(
          parallel_args.kv_split_rank(), effective_kv_split_size, *kv_infos);
      kv_infos = &filtered_kv_infos;
      if (kv_infos->empty()) {
        promise.setValue(true);
        return;
      }
    }
    if (!validate_transfer_mappings(*kv_infos, /*kv_split_size=*/1)) {
      promise.setValue(false);
      return;
    }
    if (heterogeneous_non_mla) {
      merge_heterogeneous_kv_blocks(
          merged_kv_infos, *kv_infos, source_shard_rank);
    } else {
      merge_kv_blocks(merged_kv_infos, *kv_infos, parallel_args);
    }
    bool success = true;
    if (!merged_kv_infos.empty()) {
      if (heterogeneous_non_mla) {
        success = this->push_kv_blocks_to_hetero_staging(merged_kv_infos,
                                                         layer_synchronizer,
                                                         is_spec_draft,
                                                         source_shard_rank,
                                                         local_tp_size);
      } else {
        success = this->push_kv_blocks(merged_kv_infos,
                                       layer_synchronizer,
                                       is_spec_draft,
                                       parallel_args.kv_split_rank(),
                                       parallel_args.kv_split_size_effective());
      }
    }
    promise.setValue(success);
  });
  return future;
}
}  // namespace xllm
