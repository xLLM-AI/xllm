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

#include "framework/kv_cache_transfer/llm_data_dist_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <numeric>

#include "common/macros.h"
#include "core/framework/config/disagg_pd_config.h"
#include "util/net.h"
#include "util/timer.h"

namespace xllm {

const std::map<torch::ScalarType, ge::DataType> kScalarTypeToDtype = {
    {torch::kBool, ge::DT_BOOL},
    {torch::kByte, ge::DT_UINT8},
    {torch::kChar, ge::DT_INT8},
    {torch::kShort, ge::DT_INT16},
    {torch::kInt, ge::DT_INT32},
    {torch::kLong, ge::DT_INT64},
    {torch::kBFloat16, ge::DT_BF16},
    {torch::kHalf, ge::DT_FLOAT16},
    {torch::kFloat, ge::DT_FLOAT},
    {torch::kDouble, ge::DT_DOUBLE},
};

ge::DataType dtype_to_ge_dtype(torch::ScalarType dtype) {
  const auto& it = kScalarTypeToDtype.find(dtype);
  CHECK(it != kScalarTypeToDtype.cend()) << "Unsupport data type : " << dtype;
  return it->second;
}

bool is_linear_state_cache(KVCacheTensorRole role) {
  return role == KVCacheTensorRole::CONV || role == KVCacheTensorRole::SSM;
}

int64_t sharded_dimension(KVCacheTensorRole role, const torch::Tensor& tensor) {
  if (role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE) {
    return 2;
  }
  if (role == KVCacheTensorRole::CONV) {
    return tensor.dim() - 1;
  }
  if (role == KVCacheTensorRole::SSM) {
    return 1;
  }
  return -1;
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

LlmDataDistTransfer::LlmDataDistTransfer(const uint16_t listen_port,
                                         const InstanceRole& instance_role,
                                         bool enable_lighting_indexer)
    : listen_port_(listen_port),
      enable_lighting_indexer_(enable_lighting_indexer),
      KVCacheTransfer() {
  if (instance_role == InstanceRole::PREFILL) {
    LOG(INFO) << "Create LlmDataDistTransfer for prefill instance.";
    role_ = LlmRole::kPrompt;
  } else if (instance_role == InstanceRole::DECODE) {
    LOG(INFO) << "Create LlmDataDistTransfer for decode instance.";
    role_ = LlmRole::kDecoder;
  } else {
    LOG(INFO) << "Create LlmDataDistTransfer for mix instance.";
    role_ = LlmRole::kMix;
  }
  host_ip_ = net::get_local_ip_addr();
  CHECK(!host_ip_.empty()) << "Failed to get NPU/host IP for LlmDataDist.";
  cluster_id_ = net::convert_ip_port_to_uint64(host_ip_, listen_port);
  llm_data_dist_ = std::make_shared<LlmDataDist>(cluster_id_, role_);
}

void LlmDataDistTransfer::initialize(int32_t device_id) {
  std::map<AscendString, AscendString> options;
  options[OPTION_DEVICE_ID] = std::to_string(device_id).c_str();

  // Prompt(Prefill) must publish listen endpoint; Decoder only needs device_id.
  if (role_ == LlmRole::kPrompt) {
    std::string local_ip_info = host_ip_ + ":" + std::to_string(listen_port_);
    options[OPTION_LISTEN_IP_INFO] = local_ip_info.c_str();
  }

  auto ret = llm_data_dist_->Initialize(options);
  CHECK(ret == LLM_SUCCESS)
      << "Initialize LlmDataList failed, ret = " << std::hex << ret;
  LOG(INFO) << "Initialize LlmDataList success.";
}

void LlmDataDistTransfer::finalize() { llm_data_dist_->Finalize(); }

void LlmDataDistTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  register_layer_registered_caches(kv_caches, layer_registered_caches_);
}

void LlmDataDistTransfer::free_kv_cache() {
  layer_registered_caches_.clear();
  has_grouped_cache_layout_ = false;
}

void LlmDataDistTransfer::get_cache_info(uint64_t& cluster_id,
                                         std::string& addr) {
  cluster_id = cluster_id_;
  addr = host_ip_;
}

bool LlmDataDistTransfer::link_cluster(const uint64_t cluster_id,
                                       const std::string& remote_addr,
                                       const uint16_t port) {
  if (linked_cluster_ids.find(cluster_id) != linked_cluster_ids.end()) {
    // The cluster is connected.
    return true;
  }

  std::vector<llm_datadist::Status> rets;
  std::vector<ClusterInfo> clusters;
  ClusterInfo cluster_info = create_cluster_info(cluster_id, remote_addr, port);
  clusters.emplace_back(std::move(cluster_info));

  auto ret = llm_data_dist_->LinkLlmClusters(
      clusters, rets, /*timeout_in_millis=*/60000);
  if (ret != LLM_SUCCESS) {
    LOG(ERROR) << "LinkLlmClusters failed, ret = " << std::hex << ret;
    return false;
  }
  LOG(INFO) << "LinkLlmClusters success, ip : " << remote_addr
            << ", port : " << port;
  linked_cluster_ids.insert(cluster_id);

  return true;
}

bool LlmDataDistTransfer::unlink_cluster(const uint64_t& cluster_id,
                                         const std::string& remote_addr,
                                         const uint16_t remote_port,
                                         bool force_flag) {
  std::vector<llm_datadist::Status> rets;
  std::vector<ClusterInfo> clusters;
  ClusterInfo cluster_info =
      create_cluster_info(cluster_id, remote_addr, remote_port);
  clusters.emplace_back(std::move(cluster_info));

  auto ret =
      llm_data_dist_->UnlinkLlmClusters(clusters, rets, 1000, force_flag);
  if (ret != LLM_SUCCESS) {
    LOG(ERROR) << "UnlinkLlmClusters failed, ret = " << std::hex << ret;
    return false;
  }
  LOG(INFO) << "UnlinkLlmClusters success, ip : " << remote_addr
            << ", port : " << remote_port;
  linked_cluster_ids.erase(cluster_id);

  return true;
}

bool LlmDataDistTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  if (has_grouped_cache_layout_) {
    LOG(ERROR) << "Grouped KV cache layout only supports PUSH transfer mode.";
    return false;
  }
  bool result = true;
  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(layer_registered_caches_.size());
       ++layer_id) {
    const auto& registered_caches = layer_registered_caches_[layer_id];
    for (const RegisteredCache& registered_cache : registered_caches) {
      const bool linear_state_cache = registered_cache.sequence_scoped;
      const std::vector<uint64_t>& src_ids =
          linear_state_cache ? src_linear_state_ids : src_blocks;
      const std::vector<uint64_t>& dst_ids =
          linear_state_cache ? dst_linear_state_ids : dst_blocks;
      if (src_ids.empty() || dst_ids.empty()) {
        VLOG(5) << "Skip PullKvBlocks, layer = " << layer_id
                << ", role = " << registered_cache.role.to_string()
                << ", src_ids = " << src_ids.size()
                << ", dst_ids = " << dst_ids.size();
        continue;
      }
      CacheIndex cache_index{src_cluster_id, registered_cache.cache.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range = {0, 0};
      ext_param.dst_layer_range = {0, 0};
      ext_param.tensor_num_per_layer = 1;
      auto ret = llm_data_dist_->PullKvBlocks(
          cache_index, registered_cache.cache, src_ids, dst_ids, ext_param);
      if (ret != LLM_SUCCESS) {
        LOG(ERROR) << "PullKvBlocks failed, layer = " << layer_id
                   << ", role = " << registered_cache.role.to_string()
                   << ", ret = " << std::hex << ret;
        result = false;
      }
    }
  }
  return result;
}

bool LlmDataDistTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  (void)is_spec_draft;
  return push_layer_registered_caches(layer_registered_caches_,
                                      merged_kv_infos,
                                      layer_synchronizer,
                                      kv_split_rank,
                                      kv_split_size);
}

RegisteredCache LlmDataDistTransfer::register_cache_tensor(
    int64_t layer_id,
    const KVCacheTensor& cache_tensor) {
  const torch::Tensor& tensor = cache_tensor.tensor;
  CHECK(tensor.defined() && tensor.numel() > 0)
      << cache_tensor.role.to_string() << " cache is not allocated at layer "
      << layer_id;

  auto tensor_addr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  std::vector<uint64_t> addrs = {static_cast<uint64_t>(tensor_addr)};

  RegisteredCache registered_cache{cache_tensor.role,
                                   cache_tensor.group_id,
                                   cache_tensor.sequence_scoped,
                                   Cache{},
                                   tensor};
  registered_cache.cache.tensor_addrs = {tensor_addr};

  CacheDesc& desc = registered_cache.cache.cache_desc;
  desc.num_tensors = 1;
  desc.data_type = dtype_to_ge_dtype(tensor.scalar_type());
  desc.shape = tensor.sizes().vec();

  auto ret = llm_data_dist_->RegisterKvCache(
      desc, addrs, {}, registered_cache.cache.cache_id);
  CHECK(ret == LLM_SUCCESS)
      << "Register " << cache_tensor.role.to_string()
      << " cache failed at layer " << layer_id << ", ret = " << std::hex << ret;
  VLOG(5) << "Registered KV cache: layer=" << layer_id
          << ", role=" << cache_tensor.role.to_string()
          << ", cache_id=" << registered_cache.cache.cache_id
          << ", shape=" << tensor.sizes();
  return registered_cache;
}

bool LlmDataDistTransfer::pull_and_merge_sharded_caches(
    const LayerRegisteredCaches& layer_registered_caches,
    const LayerRegisteredCaches& staging_registered_caches,
    const std::vector<uint64_t>& src_cluster_ids,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  if (src_cluster_ids.size() != 2 ||
      layer_registered_caches.size() != staging_registered_caches.size() ||
      src_blocks.size() != dst_blocks.size() ||
      src_linear_state_ids.size() != dst_linear_state_ids.size()) {
    LOG(ERROR) << "Invalid heterogeneous KV pull metadata: src_tp="
               << src_cluster_ids.size() << ", src_blocks=" << src_blocks.size()
               << ", dst_blocks=" << dst_blocks.size()
               << ", src_linear_states=" << src_linear_state_ids.size()
               << ", dst_linear_states=" << dst_linear_state_ids.size();
    return false;
  }

  // The staging tensors are registered once and reused by all requests.
  // Serialize request-level restore/merge so concurrent FirstGeneration RPCs
  // cannot overwrite each other's staging rows.
  std::lock_guard<std::mutex> hetero_pull_lock(hetero_pull_mutex_);

  const int64_t shard_count = static_cast<int64_t>(src_cluster_ids.size());
  Timer breakdown_total_timer;
  const char* parallel_pull_env = std::getenv("XLLM_PD_PARALLEL_SHARD_PULL");
  // Parallel shard pulls are the default for heterogeneous TP. Keep an
  // explicit opt-out so deployments can immediately fall back to the serial
  // path without rebuilding if the transport backend rejects concurrency.
  const bool parallel_shard_pull =
      parallel_pull_env == nullptr || std::string(parallel_pull_env) != "0";
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
    int64_t checkpoint_stride = 1;
    for (const RegisteredCache& cache : layer_caches) {
      if (cache.role == KVCacheTensorRole::CONV && cache.tensor.defined()) {
        for (const RegisteredCache& candidate : layer_caches) {
          if (candidate.role == KVCacheTensorRole::SSM &&
              candidate.tensor.defined()) {
            CHECK_EQ(candidate.tensor.size(0) % cache.tensor.size(0), 0);
            checkpoint_stride = candidate.tensor.size(0) / cache.tensor.size(0);
          }
        }
      }
    }

    for (size_t cache_index = 0; cache_index < layer_caches.size();
         ++cache_index) {
      const RegisteredCache& registered_cache = layer_caches[cache_index];
      const RegisteredCache& stage_cache = layer_staging_caches[cache_index];
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

      const bool linear_state_cache =
          is_linear_state_cache(registered_cache.role);
      std::vector<uint64_t> remote_ids =
          linear_state_cache ? src_linear_state_ids : src_blocks;
      std::vector<uint64_t> final_ids =
          linear_state_cache ? dst_linear_state_ids : dst_blocks;
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
      std::vector<torch::Tensor> shard_tensors;
      shard_tensors.reserve(src_cluster_ids.size());

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
      if (parallel_shard_pull && src_cluster_ids.size() == 2) {
        TaskGroup shard_pull_group(1);
        shard_pull_threadpool_.schedule(
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
        const uint64_t staging_offset =
            static_cast<uint64_t>(source_shard * rows_per_shard);
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
        // Each Prefill shard owns a disjoint staging row range. PullKvBlocks
        // is synchronous, so the merge can consume these rows directly
        // without cloning the entire recurrent state between shard pulls.
        shard_tensors.push_back(
            stage_cache.tensor.narrow(0, staging_offset, remote_ids.size()));
      }

      if (success) {
        Timer merge_timer;
        torch::Tensor merged;
        if (registered_cache.role == KVCacheTensorRole::CONV) {
          // Each TP rank stores its causal-conv state as
          // [Q_rank, K_rank, V_rank].  Concatenating whole rank tensors would
          // produce [Q0,K0,V0,Q1,K1,V1], while a TP1 model consumes
          // [Q0,Q1,K0,K1,V0,V1].  Recover the local V width from the matching
          // SSM cache, then merge each projection segment independently.
          int64_t local_v_width = -1;
          for (const RegisteredCache& candidate : layer_staging_caches) {
            if (candidate.role == KVCacheTensorRole::SSM &&
                candidate.tensor.defined()) {
              local_v_width =
                  candidate.tensor.size(1) * candidate.tensor.size(3);
              break;
            }
          }
          CHECK_GT(local_v_width, 0)
              << "CONV cache requires a matching SSM cache at layer "
              << layer_id;
          const int64_t local_conv_width = stage_cache.tensor.size(shard_dim);
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
            auto qkv = torch::split_with_sizes(
                shard,
                {local_qk_width, local_qk_width, local_v_width},
                shard_dim);
            q_shards.push_back(qkv[0]);
            k_shards.push_back(qkv[1]);
            v_shards.push_back(qkv[2]);
          }
          merged = torch::cat({torch::cat(q_shards, shard_dim),
                               torch::cat(k_shards, shard_dim),
                               torch::cat(v_shards, shard_dim)},
                              shard_dim);
        } else {
          merged = torch::cat(shard_tensors, shard_dim);
        }
        std::vector<int64_t> signed_final_ids(final_ids.begin(),
                                              final_ids.end());
        torch::Tensor dst_indices =
            torch::tensor(signed_final_ids,
                          torch::TensorOptions().dtype(torch::kLong))
                .to(registered_cache.tensor.device(), /*non_blocking=*/false);
        registered_cache.tensor.index_copy_(0, dst_indices, merged);
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
  LOG(INFO) << "[PD-PERF] Heterogeneous pull-merge breakdown"
            << " linear_request=" << !src_linear_state_ids.empty()
            << " parallel_shard_pull=" << parallel_shard_pull
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

bool LlmDataDistTransfer::merge_pre_pushed_sharded_caches(
    const LayerRegisteredCaches& layer_registered_caches,
    const LayerRegisteredCaches& staging_registered_caches,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& dst_linear_state_ids,
    int64_t source_shard_count) {
  if (source_shard_count != 2 ||
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
    int64_t checkpoint_stride = 1;
    for (const RegisteredCache& cache : layer_caches) {
      if (cache.role != KVCacheTensorRole::CONV || !cache.tensor.defined()) {
        continue;
      }
      for (const RegisteredCache& candidate : layer_caches) {
        if (candidate.role == KVCacheTensorRole::SSM &&
            candidate.tensor.defined()) {
          CHECK_EQ(candidate.tensor.size(0) % cache.tensor.size(0), 0);
          checkpoint_stride = candidate.tensor.size(0) / cache.tensor.size(0);
          break;
        }
      }
    }

    for (size_t cache_index = 0; cache_index < layer_caches.size();
         ++cache_index) {
      const RegisteredCache& registered_cache = layer_caches[cache_index];
      const RegisteredCache& stage_cache = layer_staging_caches[cache_index];
      const int64_t shard_dim =
          sharded_dimension(registered_cache.role, registered_cache.tensor);
      CHECK_GE(shard_dim, 0);
      std::vector<uint64_t> final_ids =
          is_linear_state_cache(registered_cache.role) ? dst_linear_state_ids
                                                       : dst_blocks;
      if (registered_cache.role == KVCacheTensorRole::SSM) {
        final_ids = expand_checkpoint_ids(final_ids, checkpoint_stride);
      }
      if (final_ids.empty()) {
        continue;
      }

      CHECK_EQ(stage_cache.tensor.size(0) % source_shard_count, 0);
      const int64_t rows_per_shard =
          stage_cache.tensor.size(0) / source_shard_count;
      // Staging is a compact per-request scratch buffer, not a mirror of
      // Decode's block allocator. Both sides preserve request block order, so
      // each shard occupies a contiguous ordinal range.
      CHECK_LE(static_cast<int64_t>(final_ids.size()), rows_per_shard);
      std::vector<torch::Tensor> shard_tensors;
      shard_tensors.reserve(source_shard_count);
      for (int64_t shard = 0; shard < source_shard_count; ++shard) {
        shard_tensors.push_back(stage_cache.tensor.narrow(
            0, shard * rows_per_shard, final_ids.size()));
      }

      torch::Tensor merged;
      if (registered_cache.role == KVCacheTensorRole::CONV) {
        int64_t local_v_width = -1;
        for (const RegisteredCache& candidate : layer_staging_caches) {
          if (candidate.role == KVCacheTensorRole::SSM &&
              candidate.tensor.defined()) {
            local_v_width = candidate.tensor.size(1) * candidate.tensor.size(3);
            break;
          }
        }
        CHECK_GT(local_v_width, 0);
        const int64_t local_conv_width = stage_cache.tensor.size(shard_dim);
        CHECK_EQ((local_conv_width - local_v_width) % 2, 0);
        const int64_t local_qk_width = (local_conv_width - local_v_width) / 2;
        std::vector<torch::Tensor> q_shards;
        std::vector<torch::Tensor> k_shards;
        std::vector<torch::Tensor> v_shards;
        for (const torch::Tensor& shard : shard_tensors) {
          auto qkv = torch::split_with_sizes(
              shard,
              {local_qk_width, local_qk_width, local_v_width},
              shard_dim);
          q_shards.push_back(qkv[0]);
          k_shards.push_back(qkv[1]);
          v_shards.push_back(qkv[2]);
        }
        merged = torch::cat({torch::cat(q_shards, shard_dim),
                             torch::cat(k_shards, shard_dim),
                             torch::cat(v_shards, shard_dim)},
                            shard_dim);
      } else {
        merged = torch::cat(shard_tensors, shard_dim);
      }

      std::vector<int64_t> signed_final_ids(final_ids.begin(), final_ids.end());
      torch::Tensor final_indices =
          torch::tensor(signed_final_ids,
                        torch::TensorOptions().dtype(torch::kLong))
              .to(registered_cache.tensor.device(), /*non_blocking=*/false);
      registered_cache.tensor.index_copy_(0, final_indices, merged);
    }
  }
  return true;
}

bool LlmDataDistTransfer::push_layer_registered_caches_to_staging(
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
    int64_t checkpoint_stride = 1;
    for (const RegisteredCache& cache : layer_caches) {
      if (cache.role != KVCacheTensorRole::CONV || !cache.tensor.defined()) {
        continue;
      }
      for (const RegisteredCache& candidate : layer_caches) {
        if (candidate.role == KVCacheTensorRole::SSM &&
            candidate.tensor.defined()) {
          CHECK_EQ(candidate.tensor.size(0) % cache.tensor.size(0), 0);
          checkpoint_stride = candidate.tensor.size(0) / cache.tensor.size(0);
          break;
        }
      }
    }

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
        if (is_linear_state_cache(source_cache.role)) {
          continue;
        }
        std::vector<uint64_t> src_ids = is_linear_state_cache(source_cache.role)
                                            ? kv_info.src_linear_state_ids
                                            : kv_info.src_blocks;
        std::vector<uint64_t> dst_ids = is_linear_state_cache(source_cache.role)
                                            ? kv_info.dst_linear_state_ids
                                            : kv_info.dst_blocks;
        if (source_cache.role == KVCacheTensorRole::SSM) {
          src_ids = expand_checkpoint_ids(src_ids, checkpoint_stride);
          dst_ids = expand_checkpoint_ids(dst_ids, checkpoint_stride);
        }
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
  LOG(INFO) << "[PD-PERF] Heterogeneous staging push source_shard="
            << source_shard_rank << ", request_count=" << keys.size()
            << ", layer_wait_ms=" << layer_wait_seconds * 1000.0
            << ", push_ms=" << push_seconds * 1000.0
            << ", total_ms=" << total_timer.elapsed_seconds() * 1000.0;
  return success;
}

void LlmDataDistTransfer::register_hetero_staging_caches(
    const LayerRegisteredCaches& source_registered_caches,
    LayerRegisteredCaches& staging_registered_caches,
    int64_t source_shard_count,
    bool source_is_sharded) {
  CHECK_EQ(source_shard_count, 2)
      << "Only Prefill TP2 to Decode TP1 staging is supported.";
  staging_registered_caches.clear();
  staging_registered_caches.resize(source_registered_caches.size());
  for (size_t layer_id = 0; layer_id < source_registered_caches.size();
       ++layer_id) {
    for (const RegisteredCache& source_cache :
         source_registered_caches[layer_id]) {
      std::vector<int64_t> shape = source_cache.tensor.sizes().vec();
      const int64_t shard_dim =
          sharded_dimension(source_cache.role, source_cache.tensor);
      CHECK_GE(shard_dim, 0);
      if (!source_is_sharded) {
        CHECK_EQ(shape[shard_dim] % source_shard_count, 0);
        shape[shard_dim] /= source_shard_count;
      }
      if (source_cache.role == KVCacheTensorRole::KEY ||
          source_cache.role == KVCacheTensorRole::VALUE) {
        // max_tokens_per_batch=32768 with block_size=128.
        shape[0] = std::min<int64_t>(shape[0], 256) * source_shard_count;
      } else if (source_cache.role == KVCacheTensorRole::SSM) {
        shape[0] = std::min<int64_t>(shape[0], 4) * source_shard_count;
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

void LlmDataDistTransfer::register_layer_registered_caches(
    std::vector<xllm::KVCache>& kv_caches,
    LayerRegisteredCaches& layer_registered_caches) {
  CHECK(!kv_caches.empty()) << "KV caches must be allocated before register.";
  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());

  layer_registered_caches.clear();
  layer_registered_caches.resize(kv_caches.size());

  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    for (const KVCacheTensor& cache_tensor :
         kv_caches[layer_id].get_cache_tensors()) {
      if (!cache_tensor.sequence_scoped &&
          cache_tensor.group_id != cache_group_id(BlockType::KV)) {
        has_grouped_cache_layout_ = true;
      }
      layer_registered_caches[layer_id].emplace_back(
          register_cache_tensor(layer_id, cache_tensor));
    }
    CHECK(!layer_registered_caches[layer_id].empty())
        << "No cache tensor registered at layer " << layer_id;
  }
}

bool LlmDataDistTransfer::push_layer_registered_caches(
    const LayerRegisteredCaches& layer_registered_caches,
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  std::vector<std::string> keys;
  keys.reserve(merged_kv_infos.size());
  for (const auto& pair : merged_kv_infos) {
    keys.push_back(pair.first);
  }
  if (kv_split_size > 1) {
    keys = rotate_dst_rank(keys, kv_split_rank);
  }

  bool result = true;
  for (int64_t layer_index = 0;
       layer_index < static_cast<int64_t>(layer_registered_caches.size());
       ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    if (!layer_synchronizer->synchronize_layer(layer_index)) {
      result = false;
      continue;
    }
    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      if (kv_info.src_blocks.empty() && kv_info.src_linear_state_ids.empty() &&
          kv_info.block_transfer_groups.empty()) {
        continue;
      }

      for (const RegisteredCache& registered_cache :
           layer_registered_caches[layer_index]) {
        const std::vector<uint64_t>* src_ids = nullptr;
        const std::vector<uint64_t>* dst_ids = nullptr;
        if (registered_cache.sequence_scoped) {
          src_ids = &kv_info.src_linear_state_ids;
          dst_ids = &kv_info.dst_linear_state_ids;
        } else {
          const int32_t group_id = registered_cache.group_id;
          const auto group_it =
              std::find_if(kv_info.block_transfer_groups.begin(),
                           kv_info.block_transfer_groups.end(),
                           [group_id](const KVBlockTransferGroup& group) {
                             return group.group_id == group_id;
                           });
          if (group_it != kv_info.block_transfer_groups.end()) {
            src_ids = &group_it->local_blocks_ids;
            dst_ids = &group_it->remote_blocks_ids;
          } else if (registered_cache.group_id ==
                     cache_group_id(BlockType::KV)) {
            src_ids = &kv_info.src_blocks;
            dst_ids = &kv_info.dst_blocks;
          } else {
            LOG(ERROR) << "Missing KV cache transfer group, layer="
                       << layer_index
                       << ", role=" << registered_cache.role.to_string()
                       << ", group_id=" << group_id;
            result = false;
            continue;
          }
        }
        CHECK(src_ids != nullptr && dst_ids != nullptr);
        if (src_ids->empty() || dst_ids->empty()) {
          VLOG(5) << "Skip PushKvBlocks, layer = " << layer_index
                  << ", role = " << registered_cache.role.to_string()
                  << ", src_ids = " << src_ids->size()
                  << ", dst_ids = " << dst_ids->size();
          continue;
        }
        if (src_ids->size() != dst_ids->size()) {
          LOG(ERROR) << "KV cache block mapping size mismatch, layer="
                     << layer_index
                     << ", role=" << registered_cache.role.to_string()
                     << ", group_id=" << registered_cache.group_id
                     << ", local=" << src_ids->size()
                     << ", remote=" << dst_ids->size();
          result = false;
          continue;
        }
        CacheIndex cache_index{kv_info.dst_cluster_id,
                               registered_cache.cache.cache_id};
        KvCacheExtParam ext_param{};
        ext_param.src_layer_range = {0, 0};
        ext_param.dst_layer_range = {0, 0};
        ext_param.tensor_num_per_layer = 1;

        auto ret = llm_data_dist_->PushKvBlocks(
            registered_cache.cache, cache_index, *src_ids, *dst_ids, ext_param);
        if (ret != LLM_SUCCESS) {
          LOG(ERROR) << "PushKvBlocks failed, layer = " << layer_index
                     << ", role = " << registered_cache.role.to_string()
                     << ", ret = " << std::hex << ret;
          result = false;
        }
      }
    }
  }
  return result;
}

ClusterInfo LlmDataDistTransfer::create_cluster_info(
    const uint64_t& cluster_id,
    const std::string& remote_ip,
    const uint16_t& remote_port) {
  ClusterInfo cluster_info;
  IpInfo local_ip_info;
  IpInfo remote_ip_info;

  local_ip_info.ip = host_ip_.c_str();
  local_ip_info.port = listen_port_;
  remote_ip_info.ip = remote_ip.c_str();
  remote_ip_info.port = remote_port;
  cluster_info.remote_cluster_id = cluster_id;
  cluster_info.local_ip_infos.emplace_back(std::move(local_ip_info));
  cluster_info.remote_ip_infos.emplace_back(std::move(remote_ip_info));

  return cluster_info;
}

}  // namespace xllm
