/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <exception>

#include "common/macros.h"
#include "core/framework/config/disagg_pd_config.h"
#include "util/net.h"

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

LlmDataDistTransfer::LlmDataDistTransfer(const uint16_t listen_port,
                                         const InstanceRole& instance_role,
                                         const std::string& model_type,
                                         bool enable_lighting_indexer)
    : listen_port_(listen_port),
      enable_lighting_indexer_(enable_lighting_indexer),
      model_type_(model_type),
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

  // Prompt(Prefill) and Mix must publish listen endpoint so remote Decoders
  // can link to them. Decoder-only instances do not need to listen.
  if (role_ == LlmRole::kPrompt || role_ == LlmRole::kMix) {
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

void LlmDataDistTransfer::free_kv_cache() { layer_registered_caches_.clear(); }

void LlmDataDistTransfer::get_cache_info(uint64_t& cluster_id,
                                         std::string& addr) {
  cluster_id = cluster_id_;
  addr = host_ip_;
}

bool LlmDataDistTransfer::link_cluster(const uint64_t cluster_id,
                                       const std::string& remote_addr,
                                       const uint16_t port) {
  return link_clusters(std::vector<uint64_t>{cluster_id},
                       std::vector<std::string>{remote_addr},
                       std::vector<uint16_t>{port});
}

bool LlmDataDistTransfer::link_clusters(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs,
    const std::vector<uint16_t>& ports) {
  if (cluster_ids.size() != remote_addrs.size() ||
      cluster_ids.size() != ports.size()) {
    LOG(ERROR) << "Cluster endpoint size mismatch: cluster_ids="
               << cluster_ids.size() << ", addrs=" << remote_addrs.size()
               << ", ports=" << ports.size();
    return false;
  }

  std::vector<llm_datadist::Status> rets;
  std::vector<ClusterInfo> clusters;
  clusters.reserve(cluster_ids.size());
  for (size_t i = 0; i < cluster_ids.size(); ++i) {
    if (linked_cluster_ids.find(cluster_ids[i]) != linked_cluster_ids.end()) {
      continue;
    }
    clusters.emplace_back(
        create_cluster_info(cluster_ids[i], remote_addrs[i], ports[i]));
  }

  if (clusters.empty()) {
    return true;
  }

  uint64_t ret = LLM_SUCCESS;
  try {
    ret = llm_data_dist_->LinkLlmClusters(
        clusters, rets, /*timeout_in_millis=*/60000);
  } catch (const std::exception& e) {
    LOG(ERROR) << "LinkLlmClusters threw exception: " << e.what()
               << ", clusters = " << clusters.size();
    return false;
  }
  if (ret != LLM_SUCCESS) {
    LOG(ERROR) << "LinkLlmClusters failed, ret = " << std::hex << ret;
    return false;
  }
  LOG(INFO) << "LinkLlmClusters success, clusters = " << clusters.size();
  for (uint64_t id : cluster_ids) {
    linked_cluster_ids.insert(id);
  }

  return true;
}

bool LlmDataDistTransfer::unlink_cluster(const uint64_t& cluster_id,
                                         const std::string& remote_addr,
                                         const uint16_t remote_port,
                                         bool force_flag) {
  return unlink_clusters(std::vector<uint64_t>{cluster_id},
                         std::vector<std::string>{remote_addr},
                         std::vector<uint16_t>{remote_port},
                         force_flag);
}

bool LlmDataDistTransfer::unlink_clusters(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs,
    const std::vector<uint16_t>& remote_ports,
    bool force_flag) {
  if (cluster_ids.size() != remote_addrs.size() ||
      cluster_ids.size() != remote_ports.size()) {
    LOG(ERROR) << "Cluster endpoint size mismatch: cluster_ids="
               << cluster_ids.size() << ", addrs=" << remote_addrs.size()
               << ", ports=" << remote_ports.size();
    return false;
  }

  std::vector<llm_datadist::Status> rets;
  std::vector<ClusterInfo> clusters;
  clusters.reserve(cluster_ids.size());
  for (size_t i = 0; i < cluster_ids.size(); ++i) {
    clusters.emplace_back(
        create_cluster_info(cluster_ids[i], remote_addrs[i], remote_ports[i]));
  }

  auto ret =
      llm_data_dist_->UnlinkLlmClusters(clusters, rets, 1000, force_flag);
  if (ret != LLM_SUCCESS) {
    LOG(ERROR) << "UnlinkLlmClusters failed, ret = " << std::hex << ret;
    return false;
  }
  LOG(INFO) << "UnlinkLlmClusters success, clusters = " << clusters.size();
  for (uint64_t id : cluster_ids) {
    linked_cluster_ids.erase(id);
  }

  return true;
}

bool LlmDataDistTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  bool result = true;
  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(layer_registered_caches_.size());
       ++layer_id) {
    const auto& registered_caches = layer_registered_caches_[layer_id];
    for (const RegisteredCache& registered_cache : registered_caches) {
      const bool linear_state_cache =
          is_linear_state_cache(registered_cache.role);
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

  RegisteredCache registered_cache{cache_tensor.role, Cache{}};
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

  return registered_cache;
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
    layer_synchronizer->synchronize_layer(layer_index);
    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      if (kv_info.src_blocks.empty() && kv_info.src_linear_state_ids.empty()) {
        continue;
      }

      for (const RegisteredCache& registered_cache :
           layer_registered_caches[layer_index]) {
        const bool linear_state_cache =
            is_linear_state_cache(registered_cache.role);
        const std::vector<uint64_t>& src_ids =
            linear_state_cache ? kv_info.src_linear_state_ids
                               : kv_info.src_blocks;
        const std::vector<uint64_t>& dst_ids =
            linear_state_cache ? kv_info.dst_linear_state_ids
                               : kv_info.dst_blocks;
        if (src_ids.empty() || dst_ids.empty()) {
          VLOG(5) << "Skip PushKvBlocks, layer = " << layer_index
                  << ", role = " << registered_cache.role.to_string()
                  << ", src_ids = " << src_ids.size()
                  << ", dst_ids = " << dst_ids.size();
          continue;
        }
        CacheIndex cache_index{kv_info.dst_cluster_id,
                               registered_cache.cache.cache_id};
        KvCacheExtParam ext_param{};
        ext_param.src_layer_range = {0, 0};
        ext_param.dst_layer_range = {0, 0};
        ext_param.tensor_num_per_layer = 1;

        auto ret = llm_data_dist_->PushKvBlocks(
            registered_cache.cache, cache_index, src_ids, dst_ids, ext_param);
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
