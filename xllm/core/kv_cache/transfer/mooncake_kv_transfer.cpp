/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kv_cache/transfer/mooncake_kv_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/kv_cache/layout/cache_layout_builder.h"
#include "core/kv_cache/layout/kv_cache_layout_policy.h"
#include "core/kv_cache/storage/kv_cache_tensors.h"
#include "core/util/tensor_helper.h"
#include "util/net.h"
#include "util/uuid.h"

namespace xllm {

namespace {

std::string get_merge_key(const uint64_t dst_cluster_id,
                          const std::string& dst_addr) {
  return std::to_string(dst_cluster_id) + "_" + dst_addr;
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
std::vector<std::string> push_destination_keys(
    const std::unordered_map<std::string, KVTransfer::KVCacheInfo>& kv_infos,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  std::vector<std::string> keys;
  keys.reserve(kv_infos.size());
  for (const auto& destination : kv_infos) {
    keys.push_back(destination.first);
  }
  if (kv_split_size > 1 && keys.size() > 1) {
    std::sort(keys.begin(), keys.end());
    std::rotate(
        keys.begin(), keys.begin() + kv_split_rank % keys.size(), keys.end());
  }
  return keys;
}
#endif

std::vector<std::string> collect_failed_request_ids(
    const std::unordered_map<std::string, KVTransfer::KVCacheInfo>& kv_infos,
    const std::unordered_set<std::string>& failed_transfer_keys) {
  size_t count = 0;
  for (const std::string& key : failed_transfer_keys) {
    count += kv_infos.at(key).request_ids().size();
  }
  std::vector<std::string> request_ids;
  request_ids.reserve(count);
  for (const std::string& key : failed_transfer_keys) {
    const std::vector<std::string>& ids = kv_infos.at(key).request_ids();
    request_ids.insert(request_ids.end(), ids.begin(), ids.end());
  }
  return request_ids;
}

void validate_virtual_kv_cache_global_offsets(
    const VirtualKVCacheLayerGlobalOffsets& offsets) {
  CHECK_EQ(offsets.k_offsets.size(), offsets.v_offsets.size());
  CHECK(offsets.index_offsets.empty() ||
        offsets.index_offsets.size() == offsets.k_offsets.size());
}

void merge_virtual_kv_cache_global_offsets(
    std::vector<VirtualKVCacheLayerGlobalOffsets>& merged_layer_offsets,
    const std::vector<VirtualKVCacheLayerGlobalOffsets>& layer_offsets) {
  if (layer_offsets.empty()) {
    return;
  }
  for (const VirtualKVCacheLayerGlobalOffsets& offsets : layer_offsets) {
    validate_virtual_kv_cache_global_offsets(offsets);
  }
  if (merged_layer_offsets.empty()) {
    merged_layer_offsets = layer_offsets;
    return;
  }

  CHECK_EQ(merged_layer_offsets.size(), layer_offsets.size());
  for (size_t layer_id = 0; layer_id < layer_offsets.size(); ++layer_id) {
    validate_virtual_kv_cache_global_offsets(merged_layer_offsets[layer_id]);
    auto& index_target = merged_layer_offsets[layer_id].index_offsets;
    const auto& index_source = layer_offsets[layer_id].index_offsets;
    auto& k_target = merged_layer_offsets[layer_id].k_offsets;
    const auto& k_source = layer_offsets[layer_id].k_offsets;
    if (!k_target.empty() && !k_source.empty()) {
      CHECK_EQ(index_target.empty(), index_source.empty());
    }
    index_target.reserve(index_target.size() + index_source.size());
    index_target.insert(
        index_target.end(), index_source.begin(), index_source.end());
    k_target.reserve(k_target.size() + k_source.size());
    k_target.insert(k_target.end(), k_source.begin(), k_source.end());

    std::vector<uint64_t>& v_target = merged_layer_offsets[layer_id].v_offsets;
    const std::vector<uint64_t>& v_source = layer_offsets[layer_id].v_offsets;
    v_target.reserve(v_target.size() + v_source.size());
    v_target.insert(v_target.end(), v_source.begin(), v_source.end());
  }
}

int64_t physical_rows_per_resource(KVCacheTensorRole role,
                                   const CacheTensorLayoutContext& context) {
  if (role == KVCacheTensorRole::SSM) {
    return context.linear_ssm_checkpoint_stride;
  }
  if (role == KVCacheTensorRole::INDEX ||
      role == KVCacheTensorRole::INDEX_SCALE) {
    return num_index_pages_per_block();
  }
  if (role == KVCacheTensorRole::KEY || role == KVCacheTensorRole::VALUE ||
      role == KVCacheTensorRole::WINDOW) {
    return context.num_kv_pages_per_block;
  }
  return 1;
}

CacheTensorManifest make_cache_tensor_manifest(
    const CacheTensorLayoutContext& tensor_layout,
    KVCacheTensor& cache_tensor,
    int64_t layer_id) {
  std::string descriptor_error;
  CHECK(describe_cache_tensor(tensor_layout, &cache_tensor, &descriptor_error))
      << "Failed to describe cache tensor, layer=" << layer_id
      << ", role=" << cache_tensor.role.to_string() << ": " << descriptor_error;
  const torch::Tensor& tensor = cache_tensor.tensor;
  CacheTensorManifest manifest;
  manifest.layer_id = layer_id;
  manifest.role = static_cast<int32_t>(
      static_cast<KVCacheTensorRole::Value>(cache_tensor.role));
  manifest.group_id = cache_tensor.group_id;
  manifest.scalar_type = static_cast<int32_t>(tensor.scalar_type());
  manifest.element_bytes = tensor.element_size();
  manifest.shape = tensor.sizes().vec();
  manifest.stride = tensor.strides().vec();
  manifest.contiguous = true;
  manifest.block_token_capacity = static_cast<uint64_t>(
      std::max<int64_t>(tensor_layout.block_token_capacity, 0));
  manifest.shard = std::move(cache_tensor.shard_descriptor.value());
  return manifest;
}

void merge_kv_info(
    std::unordered_map<std::string, KVTransfer::KVCacheInfo>& merged_kv_infos,
    const TransferKVInfo& info,
    const int32_t dst_rank) {
  uint64_t dst_cluster_id = info.remote_instance_info.cluster_ids[dst_rank];
  const std::string& dst_addr = info.remote_instance_info.addrs[dst_rank];
  std::string key = get_merge_key(dst_cluster_id, dst_addr);
  auto [it, inserted] = merged_kv_infos.try_emplace(std::move(key));
  KVTransfer::KVCacheInfo& kv_info = it->second;
  if (inserted) {
    kv_info.dst_addr = dst_addr;
  }
  for (const KVTransferMapping& src_mapping : info.mappings) {
    auto it = std::find_if(kv_info.mappings.begin(),
                           kv_info.mappings.end(),
                           [&src_mapping](const KVTransferMapping& mapping) {
                             return mapping.group_id == src_mapping.group_id;
                           });
    if (it == kv_info.mappings.end()) {
      kv_info.mappings.emplace_back(src_mapping);
      continue;
    }
    it->local_ids.insert(it->local_ids.end(),
                         src_mapping.local_ids.begin(),
                         src_mapping.local_ids.end());
    it->remote_ids.insert(it->remote_ids.end(),
                          src_mapping.remote_ids.begin(),
                          src_mapping.remote_ids.end());
  }
  kv_info.append_unique_request_id(info.request_id);
  merge_virtual_kv_cache_global_offsets(
      kv_info.dst_virtual_kv_cache_layer_global_offsets,
      info.dst_virtual_kv_cache_layer_global_offsets);
}

}  // namespace

// ============================================================================
// MooncakeKVTransferBase
// ============================================================================

MooncakeKVTransferBase::MooncakeKVTransferBase(const uint16_t listen_port,
                                               const torch::Device& device)
    : listen_port_(listen_port),
      mooncake_te_(
          std::make_unique<MooncakeKVTransferEngine>(listen_port, device)) {
  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port_);
  ShortUUID uuid;
  incarnation_id_ = uuid.random();
}

void MooncakeKVTransferBase::initialize() {
  addr_ = mooncake_te_->initialize();
}

void MooncakeKVTransferBase::configure_cache_layout(
    const ParallelArgs& parallel_args,
    const ModelArgs& model_args,
    int32_t block_token_capacity) {
  CHECK_GT(parallel_args.dp_size(), 0);
  CHECK_GT(parallel_args.cp_size(), 0);
  CHECK_GT(parallel_args.world_size(), 0);
  CHECK_EQ(parallel_args.world_size() %
               (parallel_args.dp_size() * parallel_args.cp_size()),
           0);

  CacheRegistrationContext context;
  const int32_t tp_size = parallel_args.world_size() / parallel_args.dp_size() /
                          parallel_args.cp_size();
  const int32_t rank_in_dp =
      parallel_args.rank() % (parallel_args.cp_size() * tp_size);
  context.coordinates.dp_rank =
      parallel_args.rank() / (parallel_args.cp_size() * tp_size);
  context.coordinates.dp_size = parallel_args.dp_size();
  context.coordinates.tp_rank = rank_in_dp % tp_size;
  context.coordinates.tp_size = tp_size;
  context.coordinates.cp_rank = rank_in_dp / tp_size;
  context.coordinates.cp_size = parallel_args.cp_size();
  context.coordinates.kv_split_rank = parallel_args.kv_split_rank();
  context.coordinates.kv_split_size = parallel_args.kv_split_size_effective();

  CacheTensorLayoutContext& tensor_layout = context.tensor_layout;
  tensor_layout.tp_rank = context.coordinates.tp_rank;
  tensor_layout.tp_size = context.coordinates.tp_size;
  tensor_layout.block_token_capacity = block_token_capacity;
  tensor_layout.kv_head_count =
      model_args.n_kv_heads().value_or(model_args.n_heads());
  tensor_layout.linear_key_head_count = model_args.linear_num_key_heads();
  tensor_layout.linear_value_head_count = model_args.linear_num_value_heads();
  tensor_layout.linear_key_head_dim = model_args.linear_key_head_dim();
  tensor_layout.enable_mla = model_args.enable_mla();
#if defined(USE_MLU) || defined(USE_ILU)
  tensor_layout.head_major_layout = true;
  context.layout_family = "head_token_dim";
#else
  context.layout_family = "token_head_dim";
#endif

#if defined(USE_NPU)
  context.backend = "npu";
#elif defined(USE_MLU)
  context.backend = "mlu";
#elif defined(USE_DCU)
  context.backend = "dcu";
#else
  context.backend = "cpu";
#endif
  std::ostringstream fingerprint;
  fingerprint << model_args.model_type() << ":" << model_args.n_layers() << ":"
              << tensor_layout.kv_head_count << ":" << model_args.head_dim()
              << ":" << model_args.index_n_heads();
  context.fingerprint = fingerprint.str();
  pending_registration_context_ = std::move(context);
}

void MooncakeKVTransferBase::get_cache_info(uint64_t& cluster_id,
                                            std::string& addr) {
  cluster_id = cluster_id_;
  addr = addr_;

  LOG(INFO) << "get_cache_info success, cluster_id=" << cluster_id_
            << ", addr=" << addr_;
}

bool MooncakeKVTransferBase::link_clusters(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs,
    const std::vector<uint16_t>& ports) {
  if (cluster_ids.size() != ports.size()) {
    LOG(ERROR) << "MoonCake link endpoint and port counts differ.";
    return false;
  }
  return mooncake_te_->link_sessions(cluster_ids, remote_addrs);
}

bool MooncakeKVTransferBase::unlink_cluster(const uint64_t& cluster_id,
                                            const std::string& remote_addr,
                                            const uint16_t port,
                                            bool force_flag) {
  LOG(INFO) << "unlink_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->close_session(cluster_id, remote_addr);
}

// ============================================================================
// MooncakeKVTransferDefault
// ============================================================================

uint32_t MooncakeKVTransferDefault::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType) {
  CHECK_LT(cache_pool_layouts_.size(), kMaxNumCachePools)
      << "Mooncake cache pool count exceeds the RPC limit.";
  const auto cache_pool_id = static_cast<uint32_t>(cache_pool_layouts_.size());
  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());

  CacheRegistrationContext* registration =
      pending_registration_context_ ? &*pending_registration_context_ : nullptr;
  CacheTensorLayoutContext local_layout;
  CacheTensorLayoutContext& tensor_layout =
      registration != nullptr ? registration->tensor_layout : local_layout;
  tensor_layout.linear_ssm_checkpoint_stride =
      kv_cache_shape.linear_ssm_checkpoint_stride();
  tensor_layout.num_kv_pages_per_block =
      kv_cache_shape.num_kv_pages_per_block();
  if (registration != nullptr) {
    if (kv_cache_shape.kpool_layout() ==
            KPoolCacheLayout::COMPRESSED_WITH_TAIL &&
        kv_cache_shape.has_index_cache_shape()) {
      const int64_t token_axis = tensor_layout.head_major_layout ? 2 : 1;
      tensor_layout.index_block_capacity =
          kv_cache_shape.index_cache_shape().at(token_axis);
    }
    if (cache_pool_id == kPrimaryCachePoolId &&
        kv_cache_shape.has_kpool_tail_shape()) {
      const auto& tail_shape = kv_cache_shape.kpool_tail_shape();
      registration->fingerprint +=
          "|compressed_kpool_tail:" + std::to_string(tail_shape.back()) + ":" +
          std::to_string(tail_shape[2]) + ":" +
          std::to_string(tensor_layout.block_token_capacity);
    }
  }

  CachePoolLayout pool_layout(kv_caches.size());
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> buf_bytes;
  const size_t num_expected_buffers = kv_caches.size() * 4;
  addrs.reserve(num_expected_buffers);
  lens.reserve(num_expected_buffers);
  buf_bytes.reserve(num_expected_buffers);
  auto tensor_lifetime = std::make_shared<std::vector<torch::Tensor>>();
  tensor_lifetime->reserve(num_expected_buffers);
  std::vector<CacheTensorManifest> tensor_manifests;
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    std::vector<KVCacheTensor> transfer_tensors =
        kv_caches[static_cast<size_t>(layer_id)].get_cache_tensors();
    std::vector<RegisteredBufferDesc>& layer_buffers =
        pool_layout[static_cast<size_t>(layer_id)];
    layer_buffers.reserve(transfer_tensors.size());
    for (KVCacheTensor& cache_tensor : transfer_tensors) {
      const torch::Tensor& tensor = cache_tensor.tensor;
      CHECK_GT(tensor.dim(), 0);
      CHECK(tensor.is_contiguous())
          << "Mooncake registration requires a contiguous cache tensor";
      const int64_t physical_row_count = tensor.size(0);
      const int64_t rows_per_resource =
          physical_rows_per_resource(cache_tensor.role, tensor_layout);
      CHECK_GT(rows_per_resource, 0);
      CHECK_EQ(physical_row_count % rows_per_resource, 0)
          << "Cache tensor physical rows must be divisible by its logical "
             "resource geometry, role="
          << cache_tensor.role.to_string();
      const int64_t resource_count = physical_row_count / rows_per_resource;
      const size_t logical_bytes = tensor.nbytes();
      const int64_t storage_offset = tensor.storage_offset();
      CHECK_GE(storage_offset, 0);
      const size_t element_size = tensor.element_size();
      CHECK_LE(static_cast<size_t>(storage_offset),
               std::numeric_limits<size_t>::max() / element_size);
      const size_t storage_offset_bytes =
          static_cast<size_t>(storage_offset) * element_size;
      const size_t storage_bytes = tensor.storage().nbytes();
      CHECK_LE(storage_offset_bytes, storage_bytes);
      CHECK_LE(logical_bytes, storage_bytes - storage_offset_bytes)
          << "Mooncake registration exceeds tensor storage capacity";
      const int64_t buffer_id =
          num_registered_buffers_ + static_cast<int64_t>(addrs.size());
      addrs.emplace_back(tensor.data_ptr());
      lens.emplace_back(logical_bytes);
      buf_bytes.emplace_back(logical_bytes /
                             static_cast<size_t>(resource_count));
      tensor_lifetime->emplace_back(tensor);
      layer_buffers.push_back(
          {buffer_id, cache_tensor.role, cache_tensor.group_id});
      if (registration != nullptr) {
        auto tensor_manifest =
            make_cache_tensor_manifest(tensor_layout, cache_tensor, layer_id);
        tensor_manifest.cache_pool_id = cache_pool_id;
        tensor_manifest.mooncake_buffer_id = buffer_id;
        tensor_manifest.resource_count = static_cast<uint64_t>(resource_count);
        tensor_manifest.physical_rows_per_resource =
            static_cast<uint64_t>(rows_per_resource);
        tensor_manifest.resource_stride_bytes =
            static_cast<uint64_t>(tensor.stride(0)) * element_size *
            static_cast<uint64_t>(rows_per_resource);
        tensor_manifest.buffer_bytes = logical_bytes;
        tensor_manifests.emplace_back(std::move(tensor_manifest));
      }
    }
    CHECK(!layer_buffers.empty())
        << "No Mooncake cache tensor registered at layer " << layer_id;
  }
  const size_t buffer_count = addrs.size();
  if (!mooncake_te_->register_memory(std::move(addrs),
                                     std::move(lens),
                                     std::move(buf_bytes),
                                     std::move(tensor_lifetime))) {
    LOG(FATAL) << "Mooncake cache registration failed";
  }
  cache_pool_layouts_.emplace_back(std::move(pool_layout));
  num_registered_buffers_ += static_cast<int64_t>(buffer_count);
  LOG(INFO) << "Registered Mooncake cache, layers=" << kv_caches.size()
            << ", buffers=" << buffer_count;
  if (registration != nullptr) {
    publish_cache_layout(tensor_manifests, *registration);
    pending_registration_context_.reset();
  }
  return cache_pool_id;
}

void MooncakeKVTransferBase::publish_cache_layout(
    const std::vector<CacheTensorManifest>& tensor_manifests,
    const CacheRegistrationContext& registration_context) {
  CHECK(!tensor_manifests.empty());
  if (tensor_manifests.front().cache_pool_id == kPrimaryCachePoolId) {
    local_cache_layout_.tensors.clear();
    local_cache_layout_.schema_version = kCacheLayoutSchemaVersion;
    local_cache_layout_.incarnation_id = incarnation_id_;
    local_cache_layout_.fingerprint = registration_context.fingerprint;
    local_cache_layout_.backend = registration_context.backend;
    local_cache_layout_.layout_family = registration_context.layout_family;
    local_cache_layout_.cluster_id = cluster_id_;
    local_cache_layout_.addr = addr_;
    local_cache_layout_.listen_port = listen_port_;
    local_cache_layout_.coordinates = registration_context.coordinates;
  } else {
    CHECK(!local_cache_layout_.tensors.empty())
        << "The primary cache pool must be registered first.";
    CHECK_EQ(local_cache_layout_.backend, registration_context.backend);
    CHECK_EQ(local_cache_layout_.layout_family,
             registration_context.layout_family);
    // Keep the fingerprint compatible with existing peers.
    local_cache_layout_.fingerprint.append("|spec:");
    local_cache_layout_.fingerprint.append(registration_context.fingerprint);
  }
  local_cache_layout_.tensors.reserve(local_cache_layout_.tensors.size() +
                                      tensor_manifests.size());
  local_cache_layout_.tensors.insert(local_cache_layout_.tensors.end(),
                                     tensor_manifests.begin(),
                                     tensor_manifests.end());
  local_cache_layout_.layout_generation = ++layout_generation_;
  const Status status =
      mooncake_te_->set_local_cache_layout(local_cache_layout_);
  CHECK(status.ok()) << "Failed to publish cache layout: " << status.message();
}

bool MooncakeKVTransferDefault::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  CHECK(!cache_pool_layouts_.empty()) << "KV cache is not registered.";

  std::unordered_map<int32_t, const KVTransferMapping*> mappings_by_group;
  mappings_by_group.reserve(mappings.size());
  for (const KVTransferMapping& mapping : mappings) {
    if (mapping.local_ids.size() != mapping.remote_ids.size()) {
      LOG(ERROR) << "KV cache mapping size mismatch, group_id="
                 << mapping.group_id << ", local=" << mapping.local_ids.size()
                 << ", remote=" << mapping.remote_ids.size();
      return false;
    }
    if (!mappings_by_group.emplace(mapping.group_id, &mapping).second) {
      LOG(ERROR) << "Duplicate KV cache transfer mapping, group_id="
                 << mapping.group_id;
      return false;
    }
  }

  std::vector<MooncakeTransferEngine::BufferTransferMapping> buffer_mappings;
  buffer_mappings.reserve(static_cast<size_t>(num_registered_buffers_));
  for (const auto& pool_layout : cache_pool_layouts_) {
    for (size_t layer_id = 0; layer_id < pool_layout.size(); ++layer_id) {
      for (const RegisteredBufferDesc& buffer : pool_layout[layer_id]) {
        const auto mapping_it = mappings_by_group.find(buffer.group_id);
        if (mapping_it == mappings_by_group.end()) {
          LOG(ERROR) << "Missing KV cache transfer mapping, layer=" << layer_id
                     << ", buf_id=" << buffer.buf_id
                     << ", role=" << buffer.role.to_string()
                     << ", group_id=" << buffer.group_id;
          return false;
        }
        const KVTransferMapping& mapping = *mapping_it->second;
        if (mapping.local_ids.empty()) {
          continue;
        }
        buffer_mappings.push_back(
            {buffer.buf_id, mapping.local_ids, mapping.remote_ids});
      }
    }
  }
  const bool success = mooncake_te_->move_memory_groups(
      src_addr, buffer_mappings, MooncakeTransferEngine::MoveOpcode::READ);
  if (!success) {
    LOG(ERROR) << "Pull KV cache mappings failed.";
    return false;
  }
  VLOG(1) << "[Mooncake][PDTransfer] direction=pull, cluster_id="
          << src_cluster_id << ", remote=" << src_addr
          << ", mapping_groups=" << mappings.size()
          << ", buffers=" << buffer_mappings.size() << ", success=true";
  return true;
}

void MooncakeKVTransferBase::merge_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    const std::vector<TransferKVInfo>& transfer_kv_infos) {
  for (const TransferKVInfo& info : transfer_kv_infos) {
    const int32_t dst_dp_size = info.remote_instance_info.dp_size;
    const int32_t dst_world_size =
        static_cast<int32_t>(info.remote_instance_info.cluster_ids.size());
    if (dst_dp_size <= 0 || dst_world_size <= 0 ||
        dst_world_size % dst_dp_size != 0 || info.dp_rank < 0 ||
        info.dp_rank >= dst_dp_size ||
        info.remote_instance_info.addrs.size() !=
            info.remote_instance_info.cluster_ids.size()) {
      LOG(ERROR) << "Invalid destination topology while merging KV regions.";
      continue;
    }
    const int32_t dst_tp_size = dst_world_size / dst_dp_size;
    const int32_t begin = info.dp_rank * dst_tp_size;
    const int32_t end = begin + dst_tp_size;
    for (int32_t dst_rank = begin; dst_rank < end; ++dst_rank) {
      merge_kv_info(merged_kv_infos, info, dst_rank);
    }
  }
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
std::vector<std::string> MooncakeKVTransferDefault::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    uint32_t cache_pool_id,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  CHECK_LT(cache_pool_id, cache_pool_layouts_.size())
      << "KV cache pool is not registered.";
  const int64_t num_layers =
      static_cast<int64_t>(cache_pool_layouts_[cache_pool_id].size());

  const auto keys =
      push_destination_keys(merged_kv_infos, kv_split_rank, kv_split_size);

  std::unordered_set<std::string> failed_transfer_keys;
  for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
    if (!layer_synchronizer->synchronize_layer(layer_index)) {
      LOG(ERROR) << "Synchronize KV cache layer failed, layer=" << layer_index;
      failed_transfer_keys.insert(keys.begin(), keys.end());
      continue;
    }
    for (const std::string& key : keys) {
      if (failed_transfer_keys.contains(key)) {
        continue;
      }
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      std::vector<ByteRegion> regions;
      const Status bind_status =
          mooncake_te_->bind_outgoing_regions(kv_info.dst_addr,
                                              kv_info.mappings,
                                              cache_pool_id,
                                              layer_index,
                                              &regions);
      if (!bind_status.ok()) {
        LOG(ERROR) << "Bind KV byte regions failed, layer=" << layer_index
                   << ", destination=" << kv_info.dst_addr << ": "
                   << bind_status.message();
        failed_transfer_keys.insert(key);
        continue;
      }
      if (regions.empty()) {
        continue;
      }
      const bool success = mooncake_te_->move_memory_regions(
          kv_info.dst_addr, regions, MooncakeTransferEngine::MoveOpcode::WRITE);
      if (!success) {
        LOG(ERROR) << "Push KV byte regions failed, layer=" << layer_index
                   << ", destination=" << kv_info.dst_addr;
        failed_transfer_keys.insert(key);
      }
    }
  }
  VLOG(1) << "[Mooncake][PDTransfer] direction=push, destinations="
          << keys.size() << ", layers=" << num_layers
          << ", failed_tasks=" << failed_transfer_keys.size();
  return collect_failed_request_ids(merged_kv_infos, failed_transfer_keys);
}
#endif

// ============================================================================
// MooncakeVirtualKVTransfer
// ============================================================================

MooncakeVirtualKVTransfer::MooncakeVirtualKVTransfer(
    const uint16_t listen_port,
    const torch::Device& device,
    std::unique_ptr<VirtualKVTransferMemoryProvider> memory_provider)
    : MooncakeKVTransferBase(listen_port, device),
      memory_provider_(std::move(memory_provider)) {
  CHECK(memory_provider_ != nullptr);
}

uint32_t MooncakeVirtualKVTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  CHECK_EQ(num_layers_, 0) << "Virtual memory supports one cache pool.";
  num_layers_ = static_cast<int64_t>(kv_caches.size());
  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();

  CHECK(!key_cache_shape.empty());
  const int64_t data_size =
      static_cast<int64_t>(torch::scalarTypeToTypeMeta(dtype).itemsize());
  int64_t count_per_block = 1;
  for (size_t i = 1; i < key_cache_shape.size(); ++i) {
    CHECK_GT(key_cache_shape[i], 0);
    CHECK_LE(count_per_block,
             std::numeric_limits<int64_t>::max() / key_cache_shape[i]);
    count_per_block *= key_cache_shape[i];
  }
  CHECK_LE(count_per_block, std::numeric_limits<int64_t>::max() / data_size);
  const uint64_t block_size_bytes = count_per_block * data_size;

  const VirtualKVTransferMemoryRegion memory_region =
      memory_provider_->memory_region();
  CHECK(memory_region.fully_resident)
      << "Mooncake KV transfer requires a fully resident global arena; "
         "sparse elastic arenas need an extent-aware transfer protocol; use "
         "--vmm_elastic=false for legacy dense VMM transfer";
  CHECK(memory_region.base_address != nullptr);
  CHECK_GT(memory_region.size_bytes, 0);

  std::vector<CacheTensorManifest> tensor_manifests;
  if (pending_registration_context_.has_value()) {
    auto& registration = *pending_registration_context_;
    registration.layout_family.append("_virtual_memory_v2");
    CHECK_GT(memory_region.page_size_bytes, 0);
    tensor_manifests.reserve(kv_caches.size() * 3);
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      std::vector<KVCacheTensor> transfer_tensors =
          kv_caches[static_cast<size_t>(layer_id)].get_cache_tensors();
      for (KVCacheTensor& cache_tensor : transfer_tensors) {
        CHECK(cache_tensor.role == KVCacheTensorRole::KEY ||
              cache_tensor.role == KVCacheTensorRole::VALUE ||
              cache_tensor.role == KVCacheTensorRole::INDEX)
            << "VirtualMemory MoonCake transfer supports only K/V/INDEX "
               "tensors, "
               "role="
            << cache_tensor.role.to_string();
        const torch::Tensor& tensor = cache_tensor.tensor;
        CHECK(tensor.is_contiguous());
        CHECK_GT(tensor.dim(), 0);
        const int64_t resource_count = tensor.size(0);
        const uint64_t tensor_bytes = static_cast<uint64_t>(tensor.nbytes());
        const uint64_t resource_stride =
            tensor_bytes / static_cast<uint64_t>(resource_count);
        CHECK_LE(resource_stride, memory_region.page_size_bytes)
            << "VirtualMemory transfer tensor blocks must fit one page";
        CHECK_EQ(memory_region.page_size_bytes % resource_stride, 0)
            << "VirtualMemory transfer tensor blocks must divide the page size";

        auto manifest = make_cache_tensor_manifest(
            registration.tensor_layout, cache_tensor, layer_id);
        manifest.resource_count = static_cast<uint64_t>(resource_count);
        manifest.resource_stride_bytes = resource_stride;
        manifest.buffer_bytes = static_cast<uint64_t>(memory_region.size_bytes);
        manifest.explicit_resource_offsets = true;
        tensor_manifests.emplace_back(std::move(manifest));
      }
    }
  }

  if (!mooncake_te_->register_memory({memory_region.base_address},
                                     {memory_region.size_bytes},
                                     {block_size_bytes},
                                     memory_region.lifetime)) {
    LOG(FATAL) << "register GlobalVirtualMemoryRegion failed";
  }

  LOG(INFO) << "Registered Mooncake virtual cache, total_size="
            << memory_region.size_bytes
            << ", size_per_block=" << block_size_bytes;
  if (pending_registration_context_.has_value()) {
    publish_cache_layout(tensor_manifests, *pending_registration_context_);
    const Status provider_status =
        mooncake_te_->set_virtual_memory_provider(model_id_, memory_provider_);
    CHECK(provider_status.ok()) << provider_status.message();
    pending_registration_context_.reset();
  }
  return kPrimaryCachePoolId;
}

bool MooncakeVirtualKVTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  if (model_id_.empty()) {
    LOG(ERROR) << "Invalid VirtualMemory PULL model or KV mapping";
    return false;
  }

  const KVTransferMapping* kv_mapping = nullptr;
  for (const KVTransferMapping& candidate : mappings) {
    if (candidate.group_id == cache_group_id(BlockType::KV)) {
      if (kv_mapping != nullptr ||
          candidate.local_ids.size() != candidate.remote_ids.size()) {
        LOG(ERROR) << "Invalid VirtualMemory PULL model or KV mapping";
        return false;
      }
      kv_mapping = &candidate;
      continue;
    }
    if (!candidate.local_ids.empty() || !candidate.remote_ids.empty()) {
      LOG(ERROR) << "VirtualMemory PULL only supports empty auxiliary mappings";
      return false;
    }
  }
  if (kv_mapping == nullptr) {
    LOG(ERROR) << "Invalid VirtualMemory PULL model or KV mapping";
    return false;
  }
  const KVTransferMapping& mapping = *kv_mapping;
  if (mapping.local_ids.empty()) {
    return true;
  }

  // Each source worker resolves its own physical mapping. Matching logical
  // block IDs on the two instances do not imply matching physical pages.
  WorkerCacheLayoutManifest source_manifest;
  std::vector<VirtualKVCacheLayerGlobalOffsets> source_offsets;
  VirtualKVPullSession pull_session;
  if (!mooncake_te_->get_remote_virtual_offsets(src_cluster_id,
                                                src_addr,
                                                mapping.remote_ids,
                                                &source_manifest,
                                                &source_offsets,
                                                pull_session) ||
      num_layers_ <= 0 ||
      source_offsets.size() != static_cast<size_t>(num_layers_)) {
    LOG(ERROR) << "Failed to resolve VirtualMemory PULL source offsets";
    return false;
  }

  std::vector<StridedRegionTemplate> plan;
  const Status plan_status = ReshardPlanner().build_native_pull_plan(
      local_cache_layout_, source_manifest, &plan);
  if (!plan_status.ok()) {
    LOG(ERROR) << "Invalid VirtualMemory PULL layout: "
               << plan_status.message();
    return false;
  }

  // Validate all layers before writing any destination cache data.
  std::vector<std::vector<ByteRegion>> layer_regions;
  layer_regions.resize(static_cast<size_t>(num_layers_));
  RequestRegionBinder binder;
  for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
    const size_t layer_index = static_cast<size_t>(layer_id);
    std::vector<ExplicitResourceMapping> explicit_mappings;
    if (!build_explicit_mappings(mapping,
                                 source_offsets[layer_index],
                                 layer_id,
                                 explicit_mappings)) {
      return false;
    }
    const Status bind_status =
        binder.bind_explicit(plan,
                             explicit_mappings,
                             kPrimaryCachePoolId,
                             layer_id,
                             &layer_regions[layer_index]);
    if (!bind_status.ok() || layer_regions[layer_index].empty()) {
      LOG(ERROR) << "Bind VirtualMemory PULL regions failed, layer=" << layer_id
                 << ": " << bind_status.message();
      return false;
    }
  }

  for (const std::vector<ByteRegion>& regions : layer_regions) {
    if (!mooncake_te_->move_memory_regions_with_session(
            src_addr,
            pull_session,
            regions,
            MooncakeTransferEngine::MoveOpcode::READ)) {
      LOG(ERROR) << "VirtualMemory PULL transfer failed, source=" << src_addr;
      return false;
    }
  }
  VLOG(1) << "[Mooncake][PDTransfer] direction=pull, virtual_memory=true, "
             "cluster_id="
          << src_cluster_id << ", remote=" << src_addr
          << ", blocks=" << mapping.local_ids.size() << ", success=true";
  return true;
}

bool MooncakeVirtualKVTransfer::build_explicit_mappings(
    const KVTransferMapping& mapping,
    const VirtualKVCacheLayerGlobalOffsets& remote_offsets,
    int64_t layer_id,
    std::vector<ExplicitResourceMapping>& explicit_mappings) const {
  if (remote_offsets.k_offsets.size() != mapping.local_ids.size() ||
      remote_offsets.v_offsets.size() != mapping.local_ids.size() ||
      (!remote_offsets.index_offsets.empty() &&
       remote_offsets.index_offsets.size() != mapping.local_ids.size())) {
    LOG(ERROR) << "VirtualMemory remote offset count mismatch, layer="
               << layer_id;
    return false;
  }

  explicit_mappings.clear();
  explicit_mappings.reserve(3);
  const auto append_mapping = [&](KVCacheTensorRole role,
                                  const std::vector<uint64_t>& offsets) {
    ExplicitResourceMapping explicit_mapping;
    explicit_mapping.group_id = mapping.group_id;
    explicit_mapping.role =
        static_cast<int32_t>(static_cast<KVCacheTensorRole::Value>(role));
    explicit_mapping.local_ids = mapping.local_ids;
    explicit_mapping.remote_ids = mapping.remote_ids;
    explicit_mapping.remote_offsets = offsets;
    explicit_mapping.local_offsets.reserve(mapping.local_ids.size());
    explicit_mappings.emplace_back(std::move(explicit_mapping));
  };
  append_mapping(KVCacheTensorRole::KEY, remote_offsets.k_offsets);
  append_mapping(KVCacheTensorRole::VALUE, remote_offsets.v_offsets);
  if (!remote_offsets.index_offsets.empty()) {
    append_mapping(KVCacheTensorRole::INDEX, remote_offsets.index_offsets);
  }
  for (uint64_t block_id : mapping.local_ids) {
    const auto local_offsets = memory_provider_->get_global_offsets_for_block(
        model_id_, layer_id, block_id);
    if (!local_offsets.has_value() ||
        local_offsets->index_offset.has_value() !=
            !remote_offsets.index_offsets.empty()) {
      LOG(ERROR) << "Invalid VirtualMemory local mapping or INDEX ownership, "
                    "layer="
                 << layer_id << ", block=" << block_id;
      return false;
    }
    explicit_mappings[0].local_offsets.emplace_back(local_offsets->k_offset);
    explicit_mappings[1].local_offsets.emplace_back(local_offsets->v_offset);
    if (local_offsets->index_offset.has_value()) {
      explicit_mappings[2].local_offsets.emplace_back(
          *local_offsets->index_offset);
    }
  }

  return true;
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
std::vector<std::string> MooncakeVirtualKVTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    uint32_t,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  if (model_id_.empty()) {
    LOG(ERROR) << "model_id not set for VirtualMemory mode push";
    std::vector<std::string> results;
    results.reserve(merged_kv_infos.size());
    for (const auto& entry : merged_kv_infos) {
      const auto& request_ids = entry.second.request_ids();
      results.insert(results.end(), request_ids.begin(), request_ids.end());
    }
    return results;
  }

  const auto keys =
      push_destination_keys(merged_kv_infos, kv_split_rank, kv_split_size);

  std::unordered_set<std::string> failed_transfer_keys;
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    if (!layer_synchronizer->synchronize_layer(layer_index)) {
      LOG(ERROR) << "Synchronize VirtualMemory KV cache layer failed, layer="
                 << layer_index;
      failed_transfer_keys.insert(keys.begin(), keys.end());
      continue;
    }

    for (const std::string& key : keys) {
      if (failed_transfer_keys.contains(key)) {
        continue;
      }
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      const auto mapping_it = std::find_if(
          kv_info.mappings.begin(),
          kv_info.mappings.end(),
          [](const KVTransferMapping& mapping) {
            return mapping.group_id == cache_group_id(BlockType::KV);
          });
      if (mapping_it == kv_info.mappings.end()) {
        LOG(ERROR) << "Missing VirtualMemory KV transfer mapping.";
        failed_transfer_keys.insert(key);
        continue;
      }
      if (mapping_it->local_ids.size() != mapping_it->remote_ids.size()) {
        LOG(ERROR) << "VirtualMemory KV transfer mapping size mismatch, local="
                   << mapping_it->local_ids.size()
                   << ", remote=" << mapping_it->remote_ids.size();
        failed_transfer_keys.insert(key);
        continue;
      }
      const std::vector<uint64_t>& src_blocks = mapping_it->local_ids;
      if (src_blocks.empty()) {
        continue;
      }

      if (static_cast<size_t>(layer_index) >=
          kv_info.dst_virtual_kv_cache_layer_global_offsets.size()) {
        LOG(ERROR) << "Missing VirtualMemory destination offsets, layer="
                   << layer_index;
        failed_transfer_keys.insert(key);
        continue;
      }
      std::vector<ExplicitResourceMapping> explicit_mappings;
      if (!build_explicit_mappings(
              *mapping_it,
              kv_info.dst_virtual_kv_cache_layer_global_offsets[layer_index],
              layer_index,
              explicit_mappings)) {
        failed_transfer_keys.insert(key);
        continue;
      }

      std::vector<ByteRegion> regions;
      const Status bind_status =
          mooncake_te_->bind_outgoing_regions_explicit(kv_info.dst_addr,
                                                       explicit_mappings,
                                                       kPrimaryCachePoolId,
                                                       layer_index,
                                                       &regions);
      if (!bind_status.ok()) {
        LOG(ERROR) << "Bind VirtualMemory KV byte regions failed, layer="
                   << layer_index << ", destination=" << kv_info.dst_addr
                   << ": " << bind_status.message();
        failed_transfer_keys.insert(key);
        continue;
      }
      const bool ret =
          regions.empty() || mooncake_te_->move_memory_regions(
                                 kv_info.dst_addr,
                                 regions,
                                 MooncakeTransferEngine::MoveOpcode::WRITE);
      if (!ret) {
        LOG(ERROR) << "VirtualMemory PUSH transfer failed, layer="
                   << layer_index;
        failed_transfer_keys.insert(key);
      }
    }
  }

  VLOG(1) << "[Mooncake][PDTransfer] direction=push, virtual_memory=true, "
             "layers="
          << num_layers_ << ", failed_tasks=" << failed_transfer_keys.size();
  return collect_failed_request_ids(merged_kv_infos, failed_transfer_keys);
}
#endif

}  // namespace xllm
