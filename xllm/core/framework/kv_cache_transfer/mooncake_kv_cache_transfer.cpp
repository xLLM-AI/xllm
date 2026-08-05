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

#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/push_route.h"
#include "framework/xtensor/global_xtensor.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "util/net.h"

namespace xllm {

namespace {

std::string get_merge_key(const uint64_t dst_cluster_id,
                          const std::string& dst_addr) {
  return std::to_string(dst_cluster_id) + "_" + dst_addr;
}

void merge_xtensor_offsets(
    std::vector<XTensorLayerOffsets>& merged_layer_offsets,
    const std::vector<XTensorLayerOffsets>& layer_offsets) {
  if (layer_offsets.empty()) {
    return;
  }
  if (merged_layer_offsets.empty()) {
    merged_layer_offsets = layer_offsets;
    return;
  }

  for (size_t layer_id = 0; layer_id < layer_offsets.size() &&
                            layer_id < merged_layer_offsets.size();
       ++layer_id) {
    std::vector<uint64_t>& k_target = merged_layer_offsets[layer_id].k_offsets;
    const std::vector<uint64_t>& k_source = layer_offsets[layer_id].k_offsets;
    k_target.reserve(k_target.size() + k_source.size());
    k_target.insert(k_target.end(), k_source.begin(), k_source.end());

    std::vector<uint64_t>& v_target = merged_layer_offsets[layer_id].v_offsets;
    const std::vector<uint64_t>& v_source = layer_offsets[layer_id].v_offsets;
    v_target.reserve(v_target.size() + v_source.size());
    v_target.insert(v_target.end(), v_source.begin(), v_source.end());
  }
}

std::vector<KVCacheTensor> get_mooncake_tensors(const KVCache& cache) {
  return cache.get_cache_tensors();
}

void append_mappings(std::vector<KVTransferMapping>& dst,
                     const std::vector<KVTransferMapping>& src) {
  for (const KVTransferMapping& src_mapping : src) {
    auto it = std::find_if(dst.begin(),
                           dst.end(),
                           [&src_mapping](const KVTransferMapping& mapping) {
                             return mapping.group_id == src_mapping.group_id;
                           });
    if (it == dst.end()) {
      dst.emplace_back(src_mapping);
      continue;
    }
    it->local_ids.insert(it->local_ids.end(),
                         src_mapping.local_ids.begin(),
                         src_mapping.local_ids.end());
    it->remote_ids.insert(it->remote_ids.end(),
                          src_mapping.remote_ids.begin(),
                          src_mapping.remote_ids.end());
  }
}

void merge_kv_info(
    std::unordered_map<std::string, KVCacheTransfer::KVCacheInfo>&
        merged_kv_infos,
    const TransferKVInfo& info,
    const int32_t dst_rank) {
  uint64_t dst_cluster_id = info.remote_instance_info.cluster_ids[dst_rank];
  const std::string& dst_addr = info.remote_instance_info.addrs[dst_rank];
  std::string key = get_merge_key(dst_cluster_id, dst_addr);

  auto it = merged_kv_infos.find(key);
  if (it == merged_kv_infos.end()) {
    KVCacheTransfer::KVCacheInfo kv_info;
    kv_info.dst_cluster_id = dst_cluster_id;
    kv_info.dst_addr = dst_addr;
    append_mappings(kv_info.mappings, info.mappings);
    merge_xtensor_offsets(kv_info.dst_xtensor_layer_offsets,
                          info.dst_xtensor_layer_offsets);
    merged_kv_infos.emplace(key, std::move(kv_info));
    return;
  }

  append_mappings(it->second.mappings, info.mappings);
  merge_xtensor_offsets(it->second.dst_xtensor_layer_offsets,
                        info.dst_xtensor_layer_offsets);
}

}  // namespace

// ============================================================================
// MooncakeKVCacheTransferBase
// ============================================================================

MooncakeKVCacheTransferBase::MooncakeKVCacheTransferBase(
    const int32_t device_id,
    const uint16_t listen_port,
    const torch::Device& device,
    std::unique_ptr<MooncakeTransferEngine> engine)
    : device_id_(device_id),
      device_(device),
      listen_port_(listen_port),
      mooncake_te_(std::move(engine)) {
  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port_);
}

void MooncakeKVCacheTransferBase::initialize(int32_t device_id) {
  (void)device_id;
  addr_ = mooncake_te_->initialize();
}

void MooncakeKVCacheTransferBase::get_cache_info(uint64_t& cluster_id,
                                                 std::string& addr) {
  cluster_id = cluster_id_;
  addr = addr_;

  LOG(INFO) << "get_cache_info success, cluster_id=" << cluster_id_
            << ", addr=" << addr_;
}

bool MooncakeKVCacheTransferBase::link_cluster(const uint64_t cluster_id,
                                               const std::string& remote_addr,
                                               const uint16_t port) {
  LOG(INFO) << "link_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->open_session(cluster_id, remote_addr);
}

bool MooncakeKVCacheTransferBase::unlink_cluster(const uint64_t& cluster_id,
                                                 const std::string& remote_addr,
                                                 const uint16_t port,
                                                 bool force_flag) {
  LOG(INFO) << "unlink_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->close_session(cluster_id, remote_addr);
}

// ============================================================================
// MooncakeKVCacheTransferDefault
// ============================================================================

MooncakeKVCacheTransferDefault::MooncakeKVCacheTransferDefault(
    const int32_t device_id,
    const uint16_t listen_port,
    const torch::Device& device,
    const std::string& model_type)
    : MooncakeKVCacheTransferBase(
          device_id,
          listen_port,
          device,
          std::make_unique<MooncakeTransferEngine>(listen_port, device)) {
  (void)model_type;
}

MooncakeKVCacheTransferDefault::MooncakeKVCacheTransferDefault(
    const int32_t device_id,
    const uint16_t listen_port,
    const torch::Device& device,
    const std::string& model_type,
    std::unique_ptr<MooncakeTransferEngine> engine)
    : MooncakeKVCacheTransferBase(device_id,
                                  listen_port,
                                  device,
                                  std::move(engine)) {
  (void)model_type;
}

void MooncakeKVCacheTransferDefault::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  const bool is_spec_draft = main_layout_.registered;
  CHECK(!is_spec_draft || !spec_layout_.registered)
      << "Spec draft kv cache is already registered.";

  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  bool has_v_cache = true;
  if (!kv_caches.empty()) {
    torch::Tensor value_cache = kv_caches[0].get_v_cache();
    has_v_cache = value_cache.defined() && value_cache.numel() > 0;
  }

  (void)kv_cache_shape;
  (void)dtype;

  BufLayout layout;
  layout.num_layers = num_layers;
  layout.layers.resize(static_cast<size_t>(num_layers));
  if (is_spec_draft) {
    layout.offset = main_layout_.offset + main_layout_.total_buf_cnt;
  }
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    const std::vector<KVCacheTensor> transfer_tensors =
        get_mooncake_tensors(kv_caches[static_cast<size_t>(layer_id)]);
    std::vector<RegisteredBufferDesc>& layer_buffers =
        layout.layers[static_cast<size_t>(layer_id)];
    layer_buffers.reserve(transfer_tensors.size());
    for (const KVCacheTensor& cache_tensor : transfer_tensors) {
      const torch::Tensor& tensor = cache_tensor.tensor;
      CHECK(tensor.defined() && tensor.numel() > 0)
          << "Mooncake cache tensor must be allocated, layer=" << layer_id
          << ", role=" << cache_tensor.role.to_string();
      CHECK_GT(tensor.dim(), 0);
      const int64_t block_count = tensor.size(0);
      CHECK_GT(block_count, 0);
      const uint64_t logical_bytes = static_cast<uint64_t>(tensor.nbytes());
      CHECK_EQ(logical_bytes % static_cast<uint64_t>(block_count), 0);

      RegisteredBufferDesc desc{
          layout.offset + layout.total_buf_cnt,
          cache_tensor.role,
          cache_tensor.group_id,
          logical_bytes / static_cast<uint64_t>(block_count)};
      layer_buffers.emplace_back(std::move(desc));
      ++layout.total_buf_cnt;
    }
    CHECK(!layer_buffers.empty())
        << "No Mooncake cache tensor registered at layer " << layer_id;
  }
  layout.registered = true;

  if (!is_spec_draft) {
    num_layers_ = num_layers;
    has_v_cache_ = has_v_cache;
    main_layout_ = layout;
  } else {
    spec_layout_ = layout;
  }

  register_kv_cache_impl(kv_caches);
}

void MooncakeKVCacheTransferDefault::register_kv_cache_spec(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  CHECK(main_layout_.registered)
      << "Main KV cache must be registered before spec draft KV cache.";
  register_kv_cache(kv_caches, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferDefault::add_buf(
    const torch::Tensor& tensor,
    std::vector<void*>& addrs,
    std::vector<size_t>& lens,
    std::vector<uint64_t>& buf_bytes) const {
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }

  CHECK_GT(tensor.dim(), 0) << "cache tensor dim must be positive";
  CHECK(tensor.is_contiguous())
      << "Mooncake registration requires a contiguous cache tensor";
  const int64_t block_count = tensor.size(0);
  CHECK_GT(block_count, 0) << "cache tensor block dim must be positive";

  const int64_t storage_offset = tensor.storage_offset();
  CHECK_GE(storage_offset, 0) << "tensor storage offset must be non-negative";
  const size_t element_size = tensor.element_size();
  CHECK_GT(element_size, static_cast<size_t>(0))
      << "tensor element byte size must be positive";
  CHECK_LE(static_cast<size_t>(storage_offset),
           std::numeric_limits<size_t>::max() / element_size)
      << "tensor storage offset byte size overflow";
  const size_t storage_offset_bytes =
      static_cast<size_t>(storage_offset) * element_size;
  const size_t storage_bytes = tensor.storage().nbytes();
  CHECK_LE(storage_offset_bytes, storage_bytes)
      << "tensor storage offset exceeds storage capacity";
  const size_t available_bytes = storage_bytes - storage_offset_bytes;

  const size_t logical_bytes = static_cast<size_t>(tensor.nbytes());
  CHECK_EQ(logical_bytes % static_cast<size_t>(block_count),
           static_cast<size_t>(0))
      << "cache tensor bytes must be divisible by block count";
  const size_t block_bytes = logical_bytes / static_cast<size_t>(block_count);
  CHECK_GT(block_bytes, static_cast<size_t>(0))
      << "cache tensor block byte size must be positive";

  CHECK_GE(available_bytes, logical_bytes)
      << "Mooncake registration exceeds tensor storage capacity: "
      << "logical_bytes=" << logical_bytes
      << ", available_bytes=" << available_bytes
      << ", block_bytes=" << block_bytes;

  addrs.emplace_back(tensor.data_ptr());
  lens.emplace_back(logical_bytes);
  buf_bytes.emplace_back(static_cast<uint64_t>(block_bytes));
}

bool MooncakeKVCacheTransferDefault::append_buffer_mappings(
    const BufLayout& layout,
    const std::vector<int64_t>& layer_ids,
    const std::vector<KVTransferMapping>& mappings,
    std::vector<MooncakeTransferEngine::BufferTransferMapping>* buffer_mappings)
    const {
  CHECK(buffer_mappings != nullptr);
  CHECK(layout.registered) << "KV cache is not registered.";
  CHECK_EQ(layout.layers.size(), static_cast<size_t>(layout.num_layers));

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

  std::vector<int64_t> active_layer_ids;
  if (layer_ids.empty()) {
    active_layer_ids.resize(static_cast<size_t>(layout.num_layers));
    std::iota(active_layer_ids.begin(), active_layer_ids.end(), 0);
  } else {
    active_layer_ids = layer_ids;
  }

  for (int64_t layer_id : active_layer_ids) {
    CHECK_GE(layer_id, 0) << "layer_id must be non-negative";
    CHECK_LT(layer_id, layout.num_layers) << "layer_id out of range";
  }

  for (int64_t layer_id : active_layer_ids) {
    const std::vector<RegisteredBufferDesc>& layer_buffers =
        layout.layers[static_cast<size_t>(layer_id)];
    for (const RegisteredBufferDesc& buffer : layer_buffers) {
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
      MooncakeTransferEngine::BufferTransferMapping buffer_mapping;
      buffer_mapping.buf_id = buffer.buf_id;
      buffer_mapping.local_ids = mapping.local_ids;
      buffer_mapping.remote_ids = mapping.remote_ids;
      buffer_mappings->emplace_back(std::move(buffer_mapping));
    }
  }
  return true;
}

void MooncakeKVCacheTransferDefault::register_kv_cache_impl(
    const std::vector<xllm::KVCache>& kv_caches) {
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> buf_bytes;
  addrs.reserve(kv_caches.size() * 4);
  lens.reserve(kv_caches.size() * 4);
  buf_bytes.reserve(kv_caches.size() * 4);

  for (const KVCache& cache : kv_caches) {
    const std::vector<KVCacheTensor> transfer_tensors =
        get_mooncake_tensors(cache);
    for (const KVCacheTensor& cache_tensor : transfer_tensors) {
      add_buf(cache_tensor.tensor, addrs, lens, buf_bytes);
    }
  }

  if (!mooncake_te_->register_memory(addrs, lens, buf_bytes)) {
    LOG(FATAL) << "register_kv_cache_impl failed";
  }

  LOG(INFO) << "register_kv_cache_impl success, registered_layers="
            << kv_caches.size() << ", buffers=" << buf_bytes.size();
}

bool MooncakeKVCacheTransferDefault::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  (void)src_cluster_id;
  std::vector<int64_t> layer_ids;
  std::vector<MooncakeTransferEngine::BufferTransferMapping> buffer_mappings;
  if (!append_buffer_mappings(
          main_layout_, layer_ids, mappings, &buffer_mappings)) {
    return false;
  }
  if (spec_layout_.registered &&
      !append_buffer_mappings(
          spec_layout_, layer_ids, mappings, &buffer_mappings)) {
    return false;
  }
  const bool success = mooncake_te_->move_memory_groups(
      src_addr, buffer_mappings, MooncakeTransferEngine::MoveOpcode::READ);
  if (!success) {
    LOG(ERROR) << "Pull KV cache mappings failed.";
    return false;
  }
  return true;
}

void MooncakeKVCacheTransferDefault::merge_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args) {
#if !defined(USE_MLU)
  KVCacheTransfer::merge_kv_blocks(
      merged_kv_infos, transfer_kv_infos, parallel_args);
#else
  if (has_v_cache_) {
    KVCacheTransfer::merge_kv_blocks(
        merged_kv_infos, transfer_kv_infos, parallel_args);
    return;
  }

  int32_t src_rank = parallel_args.rank();
  int32_t src_dp_size = parallel_args.dp_size();
  int32_t src_world_size = parallel_args.world_size();
  int32_t src_tp_size = src_world_size / src_dp_size;
  int32_t src_tp_rank = src_rank % src_tp_size;

  for (const TransferKVInfo& info : transfer_kv_infos) {
    int32_t dst_dp_rank = info.dp_rank;
    int32_t dst_dp_size = info.remote_instance_info.dp_size;
    int32_t dst_world_size =
        static_cast<int32_t>(info.remote_instance_info.cluster_ids.size());
    int32_t dst_tp_size = dst_world_size / dst_dp_size;

    std::unordered_set<int32_t> linked_dp_ranks;
    for (int32_t i = src_tp_rank; i < dst_world_size; i += src_tp_size) {
      int32_t linked_dp_rank = i / dst_tp_size;
      linked_dp_ranks.emplace(linked_dp_rank);
    }
    if (linked_dp_ranks.find(dst_dp_rank) == linked_dp_ranks.end()) {
      continue;
    }

    std::vector<int32_t> dst_ranks =
        get_dst_ranks(src_tp_rank, src_tp_size, dst_tp_size, dst_dp_rank);
    for (int32_t dst_rank : dst_ranks) {
      merge_kv_info(merged_kv_infos, info, dst_rank);
    }
  }
#endif
}

bool MooncakeKVCacheTransferDefault::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  const BufLayout& layout = is_spec_draft ? spec_layout_ : main_layout_;
  CHECK(layout.registered) << "KV cache is not registered.";
  const int64_t num_layers = layout.num_layers;

  std::vector<std::string> keys;
  keys.reserve(merged_kv_infos.size());
  for (const auto& pair : merged_kv_infos) {
    keys.push_back(pair.first);
  }
  if (kv_split_size > 1) {
    keys = rotate_dst_rank(keys, kv_split_rank);
  }

  bool result = true;
  for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
    if (!layer_synchronizer->synchronize_layer(layer_index)) {
      LOG(ERROR) << "Synchronize KV cache layer failed, layer=" << layer_index;
      result = false;
      continue;
    }
    std::vector<int64_t> layer_ids = {layer_index};

    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      std::vector<MooncakeTransferEngine::BufferTransferMapping>
          buffer_mappings;
      if (!append_buffer_mappings(
              layout, layer_ids, kv_info.mappings, &buffer_mappings)) {
        result = false;
        continue;
      }

      const bool success = mooncake_te_->move_memory_groups(
          kv_info.dst_addr,
          buffer_mappings,
          MooncakeTransferEngine::MoveOpcode::WRITE);
      if (!success) {
        LOG(ERROR) << "Push kv blocks failed, layer = " << layer_index
                   << ", destination=" << kv_info.dst_addr;
        result = false;
      }
    }
  }
  return result;
}

// ============================================================================
// MooncakeKVCacheTransferXTensor
// ============================================================================

MooncakeKVCacheTransferXTensor::MooncakeKVCacheTransferXTensor(
    const int32_t device_id,
    const uint16_t listen_port,
    const torch::Device& device)
    : MooncakeKVCacheTransferBase(
          device_id,
          listen_port,
          device,
          std::make_unique<MooncakeTransferEngine>(listen_port, device)) {}

void MooncakeKVCacheTransferXTensor::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = kv_caches.size();
  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();

  int64_t data_size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  int64_t count_per_block = 1;
  for (size_t i = 1; i < key_cache_shape.size(); ++i) {
    count_per_block *= key_cache_shape[i];
  }
  size_per_block_ = count_per_block * data_size;

  register_kv_cache_impl();
}

void MooncakeKVCacheTransferXTensor::register_kv_cache_impl() {
  // XTensor mode registers one shared GlobalXTensor memory region.
  auto& global_xtensor = GlobalXTensor::get_instance();
  if (!global_xtensor.is_initialized()) {
    LOG(FATAL) << "GlobalXTensor not initialized in xtensor mode";
  }

  if (global_xtensor.is_mooncake_registered()) {
    LOG(INFO) << "GlobalXTensor already registered to mooncake, skip";
    return;
  }

  std::vector<void*> addrs = {global_xtensor.base_vaddr()};
  std::vector<size_t> lens = {global_xtensor.total_size()};
  std::vector<uint64_t> buf_bytes = {static_cast<uint64_t>(size_per_block_)};

  if (!mooncake_te_->register_memory(addrs, lens, buf_bytes)) {
    LOG(FATAL) << "register GlobalXTensor failed";
  }

  global_xtensor.set_mooncake_registered(true);
  LOG(INFO) << "register_kv_cache_impl success, total_size="
            << global_xtensor.total_size()
            << ", num_pages=" << global_xtensor.num_total_pages()
            << ", size_per_block=" << size_per_block_;
}

bool MooncakeKVCacheTransferXTensor::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  (void)src_cluster_id;
  const auto mapping_it = std::find_if(
      mappings.begin(), mappings.end(), [](const KVTransferMapping& mapping) {
        return mapping.group_id == cache_group_id(BlockType::KV);
      });
  if (mapping_it == mappings.end()) {
    LOG(ERROR) << "Missing XTensor KV transfer mapping.";
    return false;
  }
  if (mapping_it->local_ids.size() != mapping_it->remote_ids.size()) {
    LOG(ERROR) << "XTensor KV transfer mapping size mismatch, local="
               << mapping_it->local_ids.size()
               << ", remote=" << mapping_it->remote_ids.size();
    return false;
  }
  if (!pull_kv_blocks_impl(
          src_addr, mapping_it->remote_ids, mapping_it->local_ids)) {
    return false;
  }
  return true;
}

bool MooncakeKVCacheTransferXTensor::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  (void)is_spec_draft;
  return push_kv_blocks_impl(
      merged_kv_infos, layer_synchronizer, kv_split_rank, kv_split_size);
}

bool MooncakeKVCacheTransferXTensor::pull_kv_blocks_impl(
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  if (model_id_.empty()) {
    LOG(ERROR) << "model_id not set for XTensor mode pull";
    return false;
  }

  auto& allocator = XTensorAllocator::get_instance();

  // For each layer, convert block_ids to GlobalXTensor offsets and transfer
  for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
    std::vector<uint64_t> src_offsets;
    std::vector<uint64_t> dst_offsets;
    src_offsets.reserve(src_blocks.size() * 2);  // K and V
    dst_offsets.reserve(dst_blocks.size() * 2);

    for (size_t i = 0; i < src_blocks.size(); ++i) {
      // Source block -> GlobalXTensor offsets
      auto [src_k_off, src_v_off] = allocator.get_global_offsets_for_block(
          model_id_, layer_id, src_blocks[i], size_per_block_);
      if (src_k_off == UINT64_MAX || src_v_off == UINT64_MAX) {
        LOG(ERROR) << "Failed to get source offsets for block " << src_blocks[i]
                   << " at layer " << layer_id;
        return false;
      }

      // Destination block -> GlobalXTensor offsets
      auto [dst_k_off, dst_v_off] = allocator.get_global_offsets_for_block(
          model_id_, layer_id, dst_blocks[i], size_per_block_);
      if (dst_k_off == UINT64_MAX || dst_v_off == UINT64_MAX) {
        LOG(ERROR) << "Failed to get dest offsets for block " << dst_blocks[i]
                   << " at layer " << layer_id;
        return false;
      }

      // K cache offsets
      src_offsets.push_back(src_k_off);
      dst_offsets.push_back(dst_k_off);
      // V cache offsets
      src_offsets.push_back(src_v_off);
      dst_offsets.push_back(dst_v_off);
    }

    auto* te = static_cast<MooncakeTransferEngine*>(mooncake_te_.get());
    auto ret = te->move_memory_by_global_offsets(
        src_addr,
        src_offsets,
        dst_offsets,
        size_per_block_,
        MooncakeTransferEngine::MoveOpcode::READ);
    if (!ret) {
      LOG(ERROR) << "pull_kv_blocks_impl failed at layer " << layer_id;
      return false;
    }
  }

  VLOG(1) << "pull_kv_blocks_impl success, num_blocks=" << src_blocks.size()
          << ", num_layers=" << num_layers_;
  return true;
}

bool MooncakeKVCacheTransferXTensor::push_kv_blocks_impl(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    int32_t kv_split_rank,
    int32_t kv_split_size) {
  if (model_id_.empty()) {
    LOG(ERROR) << "model_id not set for XTensor mode push";
    return false;
  }

  std::vector<std::string> keys;
  keys.reserve(merged_kv_infos.size());
  for (const auto& pair : merged_kv_infos) {
    keys.push_back(pair.first);
  }
  if (kv_split_size > 1) {
    keys = rotate_dst_rank(keys, kv_split_rank);
  }

  auto& allocator = XTensorAllocator::get_instance();

  bool result = true;
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    if (!layer_synchronizer->synchronize_layer(layer_index)) {
      LOG(ERROR) << "Synchronize XTensor KV cache layer failed, layer="
                 << layer_index;
      result = false;
      continue;
    }

    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      const auto mapping_it = std::find_if(
          kv_info.mappings.begin(),
          kv_info.mappings.end(),
          [](const KVTransferMapping& mapping) {
            return mapping.group_id == cache_group_id(BlockType::KV);
          });
      if (mapping_it == kv_info.mappings.end()) {
        LOG(ERROR) << "Missing XTensor KV transfer mapping.";
        return false;
      }
      if (mapping_it->local_ids.size() != mapping_it->remote_ids.size()) {
        LOG(ERROR) << "XTensor KV transfer mapping size mismatch, local="
                   << mapping_it->local_ids.size()
                   << ", remote=" << mapping_it->remote_ids.size();
        return false;
      }
      const std::vector<uint64_t>& src_blocks = mapping_it->local_ids;
      if (src_blocks.empty()) {
        continue;
      }

      // Check if we have XTensor offsets from D-node
      bool has_dst_offsets = !kv_info.dst_xtensor_layer_offsets.empty() &&
                             static_cast<size_t>(layer_index) <
                                 kv_info.dst_xtensor_layer_offsets.size();

      std::vector<uint64_t> src_offsets;
      std::vector<uint64_t> dst_offsets;
      src_offsets.reserve(src_blocks.size() * 2);
      dst_offsets.reserve(src_blocks.size() * 2);

      for (size_t i = 0; i < src_blocks.size(); ++i) {
        // Source block -> GlobalXTensor offsets (calculate locally on P-node)
        auto [src_k_off, src_v_off] = allocator.get_global_offsets_for_block(
            model_id_, layer_index, src_blocks[i], size_per_block_);
        if (src_k_off == UINT64_MAX || src_v_off == UINT64_MAX) {
          LOG(ERROR) << "Failed to get source offsets for block "
                     << src_blocks[i] << " at layer " << layer_index;
          return false;
        }

        // Destination offsets: use offsets from D-node if available
        uint64_t dst_k_off, dst_v_off;
        if (has_dst_offsets) {
          const auto& layer_offsets =
              kv_info.dst_xtensor_layer_offsets[layer_index];
          if (i < layer_offsets.k_offsets.size() &&
              i < layer_offsets.v_offsets.size()) {
            dst_k_off = layer_offsets.k_offsets[i];
            dst_v_off = layer_offsets.v_offsets[i];
          } else {
            LOG(ERROR) << "XTensor offset index out of range for block " << i
                       << " at layer " << layer_index;
            return false;
          }
        } else {
          LOG(ERROR) << "No XTensor destination offsets from D-node for layer "
                     << layer_index;
          return false;
        }

        // K cache offsets
        src_offsets.push_back(src_k_off);
        dst_offsets.push_back(dst_k_off);
        // V cache offsets
        src_offsets.push_back(src_v_off);
        dst_offsets.push_back(dst_v_off);
      }
      auto* xtensor_te =
          static_cast<MooncakeTransferEngine*>(mooncake_te_.get());

      auto ret = xtensor_te->move_memory_by_global_offsets(
          kv_info.dst_addr,
          src_offsets,
          dst_offsets,
          size_per_block_,
          MooncakeTransferEngine::MoveOpcode::WRITE);
      if (!ret) {
        LOG(ERROR) << "push_kv_blocks_impl failed at layer " << layer_index;
        result = false;
      }
    }
  }

  VLOG(1) << "push_kv_blocks_impl success, num_layers=" << num_layers_;
  return result;
}

}  // namespace xllm
