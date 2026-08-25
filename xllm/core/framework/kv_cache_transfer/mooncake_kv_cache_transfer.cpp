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

#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>

#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "framework/kv_cache/cache_layout_builder.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/push_route.h"
#include "framework/xtensor/global_xtensor.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "util/net.h"
#include "util/uuid.h"

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

int64_t physical_rows_per_resource(KVCacheTensorRole role,
                                   int64_t ssm_checkpoint_stride) {
  CHECK_GT(ssm_checkpoint_stride, 0);
  return role == KVCacheTensorRole::SSM ? ssm_checkpoint_stride : 1;
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
  ShortUUID uuid;
  incarnation_id_ = uuid.random();
}

void MooncakeKVCacheTransferBase::initialize(int32_t device_id) {
  (void)device_id;
  addr_ = mooncake_te_->initialize();
}

void MooncakeKVCacheTransferBase::configure_cache_layout(
    const ParallelArgs& parallel_args,
    const ModelArgs& model_args,
    int32_t block_token_capacity,
    bool is_spec_draft) {
  CHECK_GT(parallel_args.dp_size(), 0);
  CHECK_GT(parallel_args.cp_size(), 0);
  CHECK_GT(parallel_args.world_size(), 0);
  CHECK_EQ(parallel_args.world_size() % parallel_args.dp_size(), 0);
  CHECK_EQ(parallel_args.world_size() %
               (parallel_args.dp_size() * parallel_args.cp_size()),
           0);

  CacheRegistrationContext context;
  context.cache_namespace =
      is_spec_draft ? CacheNamespace::SPEC_DRAFT : CacheNamespace::MAIN;
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
  tensor_layout.index_head_count = model_args.index_n_heads();
  tensor_layout.linear_key_head_count = model_args.linear_num_key_heads();
  tensor_layout.linear_value_head_count = model_args.linear_num_value_heads();
  tensor_layout.linear_key_head_dim = model_args.linear_key_head_dim();
  tensor_layout.enable_mla = model_args.enable_mla();
#if defined(USE_MLU) || defined(USE_ILU)
  tensor_layout.head_major_layout = true;
  context.layout_family = "head_token_dim";
#else
  tensor_layout.head_major_layout = false;
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
              << ":" << tensor_layout.index_head_count;
  context.fingerprint = fingerprint.str();
  context.cluster_id = cluster_id_;
  context.addr = addr_;
  context.listen_port = listen_port_;
  pending_registration_context_ = std::move(context);
}

void MooncakeKVCacheTransferBase::get_cache_info(uint64_t& cluster_id,
                                                 std::string& addr) {
  cluster_id = cluster_id_;
  addr = addr_;

  LOG(INFO) << "get_cache_info success, cluster_id=" << cluster_id_
            << ", addr=" << addr_;
}

bool MooncakeKVCacheTransferBase::link_clusters(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs,
    const std::vector<uint16_t>& ports) {
  if (cluster_ids.size() != ports.size()) {
    LOG(ERROR) << "MoonCake link endpoint and port counts differ.";
    return false;
  }
  return mooncake_te_->link_sessions(cluster_ids, remote_addrs);
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

  (void)dtype;

  const int64_t ssm_checkpoint_stride =
      kv_cache_shape.linear_ssm_checkpoint_stride();
  if (pending_registration_context_.has_value()) {
    pending_registration_context_->tensor_layout.linear_ssm_checkpoint_stride =
        ssm_checkpoint_stride;
  }

  BufLayout layout;
  layout.num_layers = num_layers;
  layout.layers.resize(static_cast<size_t>(num_layers));
  if (is_spec_draft) {
    layout.offset = main_layout_.offset + main_layout_.total_buf_cnt;
  }
  std::vector<CacheTensorManifest> tensor_manifests;
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
      const int64_t physical_row_count = tensor.size(0);
      CHECK_GT(physical_row_count, 0);
      const int64_t rows_per_resource =
          physical_rows_per_resource(cache_tensor.role, ssm_checkpoint_stride);
      CHECK_EQ(physical_row_count % rows_per_resource, 0)
          << "Cache tensor physical rows must be divisible by its logical "
             "resource geometry, role="
          << cache_tensor.role.to_string();
      const int64_t resource_count = physical_row_count / rows_per_resource;
      const uint64_t logical_bytes = static_cast<uint64_t>(tensor.nbytes());
      CHECK_EQ(logical_bytes % static_cast<uint64_t>(resource_count), 0);

      RegisteredBufferDesc desc{
          layout.offset + layout.total_buf_cnt,
          cache_tensor.role,
          cache_tensor.group_id,
          logical_bytes / static_cast<uint64_t>(resource_count)};
      layer_buffers.emplace_back(std::move(desc));
      if (pending_registration_context_.has_value()) {
        KVCacheTensor described_tensor = cache_tensor;
        std::string descriptor_error;
        CHECK(
            describe_cache_tensor(pending_registration_context_->tensor_layout,
                                  &described_tensor,
                                  &descriptor_error))
            << "Failed to describe cache tensor, layer=" << layer_id
            << ", role=" << cache_tensor.role.to_string() << ": "
            << descriptor_error;
        CHECK(described_tensor.shard_descriptor.has_value());
        CacheTensorManifest tensor_manifest;
        tensor_manifest.cache_namespace =
            pending_registration_context_->cache_namespace;
        tensor_manifest.layer_id = layer_id;
        tensor_manifest.role = static_cast<int32_t>(
            static_cast<KVCacheTensorRole::Value>(cache_tensor.role));
        tensor_manifest.group_id = cache_tensor.group_id;
        tensor_manifest.mooncake_buffer_id =
            layout.offset + layout.total_buf_cnt;
        tensor_manifest.scalar_type =
            static_cast<int32_t>(tensor.scalar_type());
        tensor_manifest.element_bytes = tensor.element_size();
        tensor_manifest.shape = tensor.sizes().vec();
        tensor_manifest.stride = tensor.strides().vec();
        tensor_manifest.storage_offset_bytes = 0;
        tensor_manifest.contiguous = tensor.is_contiguous();
        tensor_manifest.resource_count = static_cast<uint64_t>(resource_count);
        tensor_manifest.physical_rows_per_resource =
            static_cast<uint64_t>(rows_per_resource);
        tensor_manifest.resource_stride_bytes =
            static_cast<uint64_t>(tensor.stride(0)) * tensor.element_size() *
            static_cast<uint64_t>(rows_per_resource);
        tensor_manifest.buffer_bytes = static_cast<uint64_t>(tensor.nbytes());
        tensor_manifest
            .block_token_capacity = static_cast<uint64_t>(std::max<int64_t>(
            pending_registration_context_->tensor_layout.block_token_capacity,
            0));
        tensor_manifest.explicit_resource_offsets = false;
        tensor_manifest.shard =
            std::move(described_tensor.shard_descriptor.value());
        tensor_manifests.emplace_back(std::move(tensor_manifest));
      }
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

  register_kv_cache_impl(kv_caches, ssm_checkpoint_stride);
  if (pending_registration_context_.has_value()) {
    publish_cache_layout(tensor_manifests, *pending_registration_context_);
    pending_registration_context_.reset();
  }
}

void MooncakeKVCacheTransferBase::publish_cache_layout(
    const std::vector<CacheTensorManifest>& tensor_manifests,
    const CacheRegistrationContext& registration_context) {
  CHECK(!tensor_manifests.empty());
  if (registration_context.cache_namespace == CacheNamespace::MAIN) {
    local_cache_layout_.tensors.clear();
    local_cache_layout_.schema_version = kCacheLayoutSchemaVersion;
    local_cache_layout_.incarnation_id = incarnation_id_;
    local_cache_layout_.fingerprint = registration_context.fingerprint;
    local_cache_layout_.backend = registration_context.backend;
    local_cache_layout_.layout_family = registration_context.layout_family;
    local_cache_layout_.cluster_id = registration_context.cluster_id;
    local_cache_layout_.addr = registration_context.addr;
    local_cache_layout_.listen_port = registration_context.listen_port;
    local_cache_layout_.coordinates = registration_context.coordinates;
  } else {
    CHECK(!local_cache_layout_.tensors.empty())
        << "Main cache layout must be published before SPEC_DRAFT.";
    CHECK_EQ(local_cache_layout_.backend, registration_context.backend);
    CHECK_EQ(local_cache_layout_.layout_family,
             registration_context.layout_family);
    // Worker coordinates describe the MAIN cache topology. SPEC_DRAFT tensors
    // carry their independent logical TP placement in each shard descriptor,
    // which permits a replicated TP1 MTP draft body beside a sharded target.
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
    std::vector<uint64_t>& buf_bytes,
    int64_t physical_rows_per_resource) const {
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }

  CHECK_GT(tensor.dim(), 0) << "cache tensor dim must be positive";
  CHECK(tensor.is_contiguous())
      << "Mooncake registration requires a contiguous cache tensor";
  const int64_t block_count = tensor.size(0);
  CHECK_GT(block_count, 0) << "cache tensor block dim must be positive";
  CHECK_GT(physical_rows_per_resource, 0);
  CHECK_EQ(block_count % physical_rows_per_resource, 0)
      << "cache tensor rows must be divisible by physical rows per resource";
  const int64_t resource_count = block_count / physical_rows_per_resource;

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
  CHECK_EQ(logical_bytes % static_cast<size_t>(resource_count),
           static_cast<size_t>(0))
      << "cache tensor bytes must be divisible by resource count";
  const size_t block_bytes =
      logical_bytes / static_cast<size_t>(resource_count);
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
    const std::vector<xllm::KVCache>& kv_caches,
    int64_t ssm_checkpoint_stride) {
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
      add_buf(
          cache_tensor.tensor,
          addrs,
          lens,
          buf_bytes,
          physical_rows_per_resource(cache_tensor.role, ssm_checkpoint_stride));
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
  VLOG(1) << "[Mooncake][PDTransfer] direction=pull, cluster_id="
          << src_cluster_id << ", remote=" << src_addr
          << ", mapping_groups=" << mappings.size()
          << ", buffers=" << buffer_mappings.size() << ", success=true";
  return true;
}

void MooncakeKVCacheTransferBase::merge_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args) {
  (void)parallel_args;
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
  const CacheNamespace cache_namespace =
      is_spec_draft ? CacheNamespace::SPEC_DRAFT : CacheNamespace::MAIN;
  for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
    if (!layer_synchronizer->synchronize_layer(layer_index)) {
      LOG(ERROR) << "Synchronize KV cache layer failed, layer=" << layer_index;
      result = false;
      continue;
    }
    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      std::vector<ByteRegion> regions;
      const Status bind_status =
          mooncake_te_->bind_outgoing_regions(kv_info.dst_addr,
                                              kv_info.mappings,
                                              cache_namespace,
                                              layer_index,
                                              &regions);
      if (!bind_status.ok()) {
        LOG(ERROR) << "Bind KV byte regions failed, layer=" << layer_index
                   << ", destination=" << kv_info.dst_addr << ": "
                   << bind_status.message();
        result = false;
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
        result = false;
      }
    }
  }
  VLOG(1) << "[Mooncake][PDTransfer] direction=push, destinations="
          << keys.size() << ", layers=" << num_layers << ", success=" << result;
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
  size_per_block_ = count_per_block * data_size;

  std::vector<CacheTensorManifest> tensor_manifests;
  std::optional<CacheRegistrationContext> registration_context;
  if (pending_registration_context_.has_value()) {
    registration_context = *pending_registration_context_;
    registration_context->layout_family.append("_xtensor");
    const auto& global_xtensor = GlobalXTensor::get_instance();
    tensor_manifests.reserve(kv_caches.size() * 2);
    for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
      const std::vector<KVCacheTensor> transfer_tensors =
          get_mooncake_tensors(kv_caches[static_cast<size_t>(layer_id)]);
      for (const KVCacheTensor& cache_tensor : transfer_tensors) {
        CHECK(cache_tensor.role == KVCacheTensorRole::KEY ||
              cache_tensor.role == KVCacheTensorRole::VALUE)
            << "XTensor MoonCake transfer supports only K/V tensors, role="
            << cache_tensor.role.to_string();
        const torch::Tensor& tensor = cache_tensor.tensor;
        CHECK(tensor.defined() && tensor.numel() > 0);
        CHECK(tensor.is_contiguous());
        CHECK_GT(tensor.dim(), 0);
        const int64_t resource_count = tensor.size(0);
        CHECK_GT(resource_count, 0);
        const uint64_t tensor_bytes = static_cast<uint64_t>(tensor.nbytes());
        CHECK_EQ(tensor_bytes % static_cast<uint64_t>(resource_count), 0);
        const uint64_t resource_stride =
            tensor_bytes / static_cast<uint64_t>(resource_count);
        CHECK_EQ(resource_stride, static_cast<uint64_t>(size_per_block_))
            << "XTensor K/V resources must use the allocator block size";

        KVCacheTensor described_tensor = cache_tensor;
        std::string descriptor_error;
        CHECK(describe_cache_tensor(registration_context->tensor_layout,
                                    &described_tensor,
                                    &descriptor_error))
            << "Failed to describe XTensor cache tensor, layer=" << layer_id
            << ", role=" << cache_tensor.role.to_string() << ": "
            << descriptor_error;
        CHECK(described_tensor.shard_descriptor.has_value());

        CacheTensorManifest manifest;
        manifest.cache_namespace = registration_context->cache_namespace;
        manifest.layer_id = layer_id;
        manifest.role = static_cast<int32_t>(
            static_cast<KVCacheTensorRole::Value>(cache_tensor.role));
        manifest.group_id = cache_tensor.group_id;
        manifest.mooncake_buffer_id = 0;
        manifest.scalar_type = static_cast<int32_t>(tensor.scalar_type());
        manifest.element_bytes = tensor.element_size();
        manifest.shape = tensor.sizes().vec();
        manifest.stride = tensor.strides().vec();
        manifest.contiguous = true;
        manifest.resource_count = static_cast<uint64_t>(resource_count);
        manifest.resource_stride_bytes = resource_stride;
        manifest.buffer_bytes =
            static_cast<uint64_t>(global_xtensor.total_size());
        manifest.block_token_capacity = static_cast<uint64_t>(std::max<int64_t>(
            registration_context->tensor_layout.block_token_capacity, 0));
        manifest.explicit_resource_offsets = true;
        manifest.shard = std::move(described_tensor.shard_descriptor.value());
        tensor_manifests.emplace_back(std::move(manifest));
      }
    }
  }

  register_kv_cache_impl();
  if (registration_context.has_value()) {
    publish_cache_layout(tensor_manifests, *registration_context);
    pending_registration_context_.reset();
  }
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

    auto* transfer_engine =
        static_cast<MooncakeTransferEngine*>(mooncake_te_.get());
    auto ret = transfer_engine->move_memory_by_global_offsets(
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
      ExplicitResourceMapping key_mapping;
      key_mapping.group_id = mapping_it->group_id;
      key_mapping.role = static_cast<int32_t>(KVCacheTensorRole::KEY);
      key_mapping.local_ids = mapping_it->local_ids;
      key_mapping.remote_ids = mapping_it->remote_ids;
      key_mapping.local_offsets.reserve(src_blocks.size());
      key_mapping.remote_offsets.reserve(src_blocks.size());

      ExplicitResourceMapping value_mapping;
      value_mapping.group_id = mapping_it->group_id;
      value_mapping.role = static_cast<int32_t>(KVCacheTensorRole::VALUE);
      value_mapping.local_ids = mapping_it->local_ids;
      value_mapping.remote_ids = mapping_it->remote_ids;
      value_mapping.local_offsets.reserve(src_blocks.size());
      value_mapping.remote_offsets.reserve(src_blocks.size());
      for (size_t offset_index = 0; offset_index < src_offsets.size();
           offset_index += 2) {
        key_mapping.local_offsets.emplace_back(src_offsets[offset_index]);
        key_mapping.remote_offsets.emplace_back(dst_offsets[offset_index]);
        value_mapping.local_offsets.emplace_back(src_offsets[offset_index + 1]);
        value_mapping.remote_offsets.emplace_back(
            dst_offsets[offset_index + 1]);
      }

      std::vector<ByteRegion> regions;
      const Status bind_status = xtensor_te->bind_outgoing_regions_explicit(
          kv_info.dst_addr,
          {key_mapping, value_mapping},
          CacheNamespace::MAIN,
          layer_index,
          &regions);
      if (!bind_status.ok()) {
        LOG(ERROR) << "Bind XTensor KV byte regions failed, layer="
                   << layer_index << ", destination=" << kv_info.dst_addr
                   << ": " << bind_status.message();
        result = false;
        continue;
      }
      const bool ret =
          regions.empty() || xtensor_te->move_memory_regions(
                                 kv_info.dst_addr,
                                 regions,
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
