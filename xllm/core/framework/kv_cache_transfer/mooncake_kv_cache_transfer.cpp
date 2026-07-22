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
#include <chrono>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "common/global_flags.h"
#include "core/framework/config/kv_cache_config.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/push_route.h"
#include "framework/xtensor/global_xtensor.h"
#include "framework/xtensor/xtensor_allocator.h"
#if defined(USE_DCU)
#include "platform/dcu/dcu_tensor_alloc.h"
#endif
#include "platform/mlu/mlu_tensor_alloc.h"
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
  std::vector<KVCacheTensor> transfer_tensors;
  for (const KVCacheTensor& cache_tensor : cache.get_cache_tensors()) {
    switch (cache_tensor.role) {
      case KVCacheTensorRole::KEY:
      case KVCacheTensorRole::VALUE:
      case KVCacheTensorRole::INDEX:
      case KVCacheTensorRole::INDEX_SCALE:
        transfer_tensors.emplace_back(cache_tensor);
        break;
      default:
        // Mooncake roles form an explicit protocol whitelist. A new cache role
        // must not become transferable without a corresponding protocol
        // decision and registration-order test.
        break;
    }
  }
  return transfer_tensors;
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
    kv_info.src_blocks.reserve(info.local_blocks_ids.size());
    kv_info.src_blocks.insert(kv_info.src_blocks.end(),
                              info.local_blocks_ids.begin(),
                              info.local_blocks_ids.end());
    kv_info.dst_blocks.reserve(info.remote_blocks_ids.size());
    kv_info.dst_blocks.insert(kv_info.dst_blocks.end(),
                              info.remote_blocks_ids.begin(),
                              info.remote_blocks_ids.end());
    merge_xtensor_offsets(kv_info.dst_xtensor_layer_offsets,
                          info.dst_xtensor_layer_offsets);
    merged_kv_infos.emplace(key, std::move(kv_info));
    return;
  }

  std::vector<uint64_t>& src_blocks = it->second.src_blocks;
  src_blocks.reserve(src_blocks.size() + info.local_blocks_ids.size());
  src_blocks.insert(src_blocks.end(),
                    info.local_blocks_ids.begin(),
                    info.local_blocks_ids.end());

  std::vector<uint64_t>& dst_blocks = it->second.dst_blocks;
  dst_blocks.reserve(dst_blocks.size() + info.remote_blocks_ids.size());
  dst_blocks.insert(dst_blocks.end(),
                    info.remote_blocks_ids.begin(),
                    info.remote_blocks_ids.end());
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
          std::make_unique<MooncakeTransferEngine>(listen_port, device)),
      model_type_(model_type) {}

MooncakeKVCacheTransferDefault::MooncakeKVCacheTransferDefault(
    const int32_t device_id,
    const uint16_t listen_port,
    const torch::Device& device,
    const std::string& model_type,
    std::unique_ptr<MooncakeTransferEngine> engine)
    : MooncakeKVCacheTransferBase(device_id,
                                  listen_port,
                                  device,
                                  std::move(engine)),
      model_type_(model_type) {}

void MooncakeKVCacheTransferDefault::allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = num_layers;
  allocate_kv_cache_impl(kv_caches, num_layers, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferDefault::allocate_kv_cache_spec(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  allocate_kv_cache_impl(kv_caches, num_layers, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferDefault::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  const bool is_spec_draft = main_layout_.registered;
  CHECK(!is_spec_draft || !spec_layout_.registered)
      << "Spec draft kv cache is already registered.";

  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  bool has_v_cache = true;
  if (!kv_caches.empty()) {
    torch::Tensor value_cache = kv_caches[0].get_v_cache();
    has_v_cache = value_cache.defined() && value_cache.numel() > 0;
  }

  int64_t data_size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  int64_t count_per_block = 1;
  for (size_t i = 1; i < key_cache_shape.size(); ++i) {
    count_per_block *= key_cache_shape[i];
  }
  const int64_t size_per_block = count_per_block * data_size;
  if (size_per_block_ == 0) {
    size_per_block_ = size_per_block;
  } else {
    CHECK_EQ(size_per_block_, size_per_block)
        << "Spec draft kv block size mismatch.";
  }

  BufLayout layout;
  layout.num_layers = num_layers;
  layout.layer_offsets.reserve(static_cast<size_t>(num_layers) + 1);
  layout.layer_offsets.emplace_back(0);
  for (const KVCache& cache : kv_caches) {
    const std::vector<KVCacheTensor> transfer_tensors =
        get_mooncake_tensors(cache);
    const int64_t buffer_count = static_cast<int64_t>(transfer_tensors.size());
    if (layout.layer_offsets.size() == 1) {
      layout.buf_cnt = buffer_count;
    } else if (layout.buf_cnt != buffer_count) {
      layout.buf_cnt = 0;
    }
    layout.total_buf_cnt += buffer_count;
    layout.layer_offsets.emplace_back(layout.total_buf_cnt);
  }
  if (is_spec_draft) {
    layout.offset = main_layout_.offset + main_layout_.total_buf_cnt;
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

void MooncakeKVCacheTransferDefault::allocate_kv_cache_impl(
    std::vector<xllm::KVCache>& kv_caches,
    int64_t num_layers,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
#if defined(USE_MLU)
  (void)kv_caches;
  (void)num_layers;
  (void)kv_cache_shape;
  (void)dtype;
  LOG(FATAL) << "MLU Mooncake cache allocation must use the KV cache factory.";
#elif defined(USE_DCU)
  // TODO(xllm-kv-allocator): DCU remains on its existing physical allocation
  // path in the MLU Mooncake migration. A follow-up must route it through
  // KVCacheCreateOptions::tensor_allocator without moving cache structure
  // decisions into Transfer. Do not add indexer layer-mask handling here.
  CHECK(kv_cache_shape.has_value_cache_shape())
      << "DCU Mooncake KV transfer requires a value cache shape.";
  CHECK(!kv_cache_shape.has_index_cache_shape())
      << "DCU Mooncake KV transfer does not support index cache yet.";
  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  const std::vector<int64_t>& value_cache_shape =
      kv_cache_shape.value_cache_shape();

  for (int64_t i = 0; i < num_layers; ++i) {
    torch::Tensor key_cache =
        dcu::alloc_zero_tensor(key_cache_shape, dtype, device_);
    torch::Tensor value_cache =
        dcu::alloc_zero_tensor(value_cache_shape, dtype, device_);
    kv_caches.emplace_back(KVCacheTensors{key_cache, value_cache});
  }
#else
  // TODO(xllm-kv-allocator): NPU remains on its existing physical allocation
  // path in the MLU Mooncake migration. A follow-up must route it through
  // KVCacheCreateOptions::tensor_allocator without moving cache structure
  // decisions into Transfer. Do not add indexer layer-mask handling here.
  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  const std::vector<int64_t>& value_cache_shape =
      kv_cache_shape.value_cache_shape();
  // Original mode: allocate device memory using aclrtMalloc
  // calculate the size of kv cache for each layer
  auto data_size = torch::elementSize(dtype);
  int64_t k_cache_size_per_layer = data_size;
  for (int64_t i = 0; i < key_cache_shape.size(); ++i) {
    k_cache_size_per_layer *= key_cache_shape[i];
  }
  int64_t v_cache_size_per_layer = data_size;
  for (int64_t i = 0; i < value_cache_shape.size(); ++i) {
    v_cache_size_per_layer *= value_cache_shape[i];
  }

  // allocate device memory for kv cache
  std::vector<uint64_t> k_cache_addrs;
  std::vector<uint64_t> v_cache_addrs;
  k_cache_addrs.reserve(num_layers);
  v_cache_addrs.reserve(num_layers);

  std::vector<uintptr_t> k_tensor_addrs;
  std::vector<uintptr_t> v_tensor_addrs;
  k_tensor_addrs.reserve(num_layers);
  v_tensor_addrs.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    void* k_cache_buffer = nullptr;
    void* v_cache_buffer = nullptr;
    auto acl_ret = aclrtMalloc(
        &k_cache_buffer, k_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK(acl_ret == ACL_SUCCESS) << "aclrtMalloc k cache failed.";
    acl_ret = aclrtMalloc(
        &v_cache_buffer, v_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK(acl_ret == ACL_SUCCESS) << "aclrtMalloc v cache failed.";

    k_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(k_cache_buffer));
    v_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(v_cache_buffer));

    k_tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(k_cache_buffer));
    v_tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(v_cache_buffer));
  }

  // convert memory addrs to torch tensors
  aclFormat npu_format_type = get_npu_kv_cache_format(model_type_);
  auto k_torch_tensors = convert_to_torch_tensor(
      key_cache_shape, dtype, k_tensor_addrs, npu_format_type);
  auto v_torch_tensors = convert_to_torch_tensor(
      value_cache_shape, dtype, v_tensor_addrs, npu_format_type);

  torch::Tensor key_cache, value_cache;
  for (int64_t i = 0; i < num_layers; ++i) {
    key_cache = k_torch_tensors[i];
    value_cache = v_torch_tensors[i];
    kv_caches.emplace_back(
        KVCacheTensors{std::move(key_cache), std::move(value_cache)});
  }
#endif
}

void MooncakeKVCacheTransferDefault::add_buf(
    const torch::Tensor& tensor,
    std::vector<void*>& addrs,
    std::vector<size_t>& lens,
    std::vector<uint64_t>& buf_bytes,
    RegisterLengthPolicy register_length_policy) const {
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

  size_t registered_bytes = logical_bytes;
  switch (register_length_policy) {
    case RegisterLengthPolicy::LOGICAL_BYTES:
      break;
    case RegisterLengthPolicy::RDMA_REGISTERABLE_BYTES:
#if defined(USE_MLU)
      registered_bytes = mlu::get_rdma_registerable_nbytes(tensor);
#else
      LOG(FATAL) << "RDMA registerable storage is only supported on MLU";
#endif
      break;
  }
  CHECK_EQ(registered_bytes % block_bytes, static_cast<size_t>(0))
      << "registered bytes must be divisible by block bytes";
  CHECK_GE(available_bytes, registered_bytes)
      << "Mooncake registration exceeds tensor storage capacity: "
      << "logical_bytes=" << logical_bytes
      << ", registered_bytes=" << registered_bytes
      << ", available_bytes=" << available_bytes
      << ", block_bytes=" << block_bytes;

  addrs.emplace_back(tensor.data_ptr());
  lens.emplace_back(registered_bytes);
  buf_bytes.emplace_back(static_cast<uint64_t>(block_bytes));
}

std::vector<int64_t> MooncakeKVCacheTransferDefault::get_buf_ids(
    const std::vector<int64_t>& layer_ids,
    bool is_spec_draft) const {
  const BufLayout& layout = is_spec_draft ? spec_layout_ : main_layout_;
  return get_buf_ids(layer_ids, layout);
}

std::vector<int64_t> MooncakeKVCacheTransferDefault::get_buf_ids(
    const std::vector<int64_t>& layer_ids,
    const BufLayout& layout) const {
  CHECK(layout.registered) << "KV cache is not registered.";

  std::vector<int64_t> active_layer_ids;
  if (layer_ids.empty()) {
    active_layer_ids.resize(static_cast<size_t>(layout.num_layers));
    std::iota(active_layer_ids.begin(), active_layer_ids.end(), 0);
  } else {
    active_layer_ids = layer_ids;
  }

  std::vector<int64_t> buf_ids;
  const bool has_variable_layout = !layout.layer_offsets.empty();
  if (has_variable_layout) {
    CHECK_EQ(layout.layer_offsets.size(),
             static_cast<size_t>(layout.num_layers) + 1)
        << "KV cache buffer layout offsets are invalid.";
  } else {
    CHECK_GT(layout.buf_cnt, 0) << "KV cache uniform buffer layout is invalid.";
  }
  for (int64_t layer_id : active_layer_ids) {
    CHECK_GE(layer_id, 0) << "layer_id must be non-negative";
    CHECK_LT(layer_id, layout.num_layers) << "layer_id out of range";
  }

  size_t buffer_count = 0;
  for (int64_t layer_id : active_layer_ids) {
    const int64_t begin =
        has_variable_layout
            ? layout.layer_offsets[static_cast<size_t>(layer_id)]
            : layer_id * layout.buf_cnt;
    const int64_t end =
        has_variable_layout
            ? layout.layer_offsets[static_cast<size_t>(layer_id) + 1]
            : begin + layout.buf_cnt;
    buffer_count += static_cast<size_t>(end - begin);
  }
  buf_ids.reserve(buffer_count);

  for (int64_t layer_id : active_layer_ids) {
    const int64_t begin =
        has_variable_layout
            ? layout.layer_offsets[static_cast<size_t>(layer_id)]
            : layer_id * layout.buf_cnt;
    const int64_t end =
        has_variable_layout
            ? layout.layer_offsets[static_cast<size_t>(layer_id) + 1]
            : begin + layout.buf_cnt;
    for (int64_t relative_id = begin; relative_id < end; ++relative_id) {
      buf_ids.emplace_back(layout.offset + relative_id);
    }
  }
  return buf_ids;
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
      if (cache_tensor.role == KVCacheTensorRole::INDEX_SCALE) {
#if defined(USE_MLU)
        add_buf(cache_tensor.tensor,
                addrs,
                lens,
                buf_bytes,
                RegisterLengthPolicy::RDMA_REGISTERABLE_BYTES);
#else
        add_buf(cache_tensor.tensor, addrs, lens, buf_bytes);
#endif
        continue;
      }
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
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  (void)src_cluster_id;
  (void)src_linear_state_ids;
  (void)dst_linear_state_ids;
  std::vector<int64_t> layer_ids;
  // Pull path is used by target/main KV cache blocks, not spec draft blocks.
  const bool is_spec_draft = false;
  std::vector<int64_t> buf_ids = get_buf_ids(layer_ids, is_spec_draft);
  auto ret = mooncake_te_->pull_memory_blocks(
      src_addr, src_blocks, dst_blocks, buf_ids);
  if (!ret) {
    LOG(ERROR) << "Pull kv cache blocks failed, ret = " << ret;
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

  for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
    layer_synchronizer->synchronize_layer(layer_index);
    std::vector<int64_t> layer_ids = {layer_index};
    std::vector<int64_t> buf_ids = get_buf_ids(layer_ids, is_spec_draft);

    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      if (kv_info.src_blocks.empty()) {
        continue;
      }

      const auto step_start = std::chrono::steady_clock::now();
      auto ret = mooncake_te_->push_memory_blocks(
          kv_info.dst_addr, kv_info.src_blocks, kv_info.dst_blocks, buf_ids);
      if (!ret) {
        LOG(ERROR) << "Push kv blocks failed, layer = " << layer_index
                   << ", ret = " << ret;
        return false;
      }
    }
  }
  return true;
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

void MooncakeKVCacheTransferXTensor::allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = num_layers;
  allocate_kv_cache_impl(kv_caches, num_layers, kv_cache_shape, dtype);
}

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

void MooncakeKVCacheTransferXTensor::allocate_kv_cache_impl(
    std::vector<xllm::KVCache>& kv_caches,
    int64_t num_layers,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  auto& allocator = XTensorAllocator::get_instance();
  CHECK(!model_id_.empty()) << "model_id must be set for XTensor mode";
  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  const std::vector<int64_t>& value_cache_shape =
      kv_cache_shape.value_cache_shape();

  auto k_tensors =
      allocator.create_k_tensors(model_id_, key_cache_shape, dtype, num_layers);
  auto v_tensors = allocator.create_v_tensors(
      model_id_, value_cache_shape, dtype, num_layers);

  for (int64_t i = 0; i < num_layers; ++i) {
#if defined(USE_NPU)
    auto k_tensor =
        at_npu::native::npu_format_cast(k_tensors[i], ACL_FORMAT_ND);
    auto v_tensor =
        at_npu::native::npu_format_cast(v_tensors[i], ACL_FORMAT_ND);
    kv_caches.emplace_back(KVCacheTensors{k_tensor, v_tensor});
#else
    kv_caches.emplace_back(KVCacheTensors{k_tensors[i], v_tensors[i]});
#endif
  }

  LOG(INFO) << "MooncakeKVCacheTransferXTensor: KV cache allocated"
            << ", model_id=" << model_id_ << ", num_layers=" << num_layers;
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
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  (void)src_cluster_id;
  (void)src_linear_state_ids;
  (void)dst_linear_state_ids;
  return pull_kv_blocks_impl(src_addr, src_blocks, dst_blocks);
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

  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    layer_synchronizer->synchronize_layer(layer_index);

    for (const std::string& key : keys) {
      const KVCacheInfo& kv_info = merged_kv_infos.at(key);
      if (kv_info.src_blocks.empty()) {
        continue;
      }

      // Check if we have XTensor offsets from D-node
      bool has_dst_offsets = !kv_info.dst_xtensor_layer_offsets.empty() &&
                             static_cast<size_t>(layer_index) <
                                 kv_info.dst_xtensor_layer_offsets.size();

      std::vector<uint64_t> src_offsets;
      std::vector<uint64_t> dst_offsets;
      src_offsets.reserve(kv_info.src_blocks.size() * 2);
      dst_offsets.reserve(kv_info.src_blocks.size() * 2);

      for (size_t i = 0; i < kv_info.src_blocks.size(); ++i) {
        // Source block -> GlobalXTensor offsets (calculate locally on P-node)
        auto [src_k_off, src_v_off] = allocator.get_global_offsets_for_block(
            model_id_, layer_index, kv_info.src_blocks[i], size_per_block_);
        if (src_k_off == UINT64_MAX || src_v_off == UINT64_MAX) {
          LOG(ERROR) << "Failed to get source offsets for block "
                     << kv_info.src_blocks[i] << " at layer " << layer_index;
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

      const auto step_start = std::chrono::steady_clock::now();
      auto ret = xtensor_te->move_memory_by_global_offsets(
          kv_info.dst_addr,
          src_offsets,
          dst_offsets,
          size_per_block_,
          MooncakeTransferEngine::MoveOpcode::WRITE);
      if (!ret) {
        LOG(ERROR) << "push_kv_blocks_impl failed at layer " << layer_index;
        return false;
      }
    }
  }

  VLOG(1) << "push_kv_blocks_impl success, num_layers=" << num_layers_;
  return true;
}

}  // namespace xllm
