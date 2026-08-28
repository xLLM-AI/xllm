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

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/kv_cache_store.h"

namespace xllm {
namespace {

std::string make_store_local_hostname(const std::string& configured,
                                      uint32_t worker_id) {
  constexpr uint32_t kDefaultPort = 12345;
  if (configured.empty()) {
    return "127.0.0.1:" + std::to_string(kDefaultPort + worker_id);
  }

  std::string host = configured;
  uint32_t port = kDefaultPort;
  size_t host_end = std::string::npos;
  size_t port_begin = std::string::npos;
  const size_t bracket_end = configured.find("]:");
  if (!configured.empty() && configured.front() == '[' &&
      bracket_end != std::string::npos) {
    host_end = bracket_end + 1;
    port_begin = bracket_end + 2;
  } else {
    const size_t last_colon = configured.rfind(':');
    const bool has_single_colon = last_colon != std::string::npos &&
                                  configured.find(':') == last_colon &&
                                  last_colon + 1 < configured.size();
    if (has_single_colon) {
      host_end = last_colon;
      port_begin = last_colon + 1;
    }
  }
  if (port_begin != std::string::npos) {
    const std::string port_text = configured.substr(port_begin);
    const bool numeric =
        std::all_of(port_text.begin(), port_text.end(), [](char character) {
          return character >= '0' && character <= '9';
        });
    if (numeric) {
      port = static_cast<uint32_t>(std::stoul(port_text));
      host = configured.substr(0, host_end);
    }
  }
  CHECK_LE(worker_id, 65535U);
  CHECK_LE(port, 65535U - worker_id)
      << "Mooncake local endpoint port exceeds 65535.";
  return host + ":" + std::to_string(port + worker_id);
}

bool has_tensor(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0;
}

BlockTypeTensorMap build_block_type_tensor_map(const KVCache& kv_cache,
                                               BlockType type) {
  BlockTypeTensorMap tensors;
  const torch::Tensor key_cache = kv_cache.get_k_cache();
  const torch::Tensor value_cache = kv_cache.get_v_cache();
  const torch::Tensor index_cache = kv_cache.get_index_cache();
  const torch::Tensor conv_cache = kv_cache.get_conv_cache();
  const torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  const torch::Tensor swa_cache = kv_cache.get_swa_cache();
  const std::optional<torch::Tensor> index_cache_scale =
      kv_cache.get_indexer_cache_scale();

  switch (type) {
    case BlockType::KV:
      if (has_tensor(conv_cache) || has_tensor(ssm_cache) ||
          has_tensor(swa_cache)) {
        return {};
      }
      if (has_tensor(key_cache)) {
        tensors.emplace(KVCacheTensorRole::KEY, key_cache);
      }
      if (has_tensor(value_cache)) {
        tensors.emplace(KVCacheTensorRole::VALUE, value_cache);
      }
      if (has_tensor(index_cache)) {
        tensors.emplace(KVCacheTensorRole::INDEX, index_cache);
      }
      // Quantized index state and scale must move together.
      if (index_cache_scale.has_value() &&
          has_tensor(index_cache_scale.value())) {
        tensors.emplace(KVCacheTensorRole::INDEX_SCALE,
                        index_cache_scale.value());
      }
      return tensors;
    case BlockType::LINEAR:
      if (has_tensor(conv_cache)) {
        tensors.emplace(KVCacheTensorRole::CONV, conv_cache);
      }
      if (has_tensor(ssm_cache)) {
        tensors.emplace(KVCacheTensorRole::SSM, ssm_cache);
      }
      return tensors;
    case BlockType::SWA:
      // The persistent SWA window is restored for every DSV4 layer.
      if (has_tensor(swa_cache)) {
        tensors.emplace(KVCacheTensorRole::SWA, swa_cache);
      }
      return tensors;
    case BlockType::C4:
      if (!has_tensor(swa_cache) || has_tensor(value_cache) ||
          !has_tensor(key_cache) || !has_tensor(index_cache)) {
        return {};
      }
      tensors.emplace(KVCacheTensorRole::KEY, key_cache);
      tensors.emplace(KVCacheTensorRole::INDEX, index_cache);
      if (index_cache_scale.has_value() &&
          has_tensor(index_cache_scale.value())) {
        tensors.emplace(KVCacheTensorRole::INDEX_SCALE,
                        index_cache_scale.value());
      }
      return tensors;
    case BlockType::C128:
      if (!has_tensor(swa_cache) || has_tensor(value_cache) ||
          !has_tensor(key_cache) || has_tensor(index_cache)) {
        return {};
      }
      tensors.emplace(KVCacheTensorRole::KEY, key_cache);
      return tensors;
    default:
      return {};
  }
}

}  // namespace

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(
    const Options& options,
    const torch::Device& device,
    const Stream* compute_stream,
    std::vector<xllm::KVCache>* kv_caches_ptr,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : options_(options),
      device_(device),
      kv_caches_ptr_(kv_caches_ptr),
      kv_cache_shape_(kv_cache_shape),
      create_options_(create_options) {
  CHECK(kv_caches_ptr_ != nullptr) << "kv_caches_ptr must not be null.";
  CHECK(compute_stream != nullptr) << "compute stream must not be null.";

  device_.set_device();
  device_.init_device_context();
  load_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/2,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.load");

  if (options_.host_blocks_factor() > 1.0) {
    std::map<BlockType, std::vector<int64_t>> layer_ids;
    GroupedCaches device_groups = build_device_groups(&layer_ids);
    create_host_cache(device_groups);
    HostKVTransferConfig config;
    config.layer_copy_batches = options_.layers_wise_copy_batchs();
    config.mode = options_.enable_kvcache_store() ? HostKVTransferMode::BASIC
                                                  : HostKVTransferMode::AUTO;
    host_kv_transfer_ =
        create_host_kv_transfer(create_host_kv_layout(device_groups, layer_ids),
                                device_,
                                *compute_stream,
                                config);
  }

  if (options_.enable_kvcache_store()) {
    CHECK(options_.host_blocks_factor() > 1.0)
        << "Mooncake Store requires Host cache capacity.";
    KVCacheStoreInitConfig store_config;
    const std::string store_local_hostname = make_store_local_hostname(
        options_.store_local_hostname(), options_.store_worker_id());
    store_config.localhost_name = store_local_hostname;
    store_config.protocol = options_.store_protocol();
    store_config.metadata_server = options_.store_metadata_server();
    store_config.master_server_address = options_.store_master_server_address();
    store_config.model_id = options_.store_namespace();
    store_config.tp_rank = options_.tp_rank();
    store_config.tp_size = options_.tp_size();
    LOG(INFO) << "[Mooncake][StoreEngine] initialize, endpoint="
              << store_local_hostname << ", protocol=" << store_config.protocol
              << ", tp_rank=" << store_config.tp_rank
              << ", tp_size=" << store_config.tp_size;
    kv_cache_store_ = std::make_unique<KVCacheStore>();
    CHECK(kv_cache_store_->init(store_config, &host_kv_caches_))
        << "Failed to initialize Mooncake Store.";
    LOG(INFO) << "[Mooncake][StoreEngine] ready, endpoint="
              << store_local_hostname << ", protocol=" << store_config.protocol
              << ", tp_rank=" << store_config.tp_rank;
  }
}

HierarchyKVCacheTransfer::~HierarchyKVCacheTransfer() {
  // No load task may outlive transfer resources or Host cache storage.
  load_threadpool_.reset();
  device_.set_device();
  if (host_kv_transfer_ != nullptr) {
    host_kv_transfer_->drain();
    host_kv_transfer_.reset();
  }
  kv_cache_store_.reset();
  host_kv_caches_.clear();
  std::lock_guard<std::mutex> lock(mutex_);
  load_handles_.clear();
}

HierarchyKVCacheTransfer::GroupedCaches
HierarchyKVCacheTransfer::build_device_groups(
    std::map<BlockType, std::vector<int64_t>>* layer_ids) const {
  CHECK(layer_ids != nullptr);
  GroupedCaches device_groups;
  const std::vector<BlockType> block_types = {BlockType::KV,
                                              BlockType::LINEAR,
                                              BlockType::SWA,
                                              BlockType::C4,
                                              BlockType::C128};
  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(kv_caches_ptr_->size());
       ++layer_id) {
    KVCache& kv_cache = kv_caches_ptr_->at(static_cast<size_t>(layer_id));
    for (BlockType type : block_types) {
      if (!build_block_type_tensor_map(kv_cache, type).empty()) {
        device_groups[type].emplace_back(&kv_cache);
        (*layer_ids)[type].emplace_back(layer_id);
      }
    }
  }
  return device_groups;
}

void HierarchyKVCacheTransfer::create_host_cache(
    const GroupedCaches& device_groups) {
  CHECK(!device_groups.empty()) << "device cache groups must not be empty.";
  for (const auto& [block_type, group_caches] : device_groups) {
    if (group_caches.empty()) {
      continue;
    }
    KVCacheCreateOptions host_options = create_options_;
    host_options.device(torch::Device(torch::kCPU))
        .enable_xtensor(false)
        .tensor_allocator(nullptr)
        .host_blocks_factor(options_.host_blocks_factor());
#if defined(USE_NPU)
    host_options.enable_kv_cache_huge_page_allocator(false);
#endif
    host_kv_caches_[block_type] =
        std::make_unique<KVCache>(kv_cache_shape_,
                                  host_options,
                                  block_type,
                                  static_cast<int64_t>(group_caches.size()));
  }
}

HostKVLayout HierarchyKVCacheTransfer::create_host_kv_layout(
    const GroupedCaches& device_groups,
    const std::map<BlockType, std::vector<int64_t>>& layer_ids) const {
  std::vector<HostKVGroupLayout> groups;
  groups.reserve(device_groups.size());
  for (const auto& [block_type, group_caches] : device_groups) {
    auto host_it = host_kv_caches_.find(block_type);
    auto layer_ids_it = layer_ids.find(block_type);
    CHECK(host_it != host_kv_caches_.end());
    CHECK(layer_ids_it != layer_ids.end());
    CHECK_EQ(group_caches.size(), layer_ids_it->second.size());

    HostKVGroupLayout group;
    group.group_id = cache_group_id(block_type);
    group.host_roles = host_it->second->get_block_type_tensors(block_type);
    group.layers.reserve(group_caches.size());
    for (size_t layer_slot = 0; layer_slot < group_caches.size();
         ++layer_slot) {
      HostKVLayerLayout layer;
      layer.absolute_layer_id = layer_ids_it->second[layer_slot];
      layer.group_layer_slot = static_cast<int64_t>(layer_slot);
      layer.device_roles =
          build_block_type_tensor_map(*group_caches[layer_slot], block_type);
      group.layers.emplace_back(std::move(layer));
    }
    groups.emplace_back(std::move(group));
  }
  return HostKVLayout(
      static_cast<int64_t>(options_.layers()), std::move(groups), device_);
}

HostKVRequest HierarchyKVCacheTransfer::make_request(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    TransferType transfer_type) {
  HostKVRequest request;
  request.mappings.reserve(block_transfer_info.size());
  for (const BlockTransferInfo& info : block_transfer_info) {
    CHECK(info.transfer_type == transfer_type)
        << "Host KV batch contains mixed transfer types.";
    const bool is_load = transfer_type == TransferType::H2D;
    request.mappings.emplace_back(
        HostKVMapping{cache_group_id(info.block_type),
                      is_load ? info.src_block_id : info.dst_block_id,
                      is_load ? info.dst_block_id : info.src_block_id});
  }
  return request;
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());
  device_.set_device();
  switch (block_transfer_info.front().transfer_type) {
    case TransferType::D2H2G:
      return offload(block_transfer_info);
    case TransferType::H2D: {
      if (host_kv_transfer_ == nullptr) {
        LOG(ERROR) << "Host KV load requested without Host cache.";
        return 0;
      }
      HostKVRequest request =
          make_request(block_transfer_info, TransferType::H2D);
      HostKVLoadHandle handle = host_kv_transfer_->prepare_load();
      CHECK(handle.synchronizer != nullptr)
          << "Failed to create Host KV load synchronizer.";
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (load_handles_.find(batch_id) != load_handles_.end()) {
          LOG(ERROR) << "Host KV load handle collision at batch_id=" << batch_id
                     << "; replacing the unconsumed handle.";
        }
        load_handles_[batch_id] = handle;
      }
      load_threadpool_->schedule(
          [this, request = std::move(request), handle]() mutable {
            load_from_host(request, handle);
          });
      return static_cast<uint32_t>(block_transfer_info.size());
    }
    default:
      LOG(ERROR) << "Unsupported transfer type: "
                 << static_cast<uint32_t>(
                        block_transfer_info.front().transfer_type);
      return 0;
  }
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    uint64_t /*batch_id*/,
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());
  CHECK(kv_cache_store_ != nullptr);
  if (block_transfer_info[0].transfer_type == TransferType::G2H) {
    return kv_cache_store_->batch_get(block_transfer_info);
  }
  LOG(ERROR) << "Unsupported slice transfer type: "
             << static_cast<uint32_t>(block_transfer_info[0].transfer_type);
  return 0;
}

std::vector<uint8_t> HierarchyKVCacheTransfer::prefetch_kv_blocks(
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());
  if (!options_.enable_kvcache_store() || kv_cache_store_ == nullptr ||
      block_transfer_info[0].transfer_type != TransferType::G2H) {
    LOG(ERROR) << "Unsupported prefetch transfer type: "
               << static_cast<uint32_t>(block_transfer_info[0].transfer_type);
    return std::vector<uint8_t>(block_transfer_info.size(), /*value=*/0);
  }
  std::vector<uint8_t> hits =
      kv_cache_store_->batch_get_with_status(block_transfer_info);
  const size_t hit_count =
      std::count(hits.begin(), hits.end(), static_cast<uint8_t>(1));
  VLOG(1) << "[Mooncake][PrefetchGet] type="
          << static_cast<int32_t>(block_transfer_info[0].block_type)
          << ", blocks=" << hits.size() << ", hits=" << hit_count;
  return hits;
}

uint32_t HierarchyKVCacheTransfer::offload(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (host_kv_transfer_ == nullptr) {
    return static_cast<uint32_t>(block_transfer_info.size());
  }
  HostKVRequest request =
      make_request(block_transfer_info, TransferType::D2H2G);
  if (!offload_to_host(request)) {
    LOG(ERROR) << "Offload to Host cache failed.";
    return 0;
  }
  if (options_.enable_kvcache_store()) {
    CHECK(kv_cache_store_ != nullptr);
    const uint32_t put_count = kv_cache_store_->batch_put(block_transfer_info);
    if (put_count != block_transfer_info.size()) {
      LOG(WARNING) << "Mooncake BatchPut partially failed: " << put_count << "/"
                   << block_transfer_info.size();
    }
    VLOG(1) << "[Mooncake][OffloadPut] blocks=" << block_transfer_info.size()
            << ", success=" << put_count;
  }
  return static_cast<uint32_t>(block_transfer_info.size());
}

bool HierarchyKVCacheTransfer::offload_to_host(const HostKVRequest& request) {
  return host_kv_transfer_->offload(request);
}

bool HierarchyKVCacheTransfer::load_from_host(const HostKVRequest& request,
                                              const HostKVLoadHandle& handle) {
  return host_kv_transfer_->load(request, handle);
}

void HierarchyKVCacheTransfer::set_layer_synchronizer(
    ModelInputParams& params) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = load_handles_.find(params.meta.batch_id);
  if (it == load_handles_.end()) {
    return;
  }
  params.parallel.layer_wise_load_synchronizer = it->second.synchronizer;
  params.parallel.layers_per_event = it->second.layers_per_event;
  load_handles_.erase(it);
}

}  // namespace xllm
