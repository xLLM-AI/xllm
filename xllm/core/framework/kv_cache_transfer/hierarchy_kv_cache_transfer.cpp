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
#include <limits>
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

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(const Options& options,
                                                   const torch::Device& device)
    : options_(options), device_(device) {
  device_.set_device();
  device_.init_device_context();
  load_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/2,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.load");
}

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(
    const Options& options,
    const torch::Device& device,
    const Stream* compute_stream,
    std::vector<xllm::KVCache>* kv_caches_ptr,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : HierarchyKVCacheTransfer(options, device) {
  CacheRegistration registration;
  registration.role = CacheRole::TARGET;
  registration.device_kv_caches = kv_caches_ptr;
  registration.kv_cache_shape = kv_cache_shape;
  registration.create_options = create_options;
  registration.producer_stream = compute_stream;
  registration.store_key_component = "main";
  register_cache(std::move(registration));
  CHECK(finalize_registration());
}

HierarchyKVCacheTransfer::~HierarchyKVCacheTransfer() { shutdown(); }

HierarchyKVCacheTransfer::CacheHandle HierarchyKVCacheTransfer::register_cache(
    CacheRegistration registration) {
  CHECK(!registration_finalized_)
      << "Hierarchy KV cache registration is already finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  CHECK(registration.device_kv_caches != nullptr)
      << "Device KV caches must not be null.";
  CHECK(!registration.device_kv_caches->empty())
      << "Device KV caches must not be empty.";
  CHECK(registration.producer_stream != nullptr)
      << "Producer stream must not be null.";
  const auto duplicate_role =
      std::find_if(cache_domains_.begin(),
                   cache_domains_.end(),
                   [&registration](const CacheDomain& domain) {
                     return domain.role == registration.role;
                   });
  CHECK(duplicate_role == cache_domains_.end())
      << "Cache role is already registered.";
  CHECK_LT(cache_domains_.size(),
           static_cast<size_t>(std::numeric_limits<CacheHandle>::max()));

  CacheDomain domain;
  domain.handle = static_cast<CacheHandle>(cache_domains_.size());
  domain.role = registration.role;
  domain.device_kv_caches = registration.device_kv_caches;
  domain.kv_cache_shape = std::move(registration.kv_cache_shape);
  domain.create_options = std::move(registration.create_options);
  domain.producer_stream = registration.producer_stream;
  domain.store_key_component = std::move(registration.store_key_component);
  const CacheHandle handle = domain.handle;
  cache_domains_.emplace_back(std::move(domain));
  return handle;
}

bool HierarchyKVCacheTransfer::finalize_registration() {
  CHECK(!registration_finalized_)
      << "Hierarchy KV cache registration is already finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  CHECK(!cache_domains_.empty()) << "No KV cache domain is registered.";

  const Stream* producer_stream = cache_domains_.front().producer_stream;
  for (const CacheDomain& domain : cache_domains_) {
    CHECK_EQ(domain.producer_stream, producer_stream)
        << "All KV cache domains must share one producer stream in phase two.";
  }

  if (options_.host_blocks_factor() > 1.0) {
    int64_t max_layer_count = 0;
    std::vector<HostKVGroupLayout> combined_groups;
    for (CacheDomain& domain : cache_domains_) {
      domain.device_caches_by_type = build_device_groups(&domain);
      create_host_cache(&domain);
      domain.host_layout =
          std::make_unique<HostKVLayout>(create_host_kv_layout(domain));
      max_layer_count = std::max<int64_t>(
          max_layer_count,
          static_cast<int64_t>(domain.device_kv_caches->size()));
      const std::vector<HostKVGroupLayout>& domain_groups =
          domain.host_layout->groups();
      combined_groups.insert(
          combined_groups.end(), domain_groups.begin(), domain_groups.end());
    }

    HostKVTransferConfig config;
    config.layer_copy_batches = options_.layers_wise_copy_batchs();
    config.mode = options_.enable_kvcache_store() ? HostKVTransferMode::BASIC
                                                  : HostKVTransferMode::AUTO;
    host_kv_transfer_ = create_host_kv_transfer(
        HostKVLayout(max_layer_count, std::move(combined_groups), device_),
        device_,
        *producer_stream,
        config);
  }

  if (options_.enable_kvcache_store()) {
    CHECK(options_.host_blocks_factor() > 1.0)
        << "Mooncake Store requires Host cache capacity.";
    HostCacheStoreIndex store_index;
    for (CacheDomain& domain : cache_domains_) {
      CHECK(!domain.store_key_component.empty())
          << "Mooncake Store requires a cache key component.";
      for (auto& [block_type, host_cache] : domain.host_caches_by_type) {
        store_index[block_type].emplace_back(HostCacheStoreEntry{
            domain.handle, domain.store_key_component, host_cache.get()});
      }
    }
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
    CHECK(kv_cache_store_->init(store_config, std::move(store_index)))
        << "Failed to initialize Mooncake Store.";
    LOG(INFO) << "[Mooncake][StoreEngine] ready, endpoint="
              << store_local_hostname << ", protocol=" << store_config.protocol
              << ", tp_rank=" << store_config.tp_rank;
  }
  registration_finalized_ = true;
  return true;
}

void HierarchyKVCacheTransfer::shutdown() {
  if (shutdown_) {
    return;
  }
  shutdown_ = true;
  // No load task may outlive transfer resources or Host cache storage.
  load_threadpool_.reset();
  device_.set_device();
  if (host_kv_transfer_ != nullptr) {
    host_kv_transfer_->drain();
    host_kv_transfer_.reset();
  }
  kv_cache_store_.reset();
  for (CacheDomain& domain : cache_domains_) {
    domain.host_layout.reset();
    domain.host_caches_by_type.clear();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  load_handles_.clear();
}

int32_t HierarchyKVCacheTransfer::domain_group_id(CacheHandle handle,
                                                  BlockType block_type) {
  constexpr int32_t kBlockTypeCount = cache_group_id(BlockType::LINEAR) + 1;
  CHECK_LE(handle,
           static_cast<CacheHandle>(
               (std::numeric_limits<int32_t>::max() - kBlockTypeCount + 1) /
               kBlockTypeCount));
  return static_cast<int32_t>(handle) * kBlockTypeCount +
         cache_group_id(block_type);
}

HierarchyKVCacheTransfer::GroupedCaches
HierarchyKVCacheTransfer::build_device_groups(CacheDomain* domain) const {
  CHECK(domain != nullptr);
  CHECK(domain->device_kv_caches != nullptr);
  GroupedCaches device_groups;
  const std::vector<BlockType> block_types = {BlockType::KV,
                                              BlockType::LINEAR,
                                              BlockType::SWA,
                                              BlockType::C4,
                                              BlockType::C128};
  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(domain->device_kv_caches->size());
       ++layer_id) {
    KVCache& kv_cache =
        domain->device_kv_caches->at(static_cast<size_t>(layer_id));
    for (BlockType type : block_types) {
      if (!build_block_type_tensor_map(kv_cache, type).empty()) {
        device_groups[type].emplace_back(&kv_cache);
        domain->layer_ids_by_type[type].emplace_back(layer_id);
      }
    }
  }
  return device_groups;
}

void HierarchyKVCacheTransfer::create_host_cache(CacheDomain* domain) {
  CHECK(domain != nullptr);
  CHECK(!domain->device_caches_by_type.empty())
      << "Device cache groups must not be empty.";
  for (const auto& [block_type, group_caches] : domain->device_caches_by_type) {
    if (group_caches.empty()) {
      continue;
    }
    KVCacheCreateOptions host_options = domain->create_options;
    host_options.device(torch::Device(torch::kCPU))
        .enable_xtensor(false)
        .tensor_allocator(nullptr)
        .host_blocks_factor(options_.host_blocks_factor());
#if defined(USE_NPU)
    host_options.enable_kv_cache_huge_page_allocator(false);
#endif
    domain->host_caches_by_type[block_type] =
        std::make_unique<KVCache>(domain->kv_cache_shape,
                                  host_options,
                                  block_type,
                                  static_cast<int64_t>(group_caches.size()));
  }
}

HostKVLayout HierarchyKVCacheTransfer::create_host_kv_layout(
    const CacheDomain& domain) const {
  std::vector<HostKVGroupLayout> groups;
  groups.reserve(domain.device_caches_by_type.size());
  for (const auto& [block_type, group_caches] : domain.device_caches_by_type) {
    auto host_it = domain.host_caches_by_type.find(block_type);
    auto layer_ids_it = domain.layer_ids_by_type.find(block_type);
    CHECK(host_it != domain.host_caches_by_type.end());
    CHECK(layer_ids_it != domain.layer_ids_by_type.end());
    CHECK_EQ(group_caches.size(), layer_ids_it->second.size());

    HostKVGroupLayout group;
    group.group_id = domain_group_id(domain.handle, block_type);
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
  return HostKVLayout(static_cast<int64_t>(domain.device_kv_caches->size()),
                      std::move(groups),
                      device_);
}

HostKVRequest HierarchyKVCacheTransfer::make_request(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    TransferType transfer_type) const {
  HostKVRequest request;
  request.target_mappings.reserve(block_transfer_info.size());
  request.draft_mappings.reserve(block_transfer_info.size());
  for (const BlockTransferInfo& info : block_transfer_info) {
    CHECK(info.transfer_type == transfer_type)
        << "Host KV batch contains mixed transfer types.";
    const bool is_load = transfer_type == TransferType::H2D;
    bool has_participating_domain = false;
    for (const CacheDomain& domain : cache_domains_) {
      if (domain.host_caches_by_type.find(info.block_type) ==
          domain.host_caches_by_type.end()) {
        continue;
      }
      has_participating_domain = true;
      std::vector<HostKVMapping>& mappings = domain.role == CacheRole::DRAFT
                                                 ? request.draft_mappings
                                                 : request.target_mappings;
      mappings.emplace_back(
          HostKVMapping{domain_group_id(domain.handle, info.block_type),
                        is_load ? info.src_block_id : info.dst_block_id,
                        is_load ? info.dst_block_id : info.src_block_id});
    }
    CHECK(has_participating_domain)
        << "No KV cache domain supports block type "
        << static_cast<int32_t>(info.block_type) << ".";
  }
  return request;
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  CHECK(registration_finalized_)
      << "Hierarchy KV cache registration is not finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
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
      HostKVLoadHandle handle =
          host_kv_transfer_->prepare_load(!request.draft_mappings.empty());
      CHECK(handle.synchronizer != nullptr)
          << "Failed to create Host KV load synchronizer.";
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto existing = load_handles_.find(batch_id);
        if (existing != load_handles_.end()) {
          LOG(ERROR) << "Host KV load handle collision at batch_id=" << batch_id
                     << "; replacing the unconsumed handle.";
          existing->second.synchronizer->abort();
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
  CHECK(registration_finalized_)
      << "Hierarchy KV cache registration is not finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
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
  CHECK(registration_finalized_)
      << "Hierarchy KV cache registration is not finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
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

bool HierarchyKVCacheTransfer::supports_block_type(BlockType block_type) const {
  return std::any_of(cache_domains_.begin(),
                     cache_domains_.end(),
                     [block_type](const CacheDomain& domain) {
                       return domain.host_caches_by_type.find(block_type) !=
                              domain.host_caches_by_type.end();
                     });
}

bool HierarchyKVCacheTransfer::supports_block_type(CacheRole role,
                                                   BlockType block_type) const {
  const auto domain = std::find_if(
      cache_domains_.begin(),
      cache_domains_.end(),
      [role](const CacheDomain& candidate) { return candidate.role == role; });
  return domain != cache_domains_.end() &&
         domain->host_caches_by_type.find(block_type) !=
             domain->host_caches_by_type.end();
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
  if (kv_cache_store_ != nullptr) {
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
  std::optional<HostKVLoadHandle> handle =
      take_load_handle(params.meta.batch_id);
  if (!handle.has_value()) {
    return;
  }
  params.parallel.layer_wise_load_synchronizer = handle->synchronizer;
  params.parallel.layers_per_event = handle->layers_per_event;
  params.parallel.draft_load_event_index = handle->draft_event_index;
}

std::optional<HostKVLoadHandle> HierarchyKVCacheTransfer::take_load_handle(
    uint64_t batch_id) {
  CHECK(registration_finalized_)
      << "Hierarchy KV cache registration is not finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = load_handles_.find(batch_id);
  if (it == load_handles_.end()) {
    return std::nullopt;
  }
  HostKVLoadHandle handle = std::move(it->second);
  load_handles_.erase(it);
  return handle;
}

}  // namespace xllm
