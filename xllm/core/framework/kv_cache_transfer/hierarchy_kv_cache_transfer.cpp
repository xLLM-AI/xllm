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

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/kv_cache_store.h"
#include "util/hash_util.h"

namespace xllm {
namespace {

constexpr uint32_t kTimeoutMs = 60000;
constexpr size_t kOffloadStreamCount = 4;

const std::vector<BlockType> kBlockTypes = {BlockType::KV,
                                            BlockType::LINEAR,
                                            BlockType::SWA,
                                            BlockType::C4,
                                            BlockType::C128};

std::string participant_name(CacheParticipant participant) {
  switch (participant) {
    case CacheParticipant::TARGET:
      return "TARGET";
    case CacheParticipant::DRAFT:
      return "DRAFT";
  }
  LOG(FATAL) << "Unsupported cache participant: "
             << static_cast<int32_t>(participant);
  return "UNKNOWN";
}

std::string make_store_local_hostname(const std::string& configured,
                                      uint32_t worker_id) {
  const uint32_t kDefaultPort = 12345;
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
  CHECK_LE(worker_id, 65535u);
  CHECK_LE(port, 65535u - worker_id)
      << "Mooncake local endpoint port exceeds 65535.";
  return host + ":" + std::to_string(port + worker_id);
}

using CopyStreamQueue =
    moodycamel::BlockingConcurrentQueue<std::unique_ptr<Stream>>;

class CopyStreamLease final {
 public:
  explicit CopyStreamLease(CopyStreamQueue* stream_pool)
      : stream_pool_(stream_pool) {
    CHECK(stream_pool_ != nullptr) << "copy stream pool must not be null.";
    stream_pool_->wait_dequeue(stream_);
    CHECK(stream_ != nullptr) << "copy stream must not be null.";
  }

  ~CopyStreamLease() { stream_pool_->enqueue(std::move(stream_)); }

  CopyStreamLease(const CopyStreamLease&) = delete;
  CopyStreamLease& operator=(const CopyStreamLease&) = delete;

  Stream* get() const { return stream_.get(); }

  void drain_or_die(const char* reason) const {
    try {
      const int32_t synchronize_result = stream_->synchronize();
      if (synchronize_result != 0) {
        LOG(FATAL) << "Failed to drain KV Cache copy stream: reason=" << reason
                   << ", result=" << synchronize_result;
      }
    } catch (const std::exception& error) {
      LOG(FATAL) << "Failed to drain KV Cache copy stream: reason=" << reason
                 << ", error=" << error.what();
    } catch (...) {
      LOG(FATAL) << "Failed to drain KV Cache copy stream: reason=" << reason
                 << ", unknown error.";
    }
  }

 private:
  CopyStreamQueue* stream_pool_ = nullptr;
  std::unique_ptr<Stream> stream_;
};

std::vector<HierarchyKVCacheTransfer::LayerBatchRange> build_layer_batch_ranges(
    int64_t num_layers,
    uint32_t requested_batches) {
  std::vector<HierarchyKVCacheTransfer::LayerBatchRange> ranges;
  if (num_layers <= 0) {
    return ranges;
  }
  uint32_t layers_per_batch =
      requested_batches == 0
          ? static_cast<uint32_t>(num_layers)
          : static_cast<uint32_t>(num_layers) / requested_batches;
  layers_per_batch = std::max<uint32_t>(layers_per_batch, 1);
  for (int64_t begin = 0; begin < num_layers; begin += layers_per_batch) {
    ranges.push_back(
        {begin, std::min<int64_t>(begin + layers_per_batch, num_layers)});
  }
  return ranges;
}

std::vector<int64_t> device_block_shape(const torch::Tensor& tensor) {
  CHECK(tensor.defined() && tensor.dim() > 0);
  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(tensor.dim() - 1));
  for (int64_t dim = 1; dim < tensor.dim(); ++dim) {
    shape.emplace_back(tensor.size(dim));
  }
  return shape;
}

std::vector<int64_t> host_block_layer_shape(const torch::Tensor& tensor) {
  CHECK(tensor.defined() && tensor.dim() > 1);
  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(tensor.dim() - 2));
  for (int64_t dim = 2; dim < tensor.dim(); ++dim) {
    shape.emplace_back(tensor.size(dim));
  }
  return shape;
}

std::string hash_schema_string(const std::string& schema) {
  const XXH3Key hash = hash_string(schema);
  return std::string(reinterpret_cast<const char*>(hash.data),
                     sizeof(hash.data));
}

}  // namespace

void HierarchyKVCacheTransfer::LoadTransaction::abort() {
  std::vector<std::shared_ptr<LayerSynchronizer>> synchronizers_to_abort;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (aborted) {
      return;
    }
    aborted = true;
    for (const auto& [participant, synchronizer] : synchronizers) {
      (void)participant;
      synchronizers_to_abort.emplace_back(synchronizer);
    }
  }
  for (const std::shared_ptr<LayerSynchronizer>& synchronizer :
       synchronizers_to_abort) {
    if (synchronizer != nullptr) {
      synchronizer->abort();
    }
  }
}

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(const Options& options,
                                                   const torch::Device& device)
    : options_(options), device_(device) {}

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(
    const Options& options,
    const torch::Device& device,
    const Stream* compute_stream,
    std::vector<KVCache>* kv_caches_ptr,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : HierarchyKVCacheTransfer(options, device) {
  ParticipantRegistration registration;
  registration.participant = CacheParticipant::TARGET;
  registration.actual_compute_stream = compute_stream;
  registration.device_caches = kv_caches_ptr;
  registration.cache_shape = kv_cache_shape;
  registration.create_options = create_options;
  registration.model_identity =
      create_options.model_id() + "|" + create_options.model_type();
  registration.tp_rank = options.tp_rank();
  registration.tp_size = options.tp_size();
  register_participant(std::move(registration));
  CHECK(finalize_registration());
}

HierarchyKVCacheTransfer::~HierarchyKVCacheTransfer() {
  load_threadpool_.reset();

  device_.set_device();
  std::unique_ptr<Stream> stream;
  while (copy_stream_.try_dequeue(stream)) {
    if (stream == nullptr) {
      continue;
    }
    CHECK_EQ(stream->synchronize(), 0)
        << "Failed to drain hierarchy KV copy stream during shutdown.";
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [batch_id, transaction] : load_transactions_) {
    (void)batch_id;
    transaction->abort();
  }
  load_transactions_.clear();
}

void HierarchyKVCacheTransfer::register_participant(
    ParticipantRegistration registration) {
  CHECK(registration_state_ != RegistrationState::READY)
      << "Cannot register a participant after finalization.";
  CHECK(registration.actual_compute_stream != nullptr)
      << "actual compute stream must not be null.";
  CHECK(registration.device_caches != nullptr)
      << "device caches must not be null.";
  CHECK(!registration.device_caches->empty())
      << "device caches must not be empty.";
  CHECK(!registration.model_identity.empty())
      << "participant model identity must not be empty.";
  CHECK_GT(registration.tp_size, 0u);
  CHECK_LT(registration.tp_rank, registration.tp_size);
  CHECK(participant_states_.find(registration.participant) ==
        participant_states_.end())
      << "Duplicate cache participant: "
      << participant_name(registration.participant);

  ParticipantState state;
  state.registration = std::move(registration);
  participant_states_.emplace(state.registration.participant, std::move(state));
  registration_state_ = RegistrationState::REGISTERING;
}

bool HierarchyKVCacheTransfer::finalize_registration() {
  CHECK(registration_state_ == RegistrationState::REGISTERING)
      << "Hierarchy KV cache registration is not active.";
  CHECK(!participant_states_.empty());

  device_.set_device();
  device_.init_device_context();
  for (auto& [participant, state] : participant_states_) {
    (void)participant;
    build_participant_state(state);
  }
  validate_composite_schema();
  initialize_resources();
  if (options_.enable_kvcache_store()) {
    initialize_store();
  }
  registration_state_ = RegistrationState::READY;
  return true;
}

void HierarchyKVCacheTransfer::initialize_resources() {
  CHECK_GT(options_.host_blocks_factor(), 1.0)
      << "Hierarchy KV cache transfer requires Host cache capacity.";
  const size_t load_threads = std::max<size_t>(2, participant_states_.size());
  load_threadpool_ = std::make_unique<ThreadPool>(
      load_threads,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.load");
  const size_t num_streams = load_threadpool_->size() + kOffloadStreamCount;
  for (size_t i = 0; i < num_streams; ++i) {
    copy_stream_.enqueue(device_.get_stream_from_pool(kTimeoutMs));
  }
  batch_memcpy_ = create_batch_memcpy(device_);
  CHECK(batch_memcpy_ != nullptr);
}

void HierarchyKVCacheTransfer::build_participant_state(
    ParticipantState& state) {
  CHECK_EQ(state.registration.device_caches->size(),
           static_cast<size_t>(state.registration.create_options.num_layers()))
      << "Participant cache layer count does not match allocation metadata: "
      << participant_name(state.registration.participant);
  build_device_block_type_map(state);
  CHECK(!state.device_grouped_caches.empty())
      << "Participant has no Host-cache-compatible tensors: "
      << participant_name(state.registration.participant);
  state.layer_batch_ranges = build_layer_batch_ranges(
      static_cast<int64_t>(state.registration.device_caches->size()),
      options_.layers_wise_copy_batchs());
  create_host_cache(state);
  build_and_validate_schema(state);
}

void HierarchyKVCacheTransfer::build_device_block_type_map(
    ParticipantState& state) {
  state.device_grouped_caches.clear();
  state.absolute_layer_ids.clear();
  for (int64_t layer_id = 0;
       layer_id <
       static_cast<int64_t>(state.registration.device_caches->size());
       ++layer_id) {
    KVCache& kv_cache =
        state.registration.device_caches->at(static_cast<size_t>(layer_id));
    for (BlockType block_type : kBlockTypes) {
      const BlockTypeTensorMap tensors =
          kv_cache.get_block_type_tensors(block_type);
      if (tensors.empty()) {
        continue;
      }
      state.device_grouped_caches[block_type].push_back(&kv_cache);
      state.absolute_layer_ids[block_type].push_back(layer_id);
    }
  }
}

void HierarchyKVCacheTransfer::create_host_cache(ParticipantState& state) {
  for (const auto& [block_type, group_caches] : state.device_grouped_caches) {
    CHECK(!group_caches.empty());
    KVCacheCreateOptions host_options = state.registration.create_options;
    host_options.device(torch::Device(torch::kCPU))
        .enable_xtensor(false)
        .tensor_allocator(nullptr)
        .host_blocks_factor(options_.host_blocks_factor());
#if defined(USE_NPU)
    host_options.enable_kv_cache_huge_page_allocator(false);
#endif
    state.host_grouped_caches[block_type] =
        std::make_unique<KVCache>(state.registration.cache_shape,
                                  host_options,
                                  block_type,
                                  static_cast<int64_t>(group_caches.size()));
  }
}

void HierarchyKVCacheTransfer::build_and_validate_schema(
    ParticipantState& state) {
  state.schema.tensor_specs.clear();
  state.schema.component_fingerprints.clear();
  for (const auto& [block_type, group_caches] : state.device_grouped_caches) {
    const auto layer_ids_it = state.absolute_layer_ids.find(block_type);
    const auto host_it = state.host_grouped_caches.find(block_type);
    CHECK(layer_ids_it != state.absolute_layer_ids.end());
    CHECK(host_it != state.host_grouped_caches.end() &&
          host_it->second != nullptr);
    const std::vector<int64_t>& layer_ids = layer_ids_it->second;
    CHECK_EQ(layer_ids.size(), group_caches.size());
    const BlockTypeTensorMap host_tensors =
        host_it->second->get_block_type_tensors(block_type);
    CHECK(!host_tensors.empty());

    std::string component_schema =
        participant_name(state.registration.participant) + "|" +
        state.registration.model_identity +
        "|tp=" + std::to_string(state.registration.tp_size) + ":" +
        std::to_string(state.registration.tp_rank) +
        "|type=" + std::to_string(static_cast<int32_t>(block_type));
    for (size_t layer_slot = 0; layer_slot < group_caches.size();
         ++layer_slot) {
      const BlockTypeTensorMap device_tensors =
          group_caches[layer_slot]->get_block_type_tensors(block_type);
      CHECK(!device_tensors.empty());
      for (const auto& [role, device_tensor] : device_tensors) {
        const auto host_tensor_it = host_tensors.find(role);
        CHECK(host_tensor_it != host_tensors.end())
            << "Missing required Host tensor: participant="
            << participant_name(state.registration.participant)
            << ", type=" << static_cast<int32_t>(block_type)
            << ", layer=" << layer_ids[layer_slot]
            << ", role=" << static_cast<int32_t>(role);
        const torch::Tensor& host_tensor = host_tensor_it->second;
        CHECK_EQ(device_tensor.scalar_type(), host_tensor.scalar_type());
        CHECK_EQ(host_tensor.size(1),
                 static_cast<int64_t>(group_caches.size()));
        const std::vector<int64_t> block_shape =
            device_block_shape(device_tensor);
        CHECK(block_shape == host_block_layer_shape(host_tensor));

        HostCacheTensorSpec spec;
        spec.participant = state.registration.participant;
        spec.block_type = block_type;
        spec.absolute_layer_id = layer_ids[layer_slot];
        spec.host_layer_slot = static_cast<int64_t>(layer_slot);
        spec.role = role;
        spec.coverage = HostPayloadCoverage::REQUIRED;
        spec.dtype = device_tensor.scalar_type();
        spec.block_shape = block_shape;
        state.schema.tensor_specs.emplace_back(std::move(spec));

        component_schema.append("|layer=");
        component_schema.append(std::to_string(layer_ids[layer_slot]));
        component_schema.append(",slot=");
        component_schema.append(std::to_string(layer_slot));
        component_schema.append(",role=");
        component_schema.append(std::to_string(static_cast<int32_t>(role)));
        component_schema.append(",dtype=");
        component_schema.append(
            std::to_string(static_cast<int32_t>(device_tensor.scalar_type())));
        component_schema.append(",shape=");
        for (int64_t dimension : block_shape) {
          component_schema.append(std::to_string(dimension));
          component_schema.push_back('x');
        }
      }
    }
    state.schema.component_fingerprints[block_type] =
        hash_schema_string(component_schema);
  }
}

void HierarchyKVCacheTransfer::validate_composite_schema() const {
  std::map<BlockType, int64_t> device_block_counts;
  std::map<BlockType, int64_t> host_block_counts;
  int64_t block_size = -1;
  for (const auto& [participant, state] : participant_states_) {
    (void)participant;
    const int64_t participant_block_size =
        state.registration.create_options.block_size();
    if (block_size < 0) {
      block_size = participant_block_size;
    } else {
      CHECK_EQ(block_size, participant_block_size)
          << "Composite participants use different logical block sizes.";
    }
    for (const auto& [block_type, group_caches] : state.device_grouped_caches) {
      CHECK(!group_caches.empty());
      const BlockTypeTensorMap device_tensors =
          group_caches.front()->get_block_type_tensors(block_type);
      CHECK(!device_tensors.empty());
      const int64_t device_blocks = device_tensors.begin()->second.size(0);
      auto device_count_it = device_block_counts.find(block_type);
      if (device_count_it == device_block_counts.end()) {
        device_block_counts[block_type] = device_blocks;
      } else {
        CHECK_EQ(device_count_it->second, device_blocks)
            << "Composite participants use different device block counts for "
            << static_cast<int32_t>(block_type);
      }

      const auto host_it = state.host_grouped_caches.find(block_type);
      CHECK(host_it != state.host_grouped_caches.end() &&
            host_it->second != nullptr);
      const BlockTypeTensorMap host_tensors =
          host_it->second->get_block_type_tensors(block_type);
      CHECK(!host_tensors.empty());
      const int64_t host_blocks = host_tensors.begin()->second.size(0);
      auto host_count_it = host_block_counts.find(block_type);
      if (host_count_it == host_block_counts.end()) {
        host_block_counts[block_type] = host_blocks;
      } else {
        CHECK_EQ(host_count_it->second, host_blocks)
            << "Composite participants use different Host block counts for "
            << static_cast<int32_t>(block_type);
      }
      CHECK_GE(host_blocks, device_blocks);
    }
  }
}

void HierarchyKVCacheTransfer::initialize_store() {
  CHECK_GT(options_.host_blocks_factor(), 1.0)
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
  CHECK(kv_cache_store_->init(store_config, this))
      << "Failed to initialize Mooncake Store.";
  LOG(INFO) << "[Mooncake][StoreEngine] ready, endpoint="
            << store_local_hostname << ", protocol=" << store_config.protocol
            << ", tp_rank=" << store_config.tp_rank;
}

HierarchyKVCacheTransfer::CopyPlan HierarchyKVCacheTransfer::build_copy_plan(
    const ParticipantState& state,
    const std::vector<BlockTransferInfo>& block_transfer_info,
    const LayerBatchRange& layer_batch_range) const {
  CopyPlan plan;
  if (block_transfer_info.empty()) {
    return plan;
  }
  const TransferType transfer_type = block_transfer_info.front().transfer_type;
  for (const BlockTransferInfo& info : block_transfer_info) {
    const auto device_it = state.device_grouped_caches.find(info.block_type);
    const auto layer_ids_it = state.absolute_layer_ids.find(info.block_type);
    const auto host_it = state.host_grouped_caches.find(info.block_type);
    if (device_it == state.device_grouped_caches.end() ||
        layer_ids_it == state.absolute_layer_ids.end() ||
        host_it == state.host_grouped_caches.end()) {
      continue;
    }
    const std::vector<KVCache*>& group_caches = device_it->second;
    const std::vector<int64_t>& layer_ids = layer_ids_it->second;
    const BlockTypeTensorMap host_tensors =
        host_it->second->get_block_type_tensors(info.block_type);

    int32_t host_block_id = -1;
    int32_t device_block_id = -1;
    if (transfer_type == TransferType::H2D) {
      host_block_id = info.src_block_id;
      device_block_id = info.dst_block_id;
    } else if (transfer_type == TransferType::D2H2G) {
      host_block_id = info.dst_block_id;
      device_block_id = info.src_block_id;
    } else {
      LOG(FATAL) << "Unsupported transfer type for copy plan: "
                 << static_cast<uint32_t>(transfer_type);
    }
    CHECK_GE(host_block_id, 0);
    CHECK_GE(device_block_id, 0);

    for (size_t layer_slot = 0; layer_slot < group_caches.size();
         ++layer_slot) {
      const int64_t absolute_layer_id = layer_ids[layer_slot];
      if (absolute_layer_id < layer_batch_range.begin_layer ||
          absolute_layer_id >= layer_batch_range.end_layer) {
        continue;
      }
      const BlockTypeTensorMap device_tensors =
          group_caches[layer_slot]->get_block_type_tensors(info.block_type);
      for (const auto& [role, device_tensor] : device_tensors) {
        const auto host_tensor_it = host_tensors.find(role);
        CHECK(host_tensor_it != host_tensors.end())
            << "Required Host tensor disappeared after schema finalization.";
        const torch::Tensor& host_tensor = host_tensor_it->second;
        CHECK_LT(host_block_id, host_tensor.size(0));
        CHECK_LT(device_block_id, device_tensor.size(0));
        torch::Tensor device_block = device_tensor[device_block_id];
        torch::Tensor host_block_layer =
            host_tensor[host_block_id][static_cast<int64_t>(layer_slot)];
        if (transfer_type == TransferType::H2D) {
          plan.src_tensors.emplace_back(host_block_layer);
          plan.dst_tensors.emplace_back(device_block);
        } else {
          plan.src_tensors.emplace_back(device_block);
          plan.dst_tensors.emplace_back(host_block_layer);
        }
      }
    }
  }
  return plan;
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  CHECK(registration_state_ == RegistrationState::READY);
  CHECK(!block_transfer_info.empty());
  device_.set_device();

  switch (block_transfer_info.front().transfer_type) {
    case TransferType::D2H2G:
      return offload(block_transfer_info);
    case TransferType::H2D: {
      auto transaction = std::make_shared<LoadTransaction>();
      if (!snapshot_ready_entries(block_transfer_info, transaction.get())) {
        LOG(ERROR) << "Composite Host cache entry is not READY for batch_id="
                   << batch_id;
        return 0;
      }
      for (const auto& [participant, state] : participant_states_) {
        const bool required = std::any_of(
            block_transfer_info.begin(),
            block_transfer_info.end(),
            [&state](const BlockTransferInfo& info) {
              return state.device_grouped_caches.find(info.block_type) !=
                     state.device_grouped_caches.end();
            });
        if (!required) {
          continue;
        }
        std::shared_ptr<LayerSynchronizer> synchronizer =
            create_layer_synchronizer(
                static_cast<int64_t>(state.layer_batch_ranges.size()));
        CHECK(synchronizer != nullptr)
            << "Failed to create participant layer synchronizer.";
        transaction->synchronizers[participant] = synchronizer;
        transaction->required_participant_mask |= participant_mask(participant);
      }
      CHECK_NE(transaction->required_participant_mask, 0u);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto existing = load_transactions_.find(batch_id);
        if (existing != load_transactions_.end()) {
          existing->second->abort();
          LOG(ERROR) << "Composite load transaction collision at batch_id="
                     << batch_id << "; replacing stale transaction.";
        }
        load_transactions_[batch_id] = transaction;
      }
      for (const auto& [participant, synchronizer] :
           transaction->synchronizers) {
        (void)synchronizer;
        load_threadpool_->schedule(
            [this, participant, transaction, block_transfer_info]() {
              load_from_host(participant, transaction, block_transfer_info);
            });
      }
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
  if (block_transfer_info[0].transfer_type != TransferType::G2H) {
    LOG(ERROR) << "Unsupported slice transfer type: "
               << static_cast<uint32_t>(block_transfer_info[0].transfer_type);
    return 0;
  }
  const std::vector<uint8_t> statuses = prefetch_kv_blocks(block_transfer_info);
  return static_cast<uint32_t>(
      std::count(statuses.begin(), statuses.end(), static_cast<uint8_t>(1)));
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
  update_prefetched_entries(block_transfer_info, hits);
  const size_t hit_count =
      std::count(hits.begin(), hits.end(), static_cast<uint8_t>(1));
  VLOG(1) << "[Mooncake][PrefetchGet] type="
          << static_cast<int32_t>(block_transfer_info[0].block_type)
          << ", blocks=" << hits.size() << ", hits=" << hit_count;
  return hits;
}

uint32_t HierarchyKVCacheTransfer::offload(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return 0;
  }
  if (!begin_entry_write(block_transfer_info)) {
    return 0;
  }
  Slice<BlockTransferInfo> slice(block_transfer_info);
  if (!offload_to_host(slice)) {
    abort_entry_write(block_transfer_info);
    LOG(ERROR) << "Composite offload to Host failed.";
    return 0;
  }
  commit_entry_write(block_transfer_info);
  if (options_.enable_kvcache_store()) {
    CHECK(kv_cache_store_ != nullptr);
    const uint32_t put_count = kv_cache_store_->batch_put(block_transfer_info);
    if (put_count != block_transfer_info.size()) {
      LOG(WARNING) << "Mooncake composite BatchPut partially failed: "
                   << put_count << "/" << block_transfer_info.size();
    }
    VLOG(1) << "[Mooncake][OffloadPut] blocks=" << block_transfer_info.size()
            << ", success=" << put_count;
  }
  return static_cast<uint32_t>(block_transfer_info.size());
}

bool HierarchyKVCacheTransfer::offload_to_host(
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(batch_memcpy_ != nullptr);
  const std::vector<BlockTransferInfo> transfer_info =
      static_cast<std::vector<BlockTransferInfo>>(block_transfer_info);
  CopyStreamLease stream(&copy_stream_);
  std::set<const Stream*> waited_streams;
  for (const auto& [participant, state] : participant_states_) {
    (void)participant;
    if (waited_streams.emplace(state.registration.actual_compute_stream)
            .second) {
      stream.get()->wait_stream(*state.registration.actual_compute_stream);
    }
  }

  for (const auto& [participant, state] : participant_states_) {
    bool participant_copied = true;
    for (const LayerBatchRange& range : state.layer_batch_ranges) {
      CopyPlan plan = build_copy_plan(state, transfer_info, range);
      if (plan.src_tensors.empty()) {
        continue;
      }
      if (!batch_memcpy_->copy_d2h(
              plan.src_tensors, plan.dst_tensors, stream.get())) {
        participant_copied = false;
        break;
      }
    }
    if (!participant_copied) {
      return false;
    }
    complete_participant_write(participant, transfer_info);
  }
  return true;
}

bool HierarchyKVCacheTransfer::load_from_host(
    CacheParticipant participant,
    const std::shared_ptr<LoadTransaction>& transaction,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  const auto state_it = participant_states_.find(participant);
  CHECK(state_it != participant_states_.end());
  const ParticipantState& state = state_it->second;
  const auto synchronizer_it = transaction->synchronizers.find(participant);
  CHECK(synchronizer_it != transaction->synchronizers.end());
  const std::shared_ptr<LayerSynchronizer>& synchronizer =
      synchronizer_it->second;

  CopyStreamLease stream(&copy_stream_);
  bool success = true;
  bool stream_has_async_h2d = false;
  for (size_t range_index = 0; range_index < state.layer_batch_ranges.size();
       ++range_index) {
    if (!entry_snapshot_matches(block_transfer_info, *transaction)) {
      success = false;
      break;
    }
    CopyPlan plan = build_copy_plan(
        state, block_transfer_info, state.layer_batch_ranges[range_index]);
    if (!plan.src_tensors.empty()) {
      if (!batch_memcpy_->submit_h2d(
              plan.src_tensors, plan.dst_tensors, stream.get())) {
        success = false;
        break;
      }
      stream_has_async_h2d = true;
    }
    if (!synchronizer->record_stream(static_cast<int64_t>(range_index),
                                     stream.get())) {
      if (stream_has_async_h2d) {
        stream.drain_or_die("participant layer-ready event recording failed");
      }
      success = false;
      break;
    }
  }
  if (success && !entry_snapshot_matches(block_transfer_info, *transaction)) {
    success = false;
  }
  if (!success) {
    transaction->abort();
  }
  return success;
}

void HierarchyKVCacheTransfer::set_layer_synchronizer(
    ModelInputParams& params) {
  CHECK_EQ(participant_states_.size(), 1u)
      << "Single-participant synchronizer API used by a composite transfer.";
  set_layer_synchronizer(participant_states_.begin()->first, params);
}

void HierarchyKVCacheTransfer::set_layer_synchronizer(
    CacheParticipant participant,
    ModelInputParams& params) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto transaction_it = load_transactions_.find(params.meta.batch_id);
  if (transaction_it == load_transactions_.end()) {
    return;
  }
  const std::shared_ptr<LoadTransaction>& transaction = transaction_it->second;
  const auto synchronizer_it = transaction->synchronizers.find(participant);
  if (synchronizer_it == transaction->synchronizers.end()) {
    return;
  }
  params.parallel.layer_wise_load_synchronizer = synchronizer_it->second;
  const ParticipantState& state = participant_states_.at(participant);
  params.parallel.layers_per_bacth_copy =
      state.layer_batch_ranges.empty()
          ? static_cast<uint32_t>(state.registration.device_caches->size())
          : static_cast<uint32_t>(state.layer_batch_ranges.front().end_layer -
                                  state.layer_batch_ranges.front().begin_layer);
  transaction->consumed_participant_mask |= participant_mask(participant);
  if (transaction->consumed_participant_mask ==
      transaction->required_participant_mask) {
    load_transactions_.erase(transaction_it);
  }
}

std::vector<HostCacheComponentSchema>
HierarchyKVCacheTransfer::store_components() const {
  std::vector<HostCacheComponentSchema> components;
  for (const auto& [participant, state] : participant_states_) {
    for (const auto& [block_type, fingerprint] :
         state.schema.component_fingerprints) {
      HostCacheComponentSchema component;
      component.participant = participant;
      component.block_type = block_type;
      component.model_identity = state.registration.model_identity;
      component.schema_fingerprint = fingerprint;
      component.tp_rank = state.registration.tp_rank;
      component.tp_size = state.registration.tp_size;
      components.emplace_back(std::move(component));
    }
  }
  return components;
}

std::vector<torch::Tensor> HierarchyKVCacheTransfer::component_storage_tensors(
    CacheParticipant participant,
    BlockType block_type) const {
  const ParticipantState& state = participant_states_.at(participant);
  const auto cache_it = state.host_grouped_caches.find(block_type);
  CHECK(cache_it != state.host_grouped_caches.end() &&
        cache_it->second != nullptr);
  const BlockTypeTensorMap tensors =
      cache_it->second->get_block_type_tensors(block_type);
  std::vector<torch::Tensor> storage;
  storage.reserve(tensors.size());
  for (const auto& [role, tensor] : tensors) {
    (void)role;
    storage.emplace_back(tensor);
  }
  return storage;
}

std::vector<torch::Tensor> HierarchyKVCacheTransfer::component_tensors(
    CacheParticipant participant,
    BlockType block_type,
    int32_t host_block_id) const {
  const std::vector<torch::Tensor> storage =
      component_storage_tensors(participant, block_type);
  std::vector<torch::Tensor> blocks;
  blocks.reserve(storage.size());
  for (const torch::Tensor& tensor : storage) {
    CHECK_GE(host_block_id, 0);
    CHECK_LT(host_block_id, tensor.size(0));
    torch::Tensor block = tensor[host_block_id];
    CHECK(block.is_contiguous());
    blocks.emplace_back(std::move(block));
  }
  return blocks;
}

uint32_t HierarchyKVCacheTransfer::participant_mask(
    CacheParticipant participant) {
  return 1u << static_cast<uint32_t>(participant);
}

uint64_t HierarchyKVCacheTransfer::schema_fingerprint_value(
    const std::string& fingerprint) {
  CHECK_GE(fingerprint.size(), sizeof(uint64_t));
  uint64_t value = 0;
  std::memcpy(&value, fingerprint.data(), sizeof(value));
  return value;
}

uint32_t HierarchyKVCacheTransfer::required_participant_mask(
    BlockType block_type) const {
  uint32_t mask = 0;
  for (const auto& [participant, state] : participant_states_) {
    if (participant_requires_type(state, block_type)) {
      mask |= participant_mask(participant);
    }
  }
  return mask;
}

uint64_t HierarchyKVCacheTransfer::composite_schema_fingerprint(
    BlockType block_type) const {
  std::string composite_schema;
  for (const auto& [participant, state] : participant_states_) {
    const auto fingerprint_it =
        state.schema.component_fingerprints.find(block_type);
    if (fingerprint_it == state.schema.component_fingerprints.end()) {
      continue;
    }
    composite_schema.append(participant_name(participant));
    composite_schema.append(fingerprint_it->second);
  }
  CHECK(!composite_schema.empty());
  return schema_fingerprint_value(hash_schema_string(composite_schema));
}

bool HierarchyKVCacheTransfer::participant_requires_type(
    const ParticipantState& state,
    BlockType block_type) const {
  return state.device_grouped_caches.find(block_type) !=
         state.device_grouped_caches.end();
}

bool HierarchyKVCacheTransfer::begin_entry_write(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const BlockTransferInfo& info : block_transfer_info) {
    const uint32_t required_mask = required_participant_mask(info.block_type);
    if (required_mask == 0) {
      LOG(ERROR) << "No participant owns requested BlockType "
                 << static_cast<int32_t>(info.block_type);
      return false;
    }
    HostCacheEntryMetadata& metadata =
        entry_metadata_[{info.block_type, info.dst_block_id}];
    ++metadata.generation;
    metadata.schema_fingerprint = composite_schema_fingerprint(info.block_type);
    metadata.required_participant_mask = required_mask;
    metadata.completed_participant_mask = 0;
    metadata.state = HostEntryState::WRITING;
  }
  return true;
}

void HierarchyKVCacheTransfer::complete_participant_write(
    CacheParticipant participant,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  const uint32_t mask = participant_mask(participant);
  for (const BlockTransferInfo& info : block_transfer_info) {
    HostCacheEntryMetadata& metadata =
        entry_metadata_.at({info.block_type, info.dst_block_id});
    if ((metadata.required_participant_mask & mask) != 0) {
      metadata.completed_participant_mask |= mask;
    }
  }
}

void HierarchyKVCacheTransfer::commit_entry_write(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const BlockTransferInfo& info : block_transfer_info) {
    HostCacheEntryMetadata& metadata =
        entry_metadata_.at({info.block_type, info.dst_block_id});
    CHECK_EQ(metadata.completed_participant_mask,
             metadata.required_participant_mask);
    metadata.state = HostEntryState::READY;
  }
}

void HierarchyKVCacheTransfer::abort_entry_write(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const BlockTransferInfo& info : block_transfer_info) {
    HostCacheEntryMetadata& metadata =
        entry_metadata_[{info.block_type, info.dst_block_id}];
    metadata.state = HostEntryState::INVALID;
  }
}

bool HierarchyKVCacheTransfer::snapshot_ready_entries(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    LoadTransaction* transaction) const {
  CHECK(transaction != nullptr);
  std::lock_guard<std::mutex> lock(mutex_);
  transaction->entry_generations.clear();
  for (const BlockTransferInfo& info : block_transfer_info) {
    const std::pair<BlockType, int32_t> entry_key = {info.block_type,
                                                     info.src_block_id};
    const auto metadata_it = entry_metadata_.find(entry_key);
    if (metadata_it == entry_metadata_.end()) {
      return false;
    }
    const HostCacheEntryMetadata& metadata = metadata_it->second;
    if (metadata.state != HostEntryState::READY ||
        metadata.required_participant_mask !=
            required_participant_mask(info.block_type) ||
        metadata.schema_fingerprint !=
            composite_schema_fingerprint(info.block_type)) {
      return false;
    }
    transaction->entry_generations[entry_key] = metadata.generation;
  }
  return true;
}

bool HierarchyKVCacheTransfer::entry_snapshot_matches(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    const LoadTransaction& transaction) const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const BlockTransferInfo& info : block_transfer_info) {
    const std::pair<BlockType, int32_t> entry_key = {info.block_type,
                                                     info.src_block_id};
    const auto generation_it = transaction.entry_generations.find(entry_key);
    const auto metadata_it = entry_metadata_.find(entry_key);
    if (generation_it == transaction.entry_generations.end() ||
        metadata_it == entry_metadata_.end()) {
      return false;
    }
    const HostCacheEntryMetadata& metadata = metadata_it->second;
    if (metadata.state != HostEntryState::READY ||
        metadata.generation != generation_it->second ||
        metadata.required_participant_mask !=
            required_participant_mask(info.block_type) ||
        metadata.schema_fingerprint !=
            composite_schema_fingerprint(info.block_type)) {
      return false;
    }
  }
  return true;
}

void HierarchyKVCacheTransfer::update_prefetched_entries(
    Slice<BlockTransferInfo>& block_transfer_info,
    const std::vector<uint8_t>& statuses) {
  CHECK_EQ(block_transfer_info.size(), statuses.size());
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t index = 0; index < block_transfer_info.size(); ++index) {
    const BlockTransferInfo& info = block_transfer_info[index];
    HostCacheEntryMetadata& metadata =
        entry_metadata_[{info.block_type, info.dst_block_id}];
    ++metadata.generation;
    metadata.schema_fingerprint = composite_schema_fingerprint(info.block_type);
    metadata.required_participant_mask =
        required_participant_mask(info.block_type);
    metadata.completed_participant_mask =
        statuses[index] == 1 ? metadata.required_participant_mask : 0;
    metadata.state =
        statuses[index] == 1 ? HostEntryState::READY : HostEntryState::INVALID;
  }
}

}  // namespace xllm
