/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/kv_cache/transfer/yuanrong_kv_cache_transfer.h"

#include <dlfcn.h>
#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <utility>

#include "core/kv_cache/storage/kv_cache_utils.h"
#include "util/hash_util.h"

namespace xllm {
namespace {

constexpr char kBridgeLibraryEnv[] = "XLLM_YUANRONG_BRIDGE";
constexpr char kBridgeLibraryName[] = "libxllm_yuanrong_datasystem_bridge.so";
constexpr char kDataSystemHostEnv[] = "XLLM_YUANRONG_DS_HOST";
constexpr char kDataSystemPortEnv[] = "XLLM_YUANRONG_DS_PORT";
constexpr int32_t kDefaultMGetTimeoutMs = 60000;

std::string sanitize_key_component(std::string value) {
  std::replace_if(
      value.begin(),
      value.end(),
      [](unsigned char ch) {
        return !std::isalnum(ch) && ch != '-' && ch != '_';
      },
      '_');
  return value;
}

void add_tensor_blob(
    std::vector<YuanRongKVCacheTransfer::BridgeBlob>& blob_list,
    const KVCacheTensor& cache_tensor,
    uint64_t block_id) {
  const torch::Tensor& tensor = cache_tensor.tensor;
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }
  CHECK_GT(tensor.dim(), 0);
  CHECK_LT(block_id, static_cast<uint64_t>(tensor.size(0)));
  torch::Tensor block = tensor[static_cast<int64_t>(block_id)];
  blob_list.push_back(YuanRongKVCacheTransfer::BridgeBlob{
      block.data_ptr(), static_cast<uint64_t>(block.nbytes())});
}

int32_t mget_timeout_ms() {
  const char* value = std::getenv("XLLM_YUANRONG_TIMEOUT_MS");
  if (value == nullptr) {
    return kDefaultMGetTimeoutMs;
  }
  const int64_t parsed = std::strtoll(value, nullptr, 10);
  if (parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
    return kDefaultMGetTimeoutMs;
  }
  return static_cast<int32_t>(parsed);
}

}  // namespace

YuanRongKVCacheTransfer::YuanRongKVCacheTransfer(int32_t device_id,
                                                 uint16_t listen_port,
                                                 const torch::Device& device,
                                                 std::string model_type,
                                                 std::string model_id,
                                                 bool publish_source_blocks)
    : model_type_(std::move(model_type)),
      model_id_(std::move(model_id)),
      publish_source_blocks_(publish_source_blocks),
      listen_port_(listen_port),
      device_id_(device_id),
      device_(device) {}

YuanRongKVCacheTransfer::~YuanRongKVCacheTransfer() {
  finalize();
  if (bridge_handle_ != nullptr) {
    dlclose(bridge_handle_);
    bridge_handle_ = nullptr;
  }
}

void YuanRongKVCacheTransfer::initialize(int32_t device_id) {
  device_id_ = device_id;
  CHECK(ensure_initialized(publish_source_blocks_))
      << "Failed to initialize DataSystem bridge: " << bridge_last_error();
}

void YuanRongKVCacheTransfer::finalize() {
  if (bridge_client_ != nullptr && bridge_destroy_ != nullptr) {
    bridge_destroy_(bridge_client_);
    bridge_client_ = nullptr;
  }
}

void YuanRongKVCacheTransfer::configure_cache_layout(
    const ParallelArgs& parallel_args,
    const ModelArgs& model_args,
    int32_t /*block_token_capacity*/,
    bool /*is_spec_draft*/) {
  CHECK_EQ(parallel_args.world_size() %
               (parallel_args.dp_size() * parallel_args.cp_size()),
           0);
  // Same decomposition the Mooncake backend publishes as cache coordinates:
  // P and D must derive identical ranks for the same KV shard, or they would
  // read and write different object names for the same block.
  const int32_t tp_size = parallel_args.world_size() / parallel_args.dp_size() /
                          parallel_args.cp_size();
  const int32_t rank_in_dp =
      parallel_args.rank() % (parallel_args.cp_size() * tp_size);
  dp_rank_ = parallel_args.rank() / (parallel_args.cp_size() * tp_size);
  tp_rank_ = rank_in_dp % tp_size;
  cp_rank_ = rank_in_dp / tp_size;
  kv_split_rank_ = parallel_args.kv_split_rank();

  const int64_t kv_head_count =
      model_args.n_kv_heads().value_or(model_args.n_heads());
  layout_fingerprint_ = model_args.model_type() + ":" +
                        std::to_string(model_args.n_layers()) + ":" +
                        std::to_string(kv_head_count) + ":" +
                        std::to_string(model_args.head_dim()) + ":" +
                        std::to_string(model_args.index_n_heads());
  layout_configured_ = true;
}

void YuanRongKVCacheTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& /*kv_cache_shape*/,
    torch::ScalarType /*dtype*/) {
  CHECK(!kv_caches.empty());
  CHECK(kv_caches_ == nullptr)
      << "DataSystem backend does not support speculative draft KV cache "
         "registration.";
  kv_caches_ = &kv_caches;
  num_layers_ = static_cast<int64_t>(kv_caches.size());
  LOG(INFO) << "DataSystem KV transfer registered " << num_layers_
            << " layers, role="
            << (publish_source_blocks_ ? "publisher(PREFILL)"
                                       : "receiver(DECODE)");
}

void YuanRongKVCacheTransfer::get_cache_info(uint64_t& cluster_id,
                                             std::string& addr) {
  cluster_id = cluster_id_;
  addr = addr_;
}

bool YuanRongKVCacheTransfer::link_clusters(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& remote_addrs,
    const std::vector<uint16_t>& /*ports*/) {
  CHECK_EQ(cluster_ids.size(), remote_addrs.size());
  // DataSystem reads and writes a shared host-object store, so there is no
  // peer link to establish. The mapping is still recorded so a pull can reject
  // an endpoint the coordinator never announced.
  for (size_t index = 0; index < cluster_ids.size(); ++index) {
    linked_clusters_[cluster_ids[index]] = remote_addrs[index];
  }
  return true;
}

bool YuanRongKVCacheTransfer::unlink_cluster(const uint64_t& cluster_id,
                                             const std::string& /*remote_addr*/,
                                             const uint16_t /*port*/,
                                             bool force_flag) {
  const auto it = linked_clusters_.find(cluster_id);
  if (it == linked_clusters_.end()) {
    return force_flag;
  }
  linked_clusters_.erase(it);
  return true;
}

bool YuanRongKVCacheTransfer::pull_kv_blocks(
    const uint64_t /*src_cluster_id*/,
    const std::string& /*src_addr*/,
    const std::vector<KVTransferMapping>& mappings) {
  const bool initialized = ensure_initialized(/*publish_source_blocks=*/false);
  if (!initialized) {
    LOG(ERROR) << "DataSystem pull requires a bridge client: "
               << bridge_last_error();
    return false;
  }

  const int32_t timeout_ms = mget_timeout_ms();
  for (const KVTransferMapping& mapping : mappings) {
    // Content addressing only covers the plain KV group; SWA / linear /
    // embedding caches have no prefix-stable hash and must fail loudly rather
    // than pull the wrong bytes.
    CHECK_EQ(mapping.hashes.size(), mapping.local_ids.size())
        << "DataSystem pull needs one content hash per block, group_id="
        << mapping.group_id << ", hashes=" << mapping.hashes.size()
        << ", local=" << mapping.local_ids.size();
    if (mapping.hashes.empty()) {
      continue;
    }

    std::vector<std::string> keys;
    keys.reserve(mapping.hashes.size());
    for (const std::string& hash : mapping.hashes) {
      keys.emplace_back(block_key(hash));
    }

    std::vector<std::vector<BridgeBlob>> blob_storage;
    blob_storage.reserve(mapping.local_ids.size());
    for (uint64_t block_id : mapping.local_ids) {
      blob_storage.emplace_back(make_blob_list(block_id));
    }

    std::vector<BridgeBlobList> blob_lists;
    blob_lists.reserve(blob_storage.size());
    for (const std::vector<BridgeBlob>& blobs : blob_storage) {
      blob_lists.push_back(
          BridgeBlobList{device_id_, blobs.data(), blobs.size()});
    }

    std::vector<const char*> raw_keys;
    raw_keys.reserve(keys.size());
    for (const std::string& key : keys) {
      raw_keys.push_back(key.c_str());
    }

    if (!bridge_mget_(bridge_client_,
                      raw_keys.data(),
                      blob_lists.data(),
                      raw_keys.size(),
                      timeout_ms)) {
      LOG(ERROR) << "DataSystem MGetH2D failed: " << bridge_last_error();
      return false;
    }
  }
  return true;
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
bool YuanRongKVCacheTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    bool /*is_spec_draft*/,
    int32_t /*kv_split_rank*/,
    int32_t /*kv_split_size*/) {
  if (!publish_source_blocks_) {
    return true;
  }
  const bool initialized = ensure_initialized(/*publish_source_blocks=*/true);
  if (!initialized) {
    LOG(ERROR) << "DataSystem publish requires a bridge client: "
               << bridge_last_error();
    return false;
  }
  CHECK(kv_caches_ != nullptr);

  // Every layer's KV must be on device before its bytes are copied out.
  if (!layer_synchronizer->synchronize_layer(num_layers_ - 1)) {
    LOG(ERROR) << "Failed to synchronize the last KV cache layer before "
                  "DataSystem publish.";
    return false;
  }

  for (auto& [key, kv_info] : merged_kv_infos) {
    (void)key;
    for (const KVTransferMapping& mapping : kv_info.mappings) {
      CHECK_EQ(mapping.hashes.size(), mapping.local_ids.size())
          << "DataSystem publish needs one content hash per block, group_id="
          << mapping.group_id << ", hashes=" << mapping.hashes.size()
          << ", local=" << mapping.local_ids.size();
      if (mapping.hashes.empty()) {
        continue;
      }

      std::vector<std::string> keys;
      keys.reserve(mapping.hashes.size());
      for (const std::string& hash : mapping.hashes) {
        keys.emplace_back(block_key(hash));
      }

      std::vector<BridgeBlobList> blob_lists;
      std::vector<std::vector<BridgeBlob>> blob_storage;
      blob_lists.reserve(mapping.local_ids.size());
      blob_storage.reserve(mapping.local_ids.size());
      for (uint64_t block_id : mapping.local_ids) {
        blob_storage.emplace_back(make_blob_list(block_id));
        blob_lists.push_back(BridgeBlobList{device_id_,
                                            blob_storage.back().data(),
                                            blob_storage.back().size()});
      }

      std::vector<const char*> raw_keys;
      raw_keys.reserve(keys.size());
      for (const std::string& k : keys) {
        raw_keys.push_back(k.c_str());
      }
      if (!bridge_mset_(bridge_client_,
                        raw_keys.data(),
                        blob_lists.data(),
                        raw_keys.size())) {
        LOG(ERROR) << "DataSystem MSetD2H failed: " << bridge_last_error();
        return false;
      }
    }
  }
  return true;
}
#endif

std::string YuanRongKVCacheTransfer::block_key(
    const std::string& block_hash) const {
  CHECK_EQ(block_hash.size(), XXH3_128BITS_HASH_VALUE_LEN * 2)
      << "Invalid DataSystem content hash: " << block_hash;
  const std::string layout_identity = sanitize_key_component(
      layout_fingerprint_.empty() ? model_type_ : layout_fingerprint_);
  return "xllm_ds_v1_model_" + layout_identity + "_rank_" +
         std::to_string(dp_rank_) + "_tp_" + std::to_string(tp_rank_) + "_cp_" +
         std::to_string(cp_rank_) + "_kvsplit_" +
         std::to_string(kv_split_rank_) + "_hash_" + block_hash;
}

std::vector<YuanRongKVCacheTransfer::BridgeBlob>
YuanRongKVCacheTransfer::make_blob_list(uint64_t block_id) const {
  CHECK(kv_caches_ != nullptr);
  std::vector<BridgeBlob> blob_list;
  for (const xllm::KVCache& kv_cache : *kv_caches_) {
    for (const KVCacheTensor& cache_tensor : kv_cache.get_cache_tensors()) {
      add_tensor_blob(blob_list, cache_tensor, block_id);
    }
  }
  return blob_list;
}

bool YuanRongKVCacheTransfer::load_bridge() {
  if (bridge_handle_ != nullptr) {
    return true;
  }
  const char* configured = std::getenv(kBridgeLibraryEnv);
  const char* library = configured != nullptr && configured[0] != '\0'
                            ? configured
                            : kBridgeLibraryName;
  bridge_handle_ = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (bridge_handle_ == nullptr) {
    LOG(ERROR) << "Failed to load the DataSystem bridge " << library << " (set "
               << kBridgeLibraryEnv << " to override): " << dlerror();
    return false;
  }

  auto load_symbol = [this](const char* name) -> void* {
    void* symbol = dlsym(bridge_handle_, name);
    if (symbol == nullptr) {
      LOG(ERROR) << "DataSystem bridge is missing symbol " << name;
    }
    return symbol;
  };

  bridge_create_ =
      reinterpret_cast<BridgeCreateFn>(load_symbol("xllm_yr_ds_create"));
  bridge_destroy_ =
      reinterpret_cast<BridgeDestroyFn>(load_symbol("xllm_yr_ds_destroy"));
  bridge_mset_ = reinterpret_cast<BridgeMSetFn>(load_symbol("xllm_yr_ds_mset"));
  bridge_mget_ = reinterpret_cast<BridgeMGetFn>(load_symbol("xllm_yr_ds_mget"));
  bridge_last_error_ =
      reinterpret_cast<BridgeLastErrorFn>(load_symbol("xllm_yr_ds_last_error"));
  return bridge_create_ != nullptr && bridge_destroy_ != nullptr &&
         bridge_mset_ != nullptr && bridge_mget_ != nullptr &&
         bridge_last_error_ != nullptr;
}

const char* YuanRongKVCacheTransfer::bridge_last_error() const {
  if (bridge_last_error_ == nullptr || bridge_client_ == nullptr) {
    return "bridge not loaded";
  }
  const char* message = bridge_last_error_(bridge_client_);
  return message == nullptr ? "" : message;
}

bool YuanRongKVCacheTransfer::ensure_initialized(bool publish_source_blocks) {
  if (bridge_client_ != nullptr) {
    return true;
  }
  if (!load_bridge()) {
    return false;
  }
  const char* host_env = std::getenv(kDataSystemHostEnv);
  const char* port_env = std::getenv(kDataSystemPortEnv);
  const std::string host = host_env != nullptr ? host_env : "127.0.0.1";
  const int64_t port_value =
      port_env != nullptr ? std::strtoll(port_env, nullptr, 10) : 0;
  CHECK_GE(port_value, 0);
  CHECK_LE(port_value, std::numeric_limits<uint16_t>::max());
  const int32_t port = static_cast<int32_t>(port_value);
  if (addr_.empty()) {
    addr_ = host + ":" + std::to_string(port);
  }
  // A publisher copies device -> host (MSetD2H); a receiver copies host ->
  // device (MGetH2D) and therefore needs remote H2D enabled.
  bridge_client_ = bridge_create_(host.c_str(),
                                  port,
                                  mget_timeout_ms(),
                                  device_id_,
                                  publish_source_blocks ? 0 : 1);
  if (bridge_client_ == nullptr) {
    LOG(ERROR) << "DataSystem bridge create failed for " << addr_;
    return false;
  }
  cluster_id_ = static_cast<uint64_t>(device_id_) + 1;
  return true;
}

}  // namespace xllm
