
#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <Mooncake/mooncake-store/include/utils.h>
#include <glog/logging.h>

#include <string>
#include <unordered_map>

#include "util/hash_util.h"

namespace xllm {

bool KVCacheStore::init(const KVCacheStoreInitConfig& config,
                        std::vector<xllm::KVCache>* host_kv_caches) {
  CHECK(!is_initialized_) << "KVCacheStore is initialized.";
  config_ = config;
  host_kv_caches_ = host_kv_caches;
  std::optional<std::string> device_names = std::nullopt;
  if (config_.protocol == "rdma") {
    if (getenv("DEVICE_NAMES")) {
      device_names = getenv("DEVICE_NAMES");
      LOG(INFO) << "device_names: " << device_names.value();
    } else {
      LOG(WARNING) << "env DEVICE_NAME not exist, set protocol as tcp";
      config_.protocol = "tcp";
    }
  }

  auto client_opt = mooncake::Client::Create(config_.localhost_name,
                                             config_.metadata_server,
                                             config_.protocol,
                                             device_names,
                                             config_.master_server_address);

  rep_config_.replica_num = config_.replica_num;
  // rep_config_.preferred_segment = config_.localhost_name;

  if (!client_opt.has_value()) {
    LOG(FATAL) << "mooncake::Client::Create fail! Failed to create client with "
                  "host_name: "
               << config_.localhost_name;
  }
  client_ptr_ = client_opt.value();

  auto k_cache = host_kv_caches_->at(0).get_k_cache();
  k_cache_size_per_block_ = k_cache.numel() * k_cache.element_size();
  LOG(INFO) << "key cache size per block: " << k_cache_size_per_block_;

  auto v_cache = host_kv_caches_->at(0).get_v_cache();
  if (v_cache.defined() && v_cache.numel() != 0) {
    v_cache_size_per_block_ = v_cache.numel() * v_cache.element_size();
    LOG(INFO) << "value cache size per block: " << v_cache_size_per_block_;
  }

  auto index_cache = host_kv_caches_->at(0).get_index_cache();
  if (index_cache.defined() && index_cache.numel() != 0) {
    index_cache_size_per_block_ =
        index_cache.numel() * index_cache.element_size();
    LOG(INFO) << "index cache size per block: " << index_cache_size_per_block_;
  }

  if (config_.protocol == "rdma") {
    if (config_.total_size > 0 && config_.tensor_data != nullptr) {
      auto result = client_ptr_->RegisterLocalMemory(
          config_.tensor_data, config_.total_size, "cpu:0", false, false);
      if (!result.has_value()) {
        LOG(ERROR) << "Failed to register local memory: "
                   << toString(result.error());
        return false;
      }
    } else {
      LOG(ERROR) << "rdma must RegisterLocalMemory, but got register size: "
                 << config_.total_size
                 << ", and data ptr: " << uint64_t(config_.tensor_data);
      return false;
    }
  }
  is_initialized_ = true;
  return true;
}

KVCacheStore::~KVCacheStore() {
  if (client_ptr_) {
    client_ptr_.reset();
  }
}

uint32_t KVCacheStore::batch_put(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!is_initialized_) {
    return 0;
  }
  std::vector<std::string> str_keys;
  std::vector<std::vector<mooncake::Slice>> slices;

  str_keys.reserve(block_transfer_info.size());
  slices.reserve(block_transfer_info.size());
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        XXH3_128BITS_HASH_VALUE_LEN);

    str_key.append(std::to_string(config_.tp_rank));

    auto exist = client_ptr_->IsExist(str_key);
    if (exist.has_value() && exist.value()) {
      continue;
    }

    str_keys.emplace_back(str_key);
    slices.emplace_back(
        std::move(genarate_mooncake_slice(block_info.dst_block_id)));
  }

  if (str_keys.size() == 0) {
    return block_transfer_info.size();
  }

  uint64_t success_cnt = block_transfer_info.size() - str_keys.size();
  auto results = client_ptr_->BatchPut(str_keys, slices, rep_config_);

  for (int i = 0; i < str_keys.size(); i++) {
    if (!results[i].has_value()) {
      break;
    }
    success_cnt++;
  }
  return success_cnt;
}

uint32_t KVCacheStore::batch_get(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!is_initialized_) {
    return 0;
  }
  std::unordered_map<std::string, std::vector<mooncake::Slice>> slices;
  std::vector<std::string> str_keys;

  str_keys.reserve(block_transfer_info.size());
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        XXH3_128BITS_HASH_VALUE_LEN);

    str_key.append(std::to_string(config_.tp_rank));
    auto exist = client_ptr_->IsExist(str_key);
    if (!exist.has_value() || !exist.value()) {
      break;
    }

    str_keys.emplace_back(str_key);
    slices.insert(std::make_pair(
        str_key, std::move(genarate_mooncake_slice(block_info.dst_block_id))));
  }

  if (str_keys.size() == 0) {
    return 0;
  }

  uint64_t success_cnt = 0;
  auto results = client_ptr_->BatchGet(str_keys, slices);
  for (int i = 0; i < str_keys.size(); i++) {
    if (!results[i].has_value()) {
      break;
    }
    success_cnt++;
  }
  return success_cnt;
}

uint32_t KVCacheStore::batch_exist(std::vector<std::string>&& keys) {
  if (!is_initialized_) {
    return 0;
  }
  auto exist_vec = client_ptr_->BatchIsExist(std::move(keys));
  uint32_t ret = 0;
  for (auto exist : exist_vec) {
    if (!exist.has_value() || !exist.value()) {
      break;
    }
    ret++;
  }
  return ret;
}

std::vector<mooncake::Slice> KVCacheStore::genarate_mooncake_slice(
    int32_t block_id) {
  std::vector<mooncake::Slice> slice;
  slice.reserve(3);

  void* k_cache = host_kv_caches_->at(block_id).get_k_cache().data_ptr();
  slice.emplace_back(mooncake::Slice{k_cache, k_cache_size_per_block_});

  if (v_cache_size_per_block_ != 0) {
    void* v_cache = host_kv_caches_->at(block_id).get_v_cache().data_ptr();
    slice.emplace_back(mooncake::Slice{v_cache, v_cache_size_per_block_});
  }

  if (index_cache_size_per_block_ != 0) {
    void* index_cache =
        host_kv_caches_->at(block_id).get_index_cache().data_ptr();
    slice.emplace_back(
        mooncake::Slice{index_cache, index_cache_size_per_block_});
  }
  return slice;
}

}  // namespace xllm
