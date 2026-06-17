/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/runtime/worker_rendezvous.h"

#include <glog/logging.h>

#include <cstddef>

#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#if defined(USE_NPU)
#include "framework/kv_cache_transfer/mooncake_weight_transfer.h"
#endif

namespace xllm {

#if defined(USE_NPU)
WorkerRendezvous::WorkerRendezvous(
    const std::shared_ptr<KVCacheTransfer>& kv_cache_transfer,
    const std::unique_ptr<MooncakeWeightTransfer>& weight_transfer)
    : kv_cache_transfer_(kv_cache_transfer),
      weight_transfer_(weight_transfer) {}
#else
WorkerRendezvous::WorkerRendezvous(
    const std::shared_ptr<KVCacheTransfer>& kv_cache_transfer)
    : kv_cache_transfer_(kv_cache_transfer) {}
#endif

bool WorkerRendezvous::link_cluster(const std::vector<uint64_t>& cluster_ids,
                                    const std::vector<std::string>& addrs,
                                    const std::vector<uint16_t>& ports) {
#if defined(USE_NPU) || defined(USE_MLU)
  if (!kv_cache_transfer_) {
    LOG(ERROR) << "KVCacheTransfer not initialized";
    return false;
  }
  if (!validate_cluster_endpoints(cluster_ids, addrs, ports)) {
    return false;
  }

  const size_t cluster_count = cluster_ids.size();
  for (size_t i = 0; i < cluster_count; ++i) {
    if (!kv_cache_transfer_->link_cluster(cluster_ids[i], addrs[i], ports[i])) {
      return false;
    }
  }
#endif
  return true;
}

bool WorkerRendezvous::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                      const std::vector<std::string>& addrs,
                                      const std::vector<uint16_t>& ports) {
#if defined(USE_NPU) || defined(USE_MLU)
  if (!kv_cache_transfer_) {
    LOG(ERROR) << "KVCacheTransfer not initialized";
    return false;
  }
  if (!validate_cluster_endpoints(cluster_ids, addrs, ports)) {
    return false;
  }

  const size_t cluster_count = cluster_ids.size();
  for (size_t i = 0; i < cluster_count; ++i) {
    if (!kv_cache_transfer_->unlink_cluster(
            cluster_ids[i], addrs[i], ports[i])) {
      return false;
    }
  }
#endif
  return true;
}

bool WorkerRendezvous::link_p2p(const std::string& remote_addr) {
#if defined(USE_NPU)
  if (!weight_transfer_) {
    LOG(ERROR) << "MooncakeWeightTransfer not initialized";
    return false;
  }
  return weight_transfer_->link_p2p(remote_addr);
#else
  LOG(ERROR) << "link_p2p requires USE_NPU build";
  return false;
#endif
}

bool WorkerRendezvous::unlink_p2p(const std::string& remote_addr) {
#if defined(USE_NPU)
  if (!weight_transfer_) {
    LOG(ERROR) << "MooncakeWeightTransfer not initialized";
    return false;
  }
  return weight_transfer_->unlink_p2p(remote_addr);
#else
  LOG(ERROR) << "unlink_p2p requires USE_NPU build";
  return false;
#endif
}

bool WorkerRendezvous::validate_cluster_endpoints(
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& addrs,
    const std::vector<uint16_t>& ports) const {
  const size_t cluster_count = cluster_ids.size();
  if (addrs.size() == cluster_count && ports.size() == cluster_count) {
    return true;
  }

  LOG(ERROR) << "Cluster endpoint size mismatch: cluster_ids="
             << cluster_ids.size() << ", addrs=" << addrs.size()
             << ", ports=" << ports.size();
  return false;
}

}  // namespace xllm
