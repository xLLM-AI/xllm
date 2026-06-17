/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "xtensor_manager_pool.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/framework/config/distributed_config.h"
#include "distributed_runtime/collective_service.h"
#include "options.h"
#include "remote_xtensor_manager.h"
#include "server/xllm_server_registry.h"

namespace xllm {
XTensorManagerPool::XTensorManagerPool(const xtensor::Options& options,
                                       int32_t dp_size)
    : options_(options),
      dp_size_(dp_size),
      collective_server_name_("XTensorManagerCollectiveServer") {
  if (::xllm::DistributedConfig::get_instance().master_node_addr().empty()) {
    setup_single_node_xtensor_managers();
  } else {
    collective_server_name_.append(std::to_string(options.server_idx()));
    setup_multi_node_xtensor_managers(
        ::xllm::DistributedConfig::get_instance().master_node_addr());
  }
}

XTensorManagerPool::~XTensorManagerPool() {
  XllmServer* collective_server =
      ServerRegistry::get_instance().register_server(collective_server_name_);
  if (collective_server != nullptr) {
    collective_server->stop();

    ServerRegistry::get_instance().unregister_server(collective_server_name_);
  }
}

void XTensorManagerPool::setup_single_node_xtensor_managers() {
  const auto& devices = options_.devices();
  const int32_t world_size = static_cast<int32_t>(devices.size());
  dp_local_tp_size_ = world_size / dp_size_;

  for (size_t i = 0; i < devices.size(); ++i) {
    const int32_t rank = static_cast<int32_t>(i);
    xtensor_managers_.emplace_back(
        std::make_unique<XTensorManager>(options_, devices[i]));
    xtensor_manager_clients_.emplace_back(
        std::make_unique<XTensorManagerClient>(xtensor_managers_.back().get()));
  }
}

void XTensorManagerPool::setup_multi_node_xtensor_managers(
    const std::string& master_node_addr) {
  const auto& devices = options_.devices();

  std::vector<std::atomic<bool>> dones(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    dones[i].store(false, std::memory_order_relaxed);
  }

  const int32_t each_node_ranks = static_cast<int32_t>(devices.size());
  const int32_t world_size =
      each_node_ranks * ::xllm::DistributedConfig::get_instance().nnodes();
  const int32_t base_rank =
      ::xllm::DistributedConfig::get_instance().node_rank() * each_node_ranks;
  dp_local_tp_size_ = world_size / dp_size_;

  for (size_t i = 0; i < devices.size(); ++i) {
    const int32_t rank = static_cast<int32_t>(i) + base_rank;

    xtensor_manager_servers_.emplace_back(
        std::make_unique<XTensorManagerServer>(
            i, master_node_addr, dones[i], devices[i], options_));

    if (::xllm::DistributedConfig::get_instance().node_rank() == 0) {
      auto dp_local_process_group_num =
          (dp_size_ > 1 && dp_size_ < world_size) ? dp_size_ : 0;

      std::shared_ptr<CollectiveService> collective_service =
          std::make_shared<CollectiveService>(
              dp_local_process_group_num, world_size, devices[0].index());
      XllmServer* collective_server =
          ServerRegistry::get_instance().register_server(
              collective_server_name_);
      if (!collective_server->start(
              collective_service, master_node_addr, collective_server_name_)) {
        LOG(ERROR) << "failed to start collective server on address: "
                   << master_node_addr;
        return;
      }

      auto xtensor_manager_addrs_map = collective_service->wait();

      for (size_t r = 0; r < world_size; ++r) {
        if (xtensor_manager_addrs_map.find(r) ==
            xtensor_manager_addrs_map.end()) {
          LOG(FATAL)
              << "Not all xtensor manager connect to master node. Miss rank is "
              << r;
          return;
        }
        xtensor_manager_clients_.emplace_back(
            std::make_unique<RemoteXTensorManager>(
                r, xtensor_manager_addrs_map[r], devices[r % each_node_ranks]));
      }
    }

    for (int idx = 0; idx < dones.size(); ++idx) {
      while (!dones[idx].load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }
}

int32_t XTensorManagerPool::get_manager_with_max_free_pages() const {
  if (xtensor_manager_clients_.empty()) {
    return 0;
  }

  size_t max_index = 0;
  size_t max_free = 0;

  for (size_t i = 0; i < xtensor_manager_clients_.size();
       i += dp_local_tp_size_) {
    const size_t current_free =
        xtensor_manager_clients_[i]->num_free_pages_per_layer();
    if (current_free > max_free) {
      max_free = current_free;
      max_index = i;
    }
  }
  return max_index;  // dp_rank
}

int32_t XTensorManagerPool::get_dp_rank(Sequence* sequence) const {
  int32_t dp_rank;
  if (sequence->dp_rank() >= 0) {
    dp_rank = sequence->dp_rank();
  } else {
    dp_rank = get_manager_with_max_free_pages();
    sequence->set_dp_rank(dp_rank);
  }
  return dp_rank;
}

bool XTensorManagerPool::allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  return allocate(sequence, sequence->num_tokens());
}

bool XTensorManagerPool::allocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    if (!allocate(sequence)) {
      return false;
    }
  }
  return true;
}

bool XTensorManagerPool::allocate(Sequence* sequence, size_t num_tokens) {
  int32_t dp_rank = get_dp_rank(sequence);
  int32_t seq_id = sequence->seq_id();
  for (int32_t i = dp_rank * dp_local_tp_size_;
       i < (dp_rank + 1) * dp_local_tp_size_;
       ++i) {
    xtensor_manager_clients_[i]->allocate_async(seq_id, num_tokens);
  }
  return true;
}

void XTensorManagerPool::deallocate(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    deallocate(sequence.get());
  }
}

void XTensorManagerPool::deallocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    deallocate(sequence);
  }
}

void XTensorManagerPool::deallocate(Sequence* sequence) {
  int32_t dp_rank = sequence->dp_rank();
  int32_t seq_id = sequence->seq_id();
  for (int32_t i = dp_rank * dp_local_tp_size_;
       i < (dp_rank + 1) * dp_local_tp_size_;
       ++i) {
    xtensor_manager_clients_[i]->deallocate_async(seq_id);
  }
}

std::vector<size_t> XTensorManagerPool::num_free_pages_per_layer() const {
  std::vector<folly::SemiFuture<size_t>> futures;
  futures.reserve(dp_size_);
  for (int32_t i = 0; i < dp_size_; ++i) {
    futures.push_back(xtensor_manager_clients_[i * dp_local_tp_size_]
                          ->num_free_pages_per_layer_async());
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  std::vector<size_t> num_free_pages_per_layer(dp_size_);
  for (int32_t i = 0; i < dp_size_; ++i) {
    num_free_pages_per_layer[i] = results[i].value();
  }
  return num_free_pages_per_layer;
}

std::vector<size_t> XTensorManagerPool::num_used_pages_per_layer() const {
  std::vector<folly::SemiFuture<size_t>> futures;
  futures.reserve(dp_size_);
  for (int32_t i = 0; i < dp_size_; ++i) {
    futures.push_back(xtensor_manager_clients_[i * dp_local_tp_size_]
                          ->num_used_pages_per_layer_async());
  }

  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  std::vector<size_t> num_used_pages_per_layer(dp_size_);
  for (int32_t i = 0; i < dp_size_; ++i) {
    num_used_pages_per_layer[i] = results[i].value();
  }
  return num_used_pages_per_layer;
}

double XTensorManagerPool::kv_cache_utilization() const {
  int32_t dp_rank = get_manager_with_max_free_pages();
  return xtensor_manager_clients_[dp_rank * dp_local_tp_size_]
      ->kv_cache_utilization();
}

}  // namespace xllm