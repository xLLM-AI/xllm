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

#include "xtensor_dist_server.h"

#include <brpc/channel.h>
#include <glog/logging.h>

#include <chrono>
#include <thread>

#include "common/global_flags.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/service_config.h"
#include "platform/device.h"
#include "server/xllm_server_registry.h"
#include "util/net.h"
#include "xtensor_dist_service.h"

namespace xllm {

void XTensorDistServer::create_server(const xtensor::Options& options,
                                      std::atomic<bool>& done,
                                      const std::string& master_node_addr,
                                      const torch::Device& d,
                                      int world_size,
                                      int global_rank,
                                      int local_rank) {
  Device device(d);
  device.set_device();

  auto service =
      std::make_shared<XTensorDistService>(global_rank, world_size, d);

  auto addr = net::get_local_ip_addr();
  auto server = ServerRegistry::get_instance().register_server(server_name_);
  if (!server->start(service, addr + ":0")) {
    LOG(ERROR) << "Failed to start XTensorDistServer on address: " << addr;
    return;
  }

  auto server_addr = addr + ":" + std::to_string(server->listen_port());
  LOG(INFO) << "XTensorDistServer " << global_rank
            << ": server address: " << server_addr;

  // Sync with master node
  proto::AddressInfo addr_info;
  addr_info.set_address(server_addr);
  addr_info.set_global_rank(global_rank);
  proto::CommUniqueIdList uids;
  sync_master_node(master_node_addr, addr_info, uids);

  // Mark service as initialized
  service->set_initialized(true);

  done.store(true);

  // Wait until server is stopped
  server->run();
}

XTensorDistServer::XTensorDistServer(int local_rank,
                                     const std::string& master_node_addr,
                                     std::atomic<bool>& done,
                                     const torch::Device& device,
                                     const xtensor::Options& options)
    : server_name_("XTensorDistServer") {
  const auto& devices = options.devices();
  int32_t each_node_ranks = static_cast<int32_t>(devices.size());
  int32_t world_size =
      each_node_ranks * ::xllm::DistributedConfig::get_instance().nnodes();
  int32_t global_rank =
      ::xllm::DistributedConfig::get_instance().node_rank() * each_node_ranks +
      local_rank;

  server_thread_ = std::make_unique<std::thread>([this,
                                                  &options,
                                                  &done,
                                                  &master_node_addr,
                                                  &device,
                                                  world_size,
                                                  global_rank,
                                                  local_rank] {
    create_server(options,
                  done,
                  master_node_addr,
                  device,
                  world_size,
                  global_rank,
                  local_rank);
  });
}

bool XTensorDistServer::sync_master_node(const std::string& master_node_addr,
                                         proto::AddressInfo& addr_info,
                                         proto::CommUniqueIdList& uids) {
  // Brpc connection resources
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.connection_type = "single";
  options.timeout_ms = 10000;
  options.max_retry = 3;
  if (channel.Init(master_node_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Failed to initialize BRPC channel to " << master_node_addr;
    return false;
  }
  proto::Collective_Stub stub(&channel);

  // Retry until master node ready
  int try_count = 0;
  brpc::Controller cntl;
  const int sleep_time_second = 3;
  while (try_count <
         ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    cntl.Reset();
    stub.Sync(&cntl, &addr_info, &uids, nullptr);
    if (cntl.Failed()) {
      LOG(WARNING) << "XTensorDistServer#" << addr_info.global_rank()
                   << " try connect to engine server error, try again."
                   << " Error message: " << cntl.ErrorText();
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time_second));
    } else {
      LOG(INFO) << "XTensorDistServer#" << addr_info.global_rank()
                << " connect to " << master_node_addr << " success.";
      break;
    }
    try_count++;
  }

  if (try_count >=
      ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    LOG(ERROR) << "XTensorDistServer#" << addr_info.global_rank()
               << " connect to " << master_node_addr << " failed."
               << " Error message: " << cntl.ErrorText();
    return false;
  }

  return true;
}

XTensorDistServer::~XTensorDistServer() {
  if (server_thread_ && server_thread_->joinable()) {
    server_thread_->join();
  }
}

}  // namespace xllm
