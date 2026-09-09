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

#pragma once

#include <brpc/server.h>
#include <torch/torch.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "common/macros.h"
#include "options.h"
#include "xtensor_dist.pb.h"

namespace xllm {

class XTensorDistServer {
 public:
  XTensorDistServer(int local_rank,
                    const std::string& master_node_addr,
                    std::atomic<bool>& done,
                    const torch::Device& device,
                    const xtensor::Options& options);
  ~XTensorDistServer();

 private:
  DISALLOW_COPY_AND_ASSIGN(XTensorDistServer);

  void create_server(const xtensor::Options& options,
                     std::atomic<bool>& done,
                     const std::string& master_node_addr,
                     const torch::Device& device,
                     int world_size,
                     int global_rank,
                     int local_rank);

  bool sync_master_node(const std::string& master_node_addr,
                        proto::AddressInfo& addr_info,
                        proto::CommUniqueIdList& uids);

 private:
  std::unique_ptr<std::thread> server_thread_;
  std::string server_name_;
};

}  // namespace xllm
