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

#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

namespace xllm {

// Weight transfer using Mooncake transfer engine for remote weight loading
class MooncakeWeightTransfer {
 public:
  MooncakeWeightTransfer(int16_t listen_port, const torch::Device& device);
  ~MooncakeWeightTransfer() = default;

  bool initialize();

  bool register_global_xtensor();

  bool link_p2p(const std::string& remote_addr);
  bool link_p2p(const std::vector<std::string>& remote_addrs);
  bool unlink_p2p(const std::string& remote_addr);
  bool unlink_p2p(const std::vector<std::string>& remote_addrs);

  bool pull_weights(const std::string& remote_addr,
                    uint64_t src_offset,
                    uint64_t dst_offset,
                    size_t size);

  bool push_weights(const std::string& remote_addr,
                    uint64_t src_offset,
                    uint64_t dst_offset,
                    size_t size);

 private:
  int16_t listen_port_;
  int32_t device_id_;
  std::string addr_;
  uint64_t cluster_id_ = 0;
  std::unique_ptr<MooncakeTransferEngine> mooncake_te_;
  bool initialized_ = false;
};

}  // namespace xllm
