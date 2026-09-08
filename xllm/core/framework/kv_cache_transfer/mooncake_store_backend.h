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

#pragma once

#pragma push_macro("BLOCK_SIZE")
#include <Mooncake/mooncake-store/include/real_client.h>
#pragma pop_macro("BLOCK_SIZE")

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace xllm {

struct MooncakeStoreBackendConfig {
  std::string localhost_name;
  std::string protocol;
  std::string rdma_devices;
  std::string metadata_server;
  std::string master_server_address;
  int32_t replica_num = 1;
};

struct MooncakeRegisteredRange {
  void* address = nullptr;
  size_t bytes = 0;
};

struct MooncakeMultiBuffer {
  std::vector<void*> addresses;
  std::vector<size_t> sizes;
};

class MooncakeStoreBackend final {
 public:
  MooncakeStoreBackend() = default;
  ~MooncakeStoreBackend();

  bool init(const MooncakeStoreBackendConfig& config,
            const std::vector<MooncakeRegisteredRange>& ranges);
  std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<MooncakeMultiBuffer>& buffers);
  bool get(const std::string& key, const MooncakeMultiBuffer& buffer);

 private:
  friend class MooncakeStoreBackendTestPeer;

  static std::optional<std::vector<MooncakeRegisteredRange>> unique_ranges(
      const std::vector<MooncakeRegisteredRange>& ranges);
  static bool get_succeeded(int64_t expected_bytes,
                            const std::vector<int>& results);
  static bool put_succeeded(int result);
  void unregister_ranges();

 private:
  mooncake::ReplicateConfig rep_config_;
  std::shared_ptr<mooncake::RealClient> client_ptr_;
  std::vector<MooncakeRegisteredRange> registered_ranges_;
};

}  // namespace xllm
