/* Copyright 2026 The xLLM Authors.

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

#include <Mooncake/mooncake-transfer-engine/include/transfer_engine.h>
#include <brpc/channel.h>
#include <brpc/server.h>
#include <google/protobuf/service.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/transfer/byte_region.h"
#include "core/platform/device.h"

namespace xllm {

using mooncake::SegmentHandle;
using mooncake::TransferEngine;
class MooncakeTransferEngineService;

class MooncakeTransferEngineCore final {
 public:
  static MooncakeTransferEngineCore& get_instance() {
    static MooncakeTransferEngineCore instance;
    return instance;
  }

  // Initialize the shared core. Only the first call takes effect.
  bool initialize(uint16_t listen_port, const torch::Device& device);

  TransferEngine* engine() { return engine_.get(); }

  const std::string& addr() const { return addr_; }
  const std::string& host_ip() const { return host_ip_; }

  // Session state is shared across all MooncakeTransferEngine instances.
  bool open_session(const std::string& control_endpoint,
                    const std::string& remote_addr,
                    bool increment_existing = true);
  bool close_session(const std::string& control_endpoint,
                     const std::string& remote_addr);
  SegmentHandle get_handle(const std::string& remote_addr);

  // Retain a mapping lease for as long as the transport can access the region.
  bool register_memory(void* base,
                       size_t bytes,
                       std::shared_ptr<void> lifetime = {});
  bool is_memory_registered(void* base, size_t bytes) const;
  // Call only after every user has stopped transfers through this region.
  bool unregister_memory(void* base);

  // Extensions share the legacy RPC listener without adding domain
  // dependencies.
  void register_rpc_service(std::shared_ptr<google::protobuf::Service> service);
  std::shared_ptr<brpc::Channel> get_or_create_channel(
      const std::string& control_endpoint);

  bool is_initialized() const { return initialized_; }

 private:
  std::shared_ptr<brpc::Channel> get_or_create_channel_locked(
      const std::string& control_endpoint);
  bool acquire_session_locked(const std::string& remote_addr);

  MooncakeTransferEngineCore() = default;
  ~MooncakeTransferEngineCore();
  MooncakeTransferEngineCore(const MooncakeTransferEngineCore&) = delete;
  MooncakeTransferEngineCore& operator=(const MooncakeTransferEngineCore&) =
      delete;

  mutable std::mutex mutex_;
  bool initialized_ = false;
  std::string addr_;
  std::string host_ip_;
  int32_t rpc_port_ = 0;
  uint16_t listen_port_ = 0;
  std::unique_ptr<TransferEngine> engine_;
  brpc::Server server_;
  std::shared_ptr<MooncakeTransferEngineService> service_;

  struct SessionInfo {
    SegmentHandle handle = static_cast<SegmentHandle>(-1);
    int32_t ref_count = 0;
  };
  std::unordered_map<std::string, SessionInfo> handles_;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>> channels_;
  struct MemoryRegistration {
    size_t bytes = 0;
    std::shared_ptr<void> lifetime;
  };
  std::unordered_map<void*, MemoryRegistration> memory_registrations_;
};

class MooncakeTransferEngine {
 public:
  enum class MoveOpcode { READ = 0, WRITE = 1 };

  struct BufferTransferMapping {
    int64_t buf_id = 0;
    std::vector<uint64_t> local_ids;
    std::vector<uint64_t> remote_ids;
  };

  MooncakeTransferEngine(const uint16_t listen_port,
                         const torch::Device& device);
  virtual ~MooncakeTransferEngine() = default;

  virtual std::string initialize();

  virtual bool register_memory(std::vector<void*> addrs,
                               std::vector<size_t> lens,
                               std::vector<uint64_t> buf_bytes,
                               std::shared_ptr<void> lifetime = {});
  bool unregister_memory(void* base);

  bool move_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& buf_ids,
                          MoveOpcode move_opcode);

  virtual bool move_memory_groups(
      const std::string& remote_addr,
      const std::vector<BufferTransferMapping>& mappings,
      MoveOpcode move_opcode);

  virtual bool move_memory_regions(const std::string& remote_addr,
                                   const std::vector<ByteRegion>& regions,
                                   MoveOpcode move_opcode);

  virtual bool pull_memory_blocks(const std::string& remote_addr,
                                  const std::vector<uint64_t>& src_blocks,
                                  const std::vector<uint64_t>& dst_blocks,
                                  const std::vector<int64_t>& buf_ids);

  virtual bool push_memory_blocks(const std::string& remote_addr,
                                  const std::vector<uint64_t>& src_blocks,
                                  const std::vector<uint64_t>& dst_blocks,
                                  const std::vector<int64_t>& buf_ids);

  // Transfer raw byte offsets within registered buffer[0].
  bool move_memory_by_global_offsets(const std::string& remote_addr,
                                     const std::vector<uint64_t>& src_offsets,
                                     const std::vector<uint64_t>& dst_offsets,
                                     size_t transfer_size,
                                     MoveOpcode move_opcode);

  bool open_session(const std::string& control_endpoint,
                    const std::string& remote_addr);

  bool close_session(const std::string& control_endpoint,
                     const std::string& remote_addr);

 private:
  uint16_t listen_port_;
  std::vector<uint64_t> buf_bytes_;
  Device device_;
  MooncakeTransferEngineCore& core_;
  std::mutex session_mutex_;
  std::unordered_map<std::string, int32_t> session_ref_counts_;
};

}  // namespace xllm
