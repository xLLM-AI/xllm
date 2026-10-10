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

#include "framework/virtual_memory/mooncake_weight_transfer.h"

#include <glog/logging.h>

#include "framework/virtual_memory/global_memory_region.h"
#include "util/net.h"

namespace xllm {

MooncakeWeightTransfer::MooncakeWeightTransfer(uint16_t listen_port,
                                               const torch::Device& device)
    : listen_port_(listen_port), device_id_(device.index()) {
  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port_);
  mooncake_te_ = std::make_unique<MooncakeTransferEngine>(listen_port_, device);
}

bool MooncakeWeightTransfer::initialize() {
  if (initialized_) {
    return true;
  }
  addr_ = mooncake_te_->initialize();
  initialized_ = !addr_.empty();
  return initialized_;
}

bool MooncakeWeightTransfer::register_global_memory_region() {
  auto& global_memory_region = GlobalMemoryRegion::get_instance();
  if (!global_memory_region.is_initialized()) {
    LOG(ERROR) << "GlobalMemoryRegion not initialized";
    return false;
  }

  if (global_memory_region.is_mooncake_registered()) {
    LOG(INFO) << "GlobalMemoryRegion already registered to mooncake, skip";
    return true;
  }

  std::vector<void*> addrs = {global_memory_region.base_vaddr()};
  std::vector<size_t> lens = {global_memory_region.total_size()};
  std::vector<uint64_t> buf_bytes = {
      static_cast<uint64_t>(global_memory_region.page_size())};
  if (!mooncake_te_->register_memory(addrs, lens, buf_bytes)) {
    LOG(ERROR) << "register GlobalMemoryRegion failed";
    return false;
  }

  global_memory_region.set_mooncake_registered(true);
  LOG(INFO) << "MooncakeWeightTransfer: register GlobalMemoryRegion success, "
            << "total_size=" << global_memory_region.total_size()
            << ", num_pages=" << global_memory_region.num_total_pages();
  return true;
}

bool MooncakeWeightTransfer::link_p2p(const std::string& remote_addr) {
  std::string host;
  int port = 0;
  net::parse_host_port_from_addr(remote_addr, host, port);
  auto remote_cluster_id =
      net::convert_ip_port_to_uint64(host, static_cast<uint16_t>(port));

  LOG(INFO) << "MooncakeWeightTransfer::link_p2p, remote_addr=" << remote_addr
            << ", remote_cluster_id=" << remote_cluster_id;

  return mooncake_te_->open_session(remote_cluster_id, remote_addr);
}

bool MooncakeWeightTransfer::link_p2p(
    const std::vector<std::string>& remote_addrs) {
  for (const auto& remote_addr : remote_addrs) {
    if (!link_p2p(remote_addr)) {
      return false;
    }
  }
  return true;
}

bool MooncakeWeightTransfer::unlink_p2p(const std::string& remote_addr) {
  std::string host;
  int port = 0;
  net::parse_host_port_from_addr(remote_addr, host, port);
  auto remote_cluster_id =
      net::convert_ip_port_to_uint64(host, static_cast<uint16_t>(port));

  LOG(INFO) << "MooncakeWeightTransfer::unlink_p2p, remote_addr=" << remote_addr
            << ", remote_cluster_id=" << remote_cluster_id;

  return mooncake_te_->close_session(remote_cluster_id, remote_addr);
}

bool MooncakeWeightTransfer::unlink_p2p(
    const std::vector<std::string>& remote_addrs) {
  for (const auto& remote_addr : remote_addrs) {
    if (!unlink_p2p(remote_addr)) {
      return false;
    }
  }
  return true;
}

bool MooncakeWeightTransfer::pull_weights(const std::string& remote_addr,
                                          uint64_t src_offset,
                                          uint64_t dst_offset,
                                          size_t size) {
  // Note: src_offsets/dst_offsets are swapped because we're reading from remote
  std::vector<uint64_t> src_offsets = {dst_offset};
  std::vector<uint64_t> dst_offsets = {src_offset};
  return mooncake_te_->move_memory_by_global_offsets(
      remote_addr,
      src_offsets,
      dst_offsets,
      size,
      MooncakeTransferEngine::MoveOpcode::READ);
}

bool MooncakeWeightTransfer::push_weights(const std::string& remote_addr,
                                          uint64_t src_offset,
                                          uint64_t dst_offset,
                                          size_t size) {
  std::vector<uint64_t> src_offsets = {src_offset};
  std::vector<uint64_t> dst_offsets = {dst_offset};
  return mooncake_te_->move_memory_by_global_offsets(
      remote_addr,
      src_offsets,
      dst_offsets,
      size,
      MooncakeTransferEngine::MoveOpcode::WRITE);
}

}  // namespace xllm
