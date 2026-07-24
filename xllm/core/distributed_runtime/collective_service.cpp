/* Copyright 2025-2026 The xLLM Authors.

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

#include "collective_service.h"

#include <brpc/closure_guard.h>
#include <glog/logging.h>

#include <vector>

#include "util/net.h"

namespace xllm {

CollectiveService::CollectiveService(int dp_group_num,
                                     int total_num,
                                     int device_idx)
    : total_num_(total_num) {
#if defined(USE_NPU)
  root_infos_.reserve(dp_group_num + 1);
  for (size_t i = 0; i < (dp_group_num + 1); ++i) {
    HcclRootInfo root_info;
    auto error = aclrtSetDevice(device_idx);
    CHECK_EQ(error, ACL_SUCCESS)
        << "ACL set device id " << device_idx << " failed. Error : " << error;
    auto status = HcclGetRootInfo(&root_info);
    CHECK_EQ(status, HCCL_SUCCESS) << "HCCL get root info failed.";
    root_infos_.push_back(root_info);
  }
#endif

  // Pre-allocate free TCPStore ports for the worst-case communication group
  // count. The communicator computes a slot index per group from
  // (group_kind, group_index) — see ProcessGroupSlots in
  // collective_communicator.cpp — which is bounded by total_num_ slots
  // across the world/tp/single_rank/dp_local/moe_tp/moe_ep/encoder_dp
  // sections. We over-allocate by a small constant to absorb future group
  // additions; the unused ports are released right after binding.
  //
  // OS-allocated free ports avoid collisions with random outgoing TCP source
  // ports inside the local ephemeral range (1024-65535 on common Linux
  // hosts), which the previous "master_port + offset" scheme could not.
  const int port_slot_count = std::max(1, 4 * total_num_ + 8);
  group_ports_.reserve(port_slot_count);
  for (int i = 0; i < port_slot_count; ++i) {
    const int port = net::get_local_free_port();
    CHECK_GT(port, 0) << "Failed to allocate free TCP port for process-group "
                         "rendezvous; slot="
                      << i;
    group_ports_.push_back(port);
  }
}

void CollectiveService::Sync(::google::protobuf::RpcController* controller,
                             const proto::AddressInfo* request,
                             proto::CommUniqueIdList* response,
                             ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  std::string address = request->address();
  int32_t global_rank = request->global_rank();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    addrs_map_[global_rank] = address;
  }
#if defined(USE_NPU)
  to_proto_list(root_infos_, response);
#endif
  // Send the master-allocated ports to every worker. All ranks see the same
  // ordered list, so each worker resolves its process-group slots to the
  // same TCPStore endpoints the master will bind on.
  for (const int port : group_ports_) {
    response->add_group_ports(port);
  }
}

std::unordered_map<int32_t, std::string> CollectiveService::wait() {
  int connected = 0;
  while (connected < total_num_) {
    absl::SleepFor(absl::Milliseconds(1000));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      connected = addrs_map_.size();
    }
  }

  return addrs_map_;
}

#if defined(USE_NPU)
void CollectiveService::to_proto_list(const std::vector<HcclRootInfo>& src,
                                      proto::CommUniqueIdList* dst) {
  for (const auto& id : src) {
    proto::CommUniqueId* protoId = dst->add_comm_unique_ids();
    protoId->set_comm_unique_id(id.internal, sizeof(id.internal));
  }
}

void CollectiveService::from_proto_list(const proto::CommUniqueIdList& src,
                                        std::vector<HcclRootInfo>* dst) {
  dst->clear();
  for (const auto& protoId : src.comm_unique_ids()) {
    HcclRootInfo id;
    std::memcpy(
        id.internal, protoId.comm_unique_id().data(), sizeof(id.internal));
    dst->push_back(id);
  }
}
#endif

}  // namespace xllm
