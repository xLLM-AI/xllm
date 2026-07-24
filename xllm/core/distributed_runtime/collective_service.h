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

#pragma once

#include <absl/container/flat_hash_set.h>
#if defined(USE_NPU)
#include <hccl/hccl.h>
#endif

#include <unordered_map>

#include "common/macros.h"
#include "worker.pb.h"

namespace xllm {

class CollectiveService : public proto::Collective {
 public:
  CollectiveService(int dp_group_num, int total_num, int device_idx);
  virtual ~CollectiveService() = default;

  void Sync(::google::protobuf::RpcController* controller,
            const proto::AddressInfo* request,
            proto::CommUniqueIdList* response,
            ::google::protobuf::Closure* done) override;

  // wait all worker connected
  std::unordered_map<int32_t, std::string> wait();

  // Master-allocated free TCPStore ports, in the canonical group order shared
  // with workers (see CollectiveCommunicator::create_process_groups). The
  // master node consumes this list locally; workers receive the same list
  // through CommUniqueIdList.group_ports during Sync.
  const std::vector<int32_t>& group_ports() const { return group_ports_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(CollectiveService);

#if defined(USE_NPU)
  void to_proto_list(const std::vector<HcclRootInfo>& src,
                     proto::CommUniqueIdList* dst);
  void from_proto_list(const proto::CommUniqueIdList& src,
                       std::vector<HcclRootInfo>* dst);
#endif

 private:
  int total_num_ = 0;
#if defined(USE_NPU)
  std::vector<HcclRootInfo> root_infos_;
#endif
  std::mutex mutex_;
  std::unordered_map<int32_t, std::string> addrs_map_;
  // Free TCPStore ports the master allocated up-front. Returned verbatim to
  // every worker through Sync so each rank uses the same set of ports for
  // process-group bootstrap.
  std::vector<int32_t> group_ports_;
};

}  // namespace xllm
