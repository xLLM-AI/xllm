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

#include <memory>
#include <mutex>

#include "mooncake_transfer_engine.pb.h"

namespace xllm {

// Preserve the existing service name and forward extension RPCs without
// interpreting their domain-specific messages.
class MooncakeTransferEngineService final
    : public proto::MooncakeTransferEngineService {
 public:
  void register_rpc_service(std::shared_ptr<google::protobuf::Service> service);
  void CallMethod(const google::protobuf::MethodDescriptor* method,
                  google::protobuf::RpcController* controller,
                  const google::protobuf::Message* request,
                  google::protobuf::Message* response,
                  google::protobuf::Closure* done) override;
  void OpenSession(google::protobuf::RpcController* controller,
                   const proto::SessionInfo* request,
                   proto::Status* response,
                   google::protobuf::Closure* done) override;
  void CloseSession(google::protobuf::RpcController* controller,
                    const proto::SessionInfo* request,
                    proto::Status* response,
                    google::protobuf::Closure* done) override;

 private:
  std::mutex mutex_;
  std::shared_ptr<google::protobuf::Service> extension_;
};

}  // namespace xllm
