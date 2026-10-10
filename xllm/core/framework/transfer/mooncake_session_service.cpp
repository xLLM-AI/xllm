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

#include "core/framework/transfer/mooncake_session_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <utility>

#include "core/framework/transfer/mooncake_transfer_engine.h"

namespace xllm {

void MooncakeTransferEngineService::register_rpc_service(
    std::shared_ptr<google::protobuf::Service> service) {
  CHECK(service != nullptr);
  CHECK_EQ(service->GetDescriptor(), GetDescriptor());
  std::lock_guard<std::mutex> lock(mutex_);
  // Registration is idempotent for the process-wide extension instance.
  CHECK(extension_ == nullptr || extension_ == service);
  extension_ = std::move(service);
}

void MooncakeTransferEngineService::CallMethod(
    const google::protobuf::MethodDescriptor* method,
    google::protobuf::RpcController* controller,
    const google::protobuf::Message* request,
    google::protobuf::Message* response,
    google::protobuf::Closure* done) {
  if (method == nullptr || controller == nullptr || request == nullptr ||
      response == nullptr) {
    brpc::ClosureGuard done_guard(done);
    if (controller != nullptr) {
      controller->SetFailed("Transfer RPC arguments are missing");
    }
    return;
  }
  // The generated dispatcher casts messages without validating their types.
  if (method->service() != GetDescriptor() ||
      request->GetDescriptor() != method->input_type() ||
      response->GetDescriptor() != method->output_type()) {
    brpc::ClosureGuard done_guard(done);
    controller->SetFailed("Transfer RPC descriptor or message type mismatch");
    return;
  }
  if (method->name() == "OpenSession" || method->name() == "CloseSession") {
    proto::MooncakeTransferEngineService::CallMethod(
        method, controller, request, response, done);
    return;
  }
  std::shared_ptr<google::protobuf::Service> extension;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    extension = extension_;
  }
  if (extension == nullptr) {
    brpc::ClosureGuard done_guard(done);
    controller->SetFailed("Transfer RPC extension is not registered");
    return;
  }
  extension->CallMethod(method, controller, request, response, done);
}

void MooncakeTransferEngineService::OpenSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (request->addr().empty()) {
    LOG(ERROR) << "OpenSession request missing addr";
    response->set_ok(false);
    return;
  }

  std::string remote_addr(request->addr());
  MooncakeTransferEngineCore& core = MooncakeTransferEngineCore::get_instance();
  const bool result = core.open_session(
      /*control_endpoint=*/"", remote_addr, /*increment_existing=*/true);

  response->set_ok(result);
}

void MooncakeTransferEngineService::CloseSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (request->addr().empty()) {
    LOG(ERROR) << "CloseSession request missing addr";
    response->set_ok(false);
    return;
  }

  std::string remote_addr(request->addr());
  bool result =
      MooncakeTransferEngineCore::get_instance().close_session("", remote_addr);

  response->set_ok(result);
}

}  // namespace xllm
