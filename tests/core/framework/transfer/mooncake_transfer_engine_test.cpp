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

#include "core/framework/transfer/mooncake_transfer_engine.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <google/protobuf/descriptor.pb.h>
#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "core/framework/transfer/mooncake_session_service.h"

namespace xllm {
namespace {

class RecordingClosure final : public google::protobuf::Closure {
 public:
  void Run() override { ran = true; }
  bool ran = false;
};

class RecordingExtension final : public proto::MooncakeTransferEngineService {
 public:
  void SetCachePeer(google::protobuf::RpcController* /*controller*/,
                    const proto::CachePeerRequest* /*request*/,
                    proto::Status* response,
                    google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(true);
    ++calls;
  }
  int32_t calls = 0;
};

}  // namespace

TEST(MooncakeTransferEngineServiceTest, OpenSessionRejectsMissingAddr) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller cntl;

  service.OpenSession(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest, CloseSessionRejectsMissingAddr) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller cntl;

  service.CloseSession(&cntl, &request, &response, nullptr);

  EXPECT_FALSE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest, CloseSessionWithoutHandleReturnsTrue) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  request.set_addr("127.0.0.1:5001");
  proto::Status response;
  brpc::Controller cntl;

  service.CloseSession(&cntl, &request, &response, nullptr);

  EXPECT_TRUE(response.ok());
}

TEST(MooncakeTransferEngineServiceTest,
     ExtensionUsesExistingServiceDescriptor) {
  MooncakeTransferEngineService service;
  auto extension = std::make_shared<RecordingExtension>();
  service.register_rpc_service(extension);
  service.register_rpc_service(extension);
  proto::CachePeerRequest request;
  proto::Status response;
  brpc::Controller controller;
  RecordingClosure done;

  service.CallMethod(service.GetDescriptor()->FindMethodByName("SetCachePeer"),
                     &controller,
                     &request,
                     &response,
                     &done);

  EXPECT_FALSE(controller.Failed());
  EXPECT_TRUE(response.ok());
  EXPECT_EQ(extension->calls, 1);
  EXPECT_TRUE(done.ran);
}

TEST(MooncakeTransferEngineServiceTest,
     RetainsExtensionUntilServiceIsDestroyed) {
  std::weak_ptr<RecordingExtension> weak_extension;
  {
    MooncakeTransferEngineService service;
    auto extension = std::make_shared<RecordingExtension>();
    weak_extension = extension;
    service.register_rpc_service(std::move(extension));
    EXPECT_FALSE(weak_extension.expired());
    proto::CachePeerRequest request;
    proto::Status response;
    brpc::Controller controller;
    service.CallMethod(
        service.GetDescriptor()->FindMethodByName("SetCachePeer"),
        &controller,
        &request,
        &response,
        nullptr);
    EXPECT_TRUE(response.ok());
  }
  EXPECT_TRUE(weak_extension.expired());
}

TEST(MooncakeTransferEngineServiceTest, MissingExtensionFailsAndCompletesRpc) {
  MooncakeTransferEngineService service;
  proto::CachePeerRequest request;
  proto::Status response;
  brpc::Controller controller;
  RecordingClosure done;

  service.CallMethod(service.GetDescriptor()->FindMethodByName("SetCachePeer"),
                     &controller,
                     &request,
                     &response,
                     &done);

  EXPECT_TRUE(controller.Failed());
  EXPECT_TRUE(done.ran);
}

TEST(MooncakeTransferEngineServiceTest, RejectsRequestAndResponseTypeMismatch) {
  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  proto::Empty wrong_message;
  const auto* method = service.GetDescriptor()->FindMethodByName("OpenSession");
  brpc::Controller request_controller;
  RecordingClosure request_done;
  service.CallMethod(
      method, &request_controller, &wrong_message, &response, &request_done);
  EXPECT_TRUE(request_controller.Failed());
  EXPECT_TRUE(request_done.ran);

  brpc::Controller response_controller;
  RecordingClosure response_done;
  service.CallMethod(
      method, &response_controller, &request, &wrong_message, &response_done);
  EXPECT_TRUE(response_controller.Failed());
  EXPECT_TRUE(response_done.ran);
}

TEST(MooncakeTransferEngineServiceTest, RejectsMethodFromAnotherService) {
  google::protobuf::FileDescriptorProto file;
  file.set_name("other_transfer_service.proto");
  file.add_message_type()->set_name("Message");
  auto* service_descriptor = file.add_service();
  service_descriptor->set_name("OtherService");
  auto* method_descriptor = service_descriptor->add_method();
  method_descriptor->set_name("OpenSession");
  method_descriptor->set_input_type(".Message");
  method_descriptor->set_output_type(".Message");
  google::protobuf::DescriptorPool pool;
  const auto* descriptor = pool.BuildFile(file);
  ASSERT_NE(descriptor, nullptr);

  MooncakeTransferEngineService service;
  proto::SessionInfo request;
  proto::Status response;
  brpc::Controller controller;
  RecordingClosure done;
  service.CallMethod(descriptor->service(0)->method(0),
                     &controller,
                     &request,
                     &response,
                     &done);

  EXPECT_TRUE(controller.Failed());
  EXPECT_TRUE(done.ran);
}

TEST(MooncakeTransferEngineServiceTest, MissingArgumentsCompleteRpc) {
  MooncakeTransferEngineService service;
  const auto* method = service.GetDescriptor()->FindMethodByName("OpenSession");
  proto::SessionInfo request;
  proto::Status response;
  for (int32_t missing = 0; missing < 4; ++missing) {
    brpc::Controller controller;
    RecordingClosure done;
    service.CallMethod(missing == 0 ? nullptr : method,
                       missing == 1 ? nullptr : &controller,
                       missing == 2 ? nullptr : &request,
                       missing == 3 ? nullptr : &response,
                       &done);
    EXPECT_TRUE(done.ran);
    if (missing != 1) {
      EXPECT_TRUE(controller.Failed());
    }
  }
}

TEST(MooncakeTransferEngineTest, FailedRegistrationDoesNotRetainMappingLease) {
  auto lifetime = std::make_shared<uint8_t>(0);
  std::weak_ptr<uint8_t> weak_lifetime = lifetime;
  MooncakeTransferEngineCore& core = MooncakeTransferEngineCore::get_instance();

  EXPECT_FALSE(core.register_memory(nullptr, /*bytes=*/1, std::move(lifetime)));

  EXPECT_TRUE(weak_lifetime.expired());
  EXPECT_FALSE(core.is_memory_registered(nullptr, /*bytes=*/1));
  EXPECT_TRUE(core.unregister_memory(nullptr));
}

}  // namespace xllm
