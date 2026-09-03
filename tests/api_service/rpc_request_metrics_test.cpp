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

#include "api_service/rpc_request_metrics.h"

#include <brpc/controller.h>
#include <gtest/gtest.h>

#include <utility>

#include "api_service/non_stream_call.h"
#include "completion.pb.h"
#include "core/common/metrics.h"

namespace xllm {
namespace {

class NoopClosure final : public google::protobuf::Closure {
 public:
  void Run() override {}
};

using CompletionCallForTest =
    NonStreamCall<proto::CompletionRequest, proto::CompletionResponse>;

TEST(RpcRequestMetricsTest, RecordsOkOnDestroyWhenControllerSucceeded) {
  brpc::Controller controller;
  const double in_before = COUNTER_VALUE(server_request_in_total);
  const double ok_before = COUNTER_VALUE(server_request_total_ok);
  const double fail_before = COUNTER_VALUE(server_request_total_fail);

  {
    RpcRequestMetrics metrics(&controller);
    EXPECT_EQ(COUNTER_VALUE(server_request_in_total), in_before + 1.0);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before);
  }

  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before + 1.0);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), fail_before);
}

TEST(RpcRequestMetricsTest, RecordsFailWhenControllerFailedAfterDispatch) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;
  const double in_before = COUNTER_VALUE(server_request_in_total);
  const double ok_before = COUNTER_VALUE(server_request_total_ok);
  const double fail_before = COUNTER_VALUE(server_request_total_fail);

  {
    RpcRequestMetrics metrics(&controller);
    CompletionCallForTest call(&controller,
                               &done,
                               &request,
                               &response,
                               /*use_arena=*/true,
                               /*is_http_request=*/false,
                               std::move(metrics));
    EXPECT_EQ(COUNTER_VALUE(server_request_in_total), in_before + 1.0);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before);
    EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), fail_before);
    controller.SetFailed("async handler failed");
  }

  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), fail_before + 1.0);
}

TEST(RpcRequestMetricsTest, RecordsLimitOnBrpcElimit) {
  brpc::Controller controller;
  const double fail_before = COUNTER_VALUE(server_request_total_fail);
  const double limit_before = COUNTER_VALUE(server_request_total_limit);

  {
    RpcRequestMetrics metrics(&controller);
    controller.SetFailed(brpc::ELIMIT, "busy");
  }

  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), fail_before + 1.0);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_limit), limit_before + 1.0);
}

TEST(RpcRequestMetricsTest, MoveTransfersOwnershipWithoutDoubleCount) {
  brpc::Controller controller;
  const double in_before = COUNTER_VALUE(server_request_in_total);
  const double ok_before = COUNTER_VALUE(server_request_total_ok);

  {
    RpcRequestMetrics first(&controller);
    RpcRequestMetrics second(std::move(first));
  }

  EXPECT_EQ(COUNTER_VALUE(server_request_in_total), in_before + 1.0);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before + 1.0);
}

TEST(RpcRequestMetricsTest, FinishThenDestroyControllerDoesNotUAF) {
  brpc::Controller* controller = new brpc::Controller();
  const double fail_before = COUNTER_VALUE(server_request_total_fail);

  RpcRequestMetrics metrics(controller);
  controller->SetFailed("done already ran");
  metrics.finish(controller);
  delete controller;

  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), fail_before + 1.0);
}

TEST(RpcRequestMetricsTest, DetachBeforeControllerDeleteDoesNotUAF) {
  brpc::Controller* controller = new brpc::Controller();
  const double fail_before = COUNTER_VALUE(server_request_total_fail);
  const double ok_before = COUNTER_VALUE(server_request_total_ok);

  {
    RpcRequestMetrics metrics(controller);
    metrics.detach();
    delete controller;
    metrics.mark_failed();
  }

  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), fail_before + 1.0);
}

TEST(RpcRequestMetricsTest, DetachThenMarkFailedRecordsFail) {
  brpc::Controller controller;
  const double fail_before = COUNTER_VALUE(server_request_total_fail);
  const double ok_before = COUNTER_VALUE(server_request_total_ok);

  {
    RpcRequestMetrics metrics(&controller);
    metrics.detach();
    metrics.mark_failed();
  }

  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), fail_before + 1.0);
}

TEST(RpcRequestMetricsTest, CallMarkRpcFailedRecordsFail) {
  brpc::Controller controller;
  proto::CompletionRequest request;
  proto::CompletionResponse response;
  NoopClosure done;
  const double fail_before = COUNTER_VALUE(server_request_total_fail);
  const double ok_before = COUNTER_VALUE(server_request_total_ok);

  {
    RpcRequestMetrics metrics(&controller);
    CompletionCallForTest call(&controller,
                               &done,
                               &request,
                               &response,
                               /*use_arena=*/true,
                               /*is_http_request=*/false,
                               std::move(metrics));
    call.mark_rpc_failed();
  }

  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_fail), fail_before + 1.0);
}

TEST(RpcRequestMetricsTest, InactiveDefaultDoesNotRecord) {
  const double in_before = COUNTER_VALUE(server_request_in_total);
  const double ok_before = COUNTER_VALUE(server_request_total_ok);

  {
    RpcRequestMetrics inactive;
  }

  EXPECT_EQ(COUNTER_VALUE(server_request_in_total), in_before);
  EXPECT_EQ(COUNTER_VALUE(server_request_total_ok), ok_before);
}

}  // namespace
}  // namespace xllm
