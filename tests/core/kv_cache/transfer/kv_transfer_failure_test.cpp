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

#include <gtest/gtest.h>
#include <torch/types.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "core/common/types.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/kv_cache/transfer/kv_transfer_completion.h"
#include "core/platform/device.h"

namespace xllm {
namespace {

class FailureProcessGroup final : public ProcessGroup {
 public:
  explicit FailureProcessGroup(bool remote_failure = false)
      : ProcessGroup(/*rank=*/0, /*world_size=*/2, torch::Device(torch::kCPU)),
        remote_failure_(remote_failure) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    ++gathers_;
    return torch::stack({input, input});
  }

  void allreduce(torch::Tensor& input) override {
    ++reductions_;
    if (remote_failure_) {
      input.index_put_({1}, 1);
    }
  }

  int32_t gathers_ = 0;
  int32_t reductions_ = 0;

 private:
  bool remote_failure_;
};

TEST(KVTransferFailureTest, DisabledPushStillWaitsWithoutReplicaCollectives) {
  KVTransferCompletion completion;
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  completion.add(promise.getSemiFuture(), {"request"});
  FailureProcessGroup group;
  std::promise<void> started;
  std::future<std::vector<std::string>> result =
      std::async(std::launch::async, [&]() {
        started.set_value();
        return finalize_kv_push_failures(completion,
                                         {},
                                         /*enable_kv_push=*/false,
                                         {&group, &group},
                                         Device(torch::kCPU));
      });
  started.get_future().wait();
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(50)),
            std::future_status::timeout);
  promise.setValue(std::vector<KVTransferTaskResult>{});
  EXPECT_TRUE(result.get().empty());
  EXPECT_EQ(group.gathers_, 0);
  EXPECT_EQ(group.reductions_, 0);
}

TEST(KVTransferFailureTest, EnabledEmptyPushStillChecksReplicas) {
  KVTransferCompletion completion;
  FailureProcessGroup group;
  EXPECT_TRUE(finalize_kv_push_failures(completion,
                                        {},
                                        /*enable_kv_push=*/true,
                                        {&group, &group},
                                        Device(torch::kCPU))
                  .empty());
  EXPECT_EQ(group.gathers_, 1);
  EXPECT_EQ(group.reductions_, 0);
}

TEST(KVTransferFailureTest, MergesFailuresAcrossDistinctReplicaGroups) {
  KVTransferCompletion completion;
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  completion.add(promise.getSemiFuture(), {"first"});
  promise.setValue(std::vector<KVTransferTaskResult>{
      KVTransferTaskResult{{"first"}, KVTransferErrorCode::FAILED}});
  std::vector<TransferKVInfo> transfers(3);
  transfers[0].request_id = "second";
  transfers[1].request_id = "first";
  transfers[2].request_id = "first";
  FailureProcessGroup tp(/*remote_failure=*/true);
  FailureProcessGroup cp;
  EXPECT_EQ(finalize_kv_push_failures(completion,
                                      transfers,
                                      /*enable_kv_push=*/true,
                                      {&tp, &cp},
                                      Device(torch::kCPU)),
            (std::vector<std::string>{"first", "second"}));
  EXPECT_EQ(tp.gathers_, 1);
  EXPECT_EQ(cp.gathers_, 1);
  EXPECT_EQ(tp.reductions_, 1);
  EXPECT_EQ(cp.reductions_, 1);
}

}  // namespace
}  // namespace xllm
