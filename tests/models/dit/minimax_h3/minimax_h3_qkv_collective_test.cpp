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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <cstdlib>
#include <functional>
#include <string>

#include "core/framework/parallel_state/dit_collective_communicator.h"
#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {
namespace {

TEST(H3QkvDomainsWorld16Test, ThreeActualDomainsMatchSerialAndInvertExactly) {
  const char* enabled = std::getenv("XLLM_H3_WORLD_TEST");
  if (enabled == nullptr || std::string(enabled) != "1") {
    GTEST_SKIP()
        << "Run the isolated 16-process H3 harness with XLLM_H3_WORLD_TEST=1";
  }
  const char* rank_text = std::getenv("RANK");
  const char* size_text = std::getenv("WORLD_SIZE");
  const char* port_text = std::getenv("XLLM_H3_TEST_PORT");
  ASSERT_NE(rank_text, nullptr);
  ASSERT_NE(size_text, nullptr);
  ASSERT_NE(port_text, nullptr);
  const int32_t rank = std::stoi(rank_text);
  ASSERT_EQ(std::stoi(size_text), 16);
  ASSERT_GE(rank, 0);
  ASSERT_LT(rank, 16);
  const char* local_text = std::getenv("LOCAL_RANK");
  const char* visible_text = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  ASSERT_NE(local_text, nullptr);
  ASSERT_NE(visible_text, nullptr);
  ASSERT_EQ(std::stoi(local_text), 0);
  ASSERT_EQ(std::string(visible_text).find(','), std::string::npos);
  RecordProperty("global_rank", rank);
  RecordProperty("local_rank", 0);
  RecordProperty("visible_device", std::string(visible_text));
  const char* host_text = std::getenv("XLLM_H3_TEST_MASTER_ADDR");
  const std::string host = host_text == nullptr ? "127.0.0.1" : host_text;
  const torch::Device device("npu:0");
  DiTCollectiveCommunicator communicator(
      rank, 16, 1, 2, 8, 1, 1, 1, {.create_domains = true});
  communicator.create_process_groups(host + ":" + port_text, device);
  const ParallelArgs& parallel = *communicator.parallel_args();
  const DiTSpQkvDomains domains = parallel.dit_sp_qkv_domains_;
  ASSERT_NO_THROW(validate_dit_sp_qkv_domains(domains, parallel.dit_sp_group_));
  int64_t async_launches = 0;
  int64_t async_waits = 0;
  int64_t serial_comparisons = 0;
  int64_t inverse_calls = 0;
  int64_t dummy_zero_checks = 0;
  const std::array<ProcessGroup*, 3> groups = {domains.q, domains.k, domains.v};
  std::array<torch::Tensor, 3> input;
  std::array<std::function<torch::Tensor()>, 3> pending;
  for (int64_t axis = 0; axis < 3; ++axis) {
    input[axis] =
        (torch::arange(256, torch::kFloat32).view({1, 4, 32, 2}) / 16.0F +
         rank * 32.0F + axis * 1024.0F)
            .to(torch::kBFloat16)
            .to(device);
    input[axis].narrow(2, 28, 4).zero_();
    pending[axis] =
        parallel_state::all_to_all_4D(input[axis], 2, 1, true, groups[axis]);
    ++async_launches;
  }
  std::array<torch::Tensor, 3> result;
  for (int64_t axis = 0; axis < 3; ++axis) {
    result[axis] = pending[axis]();
    pending[axis] = nullptr;
    ++async_waits;
  }
  for (int64_t axis = 0; axis < 3; ++axis) {
    const torch::Tensor serial = parallel_state::all_to_all_4D(
        input[axis], 2, 1, false, parallel.dit_sp_group_)();
    EXPECT_TRUE(torch::equal(result[axis], serial));
    ++serial_comparisons;
    const torch::Tensor restored = parallel_state::all_to_all_4D(
        result[axis], 1, 2, false, parallel.dit_sp_group_)();
    EXPECT_TRUE(torch::equal(restored, input[axis]));
    ++inverse_calls;
    if (parallel.dit_sp_group_->rank() == 7) {
      EXPECT_EQ(result[axis].count_nonzero().item<int64_t>(), 0);
      ++dummy_zero_checks;
    }
  }
  torch::Tensor dummy_checks =
      torch::tensor({dummy_zero_checks},
                    torch::TensorOptions().device(device).dtype(torch::kInt64));
  parallel.process_group_->allreduce(dummy_checks);
  EXPECT_EQ(dummy_checks.cpu().item<int64_t>(), 6);
  RecordProperty("actual_domains", groups.size());
  RecordProperty("domain_async_launches", async_launches);
  RecordProperty("domain_async_waits", async_waits);
  RecordProperty("serial_comparisons", serial_comparisons);
  RecordProperty("inverse_calls", inverse_calls);
  RecordProperty("global_dummy_zero_checks",
                 dummy_checks.cpu().item<int64_t>());
  parallel.process_group_->barrier();
}

}  // namespace
}  // namespace xllm
