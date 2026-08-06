/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/eplb/expert_weight_buffer_shm.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace xllm {
namespace {

TEST(ExpertBufferShmTest, ReaderSeesTensorPublishedAfterAttach) {
  EXPECT_EXIT(
      {
        int32_t exit_code = 1;
        {
          const int32_t expert_id = static_cast<int32_t>(getpid());
          const std::string service_namespace =
              "reader_attach_" + std::to_string(getpid());
          ExpertBufferShm writer(service_namespace,
                                 expert_id,
                                 /*max_layers=*/1,
                                 /*total_size=*/4096);
          ExpertBufferShm reader(service_namespace,
                                 expert_id,
                                 /*max_layers=*/1,
                                 /*total_size=*/4096);
          const torch::Tensor expected =
              torch::tensor({1, 3, 5, 7}, torch::kInt32);
          writer.add_tensor(/*layer_id=*/0, "weight", expected);
          const torch::Tensor actual =
              reader.get_tensor(/*layer_id=*/0, "weight");
          exit_code = torch::equal(actual, expected) ? 0 : 2;
        }
        _exit(exit_code);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(ExpertBufferShmTest, LayerStorageRemainsAlignedForUnevenCapacity) {
  EXPECT_EXIT(
      {
        int32_t exit_code = 1;
        {
          const int32_t expert_id = static_cast<int32_t>(getpid());
          ExpertBufferShm buffer("alignment_" + std::to_string(getpid()),
                                 expert_id,
                                 /*max_layers=*/2,
                                 /*total_size=*/130);
          const torch::Tensor expected = torch::tensor({7}, torch::kInt32);
          buffer.add_tensor(/*layer_id=*/1, "weight", expected);
          const torch::Tensor actual =
              buffer.get_tensor(/*layer_id=*/1, "weight");
          const uintptr_t address =
              reinterpret_cast<uintptr_t>(actual.data_ptr());
          exit_code =
              address % 64 == 0 && torch::equal(actual, expected) ? 0 : 2;
        }
        _exit(exit_code);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(ExpertBufferShmTest, TensorRemainsReadableAfterBufferIsDestroyed) {
  EXPECT_EXIT(
      {
        torch::Tensor actual;
        const torch::Tensor expected =
            torch::tensor({2, 4, 6, 8}, torch::kInt32);
        {
          const int32_t expert_id = static_cast<int32_t>(getpid());
          ExpertBufferShm buffer("tensor_lifetime_" + std::to_string(getpid()),
                                 expert_id,
                                 /*max_layers=*/1,
                                 /*total_size=*/4096);
          buffer.add_tensor(/*layer_id=*/0, "weight", expected);
          actual = buffer.get_tensor(/*layer_id=*/0, "weight");
        }
        const int32_t exit_code = torch::equal(actual, expected) ? 0 : 2;
        _exit(exit_code);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(ExpertBufferShmTest, ServiceNamespacesIsolateTheSameExpertId) {
  EXPECT_EXIT(
      {
        int32_t exit_code = 1;
        {
          const int32_t expert_id = static_cast<int32_t>(getpid());
          ExpertBufferShm first("service_a_" + std::to_string(getpid()),
                                expert_id,
                                /*max_layers=*/1,
                                /*total_size=*/4096);
          ExpertBufferShm second("service_b_" + std::to_string(getpid()),
                                 expert_id,
                                 /*max_layers=*/1,
                                 /*total_size=*/4096);
          first.add_tensor(
              /*layer_id=*/0, "weight", torch::tensor({1, 2}, torch::kInt32));
          second.add_tensor(
              /*layer_id=*/0, "weight", torch::tensor({3, 4}, torch::kInt32));
          const bool isolated =
              torch::equal(first.get_tensor(/*layer_id=*/0, "weight"),
                           torch::tensor({1, 2}, torch::kInt32)) &&
              torch::equal(second.get_tensor(/*layer_id=*/0, "weight"),
                           torch::tensor({3, 4}, torch::kInt32));
          exit_code = isolated ? 0 : 2;
        }
        _exit(exit_code);
      },
      ::testing::ExitedWithCode(0),
      "");
}

}  // namespace
}  // namespace xllm
