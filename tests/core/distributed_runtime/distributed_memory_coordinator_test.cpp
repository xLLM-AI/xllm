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

#include "core/distributed_runtime/distributed_memory_coordinator.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "core/kv_cache/storage/paged_kv_cache_page_allocator.h"

namespace xllm {

class DistributedMemoryCoordinatorTestPeer final {
 public:
  static std::shared_ptr<DistributedMemoryCoordinator> create() {
    return std::shared_ptr<DistributedMemoryCoordinator>(
        new DistributedMemoryCoordinator(),
        [](DistributedMemoryCoordinator* coordinator) { delete coordinator; });
  }
};

namespace {

TEST(DistributedMemoryCoordinatorTest,
     ReservesKvAndWeightsOnEveryTargetWorker) {
  auto coordinator = DistributedMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/2,
                                /*worker_count=*/4,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_model_parallel_strategy("model",
                                           /*dp_size=*/2,
                                           /*tp_size=*/2);
  coordinator->set_weight_page_count("model", /*num_pages=*/3);
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0}, {0, 1}}};

  EXPECT_TRUE(coordinator->try_reserve_model_memory("model", plan));
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({5, 5, 3, 3}));
  EXPECT_EQ(coordinator->get_free_pages_for_model("model"), 3);
}

TEST(DistributedMemoryCoordinatorTest,
     FailedJointReservationLeavesAllWorkersUntouched) {
  auto coordinator = DistributedMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/2,
                                /*worker_count=*/4,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_model_parallel_strategy("model",
                                           /*dp_size=*/2,
                                           /*tp_size=*/2);
  coordinator->set_weight_page_count("model", /*num_pages=*/3);
  ASSERT_TRUE(coordinator->try_reserve_pages("model",
                                             /*dp_rank=*/1,
                                             /*physical_pages=*/5));
  const auto before = coordinator->get_worker_free_page_counts();
  // KV alone fits on worker 2; adding weights exceeds its five free pages.
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0}, {0, 1}}};

  EXPECT_FALSE(coordinator->try_reserve_model_memory("model", plan));
  EXPECT_EQ(coordinator->get_worker_free_page_counts(), before);
  EXPECT_EQ(coordinator->get_weight_page_count("model"), 3);
}

TEST(DistributedMemoryCoordinatorTest, ForkModelUsesOnlyItsOwnDpAndTpWorkers) {
  auto coordinator = DistributedMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/2,
                                /*worker_count=*/4,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_model_parallel_strategy("fork",
                                           /*dp_size=*/1,
                                           /*tp_size=*/2);
  coordinator->set_weight_page_count("fork", /*num_pages=*/3);
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0, 1}, {}}};

  EXPECT_TRUE(coordinator->try_reserve_model_memory("fork", plan));
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({3, 3, 10, 10}));
  EXPECT_EQ(coordinator->available_pages("fork", /*dp_rank=*/1), 0);
  EXPECT_FALSE(coordinator->try_reserve_pages("fork",
                                              /*dp_rank=*/1,
                                              /*physical_pages=*/1));
  coordinator->release_kv_mapping_budget("fork", plan);
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({7, 7, 10, 10}));
}

TEST(DistributedMemoryCoordinatorTest,
     ConcurrentJointReservationsCannotOvercommit) {
  auto coordinator = DistributedMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/1,
                                /*worker_count=*/2,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_weight_page_count("first", /*num_pages=*/4);
  coordinator->set_weight_page_count("second", /*num_pages=*/4);
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0}}};
  std::atomic<int32_t> successes{0};
  auto reserve = [&](const std::string& model_id) {
    if (coordinator->try_reserve_model_memory(model_id, plan)) {
      successes.fetch_add(1);
    }
  };
  std::thread first(reserve, "first");
  std::thread second(reserve, "second");
  first.join();
  second.join();

  EXPECT_EQ(successes.load(), 1);
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({4, 4}));
}

}  // namespace
}  // namespace xllm
