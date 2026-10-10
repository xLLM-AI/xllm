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

#include "core/distributed_runtime/memory/worker_page_budget.h"

#include <gtest/gtest.h>

#include <vector>

namespace xllm {
namespace {

TEST(WorkerPageBudgetTest, RangeReservationIsAtomicAndIsolated) {
  WorkerPageBudget budget(/*total_pages=*/8, /*worker_count=*/4);

  EXPECT_TRUE(budget.try_reserve(/*start_worker=*/1,
                                 /*end_worker=*/3,
                                 /*pages_per_worker=*/6));
  EXPECT_EQ(budget.min_free_pages(1, 3), 2);
  EXPECT_EQ(budget.free_pages(0), 8);
  EXPECT_EQ(budget.free_pages(3), 8);

  EXPECT_FALSE(budget.try_reserve(1, 3, 3));
  EXPECT_EQ(budget.min_free_pages(1, 3), 2);
  EXPECT_EQ(budget.used_pages(), std::vector<size_t>({0, 6, 6, 0}));
}

TEST(WorkerPageBudgetTest, CombinedReservationChecksAllWorkersBeforeMutation) {
  WorkerPageBudget budget(/*total_pages=*/10, /*worker_count=*/3);
  ASSERT_TRUE(budget.try_reserve(0, 1, 7));

  EXPECT_FALSE(budget.try_reserve(std::vector<size_t>{2, 4, 11}));
  EXPECT_EQ(budget.used_pages(), std::vector<size_t>({7, 0, 0}));

  EXPECT_TRUE(budget.try_reserve(std::vector<size_t>{2, 4, 10}));
  EXPECT_EQ(budget.free_pages_snapshot(), std::vector<size_t>({1, 6, 0}));
}

TEST(WorkerPageBudgetTest, ReleaseSaturatesAndSnapshotIsIndependent) {
  WorkerPageBudget budget(/*total_pages=*/4, /*worker_count=*/2);
  ASSERT_TRUE(budget.try_reserve(std::vector<size_t>{3, 2}));

  const std::vector<size_t> snapshot = budget.free_pages_snapshot();
  EXPECT_EQ(snapshot, std::vector<size_t>({1, 2}));

  budget.release(0, 2, 10);
  EXPECT_EQ(budget.free_pages_snapshot(), std::vector<size_t>({4, 4}));
  budget.release(std::vector<size_t>{1, 1, 1});
  EXPECT_EQ(budget.used_pages(), std::vector<size_t>({0, 0}));
}

TEST(WorkerPageBudgetTest, InvalidReservationsDoNotMutateState) {
  WorkerPageBudget budget(/*total_pages=*/4, /*worker_count=*/2);

  EXPECT_FALSE(budget.try_reserve(-1, 1, 1));
  EXPECT_FALSE(budget.try_reserve(1, 3, 1));
  EXPECT_FALSE(budget.try_reserve(std::vector<size_t>{1}));
  EXPECT_EQ(budget.used_pages(), std::vector<size_t>({0, 0}));
  EXPECT_EQ(budget.free_pages(-1), 0);
  EXPECT_EQ(budget.free_pages(2), 0);
}

}  // namespace
}  // namespace xllm
