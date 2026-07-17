/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "framework/parallel_state/parallel_args.h"

namespace xllm {

// Runtime CP<->DP switch: DualParallelArgs holds two frozen ParallelArgs
// snapshots (one for CP_PREFILL, one for DP_DECODE) and an atomic Mode
// discriminator. set_mode flips the discriminator; active() picks the right
// snapshot. These tests pin the invariants that make the flip safe:
// - flipping to the current mode is a no-op (idempotent);
// - active() reads the flipped snapshot immediately;
// - concurrent readers never see a torn ParallelArgs (release/acquire).
namespace {

ParallelArgs make_cp_args() {
  // world=4, dp=1, cp=4, ep=1 -- the CP_PREFILL layout for a tp=1/cp=4 setup.
  return ParallelArgs(/*rank=*/0,
                      /*world_size=*/4,
                      /*dp_size=*/1,
                      /*cp_size=*/4,
                      /*process_group=*/nullptr,
                      /*ep_size=*/1);
}

ParallelArgs make_dp_args() {
  // Same world_size, but dp=4/cp=1: the DP_DECODE layout.
  return ParallelArgs(/*rank=*/0,
                      /*world_size=*/4,
                      /*dp_size=*/4,
                      /*cp_size=*/1,
                      /*process_group=*/nullptr,
                      /*ep_size=*/1);
}

}  // namespace

TEST(DualParallelArgsTest, ActiveReflectsInitialMode) {
  DualParallelArgs cp_first(
      make_cp_args(), make_dp_args(), DualParallelArgs::Mode::CP_PREFILL);
  EXPECT_EQ(cp_first.mode(), DualParallelArgs::Mode::CP_PREFILL);
  EXPECT_EQ(cp_first.active().cp_size(), 4);
  EXPECT_EQ(cp_first.active().dp_size(), 1);

  DualParallelArgs dp_first(
      make_cp_args(), make_dp_args(), DualParallelArgs::Mode::DP_DECODE);
  EXPECT_EQ(dp_first.mode(), DualParallelArgs::Mode::DP_DECODE);
  EXPECT_EQ(dp_first.active().cp_size(), 1);
  EXPECT_EQ(dp_first.active().dp_size(), 4);
}

TEST(DualParallelArgsTest, SetModeFlipsActive) {
  DualParallelArgs args(
      make_cp_args(), make_dp_args(), DualParallelArgs::Mode::CP_PREFILL);
  ASSERT_EQ(args.active().cp_size(), 4);

  args.set_mode(DualParallelArgs::Mode::DP_DECODE);
  EXPECT_EQ(args.mode(), DualParallelArgs::Mode::DP_DECODE);
  EXPECT_EQ(args.active().cp_size(), 1);
  EXPECT_EQ(args.active().dp_size(), 4);

  args.set_mode(DualParallelArgs::Mode::CP_PREFILL);
  EXPECT_EQ(args.mode(), DualParallelArgs::Mode::CP_PREFILL);
  EXPECT_EQ(args.active().cp_size(), 4);
  EXPECT_EQ(args.active().dp_size(), 1);
}

TEST(DualParallelArgsTest, SetModeIdempotent) {
  // Flipping to the mode we're already in must be a no-op that leaves
  // active() pointing at the same snapshot; ModeSwitchService relies on
  // this to make the RPC idempotent for retries.
  DualParallelArgs args(
      make_cp_args(), make_dp_args(), DualParallelArgs::Mode::CP_PREFILL);
  const void* first_addr = &args.active();

  args.set_mode(DualParallelArgs::Mode::CP_PREFILL);
  EXPECT_EQ(args.mode(), DualParallelArgs::Mode::CP_PREFILL);
  EXPECT_EQ(&args.active(), first_addr);

  args.set_mode(DualParallelArgs::Mode::CP_PREFILL);
  EXPECT_EQ(&args.active(), first_addr);
}

TEST(DualParallelArgsTest, ConcurrentReadersSeeConsistentSnapshot) {
  // Under a burst of concurrent flips, every reader should observe an
  // internally-consistent ParallelArgs (the whole struct comes from ONE
  // frozen snapshot, not a mix of pre- and post-flip fields). The
  // atomic<Mode> discriminator reads from either cp_args_ or dp_args_ as
  // an atomic reference switch; without release/acquire semantics a
  // reader could theoretically see cp_size from CP but dp_size from DP.
  DualParallelArgs args(
      make_cp_args(), make_dp_args(), DualParallelArgs::Mode::CP_PREFILL);

  std::atomic<bool> stop{false};
  std::atomic<int64_t> reader_ok{0};
  std::atomic<int64_t> reader_torn{0};

  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        const auto& a = args.active();
        // Both snapshots satisfy world_size == cp_size * dp_size. If the
        // reader observed a torn read, this invariant would fail.
        if (a.world_size() == a.cp_size() * a.dp_size() &&
            (a.cp_size() == 4 || a.dp_size() == 4)) {
          reader_ok.fetch_add(1, std::memory_order_relaxed);
        } else {
          reader_torn.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // Flip 1000 times while readers are running.
  for (int i = 0; i < 1000; ++i) {
    args.set_mode(i % 2 == 0 ? DualParallelArgs::Mode::DP_DECODE
                             : DualParallelArgs::Mode::CP_PREFILL);
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto& t : readers) {
    t.join();
  }

  EXPECT_GT(reader_ok.load(), 0);
  EXPECT_EQ(reader_torn.load(), 0)
      << "Torn read observed under concurrent flip; release/acquire on "
         "active_ is not enough or ParallelArgs is not internally frozen.";
}

}  // namespace xllm
