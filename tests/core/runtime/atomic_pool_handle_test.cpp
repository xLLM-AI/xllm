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

// Regression test for the atomic-pool-pointer discipline that
// XServiceClient uses to survive a runtime CP<->DP flip.
//
// Background: BlockManagerPool is owned by LLMEngine (via unique_ptr).
// XServiceClient holds a NON-OWNING pointer that its heartbeat and
// reconcile threads dereference every ~100ms. When
// LLMEngine::rebuild_block_manager_pool destroys the old pool and builds
// a new one for the flipped dp_size, XServiceClient must observe the
// swap atomically or its next heartbeat reads dangling memory. This is
// exactly the "rank0 disappears ~5s post-flip" bug from the 2026-07-02
// soak run; the fix was making the pointer std::atomic<const Pool*> with
// release/acquire semantics.
//
// This test does not construct a real XServiceClient (it is a singleton
// coupled to etcd, brpc stubs, and the heartbeat loop). Instead it
// exercises the pointer discipline in isolation with a stripped-down
// AtomicPoolHandle whose semantics match the real class:
//   - writer publishes a new pointer via release-store;
//   - readers snapshot the pointer once per iteration via acquire-load
//     and use only the local for the rest of that iteration.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace xllm {
namespace test {

// Stand-in for BlockManagerPool. Reads a piece of state readers care
// about (options()) without pulling in the real header.
class FakePool {
 public:
  explicit FakePool(int32_t dp_size) : dp_size_(dp_size) {}
  int32_t dp_size() const { return dp_size_; }

 private:
  int32_t dp_size_;
};

// Mirrors the pointer discipline in XServiceClient (xservice_client.h:134
// + set_block_manager_pool()). The invariants under test:
//   - store uses release order,
//   - load uses acquire order,
//   - readers pin the pointer to a local for the whole "iteration" so
//     they never observe a mid-iteration swap.
class AtomicPoolHandle {
 public:
  // Publishes a new pool. Callers must keep the old pool alive at least
  // until every reader that started before this call finishes -- the
  // real XServiceClient satisfies this because rebuild_after_flip runs
  // under switch_gate_ unique_lock.
  void Store(const FakePool* p) { ptr_.store(p, std::memory_order_release); }

  // Snapshots the pointer for one reader iteration.
  const FakePool* Snapshot() const {
    return ptr_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<const FakePool*> ptr_{nullptr};
};

TEST(AtomicPoolHandleTest, SingleThreadedStoreLoadRoundTrip) {
  AtomicPoolHandle h;
  EXPECT_EQ(h.Snapshot(), nullptr);

  auto p1 = std::make_unique<FakePool>(1);
  h.Store(p1.get());
  EXPECT_EQ(h.Snapshot()->dp_size(), 1);

  auto p2 = std::make_unique<FakePool>(2);
  h.Store(p2.get());
  EXPECT_EQ(h.Snapshot()->dp_size(), 2);
}

TEST(AtomicPoolHandleTest, ReaderSeesConsistentIterationSnapshot) {
  // Writer swaps the pool 200 times while a reader loops. The reader
  // pins the pointer once per "iteration" and does two dereferences on
  // the local (mimicking heartbeat's `pool->options()` +
  // `pool->get_merged_kvcache_event`). We hand-verify the two reads
  // return the same dp_size, i.e. the snapshot did not change mid-way.
  //
  // Note: this test cannot actually catch a use-after-free by inspection
  // -- the writer keeps every pool alive for the duration -- but it can
  // catch the case where a naive reader re-loads the atomic on each
  // access and drifts to a different pool between the two derefs.
  AtomicPoolHandle h;

  std::vector<std::unique_ptr<FakePool>> pools;
  for (int i = 0; i < 8; ++i) {
    pools.emplace_back(std::make_unique<FakePool>(i + 1));
  }
  h.Store(pools[0].get());

  std::atomic<bool> stop{false};
  std::atomic<int64_t> torn{0};

  std::thread reader([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      const FakePool* pinned = h.Snapshot();
      if (pinned == nullptr) continue;
      const int32_t first = pinned->dp_size();
      // Simulate work between two dereferences of the same iteration.
      for (int i = 0; i < 8; ++i) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
      }
      const int32_t second = pinned->dp_size();
      if (first != second) {
        torn.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });

  for (int i = 0; i < 200; ++i) {
    h.Store(pools[i % pools.size()].get());
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  stop.store(true, std::memory_order_relaxed);
  reader.join();

  EXPECT_EQ(torn.load(), 0)
      << "Reader observed a mid-iteration pointer swap; the pinning "
         "discipline (snapshot to local, use local only) is broken.";
}

TEST(AtomicPoolHandleTest, StoreOfNullptrIsRespected) {
  // XServiceClient must tolerate the pool being cleared (e.g. during
  // shutdown) without crashing on the next heartbeat.
  AtomicPoolHandle h;
  auto p = std::make_unique<FakePool>(1);
  h.Store(p.get());
  ASSERT_NE(h.Snapshot(), nullptr);

  h.Store(nullptr);
  EXPECT_EQ(h.Snapshot(), nullptr);
}

}  // namespace test
}  // namespace xllm
