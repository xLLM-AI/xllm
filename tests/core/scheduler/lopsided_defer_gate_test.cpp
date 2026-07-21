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

// Unit test for the lopsided-batch defer algorithm used by
// ContinuousScheduler::step() in DP mode. The scheduler must skip a step
// if any dp_rank has an empty batch (the DP collective would hang or hit
// the fake-input path), but must eventually force the step if the imbalance
// persists longer than a backdoor window (otherwise a permanently-idle
// dp_rank would starve the busy one).
//
// The algorithm lives inline in continuous_scheduler.cpp:step() as a
// function-local static thread_local; that placement was deliberate to
// avoid changing sizeof(ContinuousScheduler) and forcing every subclass
// to recompile. Because the algorithm is not addressable from outside
// the scheduler, this test re-implements it here as LopsidedDeferGate.
// The two implementations must stay in sync -- if you change the algorithm
// in one place, change the other.

#include <absl/time/time.h>
#include <gtest/gtest.h>

namespace xllm {
namespace test {

// Mirror of the lopsided-defer state machine. Kept intentionally minimal
// (no logging, no live scheduler state) so the invariants are testable in
// isolation.
class LopsidedDeferGate {
 public:
  enum class Decision {
    kProceed,         // batch is balanced (or dp_size <= 1); run engine.step
    kDefer,           // any_empty and we're within the backdoor window
    kProceedBackdoor  // any_empty but we've been stuck > backdoor; force step
  };

  explicit LopsidedDeferGate(absl::Duration backdoor = absl::Milliseconds(100))
      : backdoor_(backdoor) {}

  Decision Decide(int32_t active_dp_size, bool any_empty, absl::Time now) {
    if (active_dp_size <= 1) {
      last_defer_start_ = absl::InfinitePast();
      return Decision::kProceed;
    }
    if (!any_empty) {
      last_defer_start_ = absl::InfinitePast();
      return Decision::kProceed;
    }
    if (last_defer_start_ == absl::InfinitePast()) {
      last_defer_start_ = now;
    }
    if (now - last_defer_start_ > backdoor_) {
      last_defer_start_ = absl::InfinitePast();
      return Decision::kProceedBackdoor;
    }
    return Decision::kDefer;
  }

 private:
  absl::Duration backdoor_;
  absl::Time last_defer_start_ = absl::InfinitePast();
};

TEST(LopsidedDeferGateTest, DpDisabledAlwaysProceeds) {
  LopsidedDeferGate gate;
  const auto now = absl::FromUnixMicros(1'000'000);
  EXPECT_EQ(gate.Decide(/*active_dp_size=*/1, /*any_empty=*/false, now),
            LopsidedDeferGate::Decision::kProceed);
  EXPECT_EQ(gate.Decide(1, /*any_empty=*/true, now),
            LopsidedDeferGate::Decision::kProceed);
}

TEST(LopsidedDeferGateTest, BalancedBatchProceeds) {
  LopsidedDeferGate gate;
  const auto now = absl::FromUnixMicros(1'000'000);
  EXPECT_EQ(gate.Decide(/*active_dp_size=*/2, /*any_empty=*/false, now),
            LopsidedDeferGate::Decision::kProceed);
}

TEST(LopsidedDeferGateTest, LopsidedBatchDefersUntilBackdoor) {
  LopsidedDeferGate gate(absl::Milliseconds(100));
  const auto t0 = absl::FromUnixMicros(1'000'000);

  // Immediate lopsided: defer.
  EXPECT_EQ(gate.Decide(2, /*any_empty=*/true, t0),
            LopsidedDeferGate::Decision::kDefer);

  // Still lopsided at 50ms: still defer.
  EXPECT_EQ(gate.Decide(2, true, t0 + absl::Milliseconds(50)),
            LopsidedDeferGate::Decision::kDefer);

  // At exactly 100ms we're still within the window (>, not >=).
  EXPECT_EQ(gate.Decide(2, true, t0 + absl::Milliseconds(100)),
            LopsidedDeferGate::Decision::kDefer);

  // At 101ms we exceed the backdoor: force the step.
  EXPECT_EQ(gate.Decide(2, true, t0 + absl::Milliseconds(101)),
            LopsidedDeferGate::Decision::kProceedBackdoor);
}

TEST(LopsidedDeferGateTest, BackdoorResetsAfterFire) {
  // Once the backdoor fires, the timer resets so the next lopsided run
  // starts a fresh 100ms window (rather than firing on every step in
  // perpetuity).
  LopsidedDeferGate gate(absl::Milliseconds(100));
  const auto t0 = absl::FromUnixMicros(1'000'000);

  gate.Decide(2, true, t0);                            // starts timer
  gate.Decide(2, true, t0 + absl::Milliseconds(101));  // fires backdoor

  // Immediately after the fire, another lopsided step should defer, not
  // force.
  EXPECT_EQ(gate.Decide(2, true, t0 + absl::Milliseconds(102)),
            LopsidedDeferGate::Decision::kDefer);
}

TEST(LopsidedDeferGateTest, BalancedTickResetsTimer) {
  // A balanced batch in the middle clears the accumulated wait so the next
  // lopsided run gets its full 100ms budget rather than tripping the
  // backdoor early.
  LopsidedDeferGate gate(absl::Milliseconds(100));
  const auto t0 = absl::FromUnixMicros(1'000'000);

  gate.Decide(2, true, t0);                           // defer at t0
  gate.Decide(2, true, t0 + absl::Milliseconds(80));  // still defer at t0+80
  gate.Decide(
      2, false, t0 + absl::Milliseconds(85));  // balanced -> proceed, reset

  // Now 20ms after the reset we're lopsided again. Before the fix this
  // would trip the backdoor because 20+80=100 which would exceed. With
  // the reset, 20ms is well within the fresh window.
  EXPECT_EQ(gate.Decide(2, true, t0 + absl::Milliseconds(105)),
            LopsidedDeferGate::Decision::kDefer);
}

TEST(LopsidedDeferGateTest, DpSizeGreaterThanTwoStillDefers) {
  // The gate cares only about "any dp_rank empty", not how many. dp=4 with
  // one empty rank should still defer.
  LopsidedDeferGate gate;
  const auto now = absl::FromUnixMicros(1'000'000);
  EXPECT_EQ(gate.Decide(/*active_dp_size=*/4, /*any_empty=*/true, now),
            LopsidedDeferGate::Decision::kDefer);
  EXPECT_EQ(gate.Decide(/*active_dp_size=*/4, /*any_empty=*/false, now),
            LopsidedDeferGate::Decision::kProceed);
}

}  // namespace test
}  // namespace xllm
