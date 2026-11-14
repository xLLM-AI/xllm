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

#include "core/disaggregation/cancelled_pd_reservations.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>

namespace xllm {
namespace {

using Clock = CancelledPDReservations::Clock;
using namespace std::chrono_literals;

TEST(CancelledPDReservationsTest, MatchesBothRequestAndReservationIdentity) {
  CancelledPDReservations cancellations;
  const Clock::time_point now{};
  cancellations.record("request", "cancelled", now);
  EXPECT_TRUE(cancellations.contains("request", "cancelled", now));
  EXPECT_FALSE(cancellations.contains("request", "replacement", now));
  EXPECT_FALSE(cancellations.contains("other", "cancelled", now));
}

TEST(CancelledPDReservationsTest, ExpiresAtRetentionBoundary) {
  CancelledPDReservations cancellations;
  const Clock::time_point now{};
  cancellations.record("request", "reservation", now);
  EXPECT_TRUE(
      cancellations.contains("request", "reservation", now + 5min - 1ns));
  EXPECT_FALSE(cancellations.contains("request", "reservation", now + 5min));
}

TEST(CancelledPDReservationsTest, RefreshKeepsLaterEntriesReachableForExpiry) {
  CancelledPDReservations cancellations;
  const Clock::time_point now{};
  cancellations.record("first", "reservation", now);
  cancellations.record("second", "reservation", now + 1min);
  cancellations.record("first", "reservation", now + 2min);
  EXPECT_FALSE(cancellations.contains("second", "reservation", now + 6min));
  EXPECT_TRUE(cancellations.contains("first", "reservation", now + 6min));
  EXPECT_FALSE(cancellations.contains("first", "reservation", now + 7min));
}

TEST(CancelledPDReservationsTest, RecordPrunesExpiredEntriesAndAllowsReuse) {
  CancelledPDReservations cancellations;
  const Clock::time_point now{};
  cancellations.record("request", "old", now);
  cancellations.record("request", "new", now + 5min);
  EXPECT_FALSE(cancellations.contains("request", "old", now + 5min));
  EXPECT_TRUE(cancellations.contains("request", "new", now + 5min));
  cancellations.record("request", "old", now + 5min);
  EXPECT_TRUE(cancellations.contains("request", "old", now + 5min));
}

TEST(CancelledPDReservationsTest, RepeatedCancellationRefreshesSingleIdentity) {
  CancelledPDReservations cancellations;
  const Clock::time_point now{};
  for (int32_t index = 0; index < 1000; ++index) {
    cancellations.record("request", "reservation", now + index * 1s);
  }
  EXPECT_TRUE(cancellations.contains("request", "reservation", now + 1298s));
  EXPECT_FALSE(cancellations.contains("request", "reservation", now + 1299s));
}

}  // namespace
}  // namespace xllm
