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

#include "core/disaggregation/pd_reservation_releases.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>

namespace xllm {
namespace {

using Clock = PDReservationReleases::Clock;
using namespace std::chrono_literals;

TEST(PDReservationReleasesTest, MatchesBothRequestAndReservationIdentity) {
  PDReservationReleases releases;
  const Clock::time_point now{};
  releases.record("request", "cancelled", now);
  EXPECT_TRUE(releases.contains("request", "cancelled", now));
  EXPECT_FALSE(releases.contains("request", "replacement", now));
  EXPECT_FALSE(releases.contains("other", "cancelled", now));
}

TEST(PDReservationReleasesTest, ExpiresAtRetentionBoundary) {
  PDReservationReleases releases;
  const Clock::time_point now{};
  releases.record("request", "reservation", now);
  EXPECT_TRUE(releases.contains("request", "reservation", now + 5min - 1ns));
  EXPECT_FALSE(releases.contains("request", "reservation", now + 5min));
}

TEST(PDReservationReleasesTest, RefreshKeepsLaterEntriesReachableForExpiry) {
  PDReservationReleases releases;
  const Clock::time_point now{};
  releases.record("first", "reservation", now);
  releases.record("second", "reservation", now + 1min);
  releases.record("first", "reservation", now + 2min);
  EXPECT_FALSE(releases.contains("second", "reservation", now + 6min));
  EXPECT_TRUE(releases.contains("first", "reservation", now + 6min));
  EXPECT_FALSE(releases.contains("first", "reservation", now + 7min));
}

TEST(PDReservationReleasesTest, RecordPrunesExpiredEntriesAndAllowsReuse) {
  PDReservationReleases releases;
  const Clock::time_point now{};
  releases.record("request", "old", now);
  releases.record("request", "new", now + 5min);
  EXPECT_FALSE(releases.contains("request", "old", now + 5min));
  EXPECT_TRUE(releases.contains("request", "new", now + 5min));
  releases.record("request", "old", now + 5min);
  EXPECT_TRUE(releases.contains("request", "old", now + 5min));
}

TEST(PDReservationReleasesTest, RepeatedReleaseRefreshesSingleIdentity) {
  PDReservationReleases releases;
  const Clock::time_point now{};
  for (int32_t index = 0; index < 1000; ++index) {
    releases.record("request", "reservation", now + index * 1s);
  }
  EXPECT_TRUE(releases.contains("request", "reservation", now + 1298s));
  EXPECT_FALSE(releases.contains("request", "reservation", now + 1299s));
}

}  // namespace
}  // namespace xllm
