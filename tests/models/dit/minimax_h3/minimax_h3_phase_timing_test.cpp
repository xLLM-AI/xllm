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

#include "models/dit/utils/minimax_h3_phase_timing.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(H3PhaseTimerTest, DisabledDoesNotSynchronizeOrEmitReport) {
  int32_t synchronizations = 0;
  H3PhaseTimer timer(false, {}, [&] { ++synchronizations; });
  timer.start(H3TimingPhase::PREPARE);
  timer.advance(H3TimingPhase::DENOISE);
  timer.advance(H3TimingPhase::DECODE);
  EXPECT_TRUE(timer.finish().is_null());
  EXPECT_EQ(synchronizations, 0);
}

TEST(H3PhaseTimerTest, RequestUsesExactlyFourPhaseBoundaries) {
  int32_t synchronizations = 0;
  H3PhaseTimingMetadata metadata;
  metadata.request_id = "server-request-with-\"quote";
  metadata.profile = "a3_16card_tp2_u8_ref2va_v1";
  metadata.global_rank = 3;
  metadata.world_size = 16;
  metadata.tp_size = 2;
  metadata.sp_size = 8;
  metadata.denoise_steps = 49;
  H3PhaseTimer timer(true, metadata, [&] { ++synchronizations; });
  timer.start(H3TimingPhase::PREPARE);
  timer.advance(H3TimingPhase::DENOISE);
  timer.advance(H3TimingPhase::DECODE);
  const auto report = timer.finish();
  EXPECT_EQ(synchronizations, 4);
  EXPECT_EQ(report.at("request_id"), metadata.request_id);
  EXPECT_EQ(report.at("global_rank"), 3);
  EXPECT_EQ(report.at("world_size"), 16);
  EXPECT_EQ(report.at("denoise_steps"), 49);
  EXPECT_EQ(report.at("kind"), "h3_phase_request");
  EXPECT_GE(report.at("prepare_ms").get<double>(), 0.0);
  EXPECT_GE(report.at("denoise_ms").get<double>(), 0.0);
  EXPECT_GE(report.at("decode_ms").get<double>(), 0.0);
  EXPECT_DOUBLE_EQ(report.at("pipeline_ms").get<double>(),
                   report.at("prepare_ms").get<double>() +
                       report.at("denoise_ms").get<double>() +
                       report.at("decode_ms").get<double>());
  EXPECT_EQ(nlohmann::ordered_json::parse(report.dump()).at("request_id"),
            metadata.request_id);
}

TEST(H3PhaseTimerTest, LoadIsSeparateFromRequestSamples) {
  int32_t synchronizations = 0;
  H3PhaseTimer timer(true, {}, [&] { ++synchronizations; });
  timer.start(H3TimingPhase::LOAD);
  const auto report = timer.finish();
  EXPECT_EQ(synchronizations, 2);
  EXPECT_EQ(report.at("kind"), "h3_phase_model_load");
  EXPECT_GE(report.at("load_ms").get<double>(), 0.0);
  EXPECT_FALSE(report.contains("request_id"));
  EXPECT_FALSE(report.contains("pipeline_ms"));
}

}  // namespace
}  // namespace xllm
