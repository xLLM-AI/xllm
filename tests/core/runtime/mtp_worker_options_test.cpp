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

#include "core/runtime/mtp_worker_options.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(MtpWorkerOptionsTest, SelectsTextDraftForMultimodalTarget) {
  runtime::Options options;
  options.backend("vlm")
      .enable_schedule_overlap(true)
      .enable_task_pipeline(false)
      .enable_graph_aux_hidden_states(false)
      .num_decoding_tokens(5)
      .num_speculative_tokens(4);

  const runtime::Options target_options = mtp_detail::target_options(options);
  const runtime::Options draft_options = mtp_detail::draft_options(options);

  EXPECT_EQ(target_options.backend(), "vlm");
  EXPECT_FALSE(target_options.is_draft_engine());
  EXPECT_EQ(target_options.num_decoding_tokens(), 5);
  EXPECT_EQ(target_options.num_speculative_tokens(), 4);
  EXPECT_EQ(draft_options.backend(), "llm");
  EXPECT_TRUE(draft_options.is_draft_engine());
  EXPECT_EQ(draft_options.num_decoding_tokens(), 1);
  EXPECT_EQ(draft_options.num_speculative_tokens(), 0);
  EXPECT_FALSE(target_options.enable_schedule_overlap());
  EXPECT_FALSE(draft_options.enable_schedule_overlap());
  EXPECT_TRUE(target_options.enable_graph_aux_hidden_states());
  EXPECT_TRUE(draft_options.enable_graph_aux_hidden_states());

  EXPECT_EQ(options.backend(), "vlm");
  EXPECT_TRUE(options.enable_schedule_overlap());
  EXPECT_FALSE(options.enable_graph_aux_hidden_states());
}

TEST(MtpWorkerOptionsTest, PreservesTextTargetExecutionContract) {
  runtime::Options options;
  options.backend("llm");

  EXPECT_EQ(mtp_detail::target_options(options).backend(), "llm");
  EXPECT_EQ(mtp_detail::draft_options(options).backend(), "llm");
}

TEST(MtpWorkerOptionsTest, ReservesVerifierPrefixForTaskPipeline) {
  runtime::Options options;
  options.backend("vlm").enable_task_pipeline(true).num_speculative_tokens(4);

  const runtime::Options draft_options = mtp_detail::draft_options(options);

  EXPECT_EQ(draft_options.backend(), "llm");
  EXPECT_TRUE(draft_options.enable_task_pipeline());
  EXPECT_EQ(draft_options.num_decoding_tokens(), 5);
  EXPECT_EQ(draft_options.num_speculative_tokens(), 0);
  EXPECT_EQ(options.num_speculative_tokens(), 4);
}

TEST(MtpWorkerOptionsTest, ReservesMinimumDraftCapacityForTaskPipeline) {
  runtime::Options options;
  options.enable_task_pipeline(true).num_speculative_tokens(0);
  EXPECT_EQ(mtp_detail::draft_options(options).num_decoding_tokens(), 2);

  options.num_speculative_tokens(1);
  EXPECT_EQ(mtp_detail::draft_options(options).num_decoding_tokens(), 2);
}

}  // namespace
}  // namespace xllm
