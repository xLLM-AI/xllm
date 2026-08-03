/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "distributed_runtime/runtime_options_builder.h"

#include <gtest/gtest.h>

#include <vector>

namespace xllm {
namespace {

Options make_options() {
  Options options;
  options.model_path("target-model")
      .model_id("target-id")
      .draft_model_path("draft-model")
      .backend("vlm")
      .block_size(64)
      .max_cache_size(4096)
      .max_memory_utilization(0.75)
      .enable_prefix_cache(false)
      .max_encoder_cache_size(512)
      .max_linear_state_cache_slots(32)
      .num_speculative_tokens(3)
      .speculative_algorithm("MTP")
      .enable_mtp_draft_body_tp1(true)
      .task_type("generate")
      .master_node_addr("127.0.0.1:8000")
      .nnodes(2)
      .node_rank(1)
      .dp_size(2)
      .ep_size(2)
      .cp_size(1)
      .enable_schedule_overlap(false)
      .enable_chunked_prefill(false)
      .instance_role(InstanceRole::DECODE)
      .kv_cache_transfer_mode("PULL")
      .transfer_listen_port(26001)
      .enable_disagg_pd(true)
      .enable_service_routing(true)
      .enable_offline_inference(true)
      .enable_shm(true)
      .input_shm_size(2)
      .output_shm_size(3)
      .server_idx(4)
      .enable_graph(true)
      .kv_cache_dtype("int8")
      .enable_sleep_mode(true);
  return options;
}

TEST(RuntimeOptionsBuilderTest, PreservesCausalRuntimeSettings) {
  const Options options = make_options();
  const std::vector<torch::Device> devices = {torch::Device(torch::kCPU)};

  const runtime::Options runtime_options =
      make_runtime_options(options, devices);

  EXPECT_EQ(runtime_options.model_path(), "target-model");
  EXPECT_EQ(runtime_options.model_id(), "target-id");
  EXPECT_EQ(runtime_options.backend(), "vlm");
  EXPECT_EQ(runtime_options.devices(), devices);
  EXPECT_EQ(runtime_options.max_encoder_cache_size(), 512);
  EXPECT_EQ(runtime_options.instance_role(), InstanceRole::DECODE);
  EXPECT_TRUE(runtime_options.enable_disagg_pd());
  EXPECT_EQ(runtime_options.kv_cache_transfer_mode(), "PULL");
  EXPECT_EQ(runtime_options.kv_cache_dtype(), "int8");
  EXPECT_TRUE(runtime_options.enable_sleep_mode());
  EXPECT_EQ(runtime_options.input_shm_size(), 2U * 1024U * 1024U);
  EXPECT_EQ(runtime_options.output_shm_size(), 3U * 1024U * 1024U);
}

TEST(RuntimeOptionsBuilderTest,
     AddsSpeculativeSettingsWithoutChangingVlmTarget) {
  const Options options = make_options();
  const std::vector<torch::Device> devices = {torch::Device(torch::kCPU)};
  const std::vector<torch::Device> draft_devices = {torch::Device(torch::kCPU)};

  const runtime::Options runtime_options =
      make_speculative_runtime_options(options, devices, draft_devices);

  EXPECT_EQ(runtime_options.backend(), "vlm");
  ASSERT_TRUE(runtime_options.draft_model_path().has_value());
  EXPECT_EQ(runtime_options.draft_model_path().value(), "draft-model");
  EXPECT_EQ(runtime_options.draft_devices(), draft_devices);
  EXPECT_EQ(runtime_options.num_speculative_tokens(), 3);
  EXPECT_EQ(runtime_options.speculative_algorithm(), "MTP");
  EXPECT_TRUE(runtime_options.enable_mtp_draft_body_tp1());
  EXPECT_EQ(runtime_options.model_id(), "target-id");
  EXPECT_EQ(runtime_options.instance_role(), InstanceRole::DECODE);
  EXPECT_EQ(runtime_options.kv_cache_dtype(), "int8");
}

}  // namespace
}  // namespace xllm
