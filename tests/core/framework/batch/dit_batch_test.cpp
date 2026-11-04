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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/batch/dit_batch_factory.h"
#include "worker.pb.h"

namespace xllm {
namespace {

std::shared_ptr<DiTRequest> make_dit_request(const std::string& prompt) {
  DiTRequestState state;
  state.input_params().prompt = prompt;
  state.input_params().media_sources.add(
      "image", "image", torch::ones({3, 2, 2}));
  state.input_params().tensor_sources.add("prompt_embed", torch::ones({2, 4}));
  return std::make_shared<DiTRequest>("request", "", "", state);
}

DiTEncodedMedia encoded_image(std::string data) {
  return {.data = std::move(data),
          .mime_type = "image/png",
          .container = "png",
          .width = 2,
          .height = 2};
}

std::shared_ptr<DiTRequest> candidate_request(const std::string& id) {
  DiTGenerationParams params;
  params.num_images_per_prompt = 2;
  DiTRequestState state({}, params, {}, DiTRequestKind::kImage);
  return std::make_shared<DiTRequest>(id, "", "", std::move(state));
}

TEST(DiTBatchTest, EncodedCandidatesAreRequestMajorThenCandidateMajor) {
  auto first = candidate_request("first");
  auto second = candidate_request("second");
  DiTBatch batch;
  batch.add(first);
  batch.add(second);
  DiTForwardOutput output;
  output.encoded_media = {encoded_image("a"),
                          encoded_image("b"),
                          encoded_image("c"),
                          encoded_image("d")};
  batch.process_forward_output(output);
  output.encoded_media.clear();
  const DiTRequestOutput first_output = first->generate_output();
  const DiTRequestOutput second_output = second->generate_output();
  ASSERT_EQ(first_output.outputs.size(), 2u);
  ASSERT_EQ(second_output.outputs.size(), 2u);
  EXPECT_EQ(first_output.outputs[0].image, "a");
  EXPECT_EQ(first_output.outputs[1].image, "b");
  EXPECT_EQ(second_output.outputs[0].image, "c");
  EXPECT_EQ(second_output.outputs[1].image, "d");
  EXPECT_EQ(first_output.outputs[1].index, 1u);
  EXPECT_EQ(second_output.outputs[0].index, 0u);
}

TEST(DiTBatchTest, InvalidEncodedBatchDoesNotPartiallyReplaceRequests) {
  auto first = candidate_request("first");
  auto second = candidate_request("second");
  first->handle_forward_encoded_output(
      {encoded_image("old-a"), encoded_image("old-b")});
  second->handle_forward_encoded_output(
      {encoded_image("old-c"), encoded_image("old-d")});
  DiTBatch batch;
  batch.add(first);
  batch.add(second);
  DiTForwardOutput output;
  output.encoded_media = {
      encoded_image("a"), encoded_image("b"), encoded_image("c")};
  EXPECT_THROW(batch.process_forward_output(output), DiTForwardExecutionError);
  output.encoded_media.emplace_back(encoded_image("d"));
  output.encoded_media.back().mime_type = "video/mp4";
  EXPECT_THROW(batch.process_forward_output(output), DiTForwardExecutionError);
  output.encoded_media.back().mime_type = "image/png";
  output.text_output = {"mixed"};
  EXPECT_THROW(batch.process_forward_output(output), DiTForwardExecutionError);
  EXPECT_EQ(first->generate_output().outputs[0].image, "old-a");
  EXPECT_EQ(second->generate_output().outputs[1].image, "old-d");
}

TEST(DiTBatchTest, RuntimeEncodedProtoRoundTripOwnsBytesAndRejectsMixedMode) {
  DiTForwardOutput output;
  output.encoded_media = {encoded_image(std::string("A\0B", 3))};
  proto::DiTForwardOutput wire;
  ASSERT_TRUE(output.to_proto(&wire));
  DiTForwardOutput restored;
  ASSERT_TRUE(restored.from_proto(wire));
  wire.Clear();
  ASSERT_EQ(restored.encoded_media.size(), 1u);
  EXPECT_EQ(restored.encoded_media[0].data, std::string("A\0B", 3));
  EXPECT_EQ(restored.encoded_media[0].mime_type, "image/png");
  ASSERT_TRUE(output.to_proto(&wire));
  wire.add_text_output("mixed");
  EXPECT_FALSE(restored.from_proto(wire));
  EXPECT_EQ(restored.encoded_media[0].data, std::string("A\0B", 3));
  output.text_output = {"mixed"};
  const std::string prior_wire = wire.SerializeAsString();
  EXPECT_FALSE(output.to_proto(&wire));
  EXPECT_EQ(wire.SerializeAsString(), prior_wire);
}

TEST(DiTBatchTest, EmptyFactoryResultKeepsEngineBatchSlot) {
  DiTBatchFactory factory;
  auto batches = factory.create_batches({});
  ASSERT_EQ(batches.size(), 1);
  EXPECT_TRUE(batches[0].empty());
}

TEST(DiTBatchTest, BuilderPreservesPromptOrderAndStacksTensorInputs) {
  DiTBatchFactory factory;
  auto first = make_dit_request("first");
  auto second = make_dit_request("second");
  second->state().input_params().media_sources.at(0).tensor =
      torch::full({3, 2, 2}, 2.0f);
  auto batches = factory.create_batches({first, second});
  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0].size(), 2);
  auto input = batches[0].prepare_forward_input();
  EXPECT_EQ(input.model_params.batch_size, 2);
  EXPECT_EQ(input.model_params.prompts,
            (std::vector<std::string>{"first", "second"}));
  ASSERT_EQ(input.model_params.media_sources.size(), 1U);
  EXPECT_EQ(input.model_params.media_sources.at(0).name, "image");
  EXPECT_TRUE(
      torch::equal(input.model_params.media_sources.at(0).tensor[0],
                   first->state().input_params().media_sources.at(0).tensor));
  EXPECT_TRUE(
      torch::equal(input.model_params.media_sources.at(0).tensor[1],
                   second->state().input_params().media_sources.at(0).tensor));
  auto prompt_embeds = input.model_params.tensor_sources.get("prompt_embed");
  ASSERT_TRUE(prompt_embeds.has_value());
  EXPECT_EQ(prompt_embeds->sizes().vec(), (std::vector<int64_t>{2, 2, 4}));
  EXPECT_TRUE(input.model_params.generation_params ==
              first->state().generation_params());
}

TEST(DiTBatchTest, BatchKeepsRequestsAliveAfterSchedulerReleasesThem) {
  DiTBatchFactory factory;
  std::weak_ptr<DiTRequest> retained;
  std::vector<DiTBatch> batches;
  {
    auto request = make_dit_request("retained");
    retained = request;
    batches = factory.create_batches({request});
  }
  EXPECT_FALSE(retained.expired());
  EXPECT_EQ(batches[0].prepare_forward_input().model_params.prompts,
            (std::vector<std::string>{"retained"}));
  batches.clear();
  EXPECT_TRUE(retained.expired());
}

TEST(DiTBatchDeathTest, RejectsNullRequests) {
  DiTBatchFactory factory;
  EXPECT_DEATH(factory.create_batches({nullptr}), "request != nullptr");
}

TEST(DiTBatchDeathTest, RejectsMismatchedGenerationParameters) {
  DiTBatchFactory factory;
  auto first = make_dit_request("first");
  auto second = make_dit_request("second");
  second->state().generation_params().width += 1;
  auto batches = factory.create_batches({first, second});
  EXPECT_DEATH(batches[0].prepare_forward_input(), "generation params");
}

TEST(DiTBatchTest, SingleRequestSourcesUseBatchViews) {
  DiTInputParams input_params;
  torch::Tensor image =
      torch::arange(12, torch::dtype(torch::kUInt8)).reshape({3, 2, 2});
  torch::Tensor prompt_embed = torch::randn({4, 8});
  torch::Tensor prompt_audio = torch::randn({1, 32});
  input_params.media_sources.add("image", "image", image);
  input_params.tensor_sources.add("prompt_embed", prompt_embed);
  input_params.tensor_sources.add("prompt_audio", prompt_audio);

  DiTGenerationParams generation_params;
  DiTOutputFunc output_func;
  DiTRequestState state(
      input_params, generation_params, output_func, DiTRequestKind::kImage);
  auto request = std::make_shared<DiTRequest>("request", "rid", "rtime", state);

  DiTBatch batch;
  batch.add(request);
  DiTForwardInput forward_input = batch.prepare_forward_input();

  const torch::Tensor& batched_image =
      forward_input.model_params.media_sources.at(0).tensor;
  const torch::Tensor batched_embed =
      *forward_input.model_params.tensor_sources.get("prompt_embed");
  const torch::Tensor batched_audio =
      *forward_input.model_params.tensor_sources.get("prompt_audio");

  EXPECT_EQ(batched_image.sizes(), torch::IntArrayRef({1, 3, 2, 2}));
  EXPECT_EQ(batched_embed.sizes(), torch::IntArrayRef({1, 4, 8}));
  EXPECT_EQ(batched_audio.sizes(), torch::IntArrayRef({1, 1, 32}));
  EXPECT_EQ(batched_image.data_ptr(), image.data_ptr());
  EXPECT_EQ(batched_embed.data_ptr(), prompt_embed.data_ptr());
  EXPECT_EQ(batched_audio.data_ptr(), prompt_audio.data_ptr());
}

}  // namespace
}  // namespace xllm
