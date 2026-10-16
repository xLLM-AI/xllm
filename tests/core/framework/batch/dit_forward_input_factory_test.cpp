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

#include "core/framework/batch/dit_forward_input_factory.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/batch/dit_batch.h"

namespace xllm {
namespace {

std::shared_ptr<DiTRequest> make_request(const std::string& prompt,
                                         float tensor_value) {
  DiTRequestState state;
  state.input_params().prompt = prompt;
  state.input_params().media_sources.add(
      "image", "image", torch::full({3, 2, 2}, tensor_value));
  state.input_params().tensor_sources.add("prompt_embed",
                                          torch::full({2, 4}, tensor_value));
  return std::make_shared<DiTRequest>(prompt, "", "", state);
}

TEST(DiTForwardInputFactoryTest, PreservesRequestOrderAndNamedInputs) {
  auto first = make_request("first", /*tensor_value=*/1.0f);
  auto second = make_request("second", /*tensor_value=*/2.0f);
  DiTBatch batch;
  batch.add(first);
  batch.add(second);
  DiTForwardInputFactory factory;
  auto input = factory.create_input(batch);

  EXPECT_EQ(input.batch_size, 2);
  EXPECT_EQ(input.prompts, (std::vector<std::string>{"first", "second"}));
  EXPECT_TRUE(input.generation_params == first->state().generation_params());
  ASSERT_EQ(input.media_sources.size(), 1u);
  EXPECT_EQ(input.media_sources.at(0).name, "image");
  EXPECT_TRUE(
      torch::equal(input.media_sources.at(0).tensor[0],
                   first->state().input_params().media_sources.at(0).tensor));
  EXPECT_TRUE(
      torch::equal(input.media_sources.at(0).tensor[1],
                   second->state().input_params().media_sources.at(0).tensor));
  const auto prompt_embed = input.tensor_sources.get("prompt_embed");
  ASSERT_TRUE(prompt_embed.has_value());
  EXPECT_EQ(prompt_embed->sizes().vec(), (std::vector<int64_t>{2, 2, 4}));
  EXPECT_TRUE(torch::equal(
      (*prompt_embed)[0],
      *first->state().input_params().tensor_sources.get("prompt_embed")));
  EXPECT_TRUE(torch::equal(
      (*prompt_embed)[1],
      *second->state().input_params().tensor_sources.get("prompt_embed")));
}

TEST(DiTForwardInputFactoryTest, ReusesSingleRequestTensorStorage) {
  auto request = make_request("single", /*tensor_value=*/3.0f);
  DiTBatch batch;
  batch.add(request);
  DiTForwardInputFactory factory;
  auto input = factory.create_input(batch);

  const torch::Tensor& image = input.media_sources.at(0).tensor;
  EXPECT_EQ(image.sizes(), torch::IntArrayRef({1, 3, 2, 2}));
  EXPECT_EQ(
      image.data_ptr(),
      request->state().input_params().media_sources.at(0).tensor.data_ptr());
  const auto prompt_embed = input.tensor_sources.get("prompt_embed");
  ASSERT_TRUE(prompt_embed.has_value());
  EXPECT_EQ(prompt_embed->data_ptr(),
            request->state()
                .input_params()
                .tensor_sources.get("prompt_embed")
                ->data_ptr());
}

TEST(DiTForwardInputFactoryDeathTest, RejectsMixedGenerationParameters) {
  auto first = make_request("first", /*tensor_value=*/1.0f);
  auto second = make_request("second", /*tensor_value=*/2.0f);
  second->state().generation_params().width += 1;
  DiTBatch batch;
  batch.add(first);
  batch.add(second);
  DiTForwardInputFactory factory;
  EXPECT_DEATH(factory.create_input(batch), "generation params");
}

}  // namespace
}  // namespace xllm
