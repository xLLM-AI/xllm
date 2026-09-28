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
#include <vector>

#include "core/framework/batch/dit_batch_factory.h"

namespace xllm {
namespace {

std::shared_ptr<DiTRequest> make_dit_request(const std::string& prompt) {
  DiTRequestState state;
  state.input_params().prompt = prompt;
  state.input_params().image_sources.add("image", torch::ones({3, 2, 2}));
  state.input_params().tensor_sources.add("prompt_embed", torch::ones({2, 4}));
  return std::make_shared<DiTRequest>("request", "", "", state);
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
  second->state().input_params().image_sources.at(0).tensor =
      torch::full({3, 2, 2}, 2.0f);
  auto batches = factory.create_batches({first, second});
  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0].size(), 2);
  auto input = batches[0].prepare_forward_input();
  EXPECT_EQ(input.batch_size, 2);
  EXPECT_EQ(input.prompts, (std::vector<std::string>{"first", "second"}));
  ASSERT_EQ(input.image_sources.size(), 1U);
  EXPECT_EQ(input.image_sources.at(0).name, "image");
  EXPECT_TRUE(
      torch::equal(input.image_sources.at(0).tensor[0],
                   first->state().input_params().image_sources.at(0).tensor));
  EXPECT_TRUE(
      torch::equal(input.image_sources.at(0).tensor[1],
                   second->state().input_params().image_sources.at(0).tensor));
  auto prompt_embeds = input.tensor_sources.get("prompt_embed");
  ASSERT_TRUE(prompt_embeds.has_value());
  EXPECT_EQ(prompt_embeds->sizes().vec(), (std::vector<int64_t>{2, 2, 4}));
  EXPECT_TRUE(input.generation_params == first->state().generation_params());
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
  EXPECT_EQ(batches[0].prepare_forward_input().prompts,
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

}  // namespace
}  // namespace xllm
