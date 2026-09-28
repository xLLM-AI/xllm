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

#include <gtest/gtest.h>

#include "core/framework/request/dit_input_sources.h"

namespace xllm {
namespace {

TEST(DiTTensorSourcesTest, GetsTensorStrictlyByName) {
  DiTTensorSources inputs;
  inputs.add("latent", torch::tensor({1}));
  inputs.add("prompt_embed", torch::tensor({2}));

  ASSERT_TRUE(inputs.get("prompt_embed").has_value());
  EXPECT_EQ(inputs.get("prompt_embed")->item<int64_t>(), 2);
  EXPECT_FALSE(inputs.get("pooled_prompt_embed").has_value());
}

TEST(DiTTensorSourcesTest, PromptAudioStaysFloat32OnTransfer) {
  DiTTensorSources sources;
  sources.add("prompt_audio", torch::ones({1, 4}, torch::kFloat32));
  sources.add("prompt_embed", torch::ones({1, 4}, torch::kFloat32));

  DiTTensorSources converted =
      sources.to(torch::Device(torch::kCPU), torch::kBFloat16);

  EXPECT_EQ(converted.get("prompt_audio")->scalar_type(), torch::kFloat32);
  EXPECT_EQ(converted.get("prompt_embed")->scalar_type(), torch::kBFloat16);
}

TEST(DiTTensorSourcesTest, BatchSignatureMatchesByName) {
  DiTTensorSources reference;
  reference.add("prompt_embed", torch::zeros({2, 4}));
  reference.add("latent", torch::zeros({4, 8, 8}));

  DiTTensorSources reordered;
  reordered.add("latent", torch::ones({4, 8, 8}));
  reordered.add("prompt_embed", torch::ones({2, 4}));
  EXPECT_TRUE(reference.batch_signature_matches(reordered));

  DiTTensorSources missing;
  missing.add("prompt_embed", torch::ones({2, 4}));
  EXPECT_FALSE(reference.batch_signature_matches(missing));
}

}  // namespace
}  // namespace xllm
