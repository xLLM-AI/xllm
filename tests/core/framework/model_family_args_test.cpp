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

#include "core/framework/model/model_family_args.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(ModelFamilyArgsTest,
     SharesFieldsBetweenCapabilitiesAndCompatibilityAccess) {
  RawModelArgs config{};
  LlmModelArgs llm(config);
  LlmModelArgs& transformer = llm;

  EXPECT_EQ(&llm.enable_mla(), &llm.attention_params().enable_mla());
  EXPECT_EQ(&llm.enable_mla(), &transformer.enable_mla());
  EXPECT_EQ(&llm.use_moe(), &llm.moe_params().use_moe());
  EXPECT_EQ(&llm.use_moe(), &transformer.moe_params().use_moe());
  EXPECT_EQ(&llm.draft_vocab_size(),
            &llm.speculative_params().draft_vocab_size());
  EXPECT_EQ(&llm.draft_vocab_size(),
            &transformer.speculative_params().draft_vocab_size());
  EXPECT_EQ(&transformer.full_attention_interval(),
            &llm.linear_attention_params().full_attention_interval());
  EXPECT_EQ(&transformer.max_seq_len(), &llm.kv_cache_params().max_seq_len());
  EXPECT_EQ(&transformer.hidden_size(), &llm.execution_params().hidden_size());

  llm.enable_mla() = true;
  transformer.moe_params().use_moe() = true;
  llm.speculative_params().draft_vocab_size() = 256;
  EXPECT_TRUE(transformer.attention_params().enable_mla());
  EXPECT_TRUE(llm.moe_params().use_moe());
  EXPECT_EQ(llm.draft_vocab_size(), 256);

  transformer.attention_params().enable_mla() = false;
  llm.moe_params().use_moe() = false;
  llm.draft_vocab_size() = 512;
  EXPECT_FALSE(llm.enable_mla());
  EXPECT_FALSE(transformer.moe_params().use_moe());
  EXPECT_EQ(transformer.speculative_params().draft_vocab_size(), 512);

  VlmModelArgs vlm(config);
  EXPECT_EQ(&vlm.encoder_embedding_mode(),
            &vlm.embedding_params().encoder_embedding_mode());
  EXPECT_EQ(&vlm.encoder_embedding_mode(),
            &vlm.embedding_params().encoder_embedding_mode());
  vlm.embedding_params().encoder_embedding_mode() = true;
  EXPECT_TRUE(vlm.encoder_embedding_mode());
  vlm.encoder_embedding_mode() = false;
  EXPECT_FALSE(vlm.embedding_params().encoder_embedding_mode());
}

TEST(ModelFamilyArgsTest, CopiesOwnIndependentCapabilityStorage) {
  RawModelArgs config{};
  config.draft_vocab_size() = 256;
  config.layers_to_capture() = {1, 3};
  const LlmModelArgs original(config);
  LlmModelArgs copied = original;

  copied.draft_vocab_size() = 512;
  copied.speculative_params().layers_to_capture().front() = 2;
  copied.enable_mla() = true;

  EXPECT_EQ(original.draft_vocab_size(), 256);
  EXPECT_EQ(original.speculative_params().layers_to_capture().front(), 1);
  EXPECT_FALSE(original.enable_mla());
  EXPECT_EQ(copied.speculative_params().draft_vocab_size(), 512);
  EXPECT_EQ(copied.layers_to_capture().front(), 2);
  EXPECT_TRUE(copied.attention_params().enable_mla());
  EXPECT_NE(&original.attention_params(), &copied.attention_params());
  EXPECT_EQ(&copied.draft_vocab_size(),
            &copied.speculative_params().draft_vocab_size());
}

TEST(ModelArgsTest, KeepsVlmComponentCapabilitiesWithItsOwnedArguments) {
  RawModelArgs config{};
  config.head_dim() = 128;
  const ModelArgs original(
      config, ModelFamily::VLM, DiTComponentType::TEXT_ENCODER);
  EXPECT_EQ(&original.attention_params(),
            &original.dit_model_args().vlm().attention_params());
  EXPECT_EQ(&original.embedding_params(),
            &original.vlm_model_args().embedding_params());

  ModelArgs copied = original;
  copied.set_encoder_embedding_mode(true);

  EXPECT_TRUE(copied.embedding_params().encoder_embedding_mode());
  EXPECT_TRUE(copied.dit_model_args().vlm().encoder_embedding_mode());
  EXPECT_FALSE(original.embedding_params().encoder_embedding_mode());
  EXPECT_NE(&copied.attention_params(), &original.attention_params());
  EXPECT_EQ(copied.attention_params().head_dim(), 128);
}

}  // namespace
}  // namespace xllm
