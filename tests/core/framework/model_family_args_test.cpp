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

#include <string_view>
#include <utility>

namespace xllm {
namespace {

TEST(ModelFamilyTest, ParsesAndNamesSupportedFamilies) {
  const std::pair<std::string_view, ModelFamily::Value> cases[] = {
      {"llm", ModelFamily::LLM},
      {"vlm", ModelFamily::VLM},
      {"rec", ModelFamily::REC},
      {"dit", ModelFamily::DIT},
  };

  for (const auto& [name, expected] : cases) {
    SCOPED_TRACE(name);
    const ModelFamily family(name);
    EXPECT_EQ(family, expected);
    EXPECT_EQ(std::string_view(family.to_string()), name);
    EXPECT_EQ(ModelFamily(family.to_string()), family);
  }
}

TEST(ModelFamilyTest, PreservesUnknownFamilyHandling) {
  const ModelFamily default_family;
  EXPECT_EQ(default_family, ModelFamily::UNKNOWN);
  EXPECT_STREQ(default_family.to_string(), "unknown");

  for (const std::string_view name :
       {"", "unknown", "LLM", "VLM", "REC", "DIT", "llm_extra"}) {
    SCOPED_TRACE(name);
    const ModelFamily family(name);
    EXPECT_EQ(family, ModelFamily::UNKNOWN);
    EXPECT_STREQ(family.to_string(), "unknown");
  }
}

TEST(ModelFamilyTest, SupportsEnumStyleOperations) {
  constexpr ModelFamily llm = ModelFamily::LLM;
  constexpr ModelFamily vlm("vlm");
  static_assert(llm == ModelFamily::LLM);
  static_assert(vlm == ModelFamily::VLM);
  static_assert(llm != vlm);

  EXPECT_EQ(ModelFamily::LLM, llm);
  EXPECT_NE(ModelFamily::VLM, llm);
  EXPECT_EQ(ModelFamily("llm"), llm);
  EXPECT_NE(ModelFamily("rec"), llm);

  ModelFamily family = ModelFamily::UNKNOWN;
  family = ModelFamily::VLM;
  EXPECT_EQ(family, vlm);
  switch (family) {
    case ModelFamily::VLM:
      break;
    default:
      FAIL() << "Enum-style switch did not select the VLM family";
  }
}

TEST(DiTComponentTypeTest, ParsesSupportedComponentAliases) {
  const std::pair<std::string_view, DiTComponentType::Value> cases[] = {
      {"transformer", DiTComponentType::TRANSFORMER},
      {"transformer_2", DiTComponentType::TRANSFORMER},
      {"cola_dit", DiTComponentType::TRANSFORMER},
      {"vae", DiTComponentType::VAE},
      {"audio_vae", DiTComponentType::VAE},
      {"cola_vae", DiTComponentType::VAE},
      {"text_encoder", DiTComponentType::TEXT_ENCODER},
      {"text_encoder_2", DiTComponentType::TEXT_ENCODER},
      {"processor", DiTComponentType::TEXT_ENCODER},
      {"image_processor", DiTComponentType::TEXT_ENCODER},
      {"scheduler", DiTComponentType::SCHEDULER},
      {"audio_scheduler", DiTComponentType::SCHEDULER},
  };

  for (const auto& [component, expected] : cases) {
    SCOPED_TRACE(component);
    EXPECT_EQ(DiTComponentType(component), expected);
  }
}

TEST(DiTComponentTypeTest, PreservesUnknownComponentHandling) {
  for (const std::string_view component :
       {"", "unknown", "TRANSFORMER", "transformer_3", "vae_extra"}) {
    SCOPED_TRACE(component);
    EXPECT_EQ(DiTComponentType(component), DiTComponentType::UNKNOWN);
  }
}

TEST(DiTComponentTypeTest, SupportsEnumStyleOperations) {
  constexpr DiTComponentType transformer = DiTComponentType::TRANSFORMER;
  constexpr DiTComponentType vae = DiTComponentType::VAE;
  static_assert(transformer == DiTComponentType::TRANSFORMER);
  static_assert(transformer != vae);

  const DiTComponentType parsed_transformer("transformer_2");
  EXPECT_EQ(parsed_transformer, transformer);
  EXPECT_NE(parsed_transformer, vae);
  EXPECT_EQ(DiTComponentType::TRANSFORMER, parsed_transformer);
  EXPECT_NE(DiTComponentType::VAE, parsed_transformer);

  DiTComponentType component = DiTComponentType::UNKNOWN;
  component = DiTComponentType::VAE;
  EXPECT_EQ(component, vae);
  switch (component) {
    case DiTComponentType::VAE:
      break;
    default:
      FAIL() << "Enum-style switch did not select the VAE component";
  }
}

TEST(ModelFamilyArgsTest,
     SharesFieldsBetweenCapabilitiesAndCompatibilityAccess) {
  ModelArgs config{};
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
  ModelArgs config{};
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

TEST(ModelArgsBundleTest, KeepsVlmComponentCapabilitiesWithItsOwnedArguments) {
  ModelArgs config{};
  config.head_dim() = 128;
  const ModelArgsBundle original(
      config, ModelFamily::VLM, DiTComponentType::TEXT_ENCODER);
  EXPECT_EQ(&original.attention_params(),
            &original.dit_model_args().vlm().attention_params());
  EXPECT_EQ(&original.embedding_params(),
            &original.vlm_model_args().embedding_params());

  ModelArgsBundle copied = original;
  copied.set_encoder_embedding_mode(true);

  EXPECT_TRUE(copied.embedding_params().encoder_embedding_mode());
  EXPECT_TRUE(copied.dit_model_args().vlm().encoder_embedding_mode());
  EXPECT_FALSE(original.embedding_params().encoder_embedding_mode());
  EXPECT_NE(&copied.attention_params(), &original.attention_params());
  EXPECT_EQ(copied.attention_params().head_dim(), 128);
}

TEST(ModelArgsBundleTest, SelectsArgumentsFromParsedModelFamilies) {
  ModelArgs config{};
  config.hidden_size() = 256;

  const ModelArgsBundle llm(config, ModelFamily("llm"));
  EXPECT_EQ(llm.family(), ModelFamily::LLM);
  EXPECT_EQ(llm.llm_model_args().hidden_size(), 256);

  const ModelArgsBundle vlm(config, ModelFamily("vlm"));
  EXPECT_EQ(vlm.family(), ModelFamily::VLM);
  EXPECT_EQ(vlm.vlm_model_args().hidden_size(), 256);

  const ModelArgsBundle rec(config, ModelFamily("rec"));
  EXPECT_EQ(rec.family(), ModelFamily::REC);
  EXPECT_EQ(rec.rec_model_args().hidden_size(), 256);

  const ModelArgsBundle dit(config, ModelFamily("dit"));
  EXPECT_EQ(dit.family(), ModelFamily::DIT);
  EXPECT_EQ(dit.dit_model_args().transformer().hidden_size(), 256);
}

TEST(ModelArgsBundleTest, SelectsArgumentsFromParsedDitComponentTypes) {
  ModelArgs config{};
  config.hidden_size() = 256;
  config.vocab_size() = 512;
  config.num_train_timesteps() = 1000;

  const ModelArgsBundle transformer(
      config, ModelFamily::DIT, DiTComponentType("cola_dit"));
  EXPECT_EQ(transformer.dit_model_args().transformer().hidden_size(), 256);

  const ModelArgsBundle vae(
      config, ModelFamily::DIT, DiTComponentType("audio_vae"));
  EXPECT_EQ(vae.dit_model_args().vae().vocab_size(), 512);

  const ModelArgsBundle text_encoder(
      config, ModelFamily::DIT, DiTComponentType("text_encoder_2"));
  EXPECT_EQ(text_encoder.dit_model_args().text_encoder().vocab_size(), 512);

  const ModelArgsBundle scheduler(
      config, ModelFamily::DIT, DiTComponentType("audio_scheduler"));
  EXPECT_EQ(scheduler.dit_model_args().scheduler().num_train_timesteps(), 1000);

  const ModelArgsBundle default_component(config, ModelFamily::DIT);
  EXPECT_EQ(default_component.dit_model_args().component_type(),
            DiTComponentType::TRANSFORMER);
  EXPECT_EQ(default_component.dit_model_args().transformer().hidden_size(),
            256);
}

}  // namespace
}  // namespace xllm
