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

#include "core/framework/model_loader/hf_model_loader_profile.h"

#include <gtest/gtest.h>

#include <string>

#include "models/model_registry.h"

namespace xllm {
namespace {

TEST(HFModelLoaderProfileTest, RecExecutionPreservesSharedQwenTokenizerPolicy) {
  ResolvedModelDescriptor descriptor;
  descriptor.canonical_name = "qwen2";
  descriptor.registration_name = "qwen2";
  descriptor.family = ModelFamily::LLM;
  descriptor.family_name = "llm";

  HFModelLoaderProfile profile;
  std::string error;
  ASSERT_TRUE(
      select_hf_model_loader_profile(descriptor, "rec", &profile, &error))
      << error;
  EXPECT_EQ(profile.family, ModelFamily::REC);
  EXPECT_FALSE(profile.load_rec_vocab);
  EXPECT_FALSE(profile.load_media_preprocessor_args);
}

TEST(HFModelLoaderProfileTest, PythonVlmExecutionSelectsMediaPreprocessing) {
  ResolvedModelDescriptor descriptor;
  descriptor.canonical_name = "glm5_next";
  descriptor.registration_name = "glm5_next";
  descriptor.family = ModelFamily::LLM;
  descriptor.family_name = "llm";

  HFModelLoaderProfile profile;
  std::string error;
  ASSERT_TRUE(
      select_hf_model_loader_profile(descriptor, "vlm", &profile, &error))
      << error;
  EXPECT_EQ(profile.family, ModelFamily::VLM);
  EXPECT_TRUE(profile.load_media_preprocessor_args);
  EXPECT_FALSE(profile.load_rec_vocab);
}

TEST(HFModelLoaderProfileTest, OneRecVocabularyUsesCanonicalRegistration) {
  ResolvedModelDescriptor descriptor;
  descriptor.canonical_name = "onerec";
  descriptor.registration_name = "onerec_atb";
  descriptor.family = ModelFamily::REC;
  descriptor.family_name = "rec";

  HFModelLoaderProfile profile;
  std::string error;
  ASSERT_TRUE(select_hf_model_loader_profile(descriptor, "", &profile, &error))
      << error;
  EXPECT_EQ(profile.family, ModelFamily::REC);
  EXPECT_TRUE(profile.load_rec_vocab);
  EXPECT_FALSE(profile.load_media_preprocessor_args);
}

TEST(HFModelLoaderProfileTest, UsesRegisteredFamilyWithoutExecutionRequest) {
  ResolvedModelDescriptor descriptor;
  descriptor.registration_name = "vlm_profile_test";
  descriptor.family = ModelFamily::VLM;

  HFModelLoaderProfile profile;
  std::string error;
  ASSERT_TRUE(select_hf_model_loader_profile(descriptor, "", &profile, &error))
      << error;
  EXPECT_EQ(profile.family, ModelFamily::VLM);
  EXPECT_TRUE(profile.load_media_preprocessor_args);
}

TEST(HFModelLoaderProfileTest, ParserOnlyRegistrationKeepsSharedTextLoading) {
  ResolvedModelDescriptor descriptor;
  descriptor.registration_name = "parser_only_profile_test";

  HFModelLoaderProfile profile;
  std::string error;
  ASSERT_TRUE(select_hf_model_loader_profile(descriptor, "", &profile, &error))
      << error;
  EXPECT_EQ(profile.family, ModelFamily::LLM);
  EXPECT_FALSE(profile.load_media_preprocessor_args);
  EXPECT_FALSE(profile.load_rec_vocab);

  ASSERT_TRUE(
      select_hf_model_loader_profile(descriptor, "vlm", &profile, &error))
      << error;
  EXPECT_EQ(profile.family, ModelFamily::VLM);
  EXPECT_TRUE(profile.load_media_preprocessor_args);
}

TEST(HFModelLoaderProfileTest, RequestedLlmOverridesRegisteredVlm) {
  ResolvedModelDescriptor descriptor;
  descriptor.registration_name = "vlm_profile_test";
  descriptor.family = ModelFamily::VLM;
  descriptor.family_name = "vlm";

  HFModelLoaderProfile profile;
  std::string error = "previous error";
  ASSERT_TRUE(
      select_hf_model_loader_profile(descriptor, "llm", &profile, &error))
      << error;
  EXPECT_EQ(profile.family, ModelFamily::LLM);
  EXPECT_FALSE(profile.load_media_preprocessor_args);
  EXPECT_FALSE(profile.load_rec_vocab);
  EXPECT_TRUE(error.empty());
}

TEST(HFModelLoaderProfileTest, RejectsUnsupportedNamesWithoutChangingProfile) {
  ResolvedModelDescriptor descriptor;
  descriptor.registration_name = "family_profile_test";
  descriptor.family = ModelFamily::LLM;
  descriptor.family_name = "llm";

  for (const std::string family_name : {"dit", "unknown", "LLM", "invalid"}) {
    SCOPED_TRACE(family_name);
    HFModelLoaderProfile profile;
    profile.family = ModelFamily::VLM;
    profile.load_media_preprocessor_args = true;
    profile.load_rec_vocab = true;
    std::string error;

    EXPECT_FALSE(select_hf_model_loader_profile(
        descriptor, family_name, &profile, &error));
    EXPECT_EQ(error, "Unsupported HF loader model family: " + family_name);
    EXPECT_EQ(profile.family, ModelFamily::VLM);
    EXPECT_TRUE(profile.load_media_preprocessor_args);
    EXPECT_TRUE(profile.load_rec_vocab);

    descriptor.family_name = family_name;
    EXPECT_FALSE(
        select_hf_model_loader_profile(descriptor, "", &profile, &error));
    EXPECT_EQ(error, "Unsupported HF loader model family: " + family_name);
    EXPECT_EQ(profile.family, ModelFamily::VLM);
    EXPECT_TRUE(profile.load_media_preprocessor_args);
    EXPECT_TRUE(profile.load_rec_vocab);
  }
}

TEST(HFModelLoaderProfileTest, RejectsPipelineAndInvalidRequests) {
  ResolvedModelDescriptor descriptor;
  descriptor.registration_name = "pipeline_profile_test";
  descriptor.family = ModelFamily::DIT;

  HFModelLoaderProfile profile;
  std::string error;
  EXPECT_FALSE(
      select_hf_model_loader_profile(descriptor, "llm", &profile, &error));
  EXPECT_NE(error.find("DiTModelLoader"), std::string::npos);

  descriptor.family = ModelFamily::LLM;
  EXPECT_FALSE(
      select_hf_model_loader_profile(descriptor, "invalid", &profile, &error));
  EXPECT_NE(error.find("invalid"), std::string::npos);

  descriptor.registration_name.clear();
  EXPECT_FALSE(
      select_hf_model_loader_profile(descriptor, "llm", &profile, &error));
  EXPECT_NE(error.find("resolved"), std::string::npos);
}

}  // namespace
}  // namespace xllm
