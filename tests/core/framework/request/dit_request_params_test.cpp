/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/request/dit_request_params.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "text_generation.pb.h"

namespace xllm {
namespace {

// Helper to build a minimal TextGenerationRequest.
proto::TextGenerationRequest MakeTextRequest(const std::string& model,
                                             const std::string& prompt) {
  proto::TextGenerationRequest request;
  request.set_model(model);
  auto* input = request.mutable_input();
  input->set_prompt(prompt);
  return request;
}

// ===========================================================================
// DiTRequestParams constructor from TextGenerationRequest
// ===========================================================================

TEST(DiTRequestParamsTest, TextRequestSetsKindToText) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.request_kind, DiTRequestKind::kText);
}

TEST(DiTRequestParamsTest, TextRequestMapsModel) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.model, "Cola-DLM");
}

TEST(DiTRequestParamsTest, TextRequestMapsPrompt) {
  auto req = MakeTextRequest("Cola-DLM", "hello world");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.input_params.prompt, "hello world");
}

TEST(DiTRequestParamsTest, TextRequestUsesProvidedRequestId) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  req.set_request_id("custom-id");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.request_id, "custom-id");
}

TEST(DiTRequestParamsTest, TextRequestGeneratesRequestIdWhenAbsent) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_FALSE(params.request_id.empty());
  EXPECT_EQ(params.request_id.substr(0, 8), "textgen-");
}

TEST(DiTRequestParamsTest, TextRequestMapsXRequestIdAndTime) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "xrid-123", "xrtime-456");

  EXPECT_EQ(params.x_request_id, "xrid-123");
  EXPECT_EQ(params.x_request_time, "xrtime-456");
}

TEST(DiTRequestParamsTest, TextRequestDefaultGenerationParams) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  DiTRequestParams params(req, "rid", "rtime");

  // Default values when no parameters set.
  EXPECT_FALSE(params.generation_params.seed_is_set);
  EXPECT_EQ(params.generation_params.max_new_tokens, 32);
  EXPECT_EQ(params.generation_params.diffusion_steps, 16);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 7.0f);
}

TEST(DiTRequestParamsTest, TextRequestMapsSeedAndSetsFlag) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_seed(42);
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.seed, 42);
  EXPECT_TRUE(params.generation_params.seed_is_set);
}

TEST(DiTRequestParamsTest, TextRequestMapsAllSamplingParams) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(32);
  p->set_guidance_scale(5.0f);
  p->set_temperature(0.8f);
  p->set_top_k(50);
  p->set_top_p(0.95f);
  p->set_repetition_penalty(1.2f);

  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.max_new_tokens, 128);
  EXPECT_EQ(params.generation_params.diffusion_steps, 32);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 5.0f);
  EXPECT_FLOAT_EQ(params.generation_params.temperature, 0.8f);
  EXPECT_EQ(params.generation_params.top_k, 50);
  EXPECT_FLOAT_EQ(params.generation_params.top_p, 0.95f);
  EXPECT_FLOAT_EQ(params.generation_params.repetition_penalty, 1.2f);
}

TEST(DiTRequestParamsTest, TextRequestWithoutParametersUsesDefaults) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  // Don't set parameters at all.
  DiTRequestParams params(req, "rid", "rtime");

  EXPECT_EQ(params.generation_params.max_new_tokens, 32);
  EXPECT_EQ(params.generation_params.diffusion_steps, 16);
  EXPECT_FLOAT_EQ(params.generation_params.guidance_scale, 7.0f);
  EXPECT_FALSE(params.generation_params.seed_is_set);
}

// ===========================================================================
// verify_params for kText
// ===========================================================================

TEST(DiTRequestParamsVerifyTest, TextValidParamsReturnsTrue) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");

  bool called = false;
  auto result = params.verify_params([&](DiTRequestOutput /*unused*/) -> bool {
    called = true;
    return true;
  });

  EXPECT_TRUE(result);
  EXPECT_FALSE(called);  // No error callback on success.
}

TEST(DiTRequestParamsVerifyTest, TextEmptyPromptReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "");
  DiTRequestParams params(req, "rid", "rtime");

  StatusCode error_code = StatusCode::OK;
  std::string error_msg;
  auto result = params.verify_params([&](DiTRequestOutput output) -> bool {
    if (output.status.has_value()) {
      error_code = output.status.value().code();
      error_msg = output.status.value().message();
    }
    return true;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(error_msg, "prompt is empty");
}

TEST(DiTRequestParamsVerifyTest, TextEmptyModelReturnsFalse) {
  auto req = MakeTextRequest("", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");

  StatusCode error_code = StatusCode::OK;
  std::string error_msg;
  auto result = params.verify_params([&](DiTRequestOutput output) -> bool {
    if (output.status.has_value()) {
      error_code = output.status.value().code();
      error_msg = output.status.value().message();
    }
    return true;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(error_msg, "model is empty");
}

TEST(DiTRequestParamsVerifyTest, TextZeroMaxNewTokensReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(0);
  p->set_diffusion_steps(16);
  DiTRequestParams params(req, "rid", "rtime");

  StatusCode error_code = StatusCode::OK;
  auto result = params.verify_params([&](DiTRequestOutput output) -> bool {
    if (output.status.has_value()) {
      error_code = output.status.value().code();
    }
    return true;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
}

TEST(DiTRequestParamsVerifyTest, TextNegativeDiffusionStepsReturnsFalse) {
  auto req = MakeTextRequest("Cola-DLM", "hello");
  auto* p = req.mutable_parameters();
  p->set_max_new_tokens(128);
  p->set_diffusion_steps(-1);
  DiTRequestParams params(req, "rid", "rtime");

  StatusCode error_code = StatusCode::OK;
  auto result = params.verify_params([&](DiTRequestOutput output) -> bool {
    if (output.status.has_value()) {
      error_code = output.status.value().code();
    }
    return true;
  });

  EXPECT_FALSE(result);
  EXPECT_EQ(error_code, StatusCode::INVALID_ARGUMENT);
}

// ===========================================================================
// DiTGenerationParams equality with new text fields
// ===========================================================================

TEST(DiTGenerationParamsEqualityTest, EqualParamsAreEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;

  EXPECT_EQ(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentMaxNewTokensAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.max_new_tokens = 128;
  b.max_new_tokens = 256;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentSeedIsSetAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.seed_is_set = true;
  b.seed_is_set = false;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentDiffusionStepsAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.diffusion_steps = 16;
  b.diffusion_steps = 32;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTemperatureAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.temperature = 0.0f;
  b.temperature = 1.0f;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTopKAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.top_k = 0;
  b.top_k = 50;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentTopPAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.top_p = 1.0f;
  b.top_p = 0.9f;

  EXPECT_NE(a, b);
}

TEST(DiTGenerationParamsEqualityTest, DifferentRepetitionPenaltyAreNotEqual) {
  DiTGenerationParams a;
  DiTGenerationParams b;
  a.repetition_penalty = 1.0f;
  b.repetition_penalty = 1.1f;

  EXPECT_NE(a, b);
}

}  // namespace
}  // namespace xllm
