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

#include "core/framework/request/minimax_h3_conditioning.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/multimodal/minimax_h3_reference_geometry.h"
#include "core/framework/request/dit_request_params.h"
#include "core/util/tensor_codec.h"
#include "tensor.pb.h"
#include "video_generation.pb.h"

namespace xllm {
namespace {

constexpr char kHiddenDigest[] =
    "84ff92691f909a05b224e1c56abb4864f01b4f8e3c854e4bb4c7baf1d3f6d652";
constexpr char kTagDigest[] =
    "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc";
constexpr char kImageDigest[] =
    "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81";

TensorParameters condition_parameters() {
  return {{"prompt_token_tags", std::vector<int64_t>{0}},
          {"minimax_h3.schema",
           std::string("xllm.minimax_h3.text_conditioning/v1")},
          {"minimax_h3.cache_abi_version", int64_t{2}},
          {"minimax_h3.source_backend", std::string("official_hf")},
          {"minimax_h3.decoder_layer_index", int64_t{49}},
          {"minimax_h3.hidden_state_slot", int64_t{50}},
          {"minimax_h3.token_count", int64_t{1}},
          {"minimax_h3.hidden_digest", std::string(kHiddenDigest)},
          {"minimax_h3.token_tags_digest", std::string(kTagDigest)},
          {"minimax_h3.condition_cache_key", std::string(64, '0')},
          {"minimax_h3.reference_pixels_digest", std::string(kImageDigest)}};
}

proto::Tensor condition_wire() {
  proto::Tensor tensor;
  tensor.set_name("prompt_embed");
  tensor.set_datatype("BF16");
  tensor.add_shape(1);
  tensor.add_shape(5120);
  tensor.mutable_contents()->set_bytes_contents(std::string(10240, '\0'));
  EXPECT_TRUE(tensor_parameters_to_proto(condition_parameters(), &tensor));
  return tensor;
}

NamedTensor condition_source() {
  return {.name = "prompt_embed",
          .tensor = torch::zeros({1, 5120}, torch::kBFloat16),
          .parameters = condition_parameters()};
}

DiTMediaSources reference_image() {
  DiTMediaSources sources;
  sources.add("image",
              "image",
              torch::tensor({1, 2, 3}, torch::kUInt8).view({3, 1, 1}));
  return sources;
}

proto::VideoGenerationRequest condition_request() {
  proto::VideoGenerationRequest request;
  request.set_request_id("h3-contract-test");
  request.set_model("MiniMax-H3");
  *request.mutable_input()->mutable_prompt_embed() = condition_wire();
  auto* image = request.mutable_input()->add_media_sources();
  image->set_type("base64");
  image->set_name("image");
  image->set_modality("image");
  // Constant RGB=(1,2,3), H=64/W=173 resolves exactly to 2048x5536.
  image->set_base64(
      "iVBORw0KGgoAAAANSUhEUgAAAK0AAABACAIAAADnF2BIAAAAnklEQVR4nO3SQQ0AIBDAsAP8"
      "eyZTAY8mFbDHZu0D87yAH/"
      "iA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+"
      "ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+ID4gPiA+"
      "ID4gPiA+ID4gPiA+ID4gPiA+IBciAkDkMMYVLMAAAAASUVORK5CYII=");
  (*request.mutable_input()
        ->mutable_prompt_embed()
        ->mutable_parameters())["minimax_h3.reference_pixels_digest"]
      .set_string_param(
          "d49ef17c9d6ecf40a0795302cc31db68b55461130ef4d6699706bc481b400d32");
  request.mutable_parameters()->set_size("1344*768");
  request.mutable_parameters()->set_num_frames(124);
  request.mutable_parameters()->set_fps(24);
  request.mutable_parameters()->set_num_videos_per_prompt(1);
  request.mutable_parameters()->set_num_inference_steps(50);
  return request;
}

class MiniMaxH3ConditioningTest : public ::testing::Test {
 protected:
  void SetUp() override {
    DiTConfig& config = DiTConfig::get_instance();
    original_profile_ = config.dit_h3_execution_profile();
    original_task_ = config.dit_h3_task_type();
    config.dit_h3_execution_profile(std::string(kMiniMaxH3ExecutionProfile));
    config.dit_h3_task_type("ref2va");
  }
  void TearDown() override {
    DiTConfig& config = DiTConfig::get_instance();
    config.dit_h3_execution_profile(original_profile_);
    config.dit_h3_task_type(original_task_);
  }

 private:
  std::string original_profile_;
  std::string original_task_;
};

TEST_F(MiniMaxH3ConditioningTest, InlineAndDecodedContractMatchesFrozenHashes) {
  const proto::Tensor tensor = condition_wire();
  ASSERT_TRUE(validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
  const NamedTensor source = condition_source();
  EXPECT_EQ(minimax_h3_tensor_sha256(source.tensor), kHiddenDigest);
  EXPECT_TRUE(validate_minimax_h3_condition(source, reference_image()).ok());
  EXPECT_TRUE(torch::equal(util::proto_to_torch(tensor), source.tensor));
}

TEST_F(MiniMaxH3ConditioningTest,
       PartialUnknownAndAmbiguousMetadataAreRejected) {
  proto::Tensor tensor = condition_wire();
  tensor.mutable_parameters()->erase("minimax_h3.schema");
  EXPECT_TRUE(has_minimax_h3_condition_metadata(tensor));
  EXPECT_FALSE(
      validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
  tensor = condition_wire();
  (*tensor.mutable_parameters())["minimax_h3.typo"].set_int64_param(1);
  EXPECT_FALSE(
      validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
  tensor = condition_wire();
  (*tensor.mutable_parameters())["minimax_h3.token_count"].set_string_param(
      "1");
  EXPECT_FALSE(
      validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
}

TEST_F(MiniMaxH3ConditioningTest, CaptureTagsAndDigestsAreValidated) {
  NamedTensor source = condition_source();
  source.parameters["minimax_h3.hidden_state_slot"] = int64_t{49};
  EXPECT_FALSE(validate_minimax_h3_condition(source, reference_image()).ok());
  source = condition_source();
  source.parameters["prompt_token_tags"] = std::vector<int64_t>{2};
  EXPECT_FALSE(validate_minimax_h3_condition(source, reference_image()).ok());
  source = condition_source();
  source.parameters["minimax_h3.hidden_digest"] = std::string(64, 'A');
  EXPECT_FALSE(validate_minimax_h3_condition(source, reference_image()).ok());
  source = condition_source();
  source.parameters["minimax_h3.token_count"] = int64_t{60161};
  EXPECT_FALSE(validate_minimax_h3_condition(source, reference_image()).ok());
}

TEST_F(MiniMaxH3ConditioningTest, BinaryRangeAndExactTransportTypesAreChecked) {
  proto::Tensor tensor = condition_wire();
  tensor.clear_contents();
  auto& parameters = *tensor.mutable_parameters();
  parameters["is_binary"].set_bool_param(true);
  parameters["offset"].set_int64_param(3);
  parameters["len"].set_int64_param(10240);
  const BinaryPayload payload(std::string(10243, '\0'));
  ASSERT_TRUE(validate_minimax_h3_condition_wire(tensor, payload).ok());
  EXPECT_TRUE(torch::equal(util::proto_to_torch(tensor, payload),
                           condition_source().tensor));
  parameters["len"].set_int64_param(10239);
  EXPECT_FALSE(validate_minimax_h3_condition_wire(tensor, payload).ok());
  parameters["len"].set_int64_param(10240);
  parameters["offset"].set_int64_param(-1);
  EXPECT_FALSE(validate_minimax_h3_condition_wire(tensor, payload).ok());
  parameters["offset"].set_int64_param(10243);
  EXPECT_FALSE(validate_minimax_h3_condition_wire(tensor, payload).ok());
  parameters["offset"].set_int64_param(3);
  parameters["offset"].set_uint64_param(3);
  EXPECT_FALSE(validate_minimax_h3_condition_wire(tensor, payload).ok());
  parameters["offset"].clear_uint64_param();
  parameters["is_binary"].set_int64_param(1);
  EXPECT_FALSE(validate_minimax_h3_condition_wire(tensor, payload).ok());
}

TEST_F(MiniMaxH3ConditioningTest,
       InlineRejectsWrongShapeDtypeAndMixedContents) {
  proto::Tensor tensor = condition_wire();
  tensor.set_datatype("unknown");
  EXPECT_FALSE(
      validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
  tensor = condition_wire();
  tensor.set_shape(1, 5121);
  EXPECT_FALSE(
      validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
  tensor = condition_wire();
  tensor.mutable_contents()->add_fp32_contents(0.0f);
  EXPECT_FALSE(
      validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
  tensor = condition_wire();
  (*tensor.mutable_parameters())["offset"].set_int64_param(0);
  EXPECT_FALSE(
      validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
  tensor = condition_wire();
  tensor.mutable_contents()->mutable_bytes_contents()->pop_back();
  EXPECT_FALSE(
      validate_minimax_h3_condition_wire(tensor, BinaryPayload()).ok());
}

TEST_F(MiniMaxH3ConditioningTest, ChangedHiddenTagsAndPixelsFailDigests) {
  NamedTensor source = condition_source();
  source.tensor.index_put_({0, 0}, 1);
  EXPECT_FALSE(validate_minimax_h3_condition(source, reference_image()).ok());
  source = condition_source();
  source.parameters["prompt_token_tags"] = std::vector<int64_t>{1};
  EXPECT_FALSE(validate_minimax_h3_condition(source, reference_image()).ok());
  source = condition_source();
  DiTMediaSources media = reference_image();
  media.at(0).tensor.index_put_({0, 0, 0}, 2);
  EXPECT_FALSE(validate_minimax_h3_condition(source, media).ok());
  source.tensor.index_put_({0, 0}, std::numeric_limits<float>::infinity());
  EXPECT_FALSE(validate_minimax_h3_condition(source, reference_image()).ok());
}

TEST_F(MiniMaxH3ConditioningTest, ReferenceMustBeOneNamedRgbImage) {
  const NamedTensor source = condition_source();
  EXPECT_FALSE(validate_minimax_h3_condition(source, DiTMediaSources()).ok());
  DiTMediaSources media = reference_image();
  media.at(0).name = "unknown";
  EXPECT_FALSE(validate_minimax_h3_condition(source, media).ok());
  media = reference_image();
  media.add("image", "image", media.at(0).tensor);
  EXPECT_FALSE(validate_minimax_h3_condition(source, media).ok());
}

TEST_F(MiniMaxH3ConditioningTest, RequestDecodesAndOwnsConditionAndReference) {
  DiTRequestParams params;
  {
    proto::VideoGenerationRequest request = condition_request();
    params = DiTRequestParams(request, "rid", "rtime");
    request.Clear();
  }
  ASSERT_TRUE(params.request_parse_status.ok())
      << params.request_parse_status.message();
  EXPECT_TRUE(params.verify_params([](DiTRequestOutput) { return true; }));
  const auto source =
      params.input_params.tensor_sources.get_named_tensor("prompt_embed");
  ASSERT_TRUE(source.has_value());
  EXPECT_TRUE(validate_minimax_h3_condition(source->get(),
                                            params.input_params.media_sources)
                  .ok());
}

TEST_F(MiniMaxH3ConditioningTest, BinaryRequestOwnsHiddenAfterPayloadRelease) {
  DiTRequestParams params;
  {
    proto::VideoGenerationRequest request = condition_request();
    proto::Tensor* tensor = request.mutable_input()->mutable_prompt_embed();
    tensor->clear_contents();
    auto& transport = *tensor->mutable_parameters();
    transport["is_binary"].set_bool_param(true);
    transport["offset"].set_int64_param(3);
    transport["len"].set_int64_param(10240);
    const BinaryPayload payload(std::string(10243, '\0'));
    params = DiTRequestParams(request, "rid", "rtime", payload);
  }
  ASSERT_TRUE(params.request_parse_status.ok())
      << params.request_parse_status.message();
  const auto source =
      params.input_params.tensor_sources.get_named_tensor("prompt_embed");
  ASSERT_TRUE(source.has_value());
  EXPECT_EQ(minimax_h3_tensor_sha256(source->get().tensor), kHiddenDigest);
  EXPECT_FALSE(source->get().parameters.contains("offset"));
  EXPECT_FALSE(source->get().parameters.contains("len"));
  EXPECT_FALSE(source->get().parameters.contains("is_binary"));
}

TEST_F(MiniMaxH3ConditioningTest, MetadataCannotEnableNativeOrFl2vaProfile) {
  const proto::VideoGenerationRequest request = condition_request();
  DiTConfig::get_instance().dit_h3_execution_profile("native");
  EXPECT_EQ(
      DiTRequestParams(request, "rid", "rtime").request_parse_status.code(),
      StatusCode::INVALID_ARGUMENT);
  DiTConfig::get_instance().dit_h3_execution_profile(
      std::string(kMiniMaxH3ExecutionProfile));
  DiTConfig::get_instance().dit_h3_task_type("fl2va");
  EXPECT_EQ(
      DiTRequestParams(request, "rid", "rtime").request_parse_status.code(),
      StatusCode::INVALID_ARGUMENT);
}

TEST_F(MiniMaxH3ConditioningTest,
       QualifiedProfileRequiresProofBeforeTensorOrMediaDecode) {
  proto::VideoGenerationRequest request = condition_request();
  request.clear_input();
  DiTRequestParams params(request, "rid", "rtime");
  EXPECT_EQ(params.request_parse_status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(params.request_parse_status.message(),
            "qualified MiniMax-H3 requires attested prompt_embed metadata");

  request = condition_request();
  request.mutable_input()->clear_prompt_embed();
  params = DiTRequestParams(request, "rid", "rtime");
  EXPECT_EQ(params.request_parse_status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(params.input_params.media_sources.size(), 0u);

  request = condition_request();
  proto::Tensor* tensor = request.mutable_input()->mutable_prompt_embed();
  tensor->clear_parameters();
  tensor->set_datatype("unknown");
  params = DiTRequestParams(request, "rid", "rtime");
  EXPECT_EQ(params.request_parse_status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(params.request_parse_status.message(),
            "qualified MiniMax-H3 requires attested prompt_embed metadata");
  EXPECT_EQ(params.input_params.tensor_sources.size(), 0u);
  EXPECT_EQ(params.input_params.media_sources.size(), 0u);
}

TEST_F(MiniMaxH3ConditioningTest,
       NativeProfileStillAcceptsUnattestedPromptEmbeddings) {
  DiTConfig::get_instance().dit_h3_execution_profile("native");
  proto::VideoGenerationRequest request = condition_request();
  request.mutable_input()->mutable_prompt_embed()->clear_parameters();
  const DiTRequestParams params(request, "rid", "rtime");
  ASSERT_TRUE(params.request_parse_status.ok())
      << params.request_parse_status.message();
  EXPECT_EQ(params.input_params.tensor_sources.size(), 1u);
  EXPECT_EQ(params.input_params.media_sources.size(), 1u);
}

TEST_F(MiniMaxH3ConditioningTest, RequestRejectsMalformedWireBeforeDecoder) {
  proto::VideoGenerationRequest request = condition_request();
  request.mutable_input()->mutable_prompt_embed()->set_datatype("unknown");
  const DiTRequestParams params(request, "rid", "rtime");
  EXPECT_EQ(params.request_parse_status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(params.input_params.tensor_sources.size(), 0u);
}

TEST_F(MiniMaxH3ConditioningTest,
       RequestRejectsMultipleCandidatesAndBadPixels) {
  proto::VideoGenerationRequest request = condition_request();
  request.mutable_parameters()->set_num_videos_per_prompt(2);
  EXPECT_EQ(
      DiTRequestParams(request, "rid", "rtime").request_parse_status.code(),
      StatusCode::INVALID_ARGUMENT);
  request = condition_request();
  (*request.mutable_input()
        ->mutable_prompt_embed()
        ->mutable_parameters())["minimax_h3.reference_pixels_digest"]
      .set_string_param(std::string(64, '0'));
  EXPECT_EQ(
      DiTRequestParams(request, "rid", "rtime").request_parse_status.code(),
      StatusCode::INVALID_ARGUMENT);
}

TEST_F(MiniMaxH3ConditioningTest,
       QualifiedGeometryRejectsEachMismatchBeforeEnqueue) {
  using MutateParameters = std::function<void(proto::VideoParameters*)>;
  const std::vector<std::pair<std::string, MutateParameters>> cases = {
      {"width", [](auto* params) { params->set_size("1280*768"); }},
      {"height", [](auto* params) { params->set_size("1344*720"); }},
      {"frames", [](auto* params) { params->set_num_frames(125); }},
      {"fps", [](auto* params) { params->set_fps(25.0); }},
      {"sampling_rate", [](auto* params) { params->set_sampling_rate(44100); }},
      {"steps", [](auto* params) { params->set_num_inference_steps(49); }},
      {"candidates",
       [](auto* params) { params->set_num_videos_per_prompt(2); }},
      {"nonfinite_fps", [](auto* params) {
         params->set_fps(std::numeric_limits<double>::quiet_NaN());
       }}};
  for (const auto& [name, mutate] : cases) {
    SCOPED_TRACE(name);
    proto::VideoGenerationRequest request = condition_request();
    // Qualification follows the explicit profile, not a model-name substring.
    request.set_model("served-h3-alias");
    mutate(request.mutable_parameters());
    const DiTRequestParams params(request, "rid", "rtime");
    EXPECT_EQ(params.request_parse_status.code(), StatusCode::INVALID_ARGUMENT);
    int32_t callbacks = 0;
    EXPECT_FALSE(params.verify_params([&callbacks](DiTRequestOutput output) {
      ++callbacks;
      EXPECT_TRUE(output.status.has_value());
      if (output.status.has_value()) {
        EXPECT_EQ(output.status->code(), StatusCode::INVALID_ARGUMENT);
      }
      return true;
    }));
    EXPECT_EQ(callbacks, 1);
  }
}

TEST_F(MiniMaxH3ConditioningTest,
       QualifiedGeometryAcceptsOmittedLegalDefaultsAndExplicitParameters) {
  proto::VideoGenerationRequest request = condition_request();
  request.mutable_parameters()->clear_sampling_rate();
  request.mutable_parameters()->clear_num_videos_per_prompt();
  const DiTRequestParams defaulted(request, "rid", "rtime");
  ASSERT_TRUE(defaulted.request_parse_status.ok());
  EXPECT_EQ(defaulted.generation_params.audio_sampling_rate, 32000);
  EXPECT_EQ(defaulted.generation_params.num_videos_per_prompt, 1u);
  EXPECT_TRUE(defaulted.verify_params([](DiTRequestOutput) { return true; }));

  request.mutable_parameters()->set_sampling_rate(32000);
  request.mutable_parameters()->set_num_videos_per_prompt(1);
  request.mutable_parameters()->set_seed(123);
  const DiTRequestParams explicit_params(request, "rid", "rtime");
  ASSERT_TRUE(explicit_params.request_parse_status.ok());
  EXPECT_TRUE(
      explicit_params.verify_params([](DiTRequestOutput) { return true; }));
  EXPECT_TRUE(explicit_params.generation_params.seed_is_set);
  EXPECT_EQ(explicit_params.generation_params.seed, 123);
}

TEST_F(MiniMaxH3ConditioningTest,
       QualifiedGeometryRejectsOmittedIncompatibleGenerationDefaults) {
  proto::VideoGenerationRequest request = condition_request();
  request.clear_parameters();
  const DiTRequestParams params(request, "rid", "rtime");
  EXPECT_EQ(params.request_parse_status.code(), StatusCode::INVALID_ARGUMENT);
  int32_t callbacks = 0;
  EXPECT_FALSE(params.verify_params([&callbacks](DiTRequestOutput output) {
    ++callbacks;
    EXPECT_TRUE(output.status.has_value());
    if (output.status.has_value()) {
      EXPECT_EQ(output.status->code(), StatusCode::INVALID_ARGUMENT);
    }
    return true;
  }));
  EXPECT_EQ(callbacks, 1);
}

TEST_F(MiniMaxH3ConditioningTest,
       NativeGeometryRetainsGeneralGenerationParameters) {
  DiTConfig::get_instance().dit_h3_execution_profile("native");
  proto::VideoGenerationRequest request = condition_request();
  request.mutable_input()->mutable_prompt_embed()->clear_parameters();
  request.mutable_parameters()->set_size("16*16");
  request.mutable_parameters()->set_num_frames(125);
  request.mutable_parameters()->set_fps(25.0);
  request.mutable_parameters()->set_num_inference_steps(28);
  request.mutable_parameters()->set_sampling_rate(24000);
  const DiTRequestParams params(request, "rid", "rtime");
  ASSERT_TRUE(params.request_parse_status.ok())
      << params.request_parse_status.message();
  EXPECT_TRUE(params.verify_params([](DiTRequestOutput) { return true; }));
  EXPECT_EQ(params.generation_params.width, 16);
  EXPECT_EQ(params.generation_params.height, 16);
  EXPECT_EQ(params.generation_params.num_frames, 125);
  EXPECT_DOUBLE_EQ(params.generation_params.video_fps, 25.0);
  EXPECT_EQ(params.generation_params.num_inference_steps, 28);
  EXPECT_EQ(params.generation_params.audio_sampling_rate, 24000);
}

TEST_F(MiniMaxH3ConditioningTest,
       QualifiedReferenceGeometryRejectsBeforeEnqueue) {
  // Every PNG has a matching raw CHW digest: this exercises geometry, not a
  // malformed image or failed provenance check. Boundary ratios are allowed
  // by native geometry, but still do not produce the qualified canvas.
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"iVBORw0KGgoAAAANSUhEUgAAAAUAAAABCAIAAACZnPOkAAAADUlEQVR4nGNgZGJGRgAA9gA"
       "fgt4GDAAAAABJRU5ErkJggg==",
       "712f7599fd71bfc42004260d9b8c690126c1142c84287a0bb097f3e30e16a030"},
      {"iVBORw0KGgoAAAANSUhEUgAAAAEAAAAFCAIAAAAL5hHIAAAADklEQVR4nGNgZGJmQMMAATY"
       "AH63p8oEAAAAASUVORK5CYII=",
       "712f7599fd71bfc42004260d9b8c690126c1142c84287a0bb097f3e30e16a030"},
      {"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgZGIGAAAOAAf"
       "Xb+R4AAAAAElFTkSuQmCC",
       "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81"},
      {"iVBORw0KGgoAAAANSUhEUgAAAAQAAAABCAIAAAB2XpiaAAAADUlEQVR4nGNgZGKGIwAAoQA"
       "ZlmqabgAAAABJRU5ErkJggg==",
       "c814bd1acd92fcf2f7e38558b07266882886117227a58838b493cdbbcda8c0db"},
      {"iVBORw0KGgoAAAANSUhEUgAAAAEAAAAECAIAAADAusJtAAAADklEQVR4nGNgZGJmQMIAAMg"
       "AGYgwva4AAAAASUVORK5CYII=",
       "c814bd1acd92fcf2f7e38558b07266882886117227a58838b493cdbbcda8c0db"},
  };
  for (const auto& [png, digest] : cases) {
    proto::VideoGenerationRequest request = condition_request();
    request.set_model("served-h3-alias");
    request.mutable_input()->mutable_media_sources(0)->set_base64(png);
    (*request.mutable_input()
          ->mutable_prompt_embed()
          ->mutable_parameters())["minimax_h3.reference_pixels_digest"]
        .set_string_param(digest);
    const DiTRequestParams params(request, "rid", "rtime");
    ASSERT_EQ(params.request_parse_status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(params.request_parse_status.message().find("reference"),
              std::string::npos);
    int32_t callbacks = 0;
    EXPECT_FALSE(params.verify_params([&callbacks](DiTRequestOutput output) {
      ++callbacks;
      EXPECT_TRUE(output.status.has_value());
      if (output.status.has_value()) {
        EXPECT_EQ(output.status->code(), StatusCode::INVALID_ARGUMENT);
      }
      return true;
    }));
    EXPECT_EQ(callbacks, 1);
  }
}

TEST_F(MiniMaxH3ConditioningTest, NativeReferenceGeometryIsNotQualified) {
  DiTConfig::get_instance().dit_h3_execution_profile("native");
  proto::VideoGenerationRequest request = condition_request();
  request.mutable_input()->mutable_prompt_embed()->clear_parameters();
  // Native parsing continues to accept a square reference. Its existing
  // visual processor, rather than the qualified ingress guard, owns geometry.
  request.mutable_input()->mutable_media_sources(0)->set_base64(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgZGIGAAAOAAfX"
      "b+R4AAAAAElFTkSuQmCC");
  const DiTRequestParams params(request, "rid", "rtime");
  ASSERT_TRUE(params.request_parse_status.ok());
  EXPECT_TRUE(params.verify_params([](DiTRequestOutput) { return true; }));
}

TEST_F(MiniMaxH3ConditioningTest,
       NativeAttestedRequestsKeepTheirExistingEarlyRejection) {
  DiTConfig::get_instance().dit_h3_execution_profile("native");
  for (const int32_t candidates : {1, 2}) {
    proto::VideoGenerationRequest request = condition_request();
    request.mutable_parameters()->set_num_videos_per_prompt(candidates);
    const DiTRequestParams params(request, "rid", "rtime");
    EXPECT_EQ(params.request_parse_status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(params.request_parse_status.message(),
              "qualified MiniMax-H3 ref2va profile is not enabled");
    EXPECT_EQ(params.input_params.media_sources.size(), 0u);
    EXPECT_EQ(params.input_params.tensor_sources.size(), 0u);
  }
}

TEST(MiniMaxH3ReferenceGeometryTest, FrozenCanvasAndNativeRounding) {
  EXPECT_EQ(minimax_h3_reference_image_size(64, 173, 2048, 32),
            (std::pair<int64_t, int64_t>{2048, 5536}));
  EXPECT_EQ(minimax_h3_reference_image_size(1, 1, 2048, 32),
            (std::pair<int64_t, int64_t>{2048, 2048}));
  EXPECT_EQ(minimax_h3_reference_image_size(1, 4, 2048, 32),
            (std::pair<int64_t, int64_t>{2048, 8192}));
  EXPECT_EQ(minimax_h3_reference_image_size(4, 1, 2048, 32),
            (std::pair<int64_t, int64_t>{8192, 2048}));
  // nearbyint ties-to-even, preserving the native operation order.
  EXPECT_EQ(minimax_h3_reference_image_size(128, 129, 2048, 32),
            (std::pair<int64_t, int64_t>{2048, 2048}));
  EXPECT_EQ(minimax_h3_reference_image_size(128, 131, 2048, 32),
            (std::pair<int64_t, int64_t>{2048, 2112}));
}

}  // namespace
}  // namespace xllm
