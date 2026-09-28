/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/request/dit_request_params.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "audio_generation.pb.h"
#include "butil/base64.h"
#include "image_generation.pb.h"
#include "text_generation.pb.h"
#include "video_generation.pb.h"

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

constexpr char kTinyPngBase64[] =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQIHWP4z8AAAAMB"
    "AQBb2/lEAAAAAElFTkSuQmCC";

const std::string& TinyPngBytes() {
  static const std::string bytes = [] {
    std::string result;
    CHECK(butil::Base64Decode(kTinyPngBase64, &result));
    return result;
  }();
  return bytes;
}

void AppendUint16Le(uint16_t value, std::string& bytes) {
  bytes.push_back(static_cast<char>(value & 0xff));
  bytes.push_back(static_cast<char>((value >> 8) & 0xff));
}

void AppendUint32Le(uint32_t value, std::string& bytes) {
  bytes.push_back(static_cast<char>(value & 0xff));
  bytes.push_back(static_cast<char>((value >> 8) & 0xff));
  bytes.push_back(static_cast<char>((value >> 16) & 0xff));
  bytes.push_back(static_cast<char>((value >> 24) & 0xff));
}

const std::string& TinyWavBytes() {
  static const std::string bytes = [] {
    constexpr uint32_t kSampleRate = 24000;
    constexpr uint16_t kChannelCount = 1;
    constexpr uint16_t kBitsPerSample = 16;
    constexpr int16_t kSamples[] = {0, 1024, -1024, 0};
    constexpr uint32_t kDataSize = sizeof(kSamples);

    std::string result;
    result.reserve(44 + kDataSize);
    result.append("RIFF", 4);
    AppendUint32Le(36 + kDataSize, result);
    result.append("WAVEfmt ", 8);
    AppendUint32Le(16, result);
    AppendUint16Le(/*value=*/1, result);
    AppendUint16Le(kChannelCount, result);
    AppendUint32Le(kSampleRate, result);
    AppendUint32Le(kSampleRate * kChannelCount * kBitsPerSample / 8, result);
    AppendUint16Le(kChannelCount * kBitsPerSample / 8, result);
    AppendUint16Le(kBitsPerSample, result);
    result.append("data", 4);
    AppendUint32Le(kDataSize, result);
    for (const int16_t sample : kSamples) {
      AppendUint16Le(static_cast<uint16_t>(sample), result);
    }
    return result;
  }();
  return bytes;
}

proto::AudioGenerationRequest MakeAudioRequest() {
  proto::AudioGenerationRequest request;
  request.set_model("LongCat-AudioDiT");
  request.mutable_input()->set_prompt("generate speech");
  request.mutable_parameters()->set_sampling_rate(24000);
  return request;
}

proto::ImageGenerationRequest MakeImageRequest() {
  proto::ImageGenerationRequest request;
  request.set_model("QwenImageEditPlus");
  request.mutable_input()->set_prompt("edit image");
  return request;
}

proto::VideoGenerationRequest MakeVideoRequest() {
  proto::VideoGenerationRequest request;
  request.set_model("WanI2V");
  request.mutable_input()->set_prompt("animate image");
  return request;
}

bool VerifyFailsWithInvalidArgument(const DiTRequestParams& params) {
  StatusCode error_code = StatusCode::OK;
  const bool valid = params.verify_params([&](DiTRequestOutput output) {
    if (output.status.has_value()) {
      error_code = output.status->code();
    }
    return true;
  });
  return !valid && error_code == StatusCode::INVALID_ARGUMENT;
}

TEST(DiTRequestParamsTest, LegacyImagesMapToOrderedUnknownSources) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  request.mutable_input()->add_images(kTinyPngBase64);
  request.mutable_input()->add_images(kTinyPngBase64);

  DiTRequestParams params(request, "rid", "rtime");

  ASSERT_TRUE(params.input_status.ok());
  ASSERT_EQ(params.input_params.image_sources.size(), 2u);
  EXPECT_EQ(params.input_params.image_sources.at(0).name, "unknown");
  EXPECT_EQ(params.input_params.image_sources.at(1).name, "unknown");
  EXPECT_TRUE(torch::equal(params.input_params.image_sources.at(0).tensor,
                           params.input_params.image_sources.at(1).tensor));
}

TEST(DiTRequestParamsTest, ImageSourcesSupportBase64AndBinaryInOrder) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Input* input = request.mutable_input();
  proto::MediaSource* base64 = input->add_image_sources();
  base64->set_type("base64");
  base64->set_base64(kTinyPngBase64);
  proto::MediaSource* binary = input->add_image_sources();
  binary->set_type("binary");
  binary->set_name("control_image");
  binary->mutable_binary()->set_offset(2);
  binary->mutable_binary()->set_length(TinyPngBytes().size());
  const std::string payload = "xx" + TinyPngBytes();

  DiTRequestParams params(request, "rid", "rtime", payload);

  ASSERT_TRUE(params.input_status.ok());
  ASSERT_EQ(params.input_params.image_sources.size(), 2u);
  EXPECT_EQ(params.input_params.image_sources.at(0).name, "unknown");
  EXPECT_EQ(params.input_params.image_sources.at(1).name, "control_image");
  EXPECT_TRUE(torch::equal(params.input_params.image_sources.at(0).tensor,
                           params.input_params.image_sources.at(1).tensor));
}

TEST(DiTRequestParamsTest, VideoImageSourcesAreDecoded) {
  proto::VideoGenerationRequest request = MakeVideoRequest();
  proto::VideoInput* input = request.mutable_input();
  const size_t image_size = TinyPngBytes().size();
  proto::MediaSource* first = input->add_image_sources();
  first->set_type("binary");
  first->set_name("image");
  first->mutable_binary()->set_length(image_size);
  proto::MediaSource* last = input->add_image_sources();
  last->set_type("binary");
  last->set_name("last_image");
  last->mutable_binary()->set_offset(image_size);
  last->mutable_binary()->set_length(image_size);
  const std::string payload = TinyPngBytes() + TinyPngBytes();

  DiTRequestParams params(request, "rid", "rtime", payload);

  ASSERT_TRUE(params.input_status.ok());
  std::vector<torch::Tensor> images =
      params.input_params.image_sources.get({"image", "last_image"});
  EXPECT_EQ(images.size(), 2u);
}

TEST(DiTRequestParamsTest, MixedImageSourcesAndLegacyImagesPreserveOrder) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Input* input = request.mutable_input();
  proto::MediaSource* base64 = input->add_image_sources();
  base64->set_type("base64");
  base64->set_name("first");
  base64->set_base64(kTinyPngBase64);
  proto::MediaSource* binary = input->add_image_sources();
  binary->set_type("binary");
  binary->set_name("second");
  binary->mutable_binary()->set_length(TinyPngBytes().size());
  input->add_images(kTinyPngBase64);

  DiTRequestParams params(request, "rid", "rtime", TinyPngBytes());

  ASSERT_TRUE(params.input_status.ok());
  ASSERT_EQ(params.input_params.image_sources.size(), 3u);
  EXPECT_EQ(params.input_params.image_sources.at(0).name, "first");
  EXPECT_EQ(params.input_params.image_sources.at(1).name, "second");
  EXPECT_EQ(params.input_params.image_sources.at(2).name, "unknown");
  EXPECT_TRUE(torch::equal(params.input_params.image_sources.at(0).tensor,
                           params.input_params.image_sources.at(1).tensor));
  EXPECT_TRUE(torch::equal(params.input_params.image_sources.at(1).tensor,
                           params.input_params.image_sources.at(2).tensor));
}

TEST(DiTRequestParamsTest, FailedParallelImageDecodeCommitsNoSources) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Input* input = request.mutable_input();
  proto::MediaSource* valid = input->add_image_sources();
  valid->set_type("base64");
  valid->set_name("valid");
  valid->set_base64(kTinyPngBase64);
  proto::MediaSource* invalid = input->add_image_sources();
  invalid->set_type("base64");
  invalid->set_name("invalid");
  std::string invalid_base64;
  butil::Base64Encode("not an image", &invalid_base64);
  invalid->set_base64(invalid_base64);

  DiTRequestParams params(request, "rid", "rtime");

  EXPECT_TRUE(VerifyFailsWithInvalidArgument(params));
  EXPECT_TRUE(params.input_params.image_sources.empty());
}

TEST(DiTRequestParamsTest, DuplicateImageSourceNamesArePreserved) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Input* input = request.mutable_input();
  for (int32_t index = 0; index < 2; ++index) {
    proto::MediaSource* source = input->add_image_sources();
    source->set_type("base64");
    source->set_name("image");
    source->set_base64(kTinyPngBase64);
  }

  DiTRequestParams params(request, "rid", "rtime");

  ASSERT_TRUE(params.input_status.ok());
  EXPECT_EQ(params.input_params.image_sources.get({"image", "image"}).size(),
            2u);
}

TEST(DiTRequestParamsTest, RejectsOutOfBoundsBinaryImage) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::MediaSource* source = request.mutable_input()->add_image_sources();
  source->set_type("binary");
  proto::BinaryRef* binary = source->mutable_binary();
  binary->set_offset(TinyPngBytes().size());
  binary->set_length(1);

  DiTRequestParams params(request, "rid", "rtime", TinyPngBytes());

  EXPECT_TRUE(VerifyFailsWithInvalidArgument(params));
}

TEST(DiTRequestParamsTest, RejectsOverflowingBinaryImageRange) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::MediaSource* source = request.mutable_input()->add_image_sources();
  source->set_type("binary");
  proto::BinaryRef* binary = source->mutable_binary();
  binary->set_offset(1);
  binary->set_length(std::numeric_limits<uint64_t>::max());

  DiTRequestParams params(request, "rid", "rtime", TinyPngBytes());

  EXPECT_TRUE(VerifyFailsWithInvalidArgument(params));
}

TEST(DiTRequestParamsTest, RejectsEmptyImageSource) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  request.mutable_input()->add_image_sources();

  DiTRequestParams params(request, "rid", "rtime");

  EXPECT_TRUE(VerifyFailsWithInvalidArgument(params));
}

TEST(DiTRequestParamsTest, BinaryTensorInputMatchesContentsInput) {
  proto::ImageGenerationRequest contents_request = MakeImageRequest();
  proto::Tensor* contents_tensor =
      contents_request.mutable_input()->mutable_prompt_embed();
  contents_tensor->set_datatype("FP32");
  contents_tensor->add_shape(2);
  contents_tensor->mutable_contents()->add_fp32_contents(1.25f);
  contents_tensor->mutable_contents()->add_fp32_contents(-2.5f);
  DiTRequestParams contents_params(contents_request, "rid", "rtime");

  proto::ImageGenerationRequest binary_request = MakeImageRequest();
  proto::Tensor* binary_tensor =
      binary_request.mutable_input()->mutable_prompt_embed();
  binary_tensor->set_datatype("FP32");
  binary_tensor->add_shape(2);
  (*binary_tensor->mutable_parameters())["is_binary"].set_bool_param(true);
  (*binary_tensor->mutable_parameters())["offset"].set_int64_param(2);
  (*binary_tensor->mutable_parameters())["len"].set_int64_param(2 *
                                                                sizeof(float));
  const float values[] = {1.25f, -2.5f};
  std::string payload = "xx";
  payload.append(reinterpret_cast<const char*>(values), sizeof(values));
  DiTRequestParams binary_params(binary_request, "rid", "rtime", payload);

  ASSERT_TRUE(binary_params.input_status.ok());
  EXPECT_TRUE(torch::equal(
      *binary_params.input_params.tensor_sources.get("prompt_embed"),
      *contents_params.input_params.tensor_sources.get("prompt_embed")));
}

TEST(DiTRequestParamsTest, VideoBinaryTensorInputMatchesContentsInput) {
  proto::VideoGenerationRequest contents_request = MakeVideoRequest();
  proto::Tensor* contents_tensor =
      contents_request.mutable_input()->mutable_prompt_embed();
  contents_tensor->set_datatype("FP32");
  contents_tensor->add_shape(2);
  contents_tensor->mutable_contents()->add_fp32_contents(1.25f);
  contents_tensor->mutable_contents()->add_fp32_contents(-2.5f);
  DiTRequestParams contents_params(contents_request, "rid", "rtime");

  proto::VideoGenerationRequest binary_request = MakeVideoRequest();
  proto::Tensor* binary_tensor =
      binary_request.mutable_input()->mutable_prompt_embed();
  binary_tensor->set_datatype("FP32");
  binary_tensor->add_shape(2);
  (*binary_tensor->mutable_parameters())["is_binary"].set_bool_param(true);
  (*binary_tensor->mutable_parameters())["offset"].set_int64_param(2);
  (*binary_tensor->mutable_parameters())["len"].set_int64_param(2 *
                                                                sizeof(float));
  const float values[] = {1.25f, -2.5f};
  std::string payload = "xx";
  payload.append(reinterpret_cast<const char*>(values), sizeof(values));
  DiTRequestParams binary_params(binary_request, "rid", "rtime", payload);

  ASSERT_TRUE(binary_params.input_status.ok());
  EXPECT_TRUE(torch::equal(
      *binary_params.input_params.tensor_sources.get("prompt_embed"),
      *contents_params.input_params.tensor_sources.get("prompt_embed")));
}

TEST(DiTRequestParamsTest, ImageOutputTypeDefaultsAndValidates) {
  proto::ImageGenerationRequest default_request = MakeImageRequest();
  DiTRequestParams default_params(default_request, "rid", "rtime");
  EXPECT_EQ(default_params.output_type, "base64");

  proto::ImageGenerationRequest binary_request = MakeImageRequest();
  binary_request.mutable_parameters()->set_output_type("binary");
  DiTRequestParams binary_params(binary_request, "rid", "rtime");
  EXPECT_EQ(binary_params.output_type, "binary");
  EXPECT_TRUE(binary_params.input_status.ok());

  proto::ImageGenerationRequest invalid_request = MakeImageRequest();
  invalid_request.mutable_parameters()->set_output_type("url");
  DiTRequestParams invalid_params(invalid_request, "rid", "rtime");
  EXPECT_TRUE(VerifyFailsWithInvalidArgument(invalid_params));
}

TEST(DiTRequestParamsTest, VideoOutputTypeDefaultsAndValidates) {
  proto::VideoGenerationRequest default_request = MakeVideoRequest();
  DiTRequestParams default_params(default_request, "rid", "rtime");
  EXPECT_EQ(default_params.output_type, "base64");

  proto::VideoGenerationRequest binary_request = MakeVideoRequest();
  binary_request.mutable_parameters()->set_output_type("binary");
  DiTRequestParams binary_params(binary_request, "rid", "rtime");
  EXPECT_EQ(binary_params.output_type, "binary");
  EXPECT_TRUE(binary_params.input_status.ok());

  proto::VideoGenerationRequest invalid_request = MakeVideoRequest();
  invalid_request.mutable_parameters()->set_output_type("url");
  DiTRequestParams invalid_params(invalid_request, "rid", "rtime");
  EXPECT_TRUE(VerifyFailsWithInvalidArgument(invalid_params));
}

TEST(DiTRequestParamsTest, PromptAudioSupportsBase64AndBinary) {
  std::string wav_base64;
  butil::Base64Encode(TinyWavBytes(), &wav_base64);

  proto::AudioGenerationRequest base64_request = MakeAudioRequest();
  proto::MediaSource* base64_source =
      base64_request.mutable_input()->mutable_prompt_audio();
  base64_source->set_type("base64");
  base64_source->set_base64(wav_base64);
  DiTRequestParams base64_params(base64_request, "rid", "rtime");

  proto::AudioGenerationRequest binary_request = MakeAudioRequest();
  proto::MediaSource* binary_source =
      binary_request.mutable_input()->mutable_prompt_audio();
  binary_source->set_type("binary");
  binary_source->mutable_binary()->set_offset(2);
  binary_source->mutable_binary()->set_length(TinyWavBytes().size());
  DiTRequestParams binary_params(
      binary_request, "rid", "rtime", "xx" + TinyWavBytes());

  ASSERT_TRUE(base64_params.input_status.ok());
  ASSERT_TRUE(binary_params.input_status.ok());
  const torch::Tensor base64_audio =
      *base64_params.input_params.tensor_sources.get("prompt_audio");
  EXPECT_EQ(base64_audio.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::equal(
      base64_audio,
      *binary_params.input_params.tensor_sources.get("prompt_audio")));
}

TEST(DiTRequestParamsTest, RejectsOutOfBoundsBinaryPromptAudio) {
  proto::AudioGenerationRequest request = MakeAudioRequest();
  proto::MediaSource* source = request.mutable_input()->mutable_prompt_audio();
  source->set_type("binary");
  source->mutable_binary()->set_offset(1);
  source->mutable_binary()->set_length(std::numeric_limits<uint64_t>::max());

  DiTRequestParams params(request, "rid", "rtime", TinyWavBytes());

  EXPECT_TRUE(VerifyFailsWithInvalidArgument(params));
}

TEST(DiTRequestParamsTest, AudioOutputTypeDefaultsAndValidates) {
  proto::AudioGenerationRequest default_request = MakeAudioRequest();
  DiTRequestParams default_params(default_request, "rid", "rtime");
  EXPECT_EQ(default_params.output_type, "base64");

  proto::AudioGenerationRequest binary_request = MakeAudioRequest();
  binary_request.mutable_parameters()->set_output_type("binary");
  DiTRequestParams binary_params(binary_request, "rid", "rtime");
  EXPECT_EQ(binary_params.output_type, "binary");
  EXPECT_TRUE(binary_params.input_status.ok());

  proto::AudioGenerationRequest invalid_request = MakeAudioRequest();
  invalid_request.mutable_parameters()->set_output_type("url");
  DiTRequestParams invalid_params(invalid_request, "rid", "rtime");
  EXPECT_TRUE(VerifyFailsWithInvalidArgument(invalid_params));
}

TEST(DiTRequestParamsTest, RejectsBinaryTensorLengthMismatch) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::Tensor* tensor = request.mutable_input()->mutable_prompt_embed();
  tensor->set_datatype("FP32");
  tensor->add_shape(2);
  (*tensor->mutable_parameters())["is_binary"].set_bool_param(true);
  (*tensor->mutable_parameters())["offset"].set_int64_param(0);
  (*tensor->mutable_parameters())["len"].set_int64_param(sizeof(float));

  DiTRequestParams params(
      request, "rid", "rtime", std::string(sizeof(float), 0));

  EXPECT_TRUE(VerifyFailsWithInvalidArgument(params));
}

TEST(DiTRequestParamsTest, RejectsMissingOrInvalidBinaryImagePayload) {
  proto::ImageGenerationRequest request = MakeImageRequest();
  proto::MediaSource* source = request.mutable_input()->add_image_sources();
  source->set_type("binary");
  source->mutable_binary()->set_length(4);

  DiTRequestParams missing_params(request, "rid", "rtime");
  DiTRequestParams invalid_params(request, "rid", "rtime", "nope");

  EXPECT_TRUE(VerifyFailsWithInvalidArgument(missing_params));
  EXPECT_TRUE(VerifyFailsWithInvalidArgument(invalid_params));
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
