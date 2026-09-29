/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_request_params.h"

#include "core/common/instance_name.h"
#include "core/common/macros.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/multimodal/mm_codec.h"
#include "core/framework/request/dit_source_decoder.h"
#include "core/util/utils.h"
#include "core/util/uuid.h"
#include "request.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;

std::string generate_request_id(const std::string& prefix) {
  return prefix + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

std::pair<int, int> split_resolution(const std::string& s) {
  size_t pos = s.find('*');
  int width = std::stoi(s.substr(0, pos));
  int height = std::stoi(s.substr(pos + 1));
  return {width, height};
}

bool decode_tensor_input(std::string_view name,
                         const proto::Tensor& proto_tensor,
                         const BinaryPayload& request_payload,
                         DiTTensorSources& tensor_sources,
                         Status& request_parse_status) {
  torch::Tensor tensor = util::proto_to_torch(proto_tensor, request_payload);
  if (!tensor.defined()) {
    request_parse_status = Status(StatusCode::INVALID_ARGUMENT,
                                  "invalid " + std::string(name) + " tensor");
    return false;
  }
  tensor_sources.add(std::string(name), std::move(tensor));
  return true;
}

template <typename InputProto>
bool fill_input_params(DiTInputParams& input_params,
                       const InputProto& input,
                       const BinaryPayload& request_payload,
                       Status& request_parse_status) {
  input_params.prompt = input.prompt();
  if (input.has_negative_prompt()) {
    input_params.negative_prompt = input.negative_prompt();
  }
  if (input.has_prompt_embed() &&
      !decode_tensor_input("prompt_embed",
                           input.prompt_embed(),
                           request_payload,
                           input_params.tensor_sources,
                           request_parse_status)) {
    return false;
  }
  if (input.has_negative_prompt_embed() &&
      !decode_tensor_input("negative_prompt_embed",
                           input.negative_prompt_embed(),
                           request_payload,
                           input_params.tensor_sources,
                           request_parse_status)) {
    return false;
  }
  return true;
}

// Shared helper: populate generation params common to Image and Video
// parameters.
template <typename ParamsProto>
void fill_generation_params(DiTGenerationParams& generation_params,
                            const ParamsProto& params) {
  if (params.has_size()) {
    auto [w, h] = split_resolution(params.size());
    generation_params.width = w;
    generation_params.height = h;
  }
  if (params.has_num_inference_steps()) {
    generation_params.num_inference_steps = params.num_inference_steps();
  }
  if (params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = params.true_cfg_scale();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Image generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::ImageGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime,
                                   const BinaryPayload& request_payload) {
  request_kind = DiTRequestKind::kImage;
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("imggen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  if (request.has_input()) {
    const auto& input = request.input();
    if (!fill_input_params(
            input_params, input, request_payload, request_parse_status)) {
      return;
    }
    // Image-only input fields
    if (input.has_prompt_2()) {
      input_params.prompt_2 = input.prompt_2();
    }
    if (input.has_negative_prompt_2()) {
      input_params.negative_prompt_2 = input.negative_prompt_2();
    }
    if (input.has_pooled_prompt_embed() &&
        !decode_tensor_input("pooled_prompt_embed",
                             input.pooled_prompt_embed(),
                             request_payload,
                             input_params.tensor_sources,
                             request_parse_status)) {
      return;
    }
    if (input.has_negative_pooled_prompt_embed() &&
        !decode_tensor_input("negative_pooled_prompt_embed",
                             input.negative_pooled_prompt_embed(),
                             request_payload,
                             input_params.tensor_sources,
                             request_parse_status)) {
      return;
    }
    if (input.has_latent() && !decode_tensor_input("latent",
                                                   input.latent(),
                                                   request_payload,
                                                   input_params.tensor_sources,
                                                   request_parse_status)) {
      return;
    }
    if (input.has_masked_image_latent() &&
        !decode_tensor_input("masked_image_latent",
                             input.masked_image_latent(),
                             request_payload,
                             input_params.tensor_sources,
                             request_parse_status)) {
      return;
    }
    DiTSourceDecoder image_decoder(request_payload);
    if (!image_decoder.add_sources(input.image_sources(),
                                   /*default_name=*/"unknown",
                                   request_parse_status)) {
      return;
    }
    image_decoder.add_sources(input.images(),
                              /*default_name=*/"unknown");
    std::vector<NamedTensor> decoded_images;
    const auto decode_image = [](std::string_view raw_bytes,
                                 torch::Tensor& tensor) {
      OpenCVImageDecoder decoder;
      return decoder.decode(raw_bytes, tensor);
    };
    if (!image_decoder.decode(
            decode_image, decoded_images, request_parse_status)) {
      return;
    }
    for (NamedTensor& image : decoded_images) {
      input_params.image_sources.add(std::move(image.name),
                                     std::move(image.tensor));
    }
  }

  generation_params.num_images_per_prompt = 1;
  if (request.has_parameters()) {
    const auto& params = request.parameters();
    fill_generation_params(generation_params, params);
    if (params.has_num_images_per_prompt()) {
      generation_params.num_images_per_prompt =
          static_cast<uint32_t>(params.num_images_per_prompt());
    }
    // Image-only generation fields
    if (params.has_enable_cfg_renorm()) {
      generation_params.enable_cfg_renorm = params.enable_cfg_renorm();
    }
    if (params.has_cfg_renorm_min()) {
      generation_params.cfg_renorm_min = params.cfg_renorm_min();
    }
    if (params.has_output_type()) {
      output_type = params.output_type();
    }
    if (output_type != "base64" && output_type != "binary") {
      request_parse_status =
          Status(StatusCode::INVALID_ARGUMENT,
                 "output_type must be either base64 or binary");
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// Text generation constructor (Cola-DLM)
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::TextGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_kind = DiTRequestKind::kText;
  // Cola-DLM official defaults (inference.py / run_cola.py), not image DiT.
  generation_params.guidance_scale = 7.0f;
  generation_params.max_new_tokens = 32;
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("textgen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  if (request.has_input()) {
    input_params.prompt = request.input().prompt();
  }

  if (request.has_parameters()) {
    const auto& params = request.parameters();
    if (params.has_seed()) {
      generation_params.seed = params.seed();
      generation_params.seed_is_set = true;
    }
    if (params.has_max_new_tokens()) {
      generation_params.max_new_tokens = params.max_new_tokens();
    }
    if (params.has_diffusion_steps()) {
      generation_params.diffusion_steps = params.diffusion_steps();
    }
    if (params.has_guidance_scale()) {
      generation_params.guidance_scale = params.guidance_scale();
    }
    if (params.has_temperature()) {
      generation_params.temperature = params.temperature();
    }
    if (params.has_top_k()) {
      generation_params.top_k = params.top_k();
    }
    if (params.has_top_p()) {
      generation_params.top_p = params.top_p();
    }
    if (params.has_repetition_penalty()) {
      generation_params.repetition_penalty = params.repetition_penalty();
    }
  }
}

// ---------------------------------------------------------------------------
// Audio generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::AudioGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime,
                                   const BinaryPayload& request_payload) {
  request_kind = DiTRequestKind::kAudio;
  if (request.has_request_id()) {
    request_id = request.request_id();
  } else {
    request_id = generate_request_id("audiogen-");
  }
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  // generation params — must be parsed before input to make audio_sampling_rate
  // available for prompt audio decoding.
  const auto& params = request.parameters();
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_audio_duration_frames()) {
    generation_params.audio_duration_frames = params.audio_duration_frames();
  }
  if (params.has_audio_steps()) {
    generation_params.audio_steps = params.audio_steps();
  }
  if (params.has_audio_guidance_method()) {
    generation_params.audio_guidance_method = params.audio_guidance_method();
  }
  if (params.has_sampling_rate()) {
    generation_params.audio_sampling_rate = params.sampling_rate();
  }
  if (params.has_output_type()) {
    output_type = params.output_type();
  }
  if (output_type != "base64" && output_type != "binary") {
    request_parse_status =
        Status(StatusCode::INVALID_ARGUMENT,
               "output_type must be either base64 or binary");
    return;
  }

  // input params — decode prompt audio using the model's sampling rate.
  const auto& input = request.input();
  input_params.prompt = input.prompt();

  if (input.has_prompt_audio()) {
    DiTSourceDecoder audio_decoder(request_payload);
    if (!audio_decoder.add_source(input.prompt_audio(),
                                  /*default_name=*/"prompt_audio",
                                  request_parse_status)) {
      return;
    }
    std::vector<NamedTensor> decoded_audio;
    const int64_t sample_rate = generation_params.audio_sampling_rate;
    const auto decode_audio = [sample_rate](std::string_view raw_bytes,
                                            torch::Tensor& tensor) {
      FFmpegAudioDecoder decoder;
      AudioMetadata metadata;
      torch::Tensor decoded;
      if (!decoder.decode(raw_bytes, decoded, metadata, sample_rate)) {
        LOG(ERROR) << "prompt_audio decode failed";
        return false;
      }
      if (decoded.dim() == 1) {
        decoded = decoded.unsqueeze(0);
      } else if (decoded.dim() == 2 && decoded.size(0) > 1) {
        decoded = decoded.mean(0, /*keepdim=*/true);
      }
      tensor = decoded.to(torch::kFloat32);
      return true;
    };
    if (!audio_decoder.decode(
            decode_audio, decoded_audio, request_parse_status)) {
      return;
    }
    input_params.tensor_sources.add("prompt_audio",
                                    std::move(decoded_audio.front().tensor));
  }
  if (input.has_prompt_text()) {
    input_params.audio_prompt_text = input.prompt_text();
  }
}

// ---------------------------------------------------------------------------
// Video generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::VideoGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime,
                                   const BinaryPayload& request_payload) {
  request_kind = DiTRequestKind::kVideo;
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("vidgen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  if (request.has_input()) {
    const auto& input = request.input();
    if (!fill_input_params(
            input_params, input, request_payload, request_parse_status)) {
      return;
    }
    DiTSourceDecoder image_decoder(request_payload);
    if (!image_decoder.add_sources(input.image_sources(),
                                   /*default_name=*/"unknown",
                                   request_parse_status)) {
      return;
    }
    std::vector<NamedTensor> decoded_images;
    const auto decode_image = [](std::string_view raw_bytes,
                                 torch::Tensor& tensor) {
      OpenCVImageDecoder decoder;
      return decoder.decode(raw_bytes, tensor);
    };
    if (!image_decoder.decode(
            decode_image, decoded_images, request_parse_status)) {
      return;
    }
    for (NamedTensor& image : decoded_images) {
      input_params.image_sources.add(std::move(image.name),
                                     std::move(image.tensor));
    }
  }

  if (request.has_parameters()) {
    const auto& params = request.parameters();
    fill_generation_params(generation_params, params);
    if (params.has_num_videos_per_prompt()) {
      generation_params.num_videos_per_prompt =
          static_cast<uint32_t>(params.num_videos_per_prompt());
    }
    if (params.has_num_frames()) {
      generation_params.num_frames = params.num_frames();
    }
    if (params.has_fps()) {
      generation_params.video_fps = params.fps();
    }
    if (params.has_guidance_scale_2()) {
      generation_params.guidance_scale_2 = params.guidance_scale_2();
    }
    if (params.has_seconds()) {
      generation_params.seconds = params.seconds();
    }
    if (params.has_boundary_ratio()) {
      generation_params.boundary_ratio = params.boundary_ratio();
    }
    if (params.has_flow_shift()) {
      generation_params.flow_shift = params.flow_shift();
    }
    if (params.has_output_type()) {
      output_type = params.output_type();
    }
    if (output_type != "base64" && output_type != "binary") {
      request_parse_status =
          Status(StatusCode::INVALID_ARGUMENT,
                 "output_type must be either base64 or binary");
      return;
    }
  }
}

bool DiTRequestParams::verify_params(
    std::function<bool(DiTRequestOutput)> callback) const {
  if (!request_parse_status.ok()) {
    callback(DiTRequestOutput(
        Status(request_parse_status.code(), request_parse_status.message())));
    return false;
  }

  if (input_params.prompt.empty() &&
      !input_params.tensor_sources.contains("prompt_embed")) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "prompt is empty");
    return false;
  }

  if (request_kind == DiTRequestKind::kText) {
    if (model.empty()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "model is empty");
      return false;
    }
    if (generation_params.max_new_tokens <= 0) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "max_new_tokens must be positive.");
      return false;
    }
    if (generation_params.diffusion_steps <= 0) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "diffusion_steps must be positive.");
      return false;
    }
    return true;
  }

  if (request_kind == DiTRequestKind::kAudio) {
    return true;
  }

  if (generation_params.width <= 0 || generation_params.height <= 0) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Invalid image dimensions: width and height must be positive.");
    return false;
  }

  if (generation_params.num_inference_steps <= 0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "num_inference_steps must be positive.");
    return false;
  }

  if (generation_params.num_images_per_prompt == 0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "num_images_per_prompt must be greater than 0.");
    return false;
  }

  if (generation_params.num_videos_per_prompt == 0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "num_videos_per_prompt must be greater than 0.");
    return false;
  }

  // Check if the image area exceeds the maximum allowed area.
  if (::xllm::DiTConfig::get_instance().dit_generation_image_area_max() > 0) {
    int64_t area = static_cast<int64_t>(generation_params.width) *
                   static_cast<int64_t>(generation_params.height);
    if (area >
        ::xllm::DiTConfig::get_instance().dit_generation_image_area_max()) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "Requested image area (" + std::to_string(area) +
              ") exceeds the maximum allowed area (" +
              std::to_string(::xllm::DiTConfig::get_instance()
                                 .dit_generation_image_area_max()) +
              ").");
      return false;
    }
  }

  return true;
}

}  // namespace xllm
