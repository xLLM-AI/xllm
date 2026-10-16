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

#include <cmath>
#include <optional>

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

constexpr int64_t kMaxH3FrameAlignment = 16;

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
  std::optional<TensorParameters> parameters =
      proto_to_tensor_parameters(proto_tensor);
  if (!parameters.has_value()) {
    request_parse_status =
        Status(StatusCode::INVALID_ARGUMENT,
               "invalid " + std::string(name) + " tensor parameters");
    return false;
  }
  tensor_sources.add(
      std::string(name), std::move(tensor), std::move(*parameters));
  return true;
}

DiTSourceDecoder::DecodeFn media_decode_fn(int32_t audio_channels,
                                           int64_t audio_sampling_rate) {
  return [audio_channels, audio_sampling_rate](
             std::string_view name,
             std::string_view modality,
             std::string_view raw_bytes,
             std::vector<MediaNamedTensor>& outputs) {
    if (modality == "audio") {
      torch::Tensor tensor;
      AudioMetadata metadata;
      FFmpegAudioDecoder decoder;
      if (!decoder.decode(raw_bytes,
                          tensor,
                          metadata,
                          audio_sampling_rate,
                          audio_channels)) {
        return false;
      }
      if (tensor.dim() == 1) {
        tensor = tensor.unsqueeze(0);
      }
      outputs.emplace_back(MediaNamedTensor{.name = std::string(name),
                                            .modality = "audio",
                                            .tensor = std::move(tensor)});
      return true;
    }
    if (modality == "video") {
      torch::Tensor tensor;
      VideoMetadata metadata;
      FFmpegVideoDecoder decoder;
      if (!decoder.decode(raw_bytes, tensor, metadata)) {
        return false;
      }
      TensorParameters parameters;
      parameters.emplace("prompt_video_fps", metadata.fps);
      outputs.emplace_back(
          MediaNamedTensor{.name = std::string(name),
                           .modality = "video",
                           .tensor = std::move(tensor),
                           .parameters = std::move(parameters)});
      return true;
    }
    if (modality == "image") {
      torch::Tensor tensor;
      OpenCVImageDecoder decoder;
      if (!decoder.decode(raw_bytes, tensor)) {
        return false;
      }
      outputs.emplace_back(MediaNamedTensor{.name = std::string(name),
                                            .modality = "image",
                                            .tensor = std::move(tensor)});
      return true;
    }
    return false;
  };
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

// Applies output_type and rejects values the response encoder cannot emit;
// shared by the image/audio/video constructors.
template <typename ParamsProto>
void apply_output_type(std::string& output_type,
                       const ParamsProto& params,
                       Status& request_parse_status) {
  if (params.has_output_type()) {
    output_type = params.output_type();
  }
  if (output_type != "base64" && output_type != "binary") {
    request_parse_status =
        Status(StatusCode::INVALID_ARGUMENT,
               "output_type must be either base64 or binary");
  }
}

template <typename ParamsProto>
void apply_seed(DiTGenerationParams& generation_params,
                const ParamsProto& params) {
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
}

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
  apply_seed(generation_params, params);
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
}

}  // namespace

// Rejects MiniMax-H3 video requests whose client-controlled inputs the
// pipeline cannot serve, at the request boundary instead of crashing the
// worker inside forward. Returns the reason, or nullptr when valid.
const char* minimax_h3_video_reject_reason(
    const std::string& model,
    const DiTInputParams& input_params,
    const DiTGenerationParams& generation_params) {
  if (model.find("MiniMax-H3") == std::string::npos &&
      model.find("MiniMaxH3") == std::string::npos) {
    return nullptr;
  }
  if (generation_params.num_videos_per_prompt != 1) {
    return "MiniMax-H3 currently supports num_videos_per_prompt=1";
  }
  if (generation_params.num_frames <= 0) {
    return "MiniMax-H3 num_frames must be positive";
  }
  if (!std::isfinite(generation_params.video_fps) ||
      generation_params.video_fps <= 0.0) {
    return "MiniMax-H3 fps must be finite and positive";
  }
  const double duration = static_cast<double>(generation_params.num_frames) /
                          generation_params.video_fps;
  const double max_aligned_duration =
      static_cast<double>(generation_params.num_frames + kMaxH3FrameAlignment) /
      generation_params.video_fps;
  if (duration < 5.0 || max_aligned_duration > 15.0) {
    return "MiniMax-H3 duration must be between 5 and 15 seconds";
  }
  const bool is_ref2va =
      DiTConfig::get_instance().dit_h3_task_type() == "ref2va";
  size_t image_count = 0;
  size_t video_count = 0;
  size_t audio_count = 0;
  size_t video_audio_count = 0;
  const auto& sources = input_params.media_sources.entries();
  for (size_t index = 0; index < sources.size(); ++index) {
    const MediaNamedTensor& source = sources[index];
    if (source.name == "image") {
      if (source.modality != "image") {
        return "MiniMax-H3 image source must use image modality";
      }
      ++image_count;
    } else if (source.name == "prompt_video") {
      if (source.modality != "video") {
        return "MiniMax-H3 prompt_video source must use video modality";
      }
      ++video_count;
    } else if (source.name == "prompt_audio") {
      if (source.modality != "audio") {
        return "MiniMax-H3 prompt_audio source must use audio modality";
      }
      ++audio_count;
    } else if (source.name == "prompt_audio_in_video") {
      if (source.modality != "audio" || index == 0 ||
          sources[index - 1].name != "prompt_video") {
        return "MiniMax-H3 embedded audio must follow a prompt_video";
      }
      ++video_audio_count;
    } else {
      return "MiniMax-H3 unsupported media source name";
    }
  }
  if (is_ref2va) {
    if (image_count > 9 || video_count > 1 || audio_count > 1 ||
        video_audio_count > video_count) {
      return "MiniMax-H3 Ref2VA accepts at most 9 images, 1 video, 1 audio";
    }
    return nullptr;
  }
  if (image_count > 2 || video_count > 0 || audio_count > 0 ||
      video_audio_count > 0) {
    return "MiniMax-H3 FL2VA accepts at most two images";
  }
  return nullptr;
}

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
    if (!image_decoder.add_sources(input.media_sources(),
                                   request_parse_status)) {
      return;
    }
    image_decoder.add_sources(input.images(),
                              /*default_name=*/"unknown");
    std::vector<MediaNamedTensor> decoded_images;
    if (!image_decoder.decode(
            media_decode_fn(/*audio_channels=*/2,
                            generation_params.audio_sampling_rate),
            decoded_images,
            request_parse_status)) {
      return;
    }
    input_params.media_sources.add_decoded(std::move(decoded_images));
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
    apply_output_type(output_type, params, request_parse_status);
    if (!request_parse_status.ok()) {
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
    apply_seed(generation_params, params);
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

  // Generation params first: audio_sampling_rate is needed below when decoding
  // prompt audio.
  const auto& params = request.parameters();
  apply_seed(generation_params, params);
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
  apply_output_type(output_type, params, request_parse_status);
  if (!request_parse_status.ok()) {
    return;
  }

  // input params — decode prompt audio using the model's sampling rate.
  const auto& input = request.input();
  input_params.prompt = input.prompt();

  if (input.has_prompt_audio()) {
    DiTSourceDecoder audio_decoder(request_payload);
    proto::MediaSource prompt_audio = input.prompt_audio();
    if (prompt_audio.modality().empty()) {
      prompt_audio.set_modality("audio");
    } else if (prompt_audio.modality() != "audio") {
      request_parse_status = Status(StatusCode::INVALID_ARGUMENT,
                                    "prompt_audio modality must be audio");
      return;
    }
    if (!audio_decoder.add_source(prompt_audio, request_parse_status)) {
      return;
    }
    std::vector<MediaNamedTensor> decoded_audio;
    if (!audio_decoder.decode(
            media_decode_fn(/*audio_channels=*/1,
                            generation_params.audio_sampling_rate),
            decoded_audio,
            request_parse_status)) {
      return;
    }
    MediaNamedTensor& audio = decoded_audio.front();
    input_params.media_sources.add("prompt_audio",
                                   std::move(audio.modality),
                                   std::move(audio.tensor),
                                   std::move(audio.parameters));
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
  generation_params.audio_sampling_rate = 32000;
  if (request.has_parameters() && request.parameters().has_sampling_rate()) {
    generation_params.audio_sampling_rate =
        request.parameters().sampling_rate();
  }

  if (request.has_input()) {
    const auto& input = request.input();
    if (!fill_input_params(
            input_params, input, request_payload, request_parse_status)) {
      return;
    }
    DiTSourceDecoder media_decoder(request_payload);
    if (!media_decoder.add_sources(input.media_sources(),
                                   request_parse_status)) {
      return;
    }
    std::vector<MediaNamedTensor> decoded_sources;
    if (!media_decoder.decode(
            media_decode_fn(/*audio_channels=*/2,
                            generation_params.audio_sampling_rate),
            decoded_sources,
            request_parse_status)) {
      return;
    }
    input_params.media_sources.add_decoded(std::move(decoded_sources));
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
    apply_output_type(output_type, params, request_parse_status);
    if (!request_parse_status.ok()) {
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

  if (const char* reject_reason = minimax_h3_video_reject_reason(
          model, input_params, generation_params)) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, reject_reason);
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
