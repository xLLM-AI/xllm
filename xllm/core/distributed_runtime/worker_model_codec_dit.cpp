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

#include <cstdint>
#include <utility>

#include "core/common/metrics.h"
#include "core/distributed_runtime/worker_model_codec.h"
#include "core/util/tensor_codec.h"
#include "core/util/timer.h"
#include "core/util/utils.h"
#include "tensor.pb.h"
#include "worker.pb.h"

namespace xllm {

bool encode_dit_failure(const Status& status,
                        proto::DITForwardResponse* response) {
  if (response == nullptr || status.ok()) {
    return false;
  }
  proto::DITForwardResponse encoded_response;
  auto* encoded_status = encoded_response.mutable_execution_status();
  encoded_status->set_code(static_cast<uint32_t>(status.code()));
  encoded_status->set_message(status.message());
  response->Swap(&encoded_response);
  return true;
}

bool encode_dit_output(const ForwardOutput& output,
                       proto::DITForwardResponse* response) {
  if (response == nullptr ||
      !valid_dit_output_mode(output.dit_forward_output)) {
    return false;
  }
  Timer timer;
  proto::DITForwardResponse encoded_response;
  proto::DiTForwardOutput* encoded_output =
      encoded_response.mutable_dit_forward_output();
  auto* encoded_tensors = encoded_output->mutable_tensors()->mutable_tensors();
  encoded_tensors->Reserve(
      static_cast<int32_t>(output.dit_forward_output.tensors.size()));
  for (const torch::Tensor& tensor : output.dit_forward_output.tensors) {
    if (!tensor.defined() || !tensor.device().is_cpu() ||
        !tensor.is_contiguous() ||
        !util::torch_to_proto(tensor, encoded_tensors->Add())) {
      LOG(ERROR) << "Failed to serialize DiT output tensor";
      return false;
    }
  }
  auto* encoded_audio_tensors =
      encoded_output->mutable_audio_tensors()->mutable_tensors();
  encoded_audio_tensors->Reserve(
      static_cast<int32_t>(output.dit_forward_output.audio_tensors.size()));
  for (const torch::Tensor& tensor : output.dit_forward_output.audio_tensors) {
    if (!tensor.defined() || !tensor.device().is_cpu() ||
        !tensor.is_contiguous() ||
        !util::torch_to_proto(tensor, encoded_audio_tensors->Add())) {
      LOG(ERROR) << "Failed to serialize DiT audio output tensor";
      return false;
    }
  }
  encoded_output->mutable_text_output()->Add(
      output.dit_forward_output.text_output.begin(),
      output.dit_forward_output.text_output.end());
  for (const DiTEncodedMedia& media : output.dit_forward_output.encoded_media) {
    if (!valid_dit_encoded_media(media)) {
      return false;
    }
    auto* encoded = encoded_output->add_encoded_media();
    encoded->set_data(media.data);
    encoded->set_mime_type(media.mime_type);
    encoded->set_container(media.container);
    encoded->set_width(media.width);
    encoded->set_height(media.height);
    encoded->set_num_frames(media.num_frames);
    encoded->set_fps(media.fps);
    encoded->set_audio_sample_rate(media.audio_sample_rate);
    encoded->set_audio_channels(media.audio_channels);
  }
  response->Swap(&encoded_response);
  COUNTER_ADD(proto_latency_seconds_o2proto, timer.elapsed_seconds());
  return true;
}

bool decode_dit_output(const proto::DITForwardResponse& response,
                       RawForwardOutput& output) {
  Timer timer;
  RawForwardOutput decoded_output;
  if (response.has_execution_status()) {
    const auto& status = response.execution_status();
    const StatusCode code =
        status.code() <= static_cast<uint32_t>(StatusCode::RATE_LIMITED)
            ? static_cast<StatusCode>(status.code())
            : StatusCode::UNKNOWN;
    decoded_output.dit_execution_status = Status(code, status.message());
    if (!decoded_output.dit_execution_status.ok()) {
      output = std::move(decoded_output);
      return true;
    }
  }
  const proto::DiTForwardOutput& encoded_output = response.dit_forward_output();
  if (encoded_output.encoded_media_size() > 0 &&
      (encoded_output.tensors().tensors_size() > 0 ||
       encoded_output.audio_tensors().tensors_size() > 0 ||
       encoded_output.text_output_size() > 0)) {
    return false;
  }
  decoded_output.dit_forward_output.tensors.reserve(
      encoded_output.tensors().tensors_size());
  for (const proto::Tensor& encoded_tensor :
       encoded_output.tensors().tensors()) {
    torch::Tensor tensor = util::proto_to_torch(encoded_tensor);
    if (!tensor.defined()) {
      LOG(ERROR) << "Failed to deserialize DiT output tensor";
      return false;
    }
    decoded_output.dit_forward_output.tensors.emplace_back(std::move(tensor));
  }
  decoded_output.dit_forward_output.audio_tensors.reserve(
      encoded_output.audio_tensors().tensors_size());
  for (const proto::Tensor& encoded_tensor :
       encoded_output.audio_tensors().tensors()) {
    torch::Tensor tensor = util::proto_to_torch(encoded_tensor);
    if (!tensor.defined()) {
      LOG(ERROR) << "Failed to deserialize DiT audio output tensor";
      return false;
    }
    decoded_output.dit_forward_output.audio_tensors.emplace_back(
        std::move(tensor));
  }
  decoded_output.dit_forward_output.text_output.assign(
      encoded_output.text_output().begin(), encoded_output.text_output().end());
  decoded_output.dit_forward_output.encoded_media.reserve(
      encoded_output.encoded_media_size());
  for (const auto& encoded : encoded_output.encoded_media()) {
    DiTEncodedMedia media;
    media.data = encoded.data();
    media.mime_type = encoded.mime_type();
    media.container = encoded.container();
    media.width = encoded.width();
    media.height = encoded.height();
    media.num_frames = encoded.num_frames();
    media.fps = encoded.fps();
    media.audio_sample_rate = encoded.audio_sample_rate();
    media.audio_channels = encoded.audio_channels();
    if (!valid_dit_encoded_media(media)) {
      return false;
    }
    decoded_output.dit_forward_output.encoded_media.emplace_back(
        std::move(media));
  }
  output = std::move(decoded_output);
  COUNTER_ADD(proto_latency_seconds_proto2o, timer.elapsed_seconds());
  return true;
}

}  // namespace xllm
