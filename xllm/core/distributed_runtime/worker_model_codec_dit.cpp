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

bool encode_dit_output(const ForwardOutput& output,
                       proto::DITForwardResponse* response) {
  if (response == nullptr) {
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
  response->Swap(&encoded_response);
  COUNTER_ADD(proto_latency_seconds_o2proto, timer.elapsed_seconds());
  return true;
}

bool decode_dit_output(const proto::DITForwardResponse& response,
                       RawForwardOutput& output) {
  Timer timer;
  RawForwardOutput decoded_output;
  const proto::DiTForwardOutput& encoded_output = response.dit_forward_output();
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
  output = std::move(decoded_output);
  COUNTER_ADD(proto_latency_seconds_proto2o, timer.elapsed_seconds());
  return true;
}

}  // namespace xllm
