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

#pragma once

#include <torch/types.h>

#include <string>

#include "core/util/binary_payload.h"

namespace xllm::proto {
class InferInputTensor;
class Tensor;
}  // namespace xllm::proto

namespace xllm::util {

torch::Tensor convert_rec_tensor_to_torch(
    const proto::InferInputTensor& input_tensor);

torch::Tensor proto_to_torch(const proto::Tensor& proto_tensor);

torch::Tensor proto_to_torch(const proto::Tensor& proto_tensor,
                             const std::string& binary_payload);

torch::Tensor proto_to_torch(const proto::Tensor& proto_tensor,
                             const BinaryPayload& binary_payload);

bool torch_to_proto(const torch::Tensor& torch_tensor,
                    proto::Tensor* proto_tensor);

bool torch_to_proto(const torch::Tensor& torch_tensor,
                    proto::Tensor* proto_tensor,
                    std::string& binary_payload);

torch::ScalarType datatype_proto_to_torch(const std::string& proto_datatype);

std::string torch_datatype_to_proto(torch::ScalarType torch_dtype);

}  // namespace xllm::util
