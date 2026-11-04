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

#pragma once

#include <string>
#include <string_view>

#include "core/common/types.h"
#include "core/framework/request/dit_input_sources.h"
#include "core/util/binary_payload.h"

namespace xllm {

inline constexpr std::string_view kMiniMaxH3ExecutionProfile =
    "a3_16card_tp2_u8_ref2va_v1";

bool has_minimax_h3_condition_metadata(const proto::Tensor& tensor);
bool has_minimax_h3_condition_metadata(const TensorParameters& parameters);
Status validate_minimax_h3_condition_wire(const proto::Tensor& tensor,
                                          const BinaryPayload& payload);
Status validate_minimax_h3_condition(const NamedTensor& source,
                                     const DiTMediaSources& media_sources);
// The caller supplies a defined contiguous CPU tensor.
std::string minimax_h3_tensor_sha256(const torch::Tensor& tensor);

}  // namespace xllm
