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

#include <cstdint>

#include "core/common/types.h"
#include "core/framework/model/dit_model_output.h"

namespace xllm {

Status encode_minimax_h3_video_output(const DiTForwardOutput& output,
                                      double fps,
                                      int32_t sample_rate,
                                      DiTEncodedMedia& encoded);

// Overload with an explicit encoder contract. Added as a new symbol; the
// four-argument form above keeps CRF 18 and truncating quantization.
Status encode_minimax_h3_video_output(const DiTForwardOutput& output,
                                      double fps,
                                      int32_t sample_rate,
                                      DiTEncodedMedia& encoded,
                                      int32_t crf,
                                      bool round_quantization);

}  // namespace xllm
