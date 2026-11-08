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
#include <vector>

namespace xllm {

class ForwardOutput;
class LlmForwardInput;
class Stream;
class VlmForwardInput;

// Adapts prior runtime output to JSON constraints and makes failed rows safe
// before decode tokens are replaced or rebuilt.
class JsonObjectDecodeAdapter final {
 public:
  static void prepare(LlmForwardInput& input,
                      const ForwardOutput& prior_output,
                      const std::vector<std::string>& prior_sample_sequence_ids,
                      const torch::Device& device,
                      Stream& compute_stream);
  static void prepare(VlmForwardInput& input,
                      const ForwardOutput& prior_output,
                      const std::vector<std::string>& prior_sample_sequence_ids,
                      const torch::Device& device,
                      Stream& compute_stream);

 private:
  template <typename Input>
  static void advance_constraints(
      Input& input,
      const ForwardOutput& prior_output,
      const std::vector<std::string>& prior_sample_sequence_ids,
      const torch::Device& device,
      Stream& compute_stream);
  template <typename Input>
  static void sanitize_error_inputs(Input& input);
};

}  // namespace xllm
