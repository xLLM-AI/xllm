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

#include <glog/logging.h>

#include <boost/algorithm/string/predicate.hpp>
#include <cstdint>
#include <string_view>

namespace xllm {

enum class DraftSamplingMode : int8_t {
  GREEDY,
  PROBABILISTIC,
};

inline constexpr std::string_view kDraftSamplingModeGreedy = "greedy";
inline constexpr std::string_view kDraftSamplingModeProbabilistic =
    "probabilistic";

inline DraftSamplingMode parse_draft_sampling_mode(std::string_view mode) {
  if (boost::iequals(mode, kDraftSamplingModeGreedy)) {
    return DraftSamplingMode::GREEDY;
  }
  if (boost::iequals(mode, kDraftSamplingModeProbabilistic)) {
    return DraftSamplingMode::PROBABILISTIC;
  }
  LOG(FATAL) << "Unsupported draft_sampling_mode: " << mode
             << ". Supported values: " << kDraftSamplingModeGreedy << ", "
             << kDraftSamplingModeProbabilistic << ".";
  return DraftSamplingMode::GREEDY;
}

inline bool draft_probs_required(DraftSamplingMode mode,
                                 bool all_greedy_sample) {
  return mode == DraftSamplingMode::PROBABILISTIC && !all_greedy_sample;
}

}  // namespace xllm
