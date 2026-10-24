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

#pragma once
#include <torch/types.h>

#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_proposal.h"

namespace xllm {

// The single sampling owner for speculative verification; the
// general-purpose sample() never runs for verify steps.
SampleOutput rejection_sample(const SamplingParameters& params,
                              const DraftProposal& draft_proposal,
                              torch::Tensor target_logits);

}  // namespace xllm
