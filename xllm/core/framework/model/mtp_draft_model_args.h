/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>

#include "core/framework/model/model_args.h"

namespace xllm {

enum class MtpDraftModelArgsStatus : int8_t {
  NORMALIZED = 0,
  NOT_APPLICABLE = 1,
  UNSUPPORTED = 2,
};

// Leaves draft_args unchanged when the target has no MTP layers or is unknown.
MtpDraftModelArgsStatus normalize_mtp_draft_model_args(
    const ModelArgs& target_args,
    ModelArgs& draft_args);

}  // namespace xllm
