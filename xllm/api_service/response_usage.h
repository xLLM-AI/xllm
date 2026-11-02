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

#include "common.pb.h"
#include "core/framework/request/usage.h"

namespace xllm::api_service {

inline void set_proto_usage(proto::Usage* proto_usage,
                            const xllm::Usage& usage) {
  proto_usage->set_prompt_tokens(usage.num_prompt_tokens);
  proto_usage->set_completion_tokens(usage.num_generated_tokens);
  proto_usage->set_total_tokens(usage.num_total_tokens);
  auto* prompt_tokens_details = proto_usage->mutable_prompt_tokens_details();
  prompt_tokens_details->set_cached_tokens(usage.num_cached_tokens);
  prompt_tokens_details->set_audio_tokens(0);

  auto* completion_tokens_details =
      proto_usage->mutable_completion_tokens_details();
  completion_tokens_details->set_reasoning_tokens(0);
  completion_tokens_details->set_audio_tokens(0);
}

}  // namespace xllm::api_service
