/* Copyright 2025-2026 The xLLM Authors.

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

#include "processors/multimodal_processor_pipeline.h"

#include <utility>

#include "common/metrics.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "util/timer.h"

namespace xllm {

MultimodalProcessorPipeline::MultimodalProcessorPipeline(
    std::shared_ptr<Tokenizer> tokenizer,
    const TokenizerArgs& tokenizer_args,
    int32_t max_sequence_length)
    : tokenizer_(std::move(tokenizer)),
      tokenizer_args_(tokenizer_args),
      max_sequence_length_(max_sequence_length) {}

MultimodalProcessorPipeline::~MultimodalProcessorPipeline() = default;

bool MultimodalProcessorPipeline::tokenize(
    const std::string& prompt,
    std::vector<int32_t>& token_ids) const {
  Timer timer;
  if (!tokenizer_->encode(prompt, &token_ids)) {
    LOG(ERROR) << "Failed to encode prompt: " + prompt;
    return false;
  }

  pad_to_max_length(token_ids);
  COUNTER_ADD(tokenization_latency_seconds, timer.elapsed_seconds());
  return true;
}

void MultimodalProcessorPipeline::assign_mm_hash_keys(const MMInput& mm_input,
                                                      MMData& mm_data) const {
  const std::vector<MMInputItem>& input_items = mm_input.items();
  MMItemVec& output_items = mm_data.items<MMItemVec>();
  CHECK_EQ(input_items.size(), output_items.size());
  for (size_t index = 0; index < input_items.size(); ++index) {
    const std::optional<XXH3Key>& hash_key = input_items[index].hash_key;
    if (hash_key.has_value()) {
      output_items[index].mutable_state().mutable_schedule_data().key =
          hash_key.value();
    }
  }
}

void MultimodalProcessorPipeline::pad_to_max_length(
    std::vector<int32_t>& token_ids) const {
  if (max_sequence_length_ <= 0 || tokenizer_args_.pad_token().empty()) {
    return;
  }

  const auto pad_id = tokenizer_->token_to_id(tokenizer_args_.pad_token());
  if (!pad_id.has_value() ||
      static_cast<int32_t>(token_ids.size()) >= max_sequence_length_) {
    return;
  }

  const int32_t pad_count =
      max_sequence_length_ - static_cast<int32_t>(token_ids.size());
  token_ids.insert(token_ids.begin(), pad_count, pad_id.value());
}

}  // namespace xllm
