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
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "processors/prompt_processor.h"

namespace xllm {

// Expands each <|AUDIO|> placeholder to the audio's token count and
// records the spans; with CTC enabled each entry folds in its trailing pad
// run, and every audio carries its [hash, ctc_pad_num] encode meta in item
// data (aggregated by MMBatchData::batch for the executor). process()/
// find_mm_spans() validate and reject bad layouts.
class JoyaiASRPromptProcessor final : public PromptProcessor {
 public:
  explicit JoyaiASRPromptProcessor(const ModelArgs& args);

  bool process(std::string& prompt, const MMData& mm_data) override;
  bool find_mm_spans(const std::vector<int32_t>& token_ids,
                     MMData& mm_data) override;

 private:
  const std::string audio_token_;
  int32_t audio_token_id_ = 0;
  int32_t ctc_pad_token_id_ = 0;
  bool ctc_enable_ = false;
};

}  // namespace xllm
