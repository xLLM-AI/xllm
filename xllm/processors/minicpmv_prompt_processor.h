/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/model_args.h"
#include "processors/prompt_processor.h"

namespace xllm::npu::model {

class MiniCPMPromptProcessor final : public PromptProcessor {
 public:
  explicit MiniCPMPromptProcessor(const ModelArgs& args);

  void process(std::string& prompt, const MMData& mm_data) override;
  void find_mm_spans(const std::vector<int32_t>& token_ids,
                     MMData& mm_data) override;

 private:
  std::string get_image_id_placeholder(int32_t idx) const;
  std::string get_grid_placeholder(
      const std::pair<int32_t, int32_t>& grid) const;
  std::string get_slice_image_placeholder(
      const std::pair<int32_t, int32_t>& image_size,
      int32_t image_idx = 0,
      int32_t max_slice_nums = -1,
      std::optional<bool> use_image_id_opt = std::nullopt) const;

  const std::string im_start_token_ = "<image>";
  const std::string im_end_token_ = "</image>";
  const std::string slice_start_token_ = "<slice>";
  const std::string slice_end_token_ = "</slice>";
  const std::string unk_token_ = "<unk>";
  const std::string im_id_start_ = "<image_id>";
  const std::string im_id_end_ = "</image_id>";

  const int32_t im_start_id_ = 151659;
  const int32_t im_end_id_ = 151658;
  const int32_t unk_id_ = 128244;

  bool slice_mode_;
  bool use_image_id_;
  int32_t max_slice_nums_;
  int32_t image_feature_size_;
  int32_t scale_resolution_;
};

}  // namespace xllm::npu::model
