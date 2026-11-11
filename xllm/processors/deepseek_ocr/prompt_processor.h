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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "processors/prompt_processor.h"

namespace xllm {

// Expands "<image>" placeholders into the DeepSeek-OCR token grid.
//
// HF modeling_deepseekocr.py infer() emits, per image, the flat grid
//   ([IMG] * q_base + [IMG]) * q_base + [IMG]
//   [+ ([IMG] * (q * wc) + [IMG]) * (q * hc) when crops exist]
// where q_base = ceil((base_size/16)/4), q = ceil((crop_size/16)/4) and
// (wc, hc) is the per-image crop grid from image_spatial_crop. Every grid
// token is image_token_id, so the decoder-side scatter fills them in order
// with the encoder's feature rows (which are deliberately [crops, global,
// separator]-ordered; see xllm/python/models/deepseek_ocr.py).
class DeepseekOCRPromptProcessor final : public PromptProcessor {
 public:
  explicit DeepseekOCRPromptProcessor(const ModelArgs& args);

  bool process(std::string& prompt, const MMData& mm_data) override;
  bool find_mm_spans(const std::vector<int32_t>& token_ids,
                     MMData& mm_data) override;

 private:
  static int32_t queries_per_edge(int32_t view_size) {
    // HF constants: patch_size 16, downsample_ratio 4.
    return (view_size / 16 + 3) / 4;
  }

  // Token count for one image's grid (process, the expansion loop and
  // find_mm_spans all derive from this single formula). Mirrors HF infer():
  // the global grid always, plus the crops grid when (wc, hc) is not 1x1.
  int32_t grid_length(int32_t width_crops, int32_t height_crops) const {
    const int32_t q_base = queries_per_edge(base_size_);
    const int32_t q_crop = queries_per_edge(crop_size_);
    int32_t length = q_base * (q_base + 1) + 1;
    if (width_crops > 1 || height_crops > 1) {
      length += (q_crop * width_crops + 1) * (q_crop * height_crops);
    }
    return length;
  }

  const std::string image_token_ = "<image>";
  int32_t image_token_id_ = 0;
  int32_t base_size_ = 1024;
  int32_t crop_size_ = 640;
};

}  // namespace xllm
