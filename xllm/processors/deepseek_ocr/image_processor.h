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

#include <torch/types.h>

#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "processors/image_processor.h"

namespace xllm {

// Gundam-recipe preprocessing (pad to base_size, aspect-ratio crops at
// crop_size) delegated to
// xllm.python.models.deepseek_ocr.preprocess_images via pybind, so the PIL
// semantics (ImageOps.pad, Image.resize bicubic) match the HF reference.
// The per-image result dict is stored verbatim into the MMDataItem; the
// executor's EncoderInputGatherVisitor then batches the keys
// ("pixel_values", "pixel_values_crops", "image_spatial_crop") across
// items and PyExecutorImpl passes them to DeepseekOCRForCausalLM
// .encode_with_inputs.
class DeepseekOCRImageProcessor final : public ImageProcessor {
 public:
  explicit DeepseekOCRImageProcessor(const ModelArgs& args);

  bool process(const std::vector<torch::Tensor>& images,
               std::vector<MMDataItem>& output_items) const override;

 private:
  int32_t base_size_ = 1024;
  int32_t crop_size_ = 640;
};

}  // namespace xllm
