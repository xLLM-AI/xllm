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

#include <torch/torch.h>

#include <vector>

#include "core/framework/state_dict/state_dict.h"
#include "core/util/json_reader.h"
#include "core/util/threadpool.h"
#include "model_loader.h"

namespace xllm {

bool load_quant_cfg(const JsonReader& reader, QuantArgs& quant_args);

class HFModelLoader : public ModelLoader {
 public:
  HFModelLoader(const std::string& model_weights_path);

  std::unique_ptr<Tokenizer> tokenizer() const override;

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override;

  int64_t get_total_weight_size() const override;

  int64_t get_non_decoder_weight_size() const override;

  int64_t get_max_decoder_layer_weight_size() const override;

 private:
  bool load_args(const std::string& model_weights_path);
  bool load_rec_vocab(const std::string& model_weights_path);
  bool load_model_args(const std::string& model_weights_path);
  bool load_quant_args(const std::string& model_weights_path);
  bool load_tokenizer_args(const std::string& model_weights_path);
  bool load_image_preprocessor_args(const std::string& model_weights_path);
  bool load_video_preprocessor_args(const std::string& model_weights_path);
  bool load_audio_preprocessor_args(const std::string& model_weights_path);
  std::string model_weights_path() const override {
    return model_weights_path_;
  }
  std::string model_weights_path_;

  // sorted model weights files
  std::vector<std::string> model_weights_files_;

  // models weights tensors
  std::vector<std::unique_ptr<StateDict>> state_dicts_;

  std::unique_ptr<ThreadPool> threadpool_;
};
}  // namespace xllm
