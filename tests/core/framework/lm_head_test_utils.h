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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model_context/model_context.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/quantization/quant_args.h"

namespace xllm {
namespace testutil {

inline ProcessGroup* test_process_group() {
  static ProcessGroup group(/*rank=*/0,
                            /*world_size=*/1,
                            torch::Device(torch::kCPU));
  return &group;
}

inline ModelContext make_lm_test_context(int64_t hidden_size,
                                         int64_t vocab_size) {
  ParallelArgs parallel_args(/*rank=*/0,
                             /*world_size=*/1,
                             test_process_group());
  parallel_args.tp_group_ = test_process_group();
  ModelArgs model_args;
  model_args.hidden_size(/*hidden_size=*/hidden_size);
  model_args.vocab_size(/*vocab_size=*/vocab_size);
#if defined(USE_NPU)
  return ModelContext(
      parallel_args,
      model_args,
      QuantArgs(),
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU),
      /*context=*/nullptr);
#else
  return ModelContext(
      parallel_args,
      model_args,
      QuantArgs(),
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
#endif
}

class HeadOnlyModelLoader final : public ModelLoader {
 public:
  explicit HeadOnlyModelLoader(torch::Tensor head_weight) {
    state_dicts_.push_back(std::make_unique<StateDict>(
        std::unordered_map<std::string, torch::Tensor>{
            {"lm_head.weight", std::move(head_weight)}}));
  }

  std::unique_ptr<Tokenizer> tokenizer() const override { return nullptr; }

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override {
    return state_dicts_;
  }

  std::string model_weights_path() const override { return {}; }

 private:
  std::vector<std::unique_ptr<StateDict>> state_dicts_;
};

}  // namespace testutil
}  // namespace xllm
