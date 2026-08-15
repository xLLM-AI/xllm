/* Copyright 2025-2026 The xLLM Authors.

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

// Adapter interface: each model implements BuildConfig + Load.

#pragma once

#include <torch/torch.h>
#include <xlite/xlite.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/state_dict/state_dict.h"

namespace xllm::xlite {

class XliteModelAdapter {
 public:
  virtual ~XliteModelAdapter() = default;

  // Must populate runtime fields (maxBatchedTokens etc.) from singletons.
  virtual XModelConfig BuildConfig(const ModelContext& context) = 0;

  // Called before XModel::Init.
  virtual void Load(const StateDict& sd,
                    XModel& m,
                    const XModelConfig& cfg,
                    const ModelArgs& args,
                    const ParallelArgs& pa,
                    const torch::Device& device,
                    std::vector<torch::Tensor>& storages) = 0;

  // EP=1 -> tp_size; EP>1 -> 1 (experts sharded by EP). Dense returns 1.
  virtual uint32_t MoeTpSize(const ParallelArgs& pa) const { return 1; }

  virtual std::string Name() const { return ""; }
};

}  // namespace xllm::xlite