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

// DeepseekV3Adapter: DeepSeek-V3/R1 (MLA + sigmoid MoE + shared expert, no
// DSA).

#pragma once

#include "core/layers/xlite/xlite_config_builder.h"
#include "core/layers/xlite/xlite_model_adapter.h"
#include "core/layers/xlite/xlite_weight_utils.h"

namespace xllm::xlite {

class DeepseekV3Adapter : public XliteModelAdapter {
 public:
  XModelConfig BuildConfig(const ModelContext& context) override {
    return XliteConfigBuilder::FromDeepseekV3(context);
  }

  void Load(const StateDict& sd,
            XModel& m,
            const XModelConfig& cfg,
            const ModelArgs& args,
            const ParallelArgs& pa,
            const torch::Device& device,
            std::vector<torch::Tensor>& storages) override {
    XliteWeightUtils::LoadMLA(sd, m, cfg, args, pa, device, storages);
  }

  uint32_t MoeTpSize(const ParallelArgs& pa) const override {
    return pa.ep_size() > 1
               ? static_cast<uint32_t>(pa.world_size() / pa.ep_size())
               : XliteTpSize(pa);
  }

  std::string Name() const override { return "DeepseekV3Adapter"; }
};

}  // namespace xllm::xlite