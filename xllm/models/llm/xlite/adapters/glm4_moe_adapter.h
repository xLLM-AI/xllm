/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

==============================================================================*/

// Glm4MoeAdapter: GLM-4 MoE (MHA + partial rotary + sigmoid MoE + shared
// expert, W8A8). BuildConfig via FromGlm4Moe (W8A8 flags conditional on
// QuantArgs, see IsW8A8); Load -> XliteWeightUtils::LoadMoE (MHA attn + dense
// MLP + MoE experts + shared expert).

#pragma once

#include "core/layers/xlite/xlite_config_builder.h"
#include "core/layers/xlite/xlite_model_adapter.h"
#include "core/layers/xlite/xlite_weight_utils.h"

namespace xllm::xlite {

class Glm4MoeAdapter : public XliteModelAdapter {
 public:
  XModelConfig BuildConfig(const ModelContext& context) override {
    return XliteConfigBuilder::FromGlm4Moe(context);
  }

  void Load(const StateDict& sd,
            XModel& m,
            const XModelConfig& cfg,
            const ModelArgs& args,
            const ParallelArgs& pa,
            const torch::Device& device,
            std::vector<torch::Tensor>& storages) override {
    XliteWeightUtils::LoadMoE(sd, m, cfg, args, pa, device, storages);
  }

  // Same as Qwen3Moe: EP>1 -> world_size/ep_size; EP==1 -> attention TP.
  // XliteCausalLMBase ctor overrides cfg_.moeTPSize with the same formula.
  uint32_t MoeTpSize(const ParallelArgs& pa) const override {
    return pa.ep_size() > 1
               ? static_cast<uint32_t>(pa.world_size() / pa.ep_size())
               : XliteTpSize(pa);
  }

  std::string Name() const override { return "Glm4MoeAdapter"; }
};

}  // namespace xllm::xlite