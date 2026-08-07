/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "lora_config.h"
#include "lora_request.h"

namespace xllm {

// LoRAAdapter is the concrete in-memory result of loading a HuggingFace PEFT
// adapter. It owns:
//   - the parsed adapter_config.json (r, alpha, target_modules, ...)
//   - one A/B tensor pair per (target_module, layer_idx)
//
// Weights are held in **CPU tensors** at load time. Movement to device is
// LoraCacheManager's job (M4). Keeping the raw CPU state here means we can
// re-materialise on any device without re-parsing safetensors.
struct LoRAAdapter {
  // Descriptor mirrored from the load call.
  LoRARequest request;

  // Rank r as reported by adapter_config.json.
  int32_t r = 0;

  // scale factor pushed into the delta path. Per PEFT convention this is
  // `lora_alpha / r`. Stored pre-computed so the forward path just does
  // one multiply.
  float scaling = 0.0f;

  // Modules the adapter targets, e.g. {"q_proj", "k_proj", "v_proj",
  // "o_proj"}. Validated against LoRAConfig::lora_target_modules at load.
  std::vector<std::string> target_modules;

  // Per-(module, layer) A / B weights.
  //
  // Keyed by "layers.{L}.self_attn.q_proj" (or the closest xllm-native form
  // after the module-name mapping step below). Values are the raw torch
  // tensors as read from safetensors; A is [r, in_features], B is
  // [out_features, r] following HuggingFace convention.
  //
  // The map is deliberately unstructured (name -> tensor) so future work
  // like packed layer merging (q/k/v -> qkv) can rewrite the keys without
  // fighting a rigid data structure.
  std::unordered_map<std::string, torch::Tensor> tensors;
};

// LoRAAdapterLoader turns a filesystem path into a LoRAAdapter. It never
// touches device memory and never registers anything. The registry / cache
// call this and route the result themselves.
class LoRAAdapterLoader {
 public:
  explicit LoRAAdapterLoader(const LoRAConfig& config) : config_(config) {}

  // Load and validate a PEFT adapter from `req.lora_path`. On success the
  // returned struct owns all the CPU tensors; on failure the returned
  // optional is empty and the reason is logged.
  //
  // Validation performed:
  //   1. adapter_config.json exists and parses.
  //   2. r > 0 and r <= config.max_lora_rank.
  //   3. base_model_name_or_path (if present) matches req.base_model_name
  //      when the caller provided one.
  //   4. target_modules is a subset of config.lora_target_modules.
  //   5. peft_type == "LORA" (DoRA / IA3 refused for now).
  //   6. bias == "none" (bias-augmented LoRA refused for now).
  //   7. adapter_model.safetensors present and readable.
  //
  // Any tensor whose name does not fit the expected pattern
  // "base_model.model.<subkey>.lora_A.weight" is skipped with a warning
  // rather than failing outright, so a slightly non-standard adapter still
  // loads whatever it can.
  std::optional<LoRAAdapter> load(const LoRARequest& req);

 private:
  // adapter_config.json parser. Returns false on any structural problem.
  bool parse_config(const std::string& path,
                    const std::string& expected_base_model_name,
                    LoRAAdapter* out) const;

  // Read `adapter_model.safetensors`, apply the module-name mapping, and
  // populate out->tensors.
  bool load_weights(const std::string& safetensors_path,
                    LoRAAdapter* out) const;

  // HF LoRA weight names look like
  //   base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight
  // We strip the fixed "base_model.model." prefix and remap ".lora_A." /
  // ".lora_B." tails to a stable canonical form we can match on later.
  //
  // Returns std::nullopt if the name is unrecognised (skipped rather than
  // fatal, per point 7 above).
  std::optional<std::string> canonicalize_weight_name(
      const std::string& raw) const;

  const LoRAConfig& config_;
};

}  // namespace xllm
