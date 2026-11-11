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

// Multimodal-processor and config registration for DeepSeek-OCR served as a
// VLM via --backend=vlm --model_impl=python (PyCausalLM + the pure-torch
// xllm.python.models.deepseek_ocr.DeepseekOCRForCausalLM).
//
// deepseek_ocr has no C++ VLM model object (the SAM/CLIP vision towers and
// the DeepSeek-V2-style decoder all live in Python). This header registers:
//   1. DeepseekOCRPromptProcessor — expands each "<image>" placeholder into
//      the token grid "[global grid][crops grid]" and pins each image's
//      token span (all positions carry image_token_id, so the span boundary
//      is the first non-image token after the grid).
//   2. DeepseekOCRImageProcessor — Gundam-recipe preprocessing (pad to
//      base_size + aspect-ratio crops), delegated to
//      xllm.python.models.deepseek_ocr.preprocess_images via pybind so the
//      PIL semantics match the HF reference byte for byte.
// The engine's create_vlm_model python fallback then instantiates PyCausalLM
// with the model_type name, which resolves to the registered Python class.

#include <glog/logging.h>

#include <nlohmann/json.hpp>

#include "models/model_registry.h"
#include "processors/deepseek_ocr_image_processor.h"
#include "processors/deepseek_ocr_prompt_processor.h"
#include "processors/multimodal_processor.h"

namespace xllm {

using DeepseekOCRMultimodalProcessor =
    MultimodalProcessor<DeepseekOCRPromptProcessor, DeepseekOCRImageProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(deepseek_ocr, DeepseekOCRMultimodalProcessor);

const ModelArgsLoader kDeepseekOCRModelArgsLoader = [](const JsonReader& json,
                                                       ModelArgs* args) {
  // Several HF releases share model_type "deepseek_vl_v2" (the architecture
  // lineage); canonical_model_registration_name maps them all here. Only
  // DeepSeek-OCR is implemented: fail with a clear message naming the actual
  // architectures instead of falling through to an opaque weight-loading
  // error. (JsonReader keys are strings only, so read the architectures
  // array off the raw document; data() copies.)
  const nlohmann::json raw = json.data();
  if (raw.contains("architectures") && raw["architectures"].is_array() &&
      !raw["architectures"].empty() &&
      raw["architectures"][0] != "DeepseekOCRForCausalLM") {
    LOG(ERROR) << "architectures '" << raw["architectures"][0]
               << "' is not supported for model_type deepseek_vl_v2; only "
               << "DeepseekOCRForCausalLM (DeepSeek-OCR) is implemented.";
    return false;
  }
  // The HF config nests the decoder under language_config and duplicates it
  // at the top level; prefer the top-level keys (they are identical).
  LOAD_ARG_OR(model_type, "model_type", "deepseek_ocr");
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(hidden_size, "hidden_size", 1280);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6848);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 12);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 10);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 10);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(vocab_size, "vocab_size", 129280);
  // MoE (language_config values equal the top-level duplicates).
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 64);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 2);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 6);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 896);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", false);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 1.0f);
  LOAD_ARG_OR(scoring_func, "scoring_func", "softmax");
  LOAD_ARG_OR(topk_method, "topk_method", "greedy");
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  // Vision / processor parameters. The SAM/CLIP geometry is checkpoint-fixed;
  // only the view sizes are configurable (Gundam recipe: 1024 base / 640
  // crops, matching the HF infer() defaults).
  LOAD_ARG_OR(image_token_id, "image_token_id", 128815);
  LOAD_ARG_OR(mm_image_base_size, "vision_config.image_size", 1024);
  LOAD_ARG_OR(mm_image_size, "vision_config.crop_size", 640);
  // Plain 1-D positions: no mposition generator is registered and the rope
  // type stays at its default, so the engine builds 1-D position ids.
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
  return true;
};

REGISTER_MODEL_ARGS_LOADER(deepseek_ocr, kDeepseekOCRModelArgsLoader);
REGISTER_MODEL_ARGS_LOADER(DeepseekOCRForCausalLM, kDeepseekOCRModelArgsLoader);

}  // namespace xllm
