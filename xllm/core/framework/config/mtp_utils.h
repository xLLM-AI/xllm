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

#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/aux_hidden_capture.h"
#include "core/framework/model/model_args.h"
#include "core/framework/speculative/draft_config.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"
#include "core/util/json_reader.h"
#include "core/util/utils.h"

namespace xllm {

// NPU MTP targets whose num_speculative_tokens flows from engine Options into
// ModelArgs; other targets keep the checkpoint default.
inline bool uses_mtp_target_spec_width(std::string_view model_type) {
  return is_glm5_next_target_model_type(model_type) ||
         util::is_deepseek_v4_model_type(model_type);
}

inline int64_t mtp_hidden_state_width(const ModelArgs& model_args) {
  if (util::is_deepseek_v4_model_type(model_args.model_type())) {
    return model_args.hc_mult() * model_args.hidden_size();
  }
  return model_args.hidden_size();
}

inline bool uses_embedded_eagle3_draft(std::string_view algorithm,
                                       const ModelArgs& target_model_args) {
  return SpeculativeConfig::is_eagle3_algorithm(algorithm) &&
         target_model_args.enable_embedded_eagle3_draft();
}

#if defined(USE_NPU)
namespace mtp_detail {

inline int32_t read_dspark_block_size(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string config_path = model_weights_path + "/config.json";
  CHECK(reader.parse(config_path))
      << "Failed to parse DSpark draft config: " << config_path;
  return reader.value_or<int32_t>("dspark_block_size", 0);
}

inline void configure_deepseek_v4_dspark_args(ModelArgs& args,
                                              const runtime::Options& options) {
  CHECK_GT(args.dspark_num_layers(), 0)
      << "DeepSeek-V4 DSpark requires at least one draft layer.";
  args.n_layers(args.dspark_num_layers());
  args.n_hash_layers(0);

  // Default to the checkpoint's block_size; --num_speculative_tokens overrides.
  const int32_t ckpt_block_size = read_dspark_block_size(options.model_path());
  const int32_t user_num_spec = options.num_speculative_tokens();
  if (user_num_spec > 0 && user_num_spec != ckpt_block_size) {
    LOG(WARNING) << "--num_speculative_tokens=" << user_num_spec
                 << " overrides DSpark checkpoint dspark_block_size="
                 << ckpt_block_size << ".";
  }
  args.dspark_block_size(user_num_spec > 0 ? user_num_spec : ckpt_block_size);

  // DSpark stages are all standard SWA layers. Their stage ids are not target
  // model layer ids, so target compress_ratios[0..N) must not be reused.
  args.compress_ratios(
      std::vector<int32_t>(static_cast<size_t>(args.dspark_num_layers()),
                           /*value=*/1));

  args.dspark_use_native_sas(
      KernelConfig::get_instance().enable_dspark_native_sas());
}

}  // namespace mtp_detail
#endif

// Applies the family-specific speculative decoding transformations to one
// side's ModelArgs: draft model_type overrides, layer-count rewrites, and
// aux-hidden capture wiring. Platform differences stay inside. No-op when
// speculative decoding is disabled.
inline void configure_speculative_model_args(ModelArgs& args,
                                             const runtime::Options& options) {
  if (!options.enable_speculative_decode()) {
    return;
  }
  const std::string& speculative_algorithm = options.speculative_algorithm();
  const bool is_block_diffusion =
      SpeculativeConfig::is_block_diffusion_algorithm(speculative_algorithm);

#if defined(USE_NPU)
  if (SpeculativeConfig::is_mtp_algorithm(speculative_algorithm) &&
      uses_mtp_target_spec_width(args.model_type())) {
    args.num_speculative_tokens(options.num_speculative_tokens());
  }
  if (is_block_diffusion) {
    if (options.is_draft_model()) {
      args.layers_to_capture({});
      const bool is_dspark =
          SpeculativeConfig::is_dspark_algorithm(speculative_algorithm);
      const bool is_deepseek_v4_dspark =
          is_dspark && util::is_deepseek_v4_model_type(args.model_type());
      std::string draft_model_type = "DFlashDraftModel";
      if (is_dspark) {
        draft_model_type = args.model_type() == kDSparkGLM5DraftModelType
                               ? std::string(kDSparkGLM5DraftModelType)
                               : "DSparkDraftModel";
      } else if (SpeculativeConfig::is_dflash2_algorithm(
                     speculative_algorithm)) {
        draft_model_type = std::string(kDFlash2DraftModelType);
        CHECK_GT(args.dflash2_block_size(), 0);
      }
      if (is_deepseek_v4_dspark) {
        draft_model_type = std::string(kDeepseekV4DSparkModelType);
      }
      LOG(INFO) << "Overriding draft model_type from " << args.model_type()
                << " to " << draft_model_type
                << " for block-diffusion speculative decoding";
      args.model_type(draft_model_type);
      // The draft's decode step is num_speculative_tokens + 1 rows wide;
      // record it on ModelArgs so dummy inputs derive the same width.
      args.num_speculative_tokens(options.num_speculative_tokens());
      if (is_deepseek_v4_dspark) {
        mtp_detail::configure_deepseek_v4_dspark_args(args, options);
      }
    } else {
      apply_block_diffusion_target_capture(
          args, options.draft_model_path().value_or(""));
    }
  } else if (::xllm::SpeculativeConfig::get_instance()
                 .enable_atb_spec_kernel()) {
    args.num_speculative_tokens(options.num_speculative_tokens());
  } else if (options.num_speculative_tokens() == 0 &&
             args.num_nextn_predict_layers() != 0) {
    const std::string& current_type = args.model_type();
    const char* mtp_model_type = nullptr;
    if (current_type == "qwen3_5_text") {
      mtp_model_type = "qwen3_5_mtp";
    } else if (current_type == "qwen3_5_moe_text") {
      mtp_model_type = "qwen3_5_moe_mtp";
    }
    if (mtp_model_type != nullptr) {
      LOG(INFO) << "Overriding draft model_type from " << current_type << " to "
                << mtp_model_type << " for speculative decoding";
      args.model_type(mtp_model_type);
      const int32_t mtp_layers = args.num_nextn_predict_layers();
      args.n_layers(mtp_layers);
      args.layer_types(std::vector<std::string>(mtp_layers, "full_attention"));
      args.full_attention_interval(1);
    }
  }
  if (!options.is_draft_model() &&
      SpeculativeConfig::requires_aux_hidden_capture(speculative_algorithm) &&
      args.layers_to_capture().empty()) {
    std::vector<int32_t> capture_layer_ids;
    if (SpeculativeConfig::is_eagle3_algorithm(speculative_algorithm) &&
        options.draft_model_path().has_value()) {
      // Eagle3.1 drafts pin their aux capture layers in the draft config;
      // legacy eagle3 configs omit them and fall back to the default below.
      capture_layer_ids = read_capture_layer_ids(*options.draft_model_path(),
                                                 /*required=*/false);
    }
    if (capture_layer_ids.empty()) {
      const int32_t num_layers = static_cast<int32_t>(args.n_layers());
      capture_layer_ids =
          AuxHiddenCapture::legacy_default_capture_layer_ids(num_layers);
    }
    args.layers_to_capture(std::move(capture_layer_ids));
  }
#else
  args.num_speculative_tokens(options.num_speculative_tokens());
  if (is_block_diffusion) {
    if (options.is_draft_model()) {
      LOG(FATAL) << speculative_algorithm
                 << " block-diffusion draft is not supported on "
                 << Platform::type_str() << ".";
    }
    apply_block_diffusion_target_capture(
        args, options.draft_model_path().value_or(""));
  }
  // When running speculative decoding, the draft model reuses the same
  // checkpoint as the target model. The draft side needs to instantiate the
  // MTP variant, so override the model_type here without mutating the
  // original config.
  if (!is_block_diffusion && options.num_speculative_tokens() == 0 &&
      args.num_nextn_predict_layers() != 0) {
    static const std::unordered_map<std::string, std::string>
        kModelTypeToMtpType = {
            {"deepseek_v3", "deepseek_v3_mtp"},
            {"deepseek_v32", "deepseek_v32_mtp"},
            {"deepseek_v4", "deepseek_v4_mtp"},
            {"glm_moe_dsa", "glm_moe_dsa_mtp"},
            {"joyai_llm_flash", "joyai_llm_flash_mtp"},
            {"mimo", "mimo_mtp"},
        };
    const std::string& current_type = args.model_type();
    auto it = kModelTypeToMtpType.find(current_type);
    if (it != kModelTypeToMtpType.end()) {
      LOG(INFO) << "Overriding draft model_type from " << current_type << " to "
                << it->second << " for speculative decoding";
      args.model_type(it->second);
    }
  }
#endif
}

}  // namespace xllm
