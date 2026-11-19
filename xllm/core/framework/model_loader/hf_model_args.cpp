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

#include "core/framework/model_loader/hf_model_args.h"

#include <glog/logging.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <iterator>
#include <optional>
#include <regex>
#include <vector>

#include "core/framework/config/speculative_config.h"
#if defined(USE_NPU)
#include "core/framework/config/kernel_config.h"
#endif
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_type.h"
#include "core/framework/model_loader/hf_model_config.h"
#include "core/framework/quantization/quant_args.h"
#include "core/platform/platform.h"
#include "core/util/json_reader.h"
#include "core/util/utils.h"
#include "models/model_registry.h"

namespace xllm {

namespace {

struct ModelArgsAlias {
  std::string_view model_type;
  const char* resolved_type;
};

constexpr ModelArgsAlias kModelArgsAliases[] = {
    {"DeepseekOCRForCausalLM", "deepseek_ocr"},
    {"Qwen3VLForConditionalGeneration", "qwen3_vl"},
#if defined(USE_NPU) || defined(USE_MLU)
    {"deepseek_v3_mtp", "deepseek_v3"},
    {"glm5_next_mtp", "glm5_next"},
#endif
#if defined(USE_NPU)
    {"deepseek_v32_mtp", "deepseek_v32"},
#endif
#if defined(USE_MLU)
    {"glm_moe_dsa_mtp", "glm_moe_dsa"},
    {"joyai_llm_flash_mtp", "joyai_llm_flash"},
#endif
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_MUSA)
    {"qwen3_5_mtp", "qwen3_5_text"},
    {"qwen3_5_moe_mtp", "qwen3_5_moe_text"},
#endif
#if defined(USE_CUDA) || defined(USE_DCU)
    {"mimo_mtp", "mimo"},
#endif
};

constexpr int64_t kCompressedBits4 = 4;
constexpr int64_t kCompressedBits8 = 8;

std::optional<int64_t> supported_compressed_bits(const nlohmann::json& bits) {
  const double numeric = bits.get<double>();
  if (!std::isfinite(numeric)) {
    return std::nullopt;
  }
  const double truncated = std::trunc(numeric);
  if (truncated == kCompressedBits4) {
    return kCompressedBits4;
  }
  if (truncated == kCompressedBits8) {
    return kCompressedBits8;
  }
  return std::nullopt;
}

bool is_compressed_tensors_fp8_scheme(const nlohmann::json& config) {
  auto type_it = config.find("type");
  auto num_bits_it = config.find("num_bits");
  return type_it != config.end() && !type_it->is_null() &&
         num_bits_it != config.end() && !num_bits_it->is_null() &&
         boost::iequals(type_it->get<std::string>(), "float") &&
         supported_compressed_bits(*num_bits_it) == kCompressedBits8;
}

const nlohmann::json* get_compressed_tensors_config(
    const JsonReader& reader,
    const std::string& config_key) {
  const auto quant_method =
      reader.value<std::string>(config_key + ".quant_method");
  if (!quant_method.has_value() ||
      !boost::iequals(*quant_method, "compressed-tensors")) {
    return nullptr;
  }

  const nlohmann::json* quant_config = reader.resolve(config_key);
  if (quant_config == nullptr || !quant_config->is_object()) {
    LOG(ERROR) << config_key << " must be an object for "
               << "compressed-tensors quantization.";
    return nullptr;
  }
  return quant_config;
}

std::optional<int64_t> compressed_int_scheme_bits(const nlohmann::json& scheme,
                                                  bool expected_dynamic) {
  const auto type_it = scheme.find("type");
  const auto bits_it = scheme.find("num_bits");
  const auto dynamic_it = scheme.find("dynamic");
  const bool dynamic = dynamic_it != scheme.end() && !dynamic_it->is_null() &&
                       dynamic_it->get<bool>();
  return type_it != scheme.end() && type_it->is_string() &&
                 boost::iequals(type_it->get<std::string>(), "int") &&
                 bits_it != scheme.end() && !bits_it->is_null() &&
                 dynamic == expected_dynamic
             ? supported_compressed_bits(*bits_it)
             : std::nullopt;
}

bool load_ct_int_config(const nlohmann::json& config, QuantArgs& args) {
  if ((config.contains("kv_cache_scheme") &&
       !config.at("kv_cache_scheme").is_null()) ||
      (config.contains("transform_config") &&
       !config.at("transform_config").is_null() &&
       !config.at("transform_config").empty())) {
    LOG(ERROR) << "Unsupported compressed-tensors KV-cache or transform scheme";
    return false;
  }
  std::vector<CompressedQuantGroup> groups;
  groups.reserve(config.at("config_groups").size());
  for (const auto& [name, group] : config.at("config_groups").items()) {
    if (!group.is_object() || !group.contains("weights") ||
        !group.at("weights").is_object() ||
        !group.contains("input_activations") ||
        !group.at("input_activations").is_object()) {
      LOG(ERROR) << "Invalid compressed-tensors group: " << name;
      return false;
    }
    const auto& weights = group.at("weights");
    const auto& activations = group.at("input_activations");
    CompressedQuantGroup scheme;
    scheme.bits =
        compressed_int_scheme_bits(weights, /*expected_dynamic=*/false)
            .value_or(0);
    scheme.preserve_smooth = weights.value("preserve_smooth", false);
    const bool w4 = scheme.bits == kCompressedBits4;
    const std::string format =
        group.value("format", config.value("format", "int-quantized"));
    if ((scheme.bits != kCompressedBits4 && scheme.bits != kCompressedBits8) ||
        compressed_int_scheme_bits(activations, /*expected_dynamic=*/true) !=
            kCompressedBits8 ||
        format != (w4 ? "int4-le-pack-quantized" : "int-quantized") ||
        !weights.value("symmetric", true) ||
        !activations.value("symmetric", true) ||
        weights.value("strategy", "channel") != (w4 ? "group" : "channel") ||
        activations.value("strategy", "token") != "token" ||
        activations.value("preserve_smooth", false) ||
        (group.contains("output_activations") &&
         !group.at("output_activations").is_null())) {
      LOG(ERROR) << "Unsupported compressed-tensors integer scheme: " << name;
      return false;
    }
    if (w4) {
      constexpr int64_t kW4A8GroupSize = 128;
      if (!weights.contains("group_size") ||
          weights.at("group_size") != kW4A8GroupSize) {
        LOG(ERROR) << "Compressed-tensors W4A8 requires group_size="
                   << kW4A8GroupSize << ": " << name;
        return false;
      }
      scheme.group_size = kW4A8GroupSize;
    }
    scheme.targets = group.value("targets", std::vector<std::string>{"Linear"});
    if (scheme.targets.empty()) {
      LOG(ERROR) << "Empty compressed-tensors targets: " << name;
      return false;
    }
    for (const auto& target : scheme.targets) {
      if (target.rfind("re:", 0) != 0) {
        continue;
      }
      try {
        const std::regex pattern(target.substr(3));
      } catch (const std::regex_error& error) {
        LOG(ERROR) << "Invalid compressed-tensors target: " << target << ": "
                   << error.what();
        return false;
      }
    }
    groups.emplace_back(std::move(scheme));
  }
  if (groups.empty()) {
    LOG(ERROR) << "Compressed-tensors config_groups is empty";
    return false;
  }
  args.compressed_groups() = std::move(groups);
  args.bits() = kCompressedBits8;
  args.is_sym() = true;
  args.activation_dynamic() = true;
  args.is_compressed_tensors_w8a8_dynamic() = true;
  return true;
}

bool load_ct_quant_config(const nlohmann::json& config, QuantArgs& quant_args) {
  quant_args.quant_method() = "compressed-tensors";
  quant_args.ignored_modules() =
      config.value("ignore", std::vector<std::string>{});
  auto config_groups_it = config.find("config_groups");
  if (config_groups_it == config.end() || !config_groups_it->is_object()) {
    LOG(ERROR) << "config_groups must be an object for "
               << "compressed-tensors quantization.";
    return false;
  }

  if (config.value("format", "int-quantized") == "mixed-precision") {
    return load_ct_int_config(config, quant_args);
  }

  for (const auto& [group_name, group] : config_groups_it->items()) {
    if (!group.is_object()) {
      continue;
    }

    auto weights_it = group.find("weights");
    auto input_activations_it = group.find("input_activations");
    if (weights_it == group.end() || input_activations_it == group.end() ||
        !weights_it->is_object() || !input_activations_it->is_object()) {
      continue;
    }

    if (!is_compressed_tensors_fp8_scheme(*weights_it) ||
        !is_compressed_tensors_fp8_scheme(*input_activations_it)) {
      if (compressed_int_scheme_bits(*weights_it, /*expected_dynamic=*/false)
              .has_value() &&
          compressed_int_scheme_bits(*input_activations_it,
                                     /*expected_dynamic=*/true) ==
              kCompressedBits8) {
        return load_ct_int_config(config, quant_args);
      }
      continue;
    }

    quant_args.quant_method() = kQuantMethodFp8;
    quant_args.bits() = kCompressedBits8;
    quant_args.moe_weight_bits() = kCompressedBits8;

    auto dynamic_it = input_activations_it->find("dynamic");
    if (dynamic_it != input_activations_it->end() && !dynamic_it->is_null()) {
      quant_args.activation_dynamic() = dynamic_it->get<bool>();
    }
    if (quant_args.activation_dynamic() && !Platform::is_dcu()) {
      LOG(ERROR) << "Compressed-tensors dynamic FP8 requires DCU.";
      return false;
    }
    return true;
  }

  LOG(ERROR) << "Failed to find a supported FP8 or dynamic INT8 scheme in "
                "compressed-tensors config_groups";
  return false;
}

bool validate_smoothquant_mixed_w4a8(const JsonReader& reader,
                                     QuantArgs& quant_args) {
  const auto expert_weight_precision =
      reader.value<std::string>("quantization_config.expert_weight_precision");
  const int64_t experts_weight_bits =
      reader.value_or<int64_t>("quantization_config.experts_weight_bits", 0);

  const bool is_w4a8 = (expert_weight_precision.has_value() &&
                        boost::iequals(*expert_weight_precision, "int4")) ||
                       experts_weight_bits == 4;
  if (!is_w4a8) {
    return true;
  }

  quant_args.moe_weight_bits() = 4;

  if (!quant_args.only_expert_per_group()) {
    LOG(ERROR) << "DeepSeek mixed W4A8 requires "
                  "quantization_config.only_expert_per_group=true.";
    return false;
  }
  if (quant_args.quant_method() != "smoothquant") {
    LOG(ERROR) << "DeepSeek mixed W4A8 only supports "
                  "quant_method=smoothquant.";
    return false;
  }
  if (quant_args.bits() != 8) {
    LOG(ERROR) << "DeepSeek mixed W4A8 requires quantization_config.bits=8, "
               << "but got bits=" << quant_args.bits();
    return false;
  }

  const auto weight_precision =
      reader.value<std::string>("quantization_config.weight_precision");
  if (weight_precision.has_value() &&
      !boost::iequals(*weight_precision, "int8")) {
    LOG(ERROR) << "DeepSeek mixed W4A8 requires weight_precision=int8, but got "
               << *weight_precision;
    return false;
  }

  const auto activation_precision =
      reader.value<std::string>("quantization_config.activation_precision");
  if (activation_precision.has_value() &&
      !boost::iequals(*activation_precision, "int8")) {
    LOG(ERROR)
        << "DeepSeek mixed W4A8 requires activation_precision=int8, but got "
        << *activation_precision;
    return false;
  }

  const auto expert_activation_precision = reader.value<std::string>(
      "quantization_config.expert_activation_precision");
  if (expert_activation_precision.has_value() &&
      !boost::iequals(*expert_activation_precision, "int8")) {
    LOG(ERROR) << "DeepSeek mixed W4A8 requires "
                  "expert_activation_precision=int8, but got "
               << *expert_activation_precision;
    return false;
  }
  return true;
}

void lift_speculators_config(nlohmann::json& config) {
  if (!config.contains("speculators_model_type") ||
      !config.contains("transformer_layer_config")) {
    return;
  }
  const auto& tlc = config["transformer_layer_config"];
  if (tlc.is_object()) {
    for (const auto& [key, value] : tlc.items()) {
      if (!value.is_null() && !config.contains(key)) {
        config[key] = value;
      }
    }
  }
  if (config.contains("rope_parameters") &&
      config["rope_parameters"].is_object()) {
    const auto& rope = config["rope_parameters"];
    if (rope.contains("rope_theta") && !config.contains("rope_theta")) {
      config["rope_theta"] = rope["rope_theta"];
    }
    if (rope.contains("factor") && !config.contains("rope_scaling")) {
      config["rope_scaling"] = rope;
    }
  }
}

}  // namespace

void normalize_config_torch_dtype(JsonReader& reader) {
  nlohmann::json& config = reader.mutable_data();
  if (!config.contains("torch_dtype") && config.contains("dtype")) {
    config["torch_dtype"] = config["dtype"];
  }
  lift_speculators_config(config);
}

void normalize_speculators_config(nlohmann::json* config) {
  const auto spec_type_it = config->find("speculators_model_type");
  if (spec_type_it == config->end() || !spec_type_it->is_string() ||
      spec_type_it->get<std::string>() != "eagle3") {
    return;
  }
  const auto layer_it = config->find("transformer_layer_config");
  CHECK(layer_it != config->end() && layer_it->is_object())
      << "speculators eagle3 draft config requires a "
         "transformer_layer_config object";
  const std::string layer_model_type = layer_it->value("model_type", "");
  const bool layer_use_qk_norm =
      layer_it->value("use_qk_norm", layer_model_type == "qwen3");
  (*config)["model_type"] = "qwen3_eagle3";
  if (!config->contains("use_qk_norm")) {
    (*config)["use_qk_norm"] = layer_use_qk_norm;
  }
}

bool load_quant_cfg(const JsonReader& reader, QuantArgs& quant_args) {
  if (!reader.contains("quantization_config") &&
      !reader.contains("compression_config")) {
    return true;
  }

  quant_args.only_expert_per_group() =
      reader.value_or<bool>("quantization_config.only_expert_per_group", false);

  if (auto v = reader.value<std::string>("quantization_config.quant_method")) {
    quant_args.quant_method() = v.value();
  }
  const nlohmann::json* ct_config =
      get_compressed_tensors_config(reader, "quantization_config");
  if (ct_config == nullptr) {
    ct_config = get_compressed_tensors_config(reader, "compression_config");
  }
  if (ct_config != nullptr) {
    return load_ct_quant_config(*ct_config, quant_args);
  }
  if (auto v = reader.value<int64_t>("quantization_config.bits")) {
    quant_args.bits() = v.value();
    quant_args.moe_weight_bits() = v.value();
  }
  if (auto v = reader.value<int64_t>("quantization_config.group_size")) {
    quant_args.group_size() = v.value();
  }
  if (auto v = reader.value<bool>("quantization_config.desc_act")) {
    quant_args.desc_act() = v.value();
  }
  if (auto v = reader.value<bool>("quantization_config.sym")) {
    quant_args.is_sym() = v.value();
  }
  if (auto v =
          reader.value<std::string>("quantization_config.activation_scheme")) {
    std::string activation_scheme = v.value();
    if (boost::iequals(activation_scheme, "static")) {
      quant_args.activation_dynamic() = false;
    } else if (boost::iequals(activation_scheme, "dynamic")) {
      quant_args.activation_dynamic() = true;
    } else {
      LOG(ERROR) << "quantization_config.activation_scheme only support "
                    "dynamic and static.";
      return false;
    }
  }
  if (auto v = reader.value<std::string>("quantization_config.fmt")) {
    // TODO(liangzhiwei20): check fp8 quantization format
    quant_args.fmt() = v.value();
  }
  if (const auto* block_size =
          reader.resolve("quantization_config.weight_block_size")) {
    quant_args.weight_block_size() = block_size->get<std::vector<int64_t>>();
  }

  return validate_smoothquant_mixed_w4a8(reader, quant_args);
}

void configure_draft_model_args(ModelArgs& args,
                                std::string_view algorithm,
                                [[maybe_unused]] bool is_python_model) {
  if (SpeculativeConfig::is_block_diffusion_algorithm(algorithm)) {
    args.layers_to_capture({});
    if (SpeculativeConfig::is_dspark_algorithm(algorithm)) {
      if (is_deepseek_v4_family(args.model_type())) {
        args.model_type(/*value=*/"deepseek_v4_dspark");
#if defined(USE_NPU)
        args.n_layers(args.dspark_num_layers());
        args.n_hash_layers(/*value=*/0);
        args.dspark_use_native_sas(
            KernelConfig::get_instance().enable_dspark_native_sas());
        args.compress_ratios(
            std::vector<int32_t>(args.n_layers(), /*value=*/1));
#endif
      } else if (args.model_type() != "glm5_dspark") {
        args.model_type(/*value=*/"qwen3_dspark");
      }
    } else {
      args.model_type(SpeculativeConfig::is_dflash2_algorithm(algorithm)
                          ? "qwen3_dflash2"
                          : "qwen3_dflash");
    }
    return;
  }
  if (!SpeculativeConfig::is_mtp_algorithm(algorithm)) {
    return;
  }

  [[maybe_unused]] const std::string_view model_type = args.model_type();
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_MUSA)
  const bool is_qwen35_dense =
      model_type == "qwen3_5" || model_type == "qwen3_5_text" ||
      (Platform::is_mlu() && model_type == "qwen3_5_mtp");
  const bool is_qwen35_moe =
      model_type == "qwen3_5_moe" || model_type == "qwen3_5_moe_text" ||
      (Platform::is_mlu() && model_type == "qwen3_5_moe_mtp");
  if (is_qwen35_dense || is_qwen35_moe) {
    int32_t draft_layers = args.num_nextn_predict_layers();
#if defined(USE_MLU)
    if (is_python_model && draft_layers <= 0) {
      return;
    }
#else
    if (draft_layers <= 0) {
      return;
    }
#endif
    args.model_type(is_qwen35_dense ? "qwen3_5_mtp" : "qwen3_5_moe_mtp");
#if defined(USE_MLU)
    if (is_python_model) {
      return;
    }
    draft_layers = std::max(draft_layers, /*minimum=*/1);
    args.num_nextn_predict_layers(draft_layers);
#endif
#if defined(USE_NPU) || defined(USE_MLU)
    args.n_layers(draft_layers);
    args.layer_types(std::vector<std::string>(draft_layers,
                                              /*value=*/"full_attention"));
    args.full_attention_interval(/*value=*/1);
#endif
    return;
  }
#endif

#if defined(USE_NPU) || defined(USE_MLU)
  if (is_glm5_next_model(model_type)) {
#if defined(USE_MLU)
    if (!is_python_model) {
      args.index_share_for_mtp_iteration(/*value=*/false);
    }
#endif
    CHECK_EQ(args.num_nextn_predict_layers(), 1)
        << "GLM MTP requires exactly one appended prediction layer.";
    CHECK_GT(args.n_layers(), 0);
    args.mtp_start_layer_idx(args.n_layers());
    args.model_type(/*value=*/"glm5_next_mtp");
    args.n_layers(/*value=*/1);
    args.first_k_dense_replace(/*value=*/0);
    args.layer_types({"deepseek_sparse_attention"});
    args.full_attn_layers({0});
    args.mlp_layer_types({"sparse"});
    args.indexer_types({"full"});
    if (args.index_share_for_mtp_iteration()) {
      args.index_topk_freq(std::max(args.index_topk_freq(), /*minimum=*/2));
      args.index_skip_topk_offset(/*value=*/0);
      args.index_topk_pattern(/*value=*/"S");
    }
    return;
  }
#endif

#if defined(USE_MLU)
  if (model_type == "glm5_next_mtp" && !is_python_model) {
    args.index_share_for_mtp_iteration(/*value=*/false);
  }
#endif
#if defined(USE_MLU) || defined(USE_CUDA) || defined(USE_DCU)
  if (args.num_nextn_predict_layers() > 0) {
    constexpr ModelArgsAlias kEmbeddedMtpModels[] = {
#if defined(USE_MLU)
        {"deepseek_v3", "deepseek_v3_mtp"},
        {"deepseek_v4", "deepseek_v4_mtp"},
        {"glm_moe_dsa", "glm_moe_dsa_mtp"},
        {"joyai_llm_flash", "joyai_llm_flash_mtp"},
#else
        {"mimo", "mimo_mtp"},
#endif
    };
    const auto model = std::ranges::find(
        kEmbeddedMtpModels, model_type, &ModelArgsAlias::model_type);
    if (model != std::end(kEmbeddedMtpModels)) {
      args.model_type(model->resolved_type);
    }
  }
#endif
}

bool load_hf_model_args(const std::filesystem::path& model_weights_path,
                        const std::string& backend,
                        ModelArgs* args) {
  JsonReader reader;
  const std::filesystem::path args_file_path =
      model_weights_path / "config.json";
  if (!reader.parse(args_file_path.string())) {
    LOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  normalize_speculators_config(&reader.mutable_data());

  const std::string model_type =
      get_model_type(reader, model_weights_path, backend);

  std::string resolved_model_type;
  std::string error_message;
  if (!resolve_model_registration_name(
          model_type, &resolved_model_type, &error_message)) {
    LOG(ERROR) << error_message;
    return false;
  }

  const auto alias = std::ranges::find(
      kModelArgsAliases, resolved_model_type, &ModelArgsAlias::model_type);
  const auto model_args_loader = ModelRegistry::get_model_args_loader(
      alias == std::end(kModelArgsAliases) ? resolved_model_type
                                           : alias->resolved_type);
  if (model_args_loader == nullptr) {
    LOG(ERROR) << "Failed to find model args loader for model type "
               << resolved_model_type;
    return false;
  }
  normalize_config_torch_dtype(reader);
  if (!model_args_loader(reader, args)) {
    LOG(ERROR) << "Failed to load model args for model type "
               << resolved_model_type;
    return false;
  }
#if defined(USE_NPU) || defined(USE_MLU)
  if (resolved_model_type == "deepseek_v3_mtp") {
    args->first_k_dense_replace(
        reader.value_or<int32_t>("first_k_dense_replace", 0));
#if defined(USE_MLU)
    args->index_head_dim(reader.value_or<int32_t>("index_head_dim", 128));
    args->index_n_heads(reader.value_or<int32_t>("index_n_heads", 64));
    args->index_topk(reader.value_or<int32_t>("index_topk", 2048));
#endif
  } else if (resolved_model_type == "deepseek_v32_mtp") {
    args->index_n_heads(reader.value_or<int32_t>("index_n_heads", 64));
    args->index_share_for_mtp_iteration(
        reader.value_or<bool>("index_share_for_mtp_iteration", false));
  }
#endif
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_MUSA)
  if (resolved_model_type == "qwen3_5_mtp" ||
      resolved_model_type == "qwen3_5_moe_mtp") {
    const int32_t draft_layers =
        std::max(args->num_nextn_predict_layers(), /*minimum=*/1);
    args->num_nextn_predict_layers(draft_layers);
    args->n_layers(draft_layers);
    args->layer_types(
        std::vector<std::string>(draft_layers, /*value=*/"full_attention"));
  }
#endif
  if (alias != std::end(kModelArgsAliases) &&
      resolved_model_type.ends_with(/*suffix=*/"_mtp")) {
    args->model_type(resolved_model_type);
  }
  // Qwen3 block drafts enable a positive window when the source omits the
  // switch. Resolve that default before reflection loses the distinction
  // between an absent switch and an explicit false; keep target defaults.
  if (resolved_model_type == "qwen3" &&
      (args->markov_rank() > 0 || args->block_size() > 0) &&
      !reader.value<bool>("use_sliding_window").has_value()) {
    args->use_sliding_window(args->sliding_window() > 0);
  }
  args->enable_mla(should_enable_mla(model_weights_path, backend));

  return true;
}

bool load_hf_audio_preprocessor_args(
    const std::filesystem::path& model_weights_path,
    ModelArgs& args) {
  JsonReader reader;
  std::filesystem::path config_path =
      model_weights_path / "audio_preprocessor_config.json";
  if (!reader.parse(config_path.string())) {
    config_path = model_weights_path / "preprocessor_config.json";
    // The shared file may describe an image/video processor instead.
    if (!reader.parse(config_path.string()) ||
        !reader.contains("sampling_rate")) {
      return true;
    }
  }
  LOG(INFO) << "Success to parse audio preprocess args file: "
            << config_path.string();
  args.mm_audio_num_mel_bins() = reader.value_or<int64_t>(
      std::vector<std::string>{"num_mel_bins", "feature_size"}, 80);
  args.mm_audio_max_frames() = reader.value_or<int64_t>("max_length", 3000);
  // The model-args registration runs first; an absent key must not clobber it.
  args.mm_audio_downsample_rate() = reader.value_or<int64_t>(
      "downsample_rate", args.mm_audio_downsample_rate());
  // The frontend enforces its 16k contract at construction.
  args.mm_audio_sampling_rate() =
      reader.value_or<int64_t>("sampling_rate", 16000);

  const int64_t sampling_rate = args.mm_audio_sampling_rate();
  const double frame_length = reader.value_or<double>("frame_length", 25.0);
  const double frame_shift = reader.value_or<double>("frame_shift", 10.0);
  const bool in_samples = frame_length > 100 || frame_shift > 100;
  const double frame_length_samples =
      in_samples ? frame_length : frame_length * sampling_rate / 1000;
  const double frame_shift_samples =
      in_samples ? frame_shift : frame_shift * sampling_rate / 1000;
  if (!std::isfinite(frame_length_samples) ||
      !std::isfinite(frame_shift_samples) || frame_length_samples < 2 ||
      frame_shift_samples < 1 || frame_length_samples >= std::ldexp(1.0, 63) ||
      frame_shift_samples >= std::ldexp(1.0, 63)) {
    LOG(ERROR) << "Invalid audio frame length or shift in preprocessor config.";
    return false;
  }
  args.mm_audio_frame_length() = static_cast<int64_t>(frame_length_samples);
  args.mm_audio_frame_shift() = static_cast<int64_t>(frame_shift_samples);

  if (const auto* means = reader.resolve("means");
      means != nullptr && means->is_array()) {
    args.mm_audio_cmvn_means() = means->get<std::vector<double>>();
  }
  if (const auto* inverse_std = reader.resolve("inverse_std_variences");
      inverse_std != nullptr && inverse_std->is_array()) {
    args.mm_audio_cmvn_inverse_std() = inverse_std->get<std::vector<double>>();
  }
  return true;
}

bool load_hf_image_preprocessor_args(
    const std::filesystem::path& model_weights_path,
    ModelArgs& args) {
  // image preprocessor args
  JsonReader image_reader;
  const std::filesystem::path flat_file_path =
      model_weights_path / "preprocessor_config.json";
  const std::filesystem::path nested_file_path =
      model_weights_path / "processor_config.json";
  bool parsed = false;
  std::string used_file_path;
  if (image_reader.parse(flat_file_path.string())) {
    parsed = true;
    used_file_path = flat_file_path.string();
  } else if (image_reader.parse(nested_file_path.string())) {
    // GLM-5.3-Flash ships a single ``processor_config.json`` whose image
    // fields live under the ``image_processor`` sub-object. Resolve every key
    // as ``image_processor.<key>`` first, then the flat top-level form, so
    // legacy flat ``preprocessor_config.json`` files keep working.
    parsed = true;
    used_file_path = nested_file_path.string();
  }
  if (parsed) {
    LOG(INFO) << "Success to parse image preprocess args file: "
              << used_file_path;
    const nlohmann::json root = image_reader.data();
    // Resolve a key against the nested ``image_processor`` sub-object first,
    // then the flat top-level layout; returns nullptr if neither exists.
    auto resolve_field = [&](const std::string& key) -> const nlohmann::json* {
      if (auto* ptr =
              JsonReader::resolve_path(root, "image_processor." + key)) {
        return ptr;
      }
      return JsonReader::resolve_path(root, key);
    };
    auto field_int = [&](const std::string& key, int default_value) -> int {
      const auto* ptr = resolve_field(key);
      if (ptr != nullptr && !ptr->is_null() && !ptr->is_object()) {
        return ptr->get<int>();
      }
      return default_value;
    };
    auto field_bool = [&](const std::string& key, bool default_value) -> bool {
      const auto* ptr = resolve_field(key);
      if (ptr != nullptr && !ptr->is_null() && !ptr->is_object()) {
        return ptr->get<bool>();
      }
      return default_value;
    };
    auto field_double = [&](const std::string& key,
                            double default_value) -> double {
      const auto* ptr = resolve_field(key);
      if (ptr != nullptr && !ptr->is_null() && !ptr->is_object()) {
        return ptr->get<double>();
      }
      return default_value;
    };
    auto field_double_vec =
        [&](const std::string& key) -> std::optional<std::vector<double>> {
      const auto* ptr = resolve_field(key);
      if (ptr != nullptr && ptr->is_array()) {
        return ptr->get<std::vector<double>>();
      }
      return std::nullopt;
    };

    args.mm_image_do_center_crop() = field_bool("do_center_crop", false);
    args.mm_image_crop_height_size() = field_int("crop_size.height", 335);
    args.mm_image_crop_width_size() = field_int("crop_size.width", 335);

    args.mm_image_do_resize() = field_bool("do_resize", false);
    args.mm_image_resize_shortest_edge() = field_int("size.shortest_edge", 335);
    args.mm_image_resample() = field_int("resample", 335);

    args.mm_image_do_rescale() = field_bool("do_rescale", false);
    args.mm_image_rescale_factor() = field_double("rescale_factor", 0);

    args.mm_image_do_normalize() = field_bool("do_normalize", false);

    if (auto v = field_double_vec("image_mean")) {
      args.mm_image_normalize_mean() = std::move(*v);
    }
    if (auto v = field_double_vec("image_std")) {
      args.mm_image_normalize_std() = std::move(*v);
    }
    if (auto v = field_double_vec("norm_mean")) {
      args.mm_image_normalize_mean() = std::move(*v);
    }
    if (auto v = field_double_vec("norm_std")) {
      args.mm_image_normalize_std() = std::move(*v);
    }

    args.mm_image_shortest_edge() = field_int("size.shortest_edge", 0);
    args.mm_image_longest_edge() = field_int("size.longest_edge", 0);

    args.mm_image_min_pixels() = field_int("min_pixels", 0);
    args.mm_image_max_pixels() = field_int("max_pixels", 0);

    // per-token resize budget (token units; multiplied by pixels_per_token
    // inside the image processor). When both are > 0 the token-budget
    // smart_resize is used, matching HF Glm5NextImageProcessor.
    args.mm_image_min_tokens() = field_int("min_image_tokens", 0);
    args.mm_image_max_tokens() = field_int("max_image_tokens", 0);

    args.mm_image_patch_size() = field_int("patch_size", 0);
    args.mm_image_temporal_patch_size() = field_int("temporal_patch_size", 0);
    args.mm_image_merge_size() = field_int("merge_size", 0);

    args.mm_image_feature_size() = field_int("image_feature_size", 0);
    args.mm_scale_resolution() = field_int("scale_resolution", 0);
    args.mm_slice_mode() = field_bool("slice_mode", false);
    args.mm_use_image_id() = field_bool("use_image_id", false);
  }

  return true;
}

bool load_hf_video_preprocessor_args(
    const std::filesystem::path& model_weights_path,
    ModelArgs& args) {
  // video preprocessor args
  JsonReader video_reader;
  const std::filesystem::path flat_file_path =
      model_weights_path / "video_preprocessor_config.json";
  const std::filesystem::path nested_file_path =
      model_weights_path / "processor_config.json";
  bool parsed = false;
  std::string used_file_path;
  if (video_reader.parse(flat_file_path.string())) {
    parsed = true;
    used_file_path = flat_file_path.string();
  } else if (video_reader.parse(nested_file_path.string())) {
    // GLM-5.3-Flash nests video fields under ``video_processor``.
    parsed = true;
    used_file_path = nested_file_path.string();
  }
  if (parsed) {
    LOG(INFO) << "Success to parse video preprocess args file: "
              << used_file_path;
    const nlohmann::json root = video_reader.data();
    auto resolve_field = [&](const std::string& key) -> const nlohmann::json* {
      if (auto* ptr =
              JsonReader::resolve_path(root, "video_processor." + key)) {
        return ptr;
      }
      return JsonReader::resolve_path(root, key);
    };
    auto field_int = [&](const std::string& key, int default_value) -> int {
      const auto* ptr = resolve_field(key);
      if (ptr != nullptr && !ptr->is_null() && !ptr->is_object()) {
        return ptr->get<int>();
      }
      return default_value;
    };
    auto field_bool = [&](const std::string& key, bool default_value) -> bool {
      const auto* ptr = resolve_field(key);
      if (ptr != nullptr && !ptr->is_null() && !ptr->is_object()) {
        return ptr->get<bool>();
      }
      return default_value;
    };
    auto field_double_vec =
        [&](const std::string& key) -> std::optional<std::vector<double>> {
      const auto* ptr = resolve_field(key);
      if (ptr != nullptr && ptr->is_array()) {
        return ptr->get<std::vector<double>>();
      }
      return std::nullopt;
    };

    args.mm_video_shortest_edge() = field_int("size.shortest_edge", 0);
    args.mm_video_longest_edge() = field_int("size.longest_edge", 0);

    if (auto v = field_double_vec("image_mean")) {
      args.mm_video_normalize_mean() = std::move(*v);
    }
    if (auto v = field_double_vec("image_std")) {
      args.mm_video_normalize_std() = std::move(*v);
    }

    args.mm_video_patch_size() = field_int("patch_size", 0);
    args.mm_video_temporal_patch_size() = field_int("temporal_patch_size", 0);
    args.mm_video_merge_size() = field_int("merge_size", 0);
    args.mm_video_do_rescale() = field_bool("do_rescale", false);

    // per-token resize budget for video (token units). When both are > 0
    // the token-budget smart_resize is used.
    args.mm_video_min_tokens() = field_int("min_image_tokens", 0);
    args.mm_video_max_tokens() = field_int("max_image_tokens", 0);
  }

  return true;
}

}  // namespace xllm
