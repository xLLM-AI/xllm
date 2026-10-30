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

#include "model_registry.h"

#include <glog/logging.h>

#include <iostream>
#include <mutex>
#include <unordered_set>
#include <utility>

#include "core/common/extension_registry.h"
#include "core/common/options.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_loader/hf_model_config.h"
#include "core/util/dit_model_discovery.h"
#include "llm/py_causal_lm.h"
#include "models.h"
#include "processors/cacheable_multimodal_processor.h"

namespace {

// Safe logging macro to avoid crashes during static initialization
#define SAFE_LOG_WARNING(message)                       \
  do {                                                  \
    if (google::IsGoogleLoggingInitialized()) {         \
      LOG(WARNING) << message;                          \
    } else {                                            \
      std::cerr << "WARNING: " << message << std::endl; \
    }                                                   \
  } while (0)

#define SAFE_LOG_ERROR(message)                       \
  do {                                                \
    if (google::IsGoogleLoggingInitialized()) {       \
      LOG(ERROR) << message;                          \
    } else {                                          \
      std::cerr << "ERROR: " << message << std::endl; \
    }                                                 \
  } while (0)

#define SAFE_LOG_INFO(message)                       \
  do {                                               \
    if (google::IsGoogleLoggingInitialized()) {      \
      LOG(INFO) << message;                          \
    } else {                                         \
      std::cerr << "INFO: " << message << std::endl; \
    }                                                \
  } while (0)

}  // anonymous namespace

namespace xllm {

namespace {

std::string canonical_model_impl(std::string model_impl) {
  if (model_impl.empty() || model_impl == "auto" || model_impl == "native" ||
      model_impl == "cpp") {
    return "native";
  }
  if (ModelConfig::is_python_model_impl(model_impl)) {
    return "python";
  }
  return model_impl;
}

#if defined(USE_NPU)
constexpr char kAutoBackend[] = "AUTO";
constexpr char kAtbBackend[] = "ATB";
constexpr char kTorchBackend[] = "TORCH";
constexpr char kXliteBackend[] = "XLITE";

bool is_torch_only_model_type(const std::string& model_type) {
  static const std::unordered_set<std::string> kTorchOnlyModelTypes = {
      "deepseek_v4",
      "deepseek_v4_dspark",
      "deepseek_v4_mtp",
      "glm5_next_mtp",
      "glm5_dspark",
      "qwen3_5",
      "qwen3_5_text",
      "qwen3_5_moe",
      "qwen3_5_moe_text",
      "qwen3_5_mtp",
      "qwen3_5_moe_mtp",
      "qwen3_next",
      "minimax_m2"};
  return kTorchOnlyModelTypes.count(model_type) != 0 ||
         is_dflash2_draft_model_type(model_type);
}
#endif

bool validate_implementation_selection(const ModelContext& context) {
  const std::string model_impl = canonical_model_impl(context.get_model_impl());
  if (model_impl != "native" && model_impl != "python") {
    LOG(ERROR) << "Unsupported model_impl=" << context.get_model_impl()
               << "; use extension_impl to select a native extension family";
    return false;
  }
  const std::string& extension_impl = context.get_extension_impl();
  if (!extension_impl.empty() &&
      !ExtensionRegistry::get_instance().has_implementation(extension_impl)) {
    LOG(ERROR) << "Unknown extension_impl=" << extension_impl;
    return false;
  }
  return true;
}

}  // namespace

bool resolve_model_registration(const std::string& model_type,
                                const std::string& requested_npu_kernel_backend,
                                std::string* effective_npu_kernel_backend,
                                std::string* resolved_name,
                                std::string* error_message) {
  return resolve_model_registration(model_type,
                                    requested_npu_kernel_backend,
                                    /*extension_impl=*/"",
                                    /*model_component=*/"",
                                    effective_npu_kernel_backend,
                                    resolved_name,
                                    error_message);
}

bool resolve_model_registration(const std::string& model_type,
                                const std::string& requested_npu_kernel_backend,
                                const std::string& extension_impl,
                                std::string_view model_component,
                                std::string* effective_npu_kernel_backend,
                                std::string* resolved_name,
                                std::string* error_message) {
  if (resolved_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "resolved_name must not be null";
    }
    return false;
  }

  const std::string canonical_model_type =
      canonical_model_registration_name(model_type);

#if defined(USE_NPU)
  bool selected_extension_model = false;
  if (!extension_impl.empty()) {
    const auto& extensions = ExtensionRegistry::get_instance();
    if (!extensions.has_implementation(extension_impl)) {
      if (error_message != nullptr) {
        *error_message = "Unknown extension_impl=" + extension_impl;
      }
      return false;
    }
    const std::string component(model_component);
    if (extensions.has_implementation(extension_impl, component)) {
      if (model_component == "causal_lm") {
        selected_extension_model =
            static_cast<bool>(ModelRegistry::get_causallm_factory(
                canonical_model_type, extension_impl));
      } else if (model_component == "causal_vlm") {
        selected_extension_model =
            static_cast<bool>(ModelRegistry::get_causalvlm_factory(
                canonical_model_type, extension_impl));
      }
      if (!selected_extension_model) {
        if (error_message != nullptr) {
          *error_message =
              "Missing " + component +
              " extension for model type: " + canonical_model_type +
              " with extension_impl=" + extension_impl;
        }
        return false;
      }
    }
  }

  const std::string backend = requested_npu_kernel_backend.empty()
                                  ? kAutoBackend
                                  : requested_npu_kernel_backend;
  if (backend != kAutoBackend && backend != kAtbBackend &&
      backend != kTorchBackend && backend != kXliteBackend) {
    if (error_message != nullptr) {
      *error_message = "Unsupported --npu_kernel_backend=" + backend +
                       ". Supported values: AUTO, ATB, TORCH, XLITE.";
    }
    return false;
  }

  if (selected_extension_model) {
#if !defined(USE_XLITE)
    if (backend == kXliteBackend) {
      if (error_message != nullptr) {
        *error_message =
            "XLITE backend is not compiled; please build with USE_XLITE=ON.";
      }
      return false;
    }
#endif
    if (effective_npu_kernel_backend != nullptr) {
      *effective_npu_kernel_backend =
          backend == kAutoBackend ? kTorchBackend : backend;
    }
    *resolved_name = canonical_model_type;
    return true;
  }

  std::string effective_backend = backend;
  if (backend == kAutoBackend) {
    effective_backend = is_torch_only_model_type(canonical_model_type)
                            ? kTorchBackend
                            : kAtbBackend;
  } else if (backend == kXliteBackend) {
#if defined(USE_XLITE)
    // Reject torch-only model types
    if (is_torch_only_model_type(canonical_model_type)) {
      if (error_message != nullptr) {
        *error_message =
            "Model type " + model_type + " only supports TORCH, not XLITE.";
      }
      return false;
    }
    // Reject non-xlite model types
    if (!xllm::xlite::is_xlite_model_type(canonical_model_type)) {
      if (error_message != nullptr) {
        *error_message = "Model type " + model_type +
                         " has no XLITE implementation; use ATB or TORCH"
                         " backend instead.";
      }
      return false;
    }
#else
    if (error_message != nullptr) {
      *error_message =
          "XLITE backend is not compiled; please build with USE_XLITE=ON.";
    }
    return false;
#endif
  } else if (canonical_model_type == "qwen3" ||
             canonical_model_type == "qwen3_moe" ||
             canonical_model_type == "deepseek_v32" ||
             canonical_model_type == "glm_moe_dsa" ||
             canonical_model_type == "glm_moe_dsa_mtp" ||
             canonical_model_type == "qwen3_vl" ||
             canonical_model_type == "deepseek_v32_mtp" ||
             is_glm5_next_target_model_type(canonical_model_type) ||
             canonical_model_type == "joyai_asr") {
    // qwen3/qwen3_moe/deepseek_v32/glm_moe_dsa/glm_moe_dsa_mtp/qwen3_vl/
    // deepseek_v32_mtp and GLM5 target aliases support both backends.
    // qwen3_vl on TORCH is used by the Python model executor
    // (--model_impl=python implements its own ViT + deepstack).
  } else if (is_torch_only_model_type(canonical_model_type)) {
    if (backend != kTorchBackend) {
      if (error_message != nullptr) {
        *error_message = "Model type " + model_type +
                         " only supports --npu_kernel_backend=TORCH.";
      }
      return false;
    }
  } else if (backend != kAtbBackend) {
    if (error_message != nullptr) {
      *error_message = "Model type " + model_type +
                       " only supports --npu_kernel_backend=ATB.";
    }
    return false;
  }

  if (effective_npu_kernel_backend != nullptr) {
    *effective_npu_kernel_backend = effective_backend;
  }
  if (backend == kXliteBackend) {
    *resolved_name = canonical_model_type + "_xlite";
  } else if (canonical_model_type == "qwen3" &&
             effective_backend == kAtbBackend) {
    *resolved_name = "qwen3_atb";
  } else if (canonical_model_type == "qwen3_moe" &&
             effective_backend == kAtbBackend) {
    *resolved_name = "qwen3_moe_atb";
  } else if (canonical_model_type == "qwen2" &&
             effective_backend == kAtbBackend) {
    *resolved_name = "qwen2_atb";
  } else if (canonical_model_type == "qwen2_5_vl" &&
             effective_backend == kAtbBackend) {
    *resolved_name = "qwen2_5_vl_atb";
  } else if (canonical_model_type == "qwen3_vl" &&
             effective_backend == kAtbBackend) {
    *resolved_name = "qwen3_vl_atb";
  } else {
    *resolved_name = canonical_model_type;
  }
  return true;
#else
  *resolved_name = canonical_model_type;
  return true;
#endif
}

#if defined(USE_NPU)
void resolve_npu_kernel_backend(Options* options,
                                std::string_view model_component) {
  CHECK(options != nullptr) << "options must not be null";

  // Python model executor builds the compute graph in Python (torch/torch_npu),
  // bypassing ATB C++ kernels entirely. Force TORCH so kernel dispatch picks
  // pure-torch implementations for reshape_and_cache and related operators.
  if (ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl())) {
    options->npu_kernel_backend("TORCH");
    KernelConfig::get_instance().npu_kernel_backend("TORCH");
    LOG(INFO) << "Forced npu_kernel_backend=TORCH for python model_impl";
    return;
  }

  const std::string model_type =
      get_model_type(options->model_path(), options->backend());
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration(model_type,
                                  options->npu_kernel_backend(),
                                  options->extension_impl(),
                                  model_component,
                                  &effective_backend,
                                  &resolved_name,
                                  &error_message)) {
    LOG(FATAL) << error_message;
  }

  options->npu_kernel_backend(effective_backend);
  KernelConfig::get_instance().npu_kernel_backend(effective_backend);
  LOG(INFO) << "Resolved npu_kernel_backend=" << effective_backend
            << " for model_type=" << model_type;
}
#endif

int64_t get_npu_linear_state_cache_block_limit(const std::string& model_type) {
#if defined(USE_NPU)
  // Native and Python Qwen3.5 use the same MegaGdn state-slot layout.
  if (is_qwen3_5_target_model_type(model_type) || model_type == "qwen3_5_mtp" ||
      model_type == "qwen3_5_moe_mtp") {
    return 1024;
  }
#endif
  return 0;
}

bool resolve_model_registration_name(const std::string& model_type,
                                     std::string* resolved_name,
                                     std::string* error_message) {
#if defined(USE_NPU)
  return resolve_model_registration(
      model_type,
      ::xllm::KernelConfig::get_instance().npu_kernel_backend(),
      nullptr,
      resolved_name,
      error_message);
#else
  return resolve_model_registration(
      model_type, "", nullptr, resolved_name, error_message);
#endif
}

bool is_npu_model_cp_capable(const std::string& resolved_name) {
  // Registers model-side CP capability for master-side validation. Note this
  // is not the same switch as the worker-side NpuCpPlan gate: deepseek_v4 and
  // deepseek_v4_mtp own their CP split inside the model (TORCH backend) and
  // deliberately keep model_supports_model_cp() false so the worker does not
  // shard a second time. Python GLM5 models are admitted by the Python CP
  // branch before this native capability check and likewise must not enable
  // the ATB worker sharder.
  static const std::unordered_set<std::string> kCpCapableModels = {
      "deepseek_v32",
      "deepseek_v32_mtp",
      "deepseek_v4",
      "deepseek_v4_mtp",
      "glm_moe_dsa",
      "glm_moe_dsa_mtp",
  };
  static std::once_flag once;
  std::call_once(once, []() {
    for (const std::string& name : kCpCapableModels) {
      ModelRegistry::register_cp_sharding_mode(name, CpShardingMode::MODEL);
    }
  });
  return kCpCapableModels.contains(resolved_name);
}

bool is_npu_model_dcp_capable(const std::string& resolved_name) {
  return ModelRegistry::supports_dcp(resolved_name);
}

bool is_mlu_model_cp_capable(const std::string& resolved_name) {
  static const std::unordered_set<std::string> kCpCapableModels = {
      "deepseek_v4",
      "glm_moe_dsa",
  };
  static std::once_flag once;
  std::call_once(once, []() {
    for (const std::string& name : kCpCapableModels) {
      ModelRegistry::register_cp_sharding_mode(name, CpShardingMode::MODEL);
    }
  });
  return kCpCapableModels.contains(resolved_name);
}

ModelRegistry* ModelRegistry::get_instance() {
  static ModelRegistry registry;

  return &registry;
}

void ModelRegistry::register_causallm_factory(const std::string& name,
                                              CausalLMFactory factory) {
  register_causallm_factory(name, std::move(factory), /*priority=*/0);
}

bool ModelRegistry::register_causallm_factory(std::string name,
                                              CausalLMFactory factory,
                                              int32_t priority) {
  return register_causallm_factory(
      std::move(name), "native", std::move(factory), priority);
}

bool ModelRegistry::register_causallm_factory(std::string name,
                                              std::string implementation,
                                              CausalLMFactory factory,
                                              int32_t priority) {
  CHECK(factory != nullptr) << "Cannot register an empty causal LM factory";
  ModelRegistry* instance = get_instance();
  auto& factories = instance->model_registry_[name].causal_lm_factories;
  implementation = canonical_model_impl(std::move(implementation));
  auto it = factories.find(implementation);
  if (it != factories.end() && priority <= it->second.priority) {
    return false;
  }

  if (implementation != "native" && implementation != "python") {
    ExtensionRegistry::get_instance().register_implementation(implementation,
                                                              "causal_lm");
  }
  factories.insert_or_assign(
      std::move(implementation),
      CausalLMRegistration{std::move(factory), priority});
  instance->model_backend_[std::move(name)] = "llm";
  return true;
}

void ModelRegistry::register_rec_model_factory(const std::string& name,
                                               RecModelFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].rec_model_factory != nullptr) {
    SAFE_LOG_WARNING("rec model factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].rec_model_factory = factory;
    std::string& backend = instance->model_backend_[name];
    if (backend.empty()) {
      backend = "rec";
    }
  }
}

void ModelRegistry::register_causalvlm_factory(const std::string& name,
                                               CausalVLMFactory factory) {
  register_causalvlm_factory(name, std::move(factory), /*priority=*/0);
}

bool ModelRegistry::register_causalvlm_factory(std::string name,
                                               CausalVLMFactory factory,
                                               int32_t priority) {
  return register_causalvlm_factory(
      std::move(name), "native", std::move(factory), priority);
}

bool ModelRegistry::register_causalvlm_factory(std::string name,
                                               std::string implementation,
                                               CausalVLMFactory factory,
                                               int32_t priority) {
  CHECK(factory != nullptr) << "Cannot register an empty causal VLM factory";
  ModelRegistry* instance = get_instance();
  auto& factories = instance->model_registry_[name].causal_vlm_factories;
  implementation = canonical_model_impl(std::move(implementation));
  auto it = factories.find(implementation);
  if (it != factories.end() && priority <= it->second.priority) {
    return false;
  }

  if (implementation != "native" && implementation != "python") {
    ExtensionRegistry::get_instance().register_implementation(implementation,
                                                              "causal_vlm");
  }
  factories.insert_or_assign(
      std::move(implementation),
      CausalVLMRegistration{std::move(factory), priority});
  instance->model_backend_[std::move(name)] = "vlm";
  return true;
}

void ModelRegistry::register_dit_model_factory(const std::string& name,
                                               DiTModelFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].dit_model_factory != nullptr) {
    SAFE_LOG_WARNING("DiT model factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].dit_model_factory = factory;
    instance->model_backend_[name] = "dit";
  }
}

void ModelRegistry::register_model_backend(const std::string& name,
                                           const std::string& backend) {
  ModelRegistry* instance = get_instance();
  auto [it, inserted] = instance->model_backend_.emplace(name, backend);
  if (!inserted && it->second != backend) {
    SAFE_LOG_WARNING("model backend for "
                     << name << " already registered as " << it->second
                     << "; ignoring conflicting backend " << backend << ".");
  }
}

void ModelRegistry::register_multimodal_processor_factory(
    const std::string& name,
    MultimodalProcessorFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].multimodal_processor_factory != nullptr) {
    SAFE_LOG_WARNING("multimodal processor factory for "
                     << name << " already registered.");
  } else {
    instance->model_registry_[name].multimodal_processor_factory =
        std::move(factory);
  }
}

void ModelRegistry::register_model_args_loader(const std::string& name,
                                               ModelArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].model_args_loader != nullptr) {
    SAFE_LOG_WARNING("model args loader for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].model_args_loader = loader;
  }
}

void ModelRegistry::register_quant_args_loader(const std::string& name,
                                               QuantArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].quant_args_loader != nullptr) {
    SAFE_LOG_WARNING("quant args loader for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].quant_args_loader = loader;
  }
}

void ModelRegistry::register_tokenizer_args_loader(const std::string& name,
                                                   TokenizerArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].tokenizer_args_loader != nullptr) {
    SAFE_LOG_WARNING("tokenizer args loader for " << name
                                                  << " already registered.");
  } else {
    instance->model_registry_[name].tokenizer_args_loader = loader;
  }
}

void ModelRegistry::register_cp_sharding_mode(const std::string& name,
                                              CpShardingMode mode) {
  ModelRegistry* instance = get_instance();
  instance->model_registry_[name].cp_sharding_mode = mode;
}

CpShardingMode ModelRegistry::get_cp_sharding_mode(const std::string& name) {
  ModelRegistry* instance = get_instance();
  const auto it = instance->model_registry_.find(name);
  if (it == instance->model_registry_.end()) {
    return CpShardingMode::NONE;
  }
  return it->second.cp_sharding_mode;
}

void ModelRegistry::register_dcp_capability(const std::string& name) {
  get_instance()->model_registry_[name].supports_dcp = true;
}

bool ModelRegistry::supports_dcp(const std::string& name) {
  const ModelRegistry* instance = get_instance();
  const auto it = instance->model_registry_.find(name);
  return it != instance->model_registry_.end() && it->second.supports_dcp;
}

CausalLMFactory ModelRegistry::get_causallm_factory(const std::string& name) {
  return get_causallm_factory(name, "native");
}

CausalLMFactory ModelRegistry::get_causallm_factory(
    const std::string& name,
    const std::string& implementation_key) {
  const ModelRegistry* instance = get_instance();
  const auto model = instance->model_registry_.find(name);
  if (model == instance->model_registry_.end()) {
    return nullptr;
  }
  const auto registration = model->second.causal_lm_factories.find(
      canonical_model_impl(implementation_key));
  if (registration == model->second.causal_lm_factories.end()) {
    return nullptr;
  }
  return registration->second.factory;
}

RecModelFactory ModelRegistry::get_rec_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].rec_model_factory;
}

CausalVLMFactory ModelRegistry::get_causalvlm_factory(const std::string& name) {
  return get_causalvlm_factory(name, "native");
}

CausalVLMFactory ModelRegistry::get_causalvlm_factory(
    const std::string& name,
    const std::string& implementation_key) {
  const ModelRegistry* instance = get_instance();
  const auto model = instance->model_registry_.find(name);
  if (model == instance->model_registry_.end()) {
    return nullptr;
  }
  const auto registration = model->second.causal_vlm_factories.find(
      canonical_model_impl(implementation_key));
  if (registration == model->second.causal_vlm_factories.end()) {
    return nullptr;
  }
  return registration->second.factory;
}

DiTModelFactory ModelRegistry::get_dit_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].dit_model_factory;
}

MultimodalProcessorFactory ModelRegistry::get_multimodal_processor_factory(
    const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].multimodal_processor_factory;
}

ModelArgsLoader ModelRegistry::get_model_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].model_args_loader;
}

QuantArgsLoader ModelRegistry::get_quant_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].quant_args_loader;
}

TokenizerArgsLoader ModelRegistry::get_tokenizer_args_loader(
    const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].tokenizer_args_loader;
}

bool ModelRegistry::has_dit_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();
  const auto it = instance->model_registry_.find(name);
  if (it == instance->model_registry_.end()) {
    return false;
  }
  return it->second.dit_model_factory != nullptr;
}

namespace util {

namespace {

std::string try_resolve_from_component_key(const std::string& key) {
  if (key.empty()) {
    return {};
  }
  if (ModelRegistry::has_dit_model_factory(key)) {
    return key;
  }

  auto try_prefix = [](const std::string& prefix) -> std::string {
    if (ModelRegistry::has_dit_model_factory(prefix)) {
      return prefix;
    }
    for (const char* suffix : {"_dlm", "_dit", "_diffusion", "_model"}) {
      const std::string candidate = prefix + suffix;
      if (ModelRegistry::has_dit_model_factory(candidate)) {
        return candidate;
      }
    }
    return {};
  };

  if (key.size() > 4 && key.substr(key.size() - 4) == "_dit") {
    if (std::string resolved = try_prefix(key.substr(0, key.size() - 4));
        !resolved.empty()) {
      return resolved;
    }
  }
  if (key.size() > 4 && key.substr(key.size() - 4) == "_vae") {
    if (std::string resolved = try_prefix(key.substr(0, key.size() - 4));
        !resolved.empty()) {
      return resolved;
    }
  }
  return {};
}

}  // namespace

std::string resolve_dit_pipeline_type(
    const std::vector<DitDiscoveredComponent>& components) {
  if (components.empty()) {
    return {};
  }

  for (const auto& component : components) {
    if (std::string resolved =
            try_resolve_from_component_key(component.component_type);
        !resolved.empty()) {
      return resolved;
    }
    if (component.name != component.component_type) {
      if (std::string resolved = try_resolve_from_component_key(component.name);
          !resolved.empty()) {
        return resolved;
      }
    }
  }

  std::string component_summary;
  for (const auto& component : components) {
    if (!component_summary.empty()) {
      component_summary += "; ";
    }
    component_summary +=
        component.name + " (model_type=" + component.component_type + ")";
  }
  LOG(FATAL) << "Unable to resolve a registered DiT pipeline type from "
                "discovered components: "
             << component_summary;
  return {};
}

}  // namespace util

void ModelRegistry::register_mtp_shared_weight_owner(const std::string& name) {
  get_instance()->model_registry_[name].owns_mtp_shared_weights = true;
}

bool ModelRegistry::owns_mtp_shared_weights(const std::string& name) {
  const auto& models = get_instance()->model_registry_;
  const auto entry = models.find(name);
  return entry != models.end() && entry->second.owns_mtp_shared_weights;
}

void ModelRegistry::register_mtp_args_adapter(const std::string& name,
                                              MtpArgsAdapter adapter) {
  get_instance()->model_registry_[name].mtp_args_adapter = std::move(adapter);
}

void ModelRegistry::configure_mtp_args(ModelArgs& args,
                                       std::string_view algorithm,
                                       bool is_draft_engine,
                                       bool is_python_model) {
  if (!is_draft_engine || !SpeculativeConfig::is_mtp_algorithm(algorithm)) {
    return;
  }
  const auto& models = get_instance()->model_registry_;
  const auto entry =
      models.find(canonical_model_registration_name(args.model_type()));
  if (entry == models.end() || !entry->second.mtp_args_adapter) {
    return;
  }
  entry->second.mtp_args_adapter(args, is_python_model);
}

std::string ModelRegistry::get_model_backend(const std::string& name) {
  ModelRegistry* instance = get_instance();
  auto it = instance->model_backend_.find(name);
  if (it != instance->model_backend_.end()) {
    return it->second;
  }
  const std::string canonical_name = canonical_model_registration_name(name);
  it = instance->model_backend_.find(canonical_name);
  return it == instance->model_backend_.end() ? "" : it->second;
}

std::unique_ptr<MultimodalProcessorBase> create_multimodal_processor(
    const ModelArgs& model_args,
    std::shared_ptr<Tokenizer> tokenizer,
    int64_t max_cache_items,
    const TokenizerArgs& tokenizer_args) {
  const std::string& model_type = model_args.model_type();
  std::string resolved_name;
  std::string error_message;
  CHECK(resolve_model_registration_name(
      model_type, &resolved_name, &error_message))
      << error_message;

  MultimodalProcessorFactory multimodal_processor_factory =
      ModelRegistry::get_multimodal_processor_factory(resolved_name);
  CHECK(multimodal_processor_factory != nullptr)
      << "Missing multimodal processor for model type: " << model_type;
  std::unique_ptr<MultimodalProcessorBase> processor =
      multimodal_processor_factory(
          model_args, std::move(tokenizer), tokenizer_args);
  if (max_cache_items == 0) {
    return processor;
  }
  return std::make_unique<CacheableMultimodalProcessor>(std::move(processor),
                                                        max_cache_items);
}

std::unique_ptr<CausalLM> create_llm_model(const ModelContext& context) {
  if (!validate_implementation_selection(context)) {
    return nullptr;
  }
  const std::string& model_impl = context.get_model_impl();
  const std::string& extension_impl = context.get_extension_impl();
  if (!extension_impl.empty() &&
      ExtensionRegistry::get_instance().has_implementation(extension_impl,
                                                           "causal_lm")) {
    if (ModelConfig::is_python_model_impl(model_impl)) {
      LOG(ERROR) << "Native model extensions require model_impl=native";
      return nullptr;
    }
    const std::string direct_name = canonical_model_registration_name(
        context.get_model_args().model_type());
    auto direct_factory =
        ModelRegistry::get_causallm_factory(direct_name, extension_impl);
    if (!direct_factory) {
      LOG(ERROR) << "Missing causal_lm extension for model type: "
                 << direct_name << " with extension_impl=" << extension_impl;
      return nullptr;
    }
    return direct_factory(context);
  }
  std::string resolved_name;
  std::string error_message;
  if (ModelConfig::is_python_model_impl(model_impl)) {
    resolved_name = canonical_model_registration_name(
        context.get_model_args().model_type());
  } else if (!resolve_model_registration_name(
                 context.get_model_args().model_type(),
                 &resolved_name,
                 &error_message)) {
    LOG(ERROR) << error_message;
    return nullptr;
  }

  auto factory = ModelRegistry::get_causallm_factory(resolved_name, model_impl);
  if (factory) {
    return factory(context);
  }

#if defined(USE_CUDA) || defined(USE_NPU)
  if (ModelConfig::is_python_model_impl(model_impl)) {
    return std::make_unique<PyCausalLM>(context);
  }
#else
  if (ModelConfig::is_python_model_impl(model_impl)) {
    LOG(ERROR) << "--model_impl=python is only supported on CUDA/NPU builds.";
    return nullptr;
  }
#endif

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type()
             << " for model_impl=" << canonical_model_impl(model_impl);

  return nullptr;
}

std::unique_ptr<RecModel> create_rec_model(const ModelContext& context) {
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration_name(context.get_model_args().model_type(),
                                       &resolved_name,
                                       &error_message)) {
    LOG(ERROR) << error_message;
    return nullptr;
  }

  auto factory = ModelRegistry::get_rec_model_factory(resolved_name);
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported rec model type: "
             << context.get_model_args().model_type();

  return nullptr;
}

std::unique_ptr<CausalVLM> create_vlm_model(const ModelContext& context) {
  if (!validate_implementation_selection(context)) {
    return nullptr;
  }
  const std::string& model_impl = context.get_model_impl();
  const std::string& extension_impl = context.get_extension_impl();
  if (!extension_impl.empty() &&
      ExtensionRegistry::get_instance().has_implementation(extension_impl,
                                                           "causal_vlm")) {
    if (ModelConfig::is_python_model_impl(model_impl)) {
      LOG(ERROR) << "Native model extensions require model_impl=native";
      return nullptr;
    }
    const std::string direct_name = canonical_model_registration_name(
        context.get_model_args().model_type());
    auto direct_factory =
        ModelRegistry::get_causalvlm_factory(direct_name, extension_impl);
    if (!direct_factory) {
      LOG(ERROR) << "Missing causal_vlm extension for model type: "
                 << direct_name << " with extension_impl=" << extension_impl;
      return nullptr;
    }
    return direct_factory(context);
  }
  std::string resolved_name;
  std::string error_message;
  if (ModelConfig::is_python_model_impl(model_impl)) {
    resolved_name = canonical_model_registration_name(
        context.get_model_args().model_type());
  } else if (!resolve_model_registration_name(
                 context.get_model_args().model_type(),
                 &resolved_name,
                 &error_message)) {
    LOG(ERROR) << error_message;
    return nullptr;
  }

  auto factory =
      ModelRegistry::get_causalvlm_factory(resolved_name, model_impl);
  if (factory) {
    return factory(context);
  }

  if (ModelConfig::is_python_model_impl(model_impl)) {
    std::string python_model_name = context.get_model_args().model_type();
    if (is_glm5_next_target_model_type(python_model_name)) {
      python_model_name = "glm5_next_vl";
    }
    return std::make_unique<PyCausalLM>(context, std::move(python_model_name));
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type()
             << " for model_impl=" << canonical_model_impl(model_impl);

  return nullptr;
}

std::unique_ptr<DiTModel> create_dit_model(const DiTModelContext& context) {
  // get the factory function for the model type from model registry
  auto factory = ModelRegistry::get_dit_model_factory(context.model_type());
  if (factory) {
    return factory(context);
  }
  LOG(INFO) << "DiT Model type: " << context.model_type();
  LOG(ERROR) << "Unsupported model type: " << context.model_type();

  return nullptr;
}

}  // namespace xllm
