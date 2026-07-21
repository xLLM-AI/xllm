/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "model_registry.h"

#include <glog/logging.h>

#include <iostream>
#include <unordered_set>

#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/util/dit_model_discovery.h"
#include "llm/py_causal_lm.h"
#include "models.h"

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

#if defined(USE_NPU)
constexpr char kAutoBackend[] = "AUTO";
constexpr char kAtbBackend[] = "ATB";
constexpr char kTorchBackend[] = "TORCH";

bool is_torch_only_model_type(const std::string& model_type) {
  static const std::unordered_set<std::string> kTorchOnlyModelTypes = {
      "deepseek_v4",
      "deepseek_v4_mtp",
      "qwen3_5",
      "qwen3_5_text",
      "qwen3_5_moe",
      "qwen3_5_moe_text",
      "qwen3_5_mtp",
      "qwen3_5_moe_mtp",
      "qwen3_next",
      "minimax_m2"};
  return kTorchOnlyModelTypes.count(model_type) != 0;
}
#endif

}  // namespace

bool resolve_model_registration(const std::string& model_type,
                                const std::string& requested_npu_kernel_backend,
                                std::string* effective_npu_kernel_backend,
                                std::string* resolved_name,
                                std::string* error_message) {
  if (resolved_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "resolved_name must not be null";
    }
    return false;
  }

#if defined(USE_NPU)
  const std::string backend = requested_npu_kernel_backend.empty()
                                  ? kAutoBackend
                                  : requested_npu_kernel_backend;
  if (backend != kAutoBackend && backend != kAtbBackend &&
      backend != kTorchBackend) {
    if (error_message != nullptr) {
      *error_message = "Unsupported --npu_kernel_backend=" + backend +
                       ". Supported values: AUTO, ATB, TORCH.";
    }
    return false;
  }

  std::string effective_backend = backend;
  if (backend == kAutoBackend) {
    effective_backend =
        is_torch_only_model_type(model_type) ? kTorchBackend : kAtbBackend;
  } else if (model_type == "qwen3" || model_type == "qwen3_moe") {
    // qwen3/qwen3_moe support both backends.
  } else if (is_torch_only_model_type(model_type)) {
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
  if (model_type == "qwen3" && effective_backend == kAtbBackend) {
    *resolved_name = "qwen3_atb";
  } else if (model_type == "qwen3_moe" && effective_backend == kAtbBackend) {
    *resolved_name = "qwen3_moe_atb";
  } else {
    *resolved_name = model_type;
  }
  return true;
#else
  *resolved_name = model_type;
  return true;
#endif
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

ModelRegistry* ModelRegistry::get_instance() {
  static ModelRegistry registry;

  return &registry;
}

void ModelRegistry::register_causallm_factory(const std::string& name,
                                              CausalLMFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].causal_lm_factory != nullptr) {
    SAFE_LOG_WARNING("causal lm factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].causal_lm_factory = factory;
    instance->model_backend_[name] = "llm";
  }
}

void ModelRegistry::register_rec_model_factory(const std::string& name,
                                               RecModelFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].rec_model_factory != nullptr) {
    SAFE_LOG_WARNING("rec model factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].rec_model_factory = factory;
    instance->model_backend_[name] = "rec";
  }
}

void ModelRegistry::register_causalvlm_factory(const std::string& name,
                                               CausalVLMFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].causal_vlm_factory != nullptr) {
    SAFE_LOG_WARNING("causal vlm factory for " << name
                                               << " already registered.");
  } else {
    instance->model_registry_[name].causal_vlm_factory = factory;
    instance->model_backend_[name] = "vlm";
  }
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

CausalLMFactory ModelRegistry::get_causallm_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].causal_lm_factory;
}

RecModelFactory ModelRegistry::get_rec_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].rec_model_factory;
}

CausalVLMFactory ModelRegistry::get_causalvlm_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].causal_vlm_factory;
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

std::string ModelRegistry::get_model_backend(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_backend_[name];
}

std::unique_ptr<CausalLM> create_llm_model(const ModelContext& context) {
  // Python model executor: build the graph via the embedded interpreter instead
  // of resolving a C++ model class from the registry.
  const auto& model_impl = context.get_model_impl();
#if defined(USE_CUDA)
  if (ModelConfig::is_python_model_impl(model_impl)) {
    return std::make_unique<PyCausalLM>(context);
  }
#else
  if (ModelConfig::is_python_model_impl(model_impl)) {
    LOG(ERROR) << "--model_impl=python is only supported on CUDA builds.";
    return nullptr;
  }
#endif

  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration_name(context.get_model_args().model_type(),
                                       &resolved_name,
                                       &error_message)) {
    LOG(ERROR) << error_message;
    return nullptr;
  }

  auto factory = ModelRegistry::get_causallm_factory(resolved_name);
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type();

  return nullptr;
}

std::unique_ptr<CausalLM> create_rec_model(const ModelContext& context) {
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
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration_name(context.get_model_args().model_type(),
                                       &resolved_name,
                                       &error_message)) {
    LOG(ERROR) << error_message;
    return nullptr;
  }

  auto factory = ModelRegistry::get_causalvlm_factory(resolved_name);
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type();

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
