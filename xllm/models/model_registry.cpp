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

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <utility>

#include "core/common/extension_registry.h"
#include "core/common/options.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_loader/dit_model_discovery.h"
#include "core/framework/model_loader/hf_model_config.h"
#include "core/platform/platform.h"
#include "llm/py_causal_lm.h"
#include "models.h"
#include "models/dit/py_dit_model.h"
#include "processors/cacheable_multimodal_processor.h"
#include "processors/multimodal_processor_pipeline.h"
#include "processors/text_prompt_processor.h"

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

}  // anonymous namespace

namespace xllm {

namespace {

constexpr char kCausalLMComponent[] = "causal_lm";

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
constexpr char kTorchBackend[] = "TORCH";

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
                                    effective_npu_kernel_backend,
                                    resolved_name,
                                    error_message);
}

bool resolve_model_registration(const std::string& model_type,
                                const std::string& requested_npu_kernel_backend,
                                const std::string& extension_impl,
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
  if (!extension_impl.empty()) {
    const auto& extensions = ExtensionRegistry::get_instance();
    if (!extensions.has_implementation(extension_impl)) {
      if (error_message != nullptr) {
        *error_message = "Unknown extension_impl=" + extension_impl;
      }
      return false;
    }
    if (extensions.has_implementation(extension_impl, kCausalLMComponent)) {
      const bool selected_extension_model =
          static_cast<bool>(ModelRegistry::get_model_factory(
              canonical_model_type, extension_impl));
      if (!selected_extension_model) {
        if (error_message != nullptr) {
          *error_message = "Missing causal_lm extension for model type: " +
                           canonical_model_type +
                           " with extension_impl=" + extension_impl;
        }
        return false;
      }
    }
  }

  const std::string backend = requested_npu_kernel_backend.empty()
                                  ? kAutoBackend
                                  : requested_npu_kernel_backend;
  if (backend != kAutoBackend && backend != kTorchBackend) {
    if (error_message != nullptr) {
      *error_message = "Unsupported --npu_kernel_backend=" + backend +
                       ". Supported values: AUTO, TORCH.";
    }
    return false;
  }

  if (effective_npu_kernel_backend != nullptr) {
    *effective_npu_kernel_backend = kTorchBackend;
  }
  *resolved_name = canonical_model_type;
  return true;
#else
  *resolved_name = canonical_model_type;
  return true;
#endif
}

#if defined(USE_NPU)
void resolve_npu_kernel_backend(Options& options) {
  // Python executors use the Torch kernel implementation.
  if (ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl())) {
    options.npu_kernel_backend(kTorchBackend);
    KernelConfig::get_instance().npu_kernel_backend(kTorchBackend);
    LOG(INFO) << "Forced npu_kernel_backend=TORCH for python model_impl";
    return;
  }

  const std::string model_type =
      get_model_type(options.model_path(), options.backend());
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;
  if (!resolve_model_registration(model_type,
                                  options.npu_kernel_backend(),
                                  options.extension_impl(),
                                  &effective_backend,
                                  &resolved_name,
                                  &error_message)) {
    LOG(FATAL) << error_message;
  }

  options.npu_kernel_backend(effective_backend);
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
  static const std::unordered_set<std::string> kCpCapableModels = {
      "deepseek_v4", "deepseek_v4_mtp"};
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
  return kCpCapableModels.contains(resolved_name);
}

ModelRegistry* ModelRegistry::get_instance() {
  static ModelRegistry registry;

  return &registry;
}

bool ModelRegistry::register_model_factory(
    std::string name,
    std::string implementation,
    ModelFactory factory,
    int32_t priority,
    MultimodalProcessorFactory processor_factory) {
  CHECK(factory != nullptr) << "Cannot register an empty causal LM factory";
  ModelRegistry* instance = get_instance();
  auto& factories =
      instance->model_registry_[std::move(name)].causal_lm_factories;
  implementation = canonical_model_impl(std::move(implementation));
  auto [it, inserted] = factories.try_emplace(implementation);
  auto& registration = it->second;
  if (!inserted && priority <= registration.priority) {
    return false;
  }
  if (processor_factory) {
    registration.processor_factory_ = std::move(processor_factory);
  }

  if (implementation != "native" && implementation != "python") {
    ExtensionRegistry::get_instance().register_implementation(
        implementation, kCausalLMComponent);
  }
  registration.factory = std::move(factory);
  registration.priority = priority;
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

void ModelRegistry::register_dcp_capability(const std::string& name) {
  get_instance()->model_registry_[name].supports_dcp = true;
}

bool ModelRegistry::supports_dcp(const std::string& name) {
  const ModelRegistry* instance = get_instance();
  const auto it = instance->model_registry_.find(name);
  return it != instance->model_registry_.end() && it->second.supports_dcp;
}

ModelFactory ModelRegistry::get_model_factory(
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

DiTModelFactory ModelRegistry::get_dit_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].dit_model_factory;
}

MultimodalProcessorFactory ModelRegistry::get_multimodal_processor_factory(
    const std::string& name,
    const std::string& extension_impl) {
  const auto& models = get_instance()->model_registry_;
  const auto model = models.find(name);
  if (model == models.end()) {
    return {};
  }

  const auto& factories = model->second.causal_lm_factories;
  if (!extension_impl.empty()) {
    const auto extension_factory = factories.find(extension_impl);
    if (extension_factory != factories.end() &&
        extension_factory->second.processor_factory_) {
      return extension_factory->second.processor_factory_;
    }
  }
  const auto native = factories.find(/*key=*/"native");
  if (native != factories.end() && native->second.processor_factory_) {
    return native->second.processor_factory_;
  }
  return model->second.multimodal_processor_factory;
}

void ModelRegistry::register_speech_model_factory(const std::string& name,
                                                  SpeechModelFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].speech_model_factory != nullptr) {
    SAFE_LOG_WARNING("speech model factory for " << name
                                                 << " already registered.");
  } else {
    instance->model_registry_[name].speech_model_factory = std::move(factory);
  }
}

SpeechModelFactory ModelRegistry::get_speech_model_factory(
    const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].speech_model_factory;
}

ModelArgsLoader ModelRegistry::get_model_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].model_args_loader;
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

void ModelRegistry::register_mtp_args_adapter(const std::string& name,
                                              MtpArgsAdapter adapter) {
  get_instance()->model_registry_[name].mtp_args_adapter = std::move(adapter);
}

void ModelRegistry::register_mtp_capabilities(
    const std::string& name,
    MtpModelCapabilities capabilities) {
  get_instance()->model_registry_[name].mtp_capabilities = capabilities;
}

MtpModelCapabilities ModelRegistry::get_mtp_capabilities(
    const std::string& name) {
  const auto& models = get_instance()->model_registry_;
  const auto entry = models.find(name);
  if (entry == models.end()) {
    return MtpModelCapabilities();
  }
  return entry->second.mtp_capabilities;
}

void ModelRegistry::register_graph_kv_capacity(const std::string& name,
                                               GraphKVCapacityAdapter adapter) {
  get_instance()->model_registry_[name].graph_kv_capacity_adapter =
      std::move(adapter);
}

int64_t ModelRegistry::get_graph_kv_capacity(const std::string& name,
                                             const ModelArgs& args,
                                             int64_t default_capacity) {
  const auto& registry = get_instance()->model_registry_;
  const auto it = registry.find(name);
  if (it == registry.end() || !it->second.graph_kv_capacity_adapter) {
    return default_capacity;
  }
  return it->second.graph_kv_capacity_adapter(args);
}

void ModelRegistry::configure_mtp_args(ModelArgs& args,
                                       std::string_view algorithm,
                                       bool is_draft_engine,
                                       bool is_python_model) {
  if (!is_draft_engine || !SpeculativeConfig::is_mtp_algorithm(algorithm)) {
    return;
  }
  const int64_t num_layers = args.n_layers();
  const auto& models = get_instance()->model_registry_;
  const auto entry =
      models.find(canonical_model_registration_name(args.model_type()));
  if (entry != models.end() && entry->second.mtp_args_adapter) {
    entry->second.mtp_args_adapter(args, is_python_model);
  }
  const int32_t num_draft_layers = args.num_nextn_predict_layers();
  if (num_draft_layers <= 0) {
    return;
  }
  if (args.n_layers() != num_draft_layers) {
    args.mtp_start_layer_idx(args.n_layers()).n_layers(num_draft_layers);
  }
  const auto& index_topk_pattern = args.index_topk_pattern();
  if (num_draft_layers < num_layers &&
      index_topk_pattern.size() == static_cast<size_t>(num_layers)) {
    args.index_topk_pattern(
        index_topk_pattern.substr(/*pos=*/0, num_draft_layers));
  }
  args.first_k_dense_replace(
      std::min(args.first_k_dense_replace(), num_draft_layers));
}

std::string ModelRegistry::get_model_backend(const std::string& name) {
  ModelRegistry* instance = get_instance();
  auto it = instance->model_backend_.find(name);
  if (it != instance->model_backend_.end()) {
    return it->second;
  }
  const std::string canonical_name = canonical_model_registration_name(name);
  it = instance->model_backend_.find(canonical_name);
  if (it != instance->model_backend_.end()) {
    return it->second;
  }
  for (const auto* model_name : {&name, &canonical_name}) {
    const auto model = instance->model_registry_.find(*model_name);
    if (model == instance->model_registry_.end()) {
      continue;
    }
    if (model->second.multimodal_processor_factory ||
        std::ranges::any_of(
            model->second.causal_lm_factories, [](const auto& model_factory) {
              return static_cast<bool>(model_factory.second.processor_factory_);
            })) {
      return "vlm";
    }
    if (!model->second.causal_lm_factories.empty()) {
      return "llm";
    }
  }
  return "";
}

std::unique_ptr<MultimodalProcessorBase> create_multimodal_processor(
    const std::string& model_path,
    const ModelArgs& model_args,
    std::shared_ptr<Tokenizer> tokenizer,
    int64_t max_cache_items,
    const TokenizerArgs& tokenizer_args,
    int32_t max_sequence_length,
    const std::string& extension_impl) {
  const std::string& model_type = model_args.model_type();
  std::string resolved_name;
  std::string error_message;
  CHECK(resolve_model_registration_name(
      model_type, &resolved_name, &error_message))
      << error_message;

  MultimodalProcessorFactory multimodal_processor_factory =
      ModelRegistry::get_multimodal_processor_factory(resolved_name,
                                                      extension_impl);
  if (multimodal_processor_factory == nullptr &&
      ModelConfig::is_python_model_impl(
          ModelConfig::get_instance().model_impl()) &&
      ModelRegistry::get_model_backend(resolved_name) == "llm") {
    return std::make_unique<MultimodalProcessor<TextPromptProcessor>>(
        model_args,
        std::move(tokenizer),
        tokenizer_args,
        max_sequence_length,
        model_path);
  }
  CHECK(multimodal_processor_factory != nullptr)
      << "Missing multimodal processor for model type: " << model_type;
  std::unique_ptr<MultimodalProcessorBase> processor =
      multimodal_processor_factory(model_path,
                                   model_args,
                                   std::move(tokenizer),
                                   tokenizer_args,
                                   max_sequence_length);
  if (max_cache_items == 0) {
    return processor;
  }
  return std::make_unique<CacheableMultimodalProcessor>(std::move(processor),
                                                        max_cache_items);
}

CausalModel create_causal_model(const ModelContext& context) {
  if (!validate_implementation_selection(context)) {
    return {};
  }
  const std::string& model_impl = context.get_model_impl();
  const std::string& extension_impl = context.get_extension_impl();
  if (!extension_impl.empty() &&
      ExtensionRegistry::get_instance().has_implementation(
          extension_impl, kCausalLMComponent)) {
    if (ModelConfig::is_python_model_impl(model_impl)) {
      LOG(ERROR) << "Native model extensions require model_impl=native";
      return {};
    }
    const std::string direct_name = canonical_model_registration_name(
        context.get_model_args().model_type());
    auto direct_factory =
        ModelRegistry::get_model_factory(direct_name, extension_impl);
    if (!direct_factory) {
      LOG(ERROR) << "Missing causal_lm extension for model type: "
                 << direct_name << " with extension_impl=" << extension_impl;
      return {};
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
    return {};
  }

  auto factory = ModelRegistry::get_model_factory(resolved_name, model_impl);
  if (factory) {
    return factory(context);
  }

  if (ModelConfig::is_python_model_impl(model_impl)) {
#if defined(USE_CUDA) || defined(USE_NPU)
    return std::make_unique<PyCausalLM>(context);
#else
    LOG(ERROR) << "--model_impl=python is only supported on CUDA/NPU builds.";
    return {};
#endif
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type()
             << " for model_impl=" << canonical_model_impl(model_impl);

  return {};
}

std::unique_ptr<DiTModel> create_dit_model(const DiTModelContext& context) {
  if (ModelConfig::is_python_model_impl(context.model_impl())) {
    if (context.model_type() == "WanImageToVideoPipeline" &&
        Platform::is_npu() &&
        context.get_tensor_options().device().type() ==
            Platform::type_torch()) {
      const auto& config = DiTConfig::get_instance();
      const auto& parallel = context.get_parallel_args();
      CHECK(context.get_dit_config().selected_policy == PolicyType::None)
          << "Python Wan2.1 I2V requires dit_cache_policy=None";
      CHECK(!config.dit_laser_attention_enabled() &&
            !config.dit_sparse_attention_enabled() &&
            !ExecutionConfig::get_instance().enable_graph() &&
            !LoadConfig::get_instance().enable_rolling_load())
          << "Python Wan2.1 I2V requires dense eager execution";
      CHECK(parallel.cfg_size() == 1 && parallel.dp_size() == 1 &&
            parallel.vae_size() == 1 && parallel.text_encoder_tp_size() == 1)
          << "Python Wan2.1 I2V supports TP/SP with CFG1, DP1, VAE1 and text "
             "encoder TP1";
      CHECK_EQ(config.max_requests_per_batch(), 1);
      CHECK(context.get_quant_args("transformer").quant_method().empty())
          << "Python Wan2.1 I2V requires dense floating point weights";
      return std::make_unique<PyDiTModel>(context);
    }
    if (context.model_type() == "MiniMaxH3ModularPipeline" &&
        Platform::is_npu() &&
        context.get_tensor_options().device().type() ==
            Platform::type_torch()) {
      const auto& config = DiTConfig::get_instance();
      const auto& parallel = context.get_parallel_args();
      CHECK_EQ(config.dit_h3_task_type(), "fl2va")
          << "Python MiniMax-H3 supports T2VA with fl2va weights";
      CHECK(context.get_dit_config().selected_policy == PolicyType::None)
          << "Python MiniMax-H3 requires dit_cache_policy=None";
      CHECK(!config.dit_sparse_attention_enabled() &&
            !ExecutionConfig::get_instance().enable_graph() &&
            !LoadConfig::get_instance().enable_rolling_load())
          << "Python MiniMax-H3 requires dense eager execution";
      CHECK(parallel.world_size() == 4 && parallel.tp_size() == 2 &&
            parallel.sp_size() == 2 && parallel.vae_size() == 4 &&
            parallel.cfg_size() == 1 && parallel.dp_size() == 1 &&
            parallel.dit_tp_group_ != nullptr &&
            parallel.dit_sp_group_ != nullptr &&
            parallel.dit_vae_group_ != nullptr)
          << "Python MiniMax-H3 requires TP2, SP2, VAE4, CFG1 and DP1";
      CHECK_EQ(config.max_requests_per_batch(), 1);
      CHECK(context.get_quant_args("transformer").quant_method().empty())
          << "Python MiniMax-H3 requires dense floating point weights";
      return std::make_unique<PyDiTModel>(context);
    }
    LOG(ERROR) << "Python DiT is unavailable for " << context.model_type()
               << " on " << context.get_tensor_options().device();
    return nullptr;
  }
  auto factory = ModelRegistry::get_dit_model_factory(context.model_type());
  if (factory) {
    return factory(context);
  }
  LOG(INFO) << "DiT Model type: " << context.model_type();
  LOG(ERROR) << "Unsupported model type: " << context.model_type();

  return nullptr;
}

}  // namespace xllm
