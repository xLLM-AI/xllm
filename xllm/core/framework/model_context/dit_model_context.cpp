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

#include "core/framework/model_context/dit_model_context.h"

#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
// #include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include <torch_npu/csrc/libs/init_npu.h>
#endif

namespace xllm {

namespace {

DiTComponentType get_dit_component_type(const std::string& component) {
  if (component == "transformer" || component == "transformer_2" ||
      component == "cola_dit") {
    return DiTComponentType::TRANSFORMER;
  }
  if (component == "vae" || component == "audio_vae" ||
      component == "cola_vae") {
    return DiTComponentType::VAE;
  }
  if (component == "text_encoder" || component == "text_encoder_2" ||
      component == "processor" || component == "image_processor") {
    return DiTComponentType::TEXT_ENCODER;
  }
  if (component == "scheduler" || component == "audio_scheduler") {
    return DiTComponentType::SCHEDULER;
  }
  return DiTComponentType::UNKNOWN;
}

bool is_vlm_component(const std::string& component, const RawModelArgs& args) {
  // Image processors are represented by the vision half of VLM arguments.
  // This keeps processor configuration in the same typed component bundle as
  // the VLM text encoder that consumes it.
  if (component == "processor" || component == "image_processor") {
    return true;
  }

  if (component != "text_encoder" && component != "text_encoder_2") {
    return false;
  }

  std::string model_type = args.model_type();
  std::transform(model_type.begin(),
                 model_type.end(),
                 model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return model_type.find("vl") != std::string::npos;
}

std::unordered_map<std::string, std::shared_ptr<ModelArgs>> make_model_args(
    const std::unordered_map<std::string, RawModelArgs>& model_args) {
  std::unordered_map<std::string, std::shared_ptr<ModelArgs>> views;
  views.reserve(model_args.size());
  for (const auto& [component, args] : model_args) {
    const DiTComponentType component_type = get_dit_component_type(component);
    if (component_type == DiTComponentType::UNKNOWN) {
      continue;
    }
    const ModelFamily family =
        is_vlm_component(component, args) ? ModelFamily::VLM : ModelFamily::DIT;
    views.emplace(component,
                  std::make_shared<ModelArgs>(args, family, component_type));
  }
  return views;
}

}  // namespace

DiTModelContext::DiTModelContext(
    const ParallelArgs& input_parallel_args,
    std::unordered_map<std::string, RawModelArgs> model_args,
    std::unordered_map<std::string, QuantArgs> quant_args,
    const torch::TensorOptions& tensor_options,
    const DiTCacheConfig& dit_config,
    std::string model_type,
    std::string registration_name)
    : parallel_args_(input_parallel_args),
      model_args_(make_model_args(model_args)),
      quant_args_(std::move(quant_args)),
      tensor_options_(tensor_options),
      dit_config_(dit_config),
      model_type_(std::move(model_type)),
      registration_name_(std::move(registration_name)) {
#if defined(USE_NPU)
  int32_t device_id = tensor_options.device().index();
  void* stream = c10_npu::getCurrentNPUStream(device_id).stream();
  atb::CreateContext(&context_);
  context_->SetExecuteStream(stream);
  context_->SetAsyncTilingCopyStatus(true);
#endif
}

const QuantArgs& DiTModelContext::get_quant_args(
    const std::string& component) const {
  const auto& itor = quant_args_.find(component);
  if (itor != quant_args_.end()) {
    return itor->second;
  } else {
    LOG(FATAL) << "qunat args not found, component:" << component;
    static QuantArgs args;
    return args;
  }
}

const DiTComponentModelArgs& DiTModelContext::get_dit_model_args(
    const std::string& component) const& {
  const auto& iterator = model_args_.find(component);
  if (iterator != model_args_.end()) {
    return iterator->second->dit_model_args();
  }
  LOG(FATAL) << "DiT model args not found, component: " << component;
  static const DiTComponentModelArgs args{RawModelArgs()};
  return args;
}

const VlmModelArgs& DiTModelContext::get_vlm_model_args(
    const std::string& component) const& {
  const auto iterator = model_args_.find(component);
  if (iterator != model_args_.end()) {
    CHECK(iterator->second->family() == ModelFamily::VLM)
        << "Component does not contain VLM arguments: " << component;
    return iterator->second->vlm_model_args();
  }
  LOG(FATAL) << "VLM model args not found, component: " << component;
  static const VlmModelArgs args;
  return args;
}

ModelContext DiTModelContext::get_model_context(
    const std::string& component,
    ModelFamily expected_family) const {
  const auto storage = model_args_.find(component);
  CHECK(storage != model_args_.end())
      << "DiT model args storage not found, component: " << component;
  const ModelFamily family = storage->second->family();
  CHECK(expected_family == ModelFamily::UNKNOWN || expected_family == family)
      << "Unexpected model family for component " << component << ": "
      << static_cast<int>(family) << ", expected "
      << static_cast<int>(expected_family);
#if defined(USE_NPU)
  return ModelContext(
      parallel_args_,
      storage->second,
      get_quant_args(component),
      tensor_options_,
      context_,
      {family, registration_name_, get_dit_component_type(component)});
#else
  return ModelContext(
      parallel_args_,
      storage->second,
      get_quant_args(component),
      tensor_options_,
      {family, registration_name_, get_dit_component_type(component)});
#endif
}

}  // namespace xllm
