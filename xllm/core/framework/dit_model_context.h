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

#if defined(USE_NPU)
#include <acl/acl.h>
#endif

#include <memory>

#include "core/framework/dit_cache/dit_cache_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/framework/quant_args.h"
#include "framework/parallel_state/parallel_args.h"

namespace xllm {

class DiTModelContext {
 public:
  DiTModelContext() : parallel_args_(1, 1, nullptr) {};

  DiTModelContext(const ParallelArgs& input_parallel_args,
                  const std::unordered_map<std::string, ModelArgs>& model_args,
                  const std::unordered_map<std::string, QuantArgs>& quant_args,
                  const torch::TensorOptions& tensor_options,
                  const DiTCacheConfig& dit_config,
                  const std::string& model_type);

  const ModelArgs& get_model_args(const std::string& component) const;

  const QuantArgs& get_quant_args(const std::string& component) const;

  bool has_component(const std::string& component) const {
    return model_args_.count(component) > 0;
  }

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  ModelContext get_model_context(const std::string& component) const;
#endif

  const ParallelArgs& get_parallel_args() const { return parallel_args_; }

  const torch::TensorOptions& get_tensor_options() const {
    return tensor_options_;
  }

  const std::string& model_type() const { return model_type_; }

  const DiTCacheConfig& get_dit_config() const { return dit_config_; }

#if defined(USE_NPU)
  const atb::Context* get_atb_context() const { return context_; }
#endif

 private:
  std::unordered_map<std::string, ModelArgs> model_args_;
  std::unordered_map<std::string, QuantArgs> quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions tensor_options_;
  DiTCacheConfig dit_config_;
  std::string model_type_;

#if defined(USE_NPU)
  // used for npu atb
  atb::Context* context_;
#endif
};

}  // namespace xllm
