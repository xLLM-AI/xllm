/* Copyright 2025-2026 The xLLM Authors.

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

#include "layers/npu/buffer/atb_workspace.h"
#endif

#include <memory>
#include <string>

#include "core/common/flash_comm1_context.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/quant_args.h"
#include "framework/parallel_state/parallel_args.h"

namespace xllm {

// Internal optimization configuration.
// This struct holds all optimization techniques that are automatically
// determined from model and quantization arguments and other internal factors,
// not set directly by users.
struct OptimizationConfig {
  // Enable use of fused computation kernels.
  bool enable_fused_spec_kernel = false;
  bool enable_fused_mla_kernel = false;
  bool enable_fused_indexer_qk = false;

  // Broadcast speculative-decoding sampling results across the TP consensus
  // group so every rank adopts rank 0's accepted/draft tokens and avoids
  // per-rank RNG divergence in the draft decoder's TP all-reduce.
  bool enable_spec_token_broadcast = false;

  // we can detailize this part later. for example:
  // PROPERTY(bool, enable_fused_mlp_kernel) = false;
};

class ModelContext {
 public:
  ModelContext() : parallel_args_(1, 1, nullptr) {};

  ModelContext(const ParallelArgs& input_parallel_args,
               const ModelArgs& model_args,
               const QuantArgs& quant_args,
               const torch::TensorOptions& tensor_options);

#if defined(USE_NPU)
  ModelContext(const ParallelArgs& input_parallel_args,
               const ModelArgs& model_args,
               const QuantArgs& quant_args,
               const torch::TensorOptions& tensor_options,
               atb::Context* context);
#endif

  const ModelArgs& get_model_args() const { return model_args_; }

  const QuantArgs& get_quant_args() const { return quant_args_; }

  const ParallelArgs& get_parallel_args() const { return parallel_args_; }

  const torch::TensorOptions& get_tensor_options() const {
    return tensor_options_;
  }

  const OptimizationConfig& get_optimization_config() const {
    return optimization_config_;
  }

  const FlashComm1Options& get_flash_comm1_options() const {
    return flash_comm1_options_;
  }

  void set_flash_comm1_options(const FlashComm1Options& options) {
    flash_comm1_options_ = options;
  }

  ModelContext with_parallel_args(const ParallelArgs& parallel_args) const;

#if defined(USE_NPU)
  const atb::Context* get_atb_context() const { return context_; }
  std::shared_ptr<AtbWorkspace> get_atb_workspace() const {
    return atb_workspace_;
  }
#endif

  void set_encoder_embedding_mode(bool encoder_embedding_mode) {
    model_args_.encoder_embedding_mode() = encoder_embedding_mode;
  }

  const std::string& get_model_id() const { return model_id_; }
  void set_model_id(const std::string& model_id) { model_id_ = model_id; }

  const std::string& get_model_impl() const { return model_impl_; }
  void set_model_impl(const std::string& model_impl) {
    model_impl_ = model_impl;
  }

 private:
  // derive optimization config based on model args, quant args and other
  // factors
  void derive_optimization_config();

  std::string model_id_;  // Model identifier for XTensor multi-model support
  std::string model_impl_;
  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions tensor_options_;
  OptimizationConfig optimization_config_;
  FlashComm1Options flash_comm1_options_;

#if defined(USE_NPU)
  // used for npu atb
  atb::Context* context_;
  std::shared_ptr<AtbWorkspace> atb_workspace_;
#endif
};

}  // namespace xllm
