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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "core/common/flash_comm1_context.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/quantization/quant_args.h"

namespace xllm {

class ModelStreamRegistry;

// Full batch capacities for model buffers and communication resources. Worker
// startup supplies effective limits; standalone contexts keep legacy defaults.
struct ModelBatchLimits {
  int32_t max_tokens_per_batch = 10240;
  int32_t max_seqs_per_batch = 200;
};

// Internal optimization configuration.
// This struct holds all optimization techniques that are automatically
// determined from model and quantization arguments and other internal factors,
// not set directly by users.
struct OptimizationConfig {
  // Enable use of fused computation kernels.
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
               const torch::TensorOptions& tensor_options,
               int32_t num_speculative_tokens = 0,
               bool enable_chunked_prefill = true,
               ModelBatchLimits batch_limits = {});

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

  const std::shared_ptr<ModelStreamRegistry>& stream_registry() const {
    return stream_registry_;
  }

  void set_flash_comm1_options(const FlashComm1Options& options) {
    flash_comm1_options_ = options;
  }

  int32_t num_speculative_tokens() const { return num_speculative_tokens_; }

  bool enable_chunked_prefill() const { return enable_chunked_prefill_; }

  const ModelBatchLimits& batch_limits() const { return batch_limits_; }

  ModelContext with_parallel_args(const ParallelArgs& parallel_args) const;

  ModelContext with_quant_args(const QuantArgs& quant_args) const;

  void set_encoder_embedding_mode(bool encoder_embedding_mode) {
    model_args_.encoder_embedding_mode() = encoder_embedding_mode;
  }

  const std::string& get_model_id() const { return model_id_; }
  void set_model_id(const std::string& model_id) { model_id_ = model_id; }

  const std::string& get_model_impl() const { return model_impl_; }
  void set_model_impl(const std::string& model_impl) {
    model_impl_ = model_impl;
  }

  const std::string& kv_cache_dtype() const { return kv_cache_dtype_; }
  void set_kv_cache_dtype(std::string dtype) {
    kv_cache_dtype_ = std::move(dtype);
  }

  const std::string& get_extension_impl() const { return extension_impl_; }
  void set_extension_impl(std::string extension_impl) {
    extension_impl_ = std::move(extension_impl);
  }

 private:
  // derive optimization config based on model args, quant args and other
  // factors
  void derive_optimization_config();

  std::string
      model_id_;  // Model identifier for VirtualMemory multi-model support
  std::string model_impl_;
  std::string extension_impl_;
  std::string kv_cache_dtype_ = "auto";
  ModelArgs model_args_;
  int32_t num_speculative_tokens_ = 0;
  bool enable_chunked_prefill_ = true;
  ModelBatchLimits batch_limits_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions tensor_options_;
  OptimizationConfig optimization_config_;
  FlashComm1Options flash_comm1_options_;
  std::shared_ptr<ModelStreamRegistry> stream_registry_;
};

}  // namespace xllm
