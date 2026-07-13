/* Copyright 2026 The xLLM Authors.

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

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "executor_impl_factory.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "runtime/executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class PyCausalLM;

class __attribute__((visibility("hidden"))) PyExecutorImpl final
    : public ExecutorImpl {
 public:
  PyExecutorImpl(CausalLM* model,
                 const ModelArgs& args,
                 const torch::Device& device,
                 const runtime::Options& options);

  ~PyExecutorImpl() override;

  ForwardInput prepare_inputs(Batch& batch) override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

 private:
  PyCausalLM* py_causal_lm_;
  ModelArgs args_;
  runtime::Options options_;
  bool enable_mla_ = false;

  pybind11::object py_executor_;
  bool kv_bound_ = false;
  int64_t kv_layer_count_ = 0;
};

REGISTER_EXECUTOR("python", PyExecutorImpl);

}  // namespace xllm
