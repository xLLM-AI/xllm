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

#include <torch/torch.h>

#include "common/macros.h"
#include "executor_impl.h"
#include "framework/model/causal_lm.h"
#include "options.h"

namespace xllm {

class ExecutorImplFactory {
 public:
  using Creator =
      std::function<std::unique_ptr<ExecutorImpl>(CausalLM*,
                                                  const ModelArgs&,
                                                  const torch::Device&,
                                                  const runtime::Options&)>;

  static ExecutorImplFactory& get_instance();

  bool register_creator(const std::string& name, Creator creator);

  std::unique_ptr<ExecutorImpl> create_executor_impl(
      CausalLM* model,
      const ModelArgs& args,
      const torch::Device& device,
      const runtime::Options& options,
      const std::string& backend);

  DISALLOW_COPY_AND_ASSIGN(ExecutorImplFactory);

 private:
  ExecutorImplFactory() = default;

  ~ExecutorImplFactory() = default;

  std::unordered_map<std::string, Creator> creators_;
};

#define REGISTER_EXECUTOR(backend, class_type)                                 \
  namespace {                                                                  \
  bool class_type##_registered = []() -> bool {                                \
    return ExecutorImplFactory::get_instance().register_creator(               \
        backend,                                                               \
        [](CausalLM* model,                                                    \
           const ModelArgs& args,                                              \
           const torch::Device& device,                                        \
           const runtime::Options& options) -> std::unique_ptr<ExecutorImpl> { \
          return std::make_unique<class_type>(model, args, device, options);   \
        });                                                                    \
  }();                                                                         \
  }

}  // namespace xllm
